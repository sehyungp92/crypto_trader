"""SQLite WAL-mode event store for relay service."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    bot_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    received_at TEXT NOT NULL,
    acked INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_events_bot_acked ON events(bot_id, acked);
CREATE INDEX IF NOT EXISTS idx_events_received ON events(received_at);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


class RelayStore:
    """SQLite-backed event buffer with dedup and watermark tracking."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._start_mono = time.monotonic()
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def insert_events(self, bot_id: str, event_type: str, events: list[dict]) -> int:
        """Insert events, deduplicating by event_id. Returns count inserted."""
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0

        for event in events:
            event_id = _extract_event_id(event)
            payload = json.dumps(event, default=str)

            try:
                self._conn.execute(
                    "INSERT INTO events (event_id, bot_id, event_type, payload, received_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (event_id, bot_id, event_type, payload, now),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # Duplicate event_id — skip
                pass

        self._conn.commit()
        return inserted

    def get_events(
        self,
        since_id: int = 0,
        limit: int = 100,
        bot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get unacked events since watermark ID."""
        if bot_id:
            rows = self._conn.execute(
                "SELECT id, event_id, bot_id, event_type, payload, received_at "
                "FROM events WHERE id > ? AND acked = 0 AND bot_id = ? "
                "ORDER BY id LIMIT ?",
                (since_id, bot_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, event_id, bot_id, event_type, payload, received_at "
                "FROM events WHERE id > ? AND acked = 0 "
                "ORDER BY id LIMIT ?",
                (since_id, limit),
            ).fetchall()

        return [
            {
                "id": r[0],
                "event_id": r[1],
                "bot_id": r[2],
                "event_type": r[3],
                "payload": json.loads(r[4]),
                "received_at": r[5],
            }
            for r in rows
        ]

    def ack_events(self, event_ids: list[str]) -> int:
        """Mark events as acknowledged. Returns count acked."""
        if not event_ids:
            return 0

        placeholders = ",".join("?" * len(event_ids))
        cursor = self._conn.execute(
            f"UPDATE events SET acked = 1 WHERE event_id IN ({placeholders})",
            event_ids,
        )
        self._conn.commit()
        return cursor.rowcount

    def get_health(self) -> dict[str, Any]:
        """Get store health stats."""
        total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        pending = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE acked = 0"
        ).fetchone()[0]
        acked = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE acked = 1"
        ).fetchone()[0]

        # Per-bot stats
        bot_rows = self._conn.execute(
            "SELECT bot_id, COUNT(*), SUM(CASE WHEN acked = 0 THEN 1 ELSE 0 END) "
            "FROM events GROUP BY bot_id"
        ).fetchall()
        per_bot = {
            r[0]: {"total": r[1], "pending": r[2] or 0}
            for r in bot_rows
        }

        last_event_rows = self._conn.execute(
            "SELECT bot_id, MAX(received_at) FROM events GROUP BY bot_id"
        ).fetchall()
        last_event_per_bot = {
            r[0]: r[1]
            for r in last_event_rows
        }

        oldest_pending_row = self._conn.execute(
            "SELECT MIN(received_at) FROM events WHERE acked = 0"
        ).fetchone()
        oldest_pending_age_seconds = None
        if oldest_pending_row and oldest_pending_row[0]:
            try:
                oldest_pending_at = datetime.fromisoformat(oldest_pending_row[0])
                oldest_pending_age_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - oldest_pending_at).total_seconds(),
                )
            except ValueError:
                oldest_pending_age_seconds = None

        return {
            "status": "ok",
            "pending_events": pending,
            "total_events": total,
            "pending": pending,
            "acked": acked,
            "per_bot": per_bot,
            "per_bot_pending": {
                bot_id: stats["pending"]
                for bot_id, stats in per_bot.items()
            },
            "last_event_per_bot": last_event_per_bot,
            "oldest_pending_age_seconds": oldest_pending_age_seconds,
            "db_size_bytes": self._db_size_bytes(),
            "uptime_seconds": time.monotonic() - self._start_mono,
        }

    def _db_size_bytes(self) -> int:
        """Return SQLite main/WAL/SHM bytes for disk-pressure monitoring."""
        db_path = Path(self._db_path)
        paths = [
            db_path,
            Path(f"{self._db_path}-wal"),
            Path(f"{self._db_path}-shm"),
        ]
        return sum(path.stat().st_size for path in paths if path.exists())

    def purge_acked(self, older_than_hours: int = 24) -> int:
        """Delete acked events older than N hours. Returns count deleted."""
        cutoff = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff = (cutoff - timedelta(hours=older_than_hours)).isoformat()

        cursor = self._conn.execute(
            "DELETE FROM events WHERE acked = 1 AND received_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount


def _extract_event_id(event: dict) -> str:
    """Extract event_id from event payload."""
    # Try metadata.event_id first
    metadata = event.get("metadata", {})
    if isinstance(metadata, dict):
        eid = metadata.get("event_id")
        if eid:
            return str(eid)

    # Fallback to top-level event_id
    eid = event.get("event_id")
    if eid:
        return str(eid)

    # Generate from hash of payload
    import hashlib
    raw = json.dumps(event, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
