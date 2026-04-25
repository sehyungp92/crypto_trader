"""SidecarForwarder — background thread that polls JSONL files and forwards to relay."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

# JSONL file types to poll
_EVENT_FILES = (
    "instrumented_trades",
    "missed_opportunities",
    "daily_snapshots",
    "errors",
)


class SidecarForwarder:
    """Background thread that polls JSONL files and forwards events to relay.

    Matches the reference architecture's sidecar pattern:
    - Reads new lines since last watermark
    - Signs payload with HMAC-SHA256
    - Gzip compresses if > 1KB
    - POSTs to relay /events endpoint
    - Persists watermarks for crash recovery
    """

    def __init__(
        self,
        state_dir: Path,
        relay_url: str,
        bot_id: str,
        shared_secret: str,
        poll_interval: float = 5.0,
        batch_size: int = 50,
    ) -> None:
        self._state_dir = state_dir
        self._relay_url = relay_url.rstrip("/")
        self._bot_id = bot_id
        self._secret = shared_secret.encode()
        self._poll_interval = poll_interval
        self._batch_size = batch_size

        self._watermarks: dict[str, int] = {}
        self._watermark_file = state_dir / ".sidecar_watermarks.json"
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the sidecar polling thread."""
        self._load_watermarks()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="sidecar")
        self._thread.start()
        log.info("sidecar.started", relay_url=self._relay_url)

    def stop(self) -> None:
        """Signal the polling thread to stop and wait."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        log.info("sidecar.stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _poll_loop(self) -> None:
        """Main loop: read new JSONL lines, batch, sign, POST to relay."""
        while not self._stop_event.is_set():
            for event_type in _EVENT_FILES:
                try:
                    path = self._state_dir / f"{event_type}.jsonl"
                    if not path.exists():
                        continue
                    new_events, new_offset = self._read_since_watermark(path, event_type)
                    if new_events:
                        if self._send_batch(new_events, event_type):
                            # Only advance watermark after successful send
                            self._watermarks[event_type] = new_offset
                            self._save_watermarks()
                except Exception:
                    log.exception("sidecar.poll_error", event_type=event_type)

            self._stop_event.wait(self._poll_interval)

    def _read_since_watermark(self, path: Path, key: str) -> tuple[list[dict], int]:
        """Read new lines since last watermark offset.

        Returns (events, new_offset). Caller is responsible for advancing
        the watermark only after successful delivery.
        """
        offset = self._watermarks.get(key, 0)
        events: list[dict] = []
        new_offset = offset

        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            log.warning("sidecar.bad_json_line", file=key)
                    if len(events) >= self._batch_size:
                        break
                new_offset = f.tell()

        except Exception:
            log.exception("sidecar.read_error", file=key)

        return events, new_offset

    def _send_batch(self, events: list[dict], event_type: str) -> bool:
        """Sign with HMAC-SHA256, gzip if needed, POST to relay /events.

        Returns True if the batch was delivered successfully.
        """
        try:
            import urllib.request

            payload = {
                "bot_id": self._bot_id,
                "event_type": event_type,
                "events": events,
            }

            # Canonical JSON for HMAC
            canonical = json.dumps(payload, sort_keys=True, default=str)
            signature = hmac.new(self._secret, canonical.encode(), hashlib.sha256).hexdigest()

            body = canonical.encode()
            headers = {
                "Content-Type": "application/json",
                "X-Bot-Id": self._bot_id,
                "X-Signature": signature,
            }

            # Gzip if > 1KB
            if len(body) > 1024:
                body = gzip.compress(body)
                headers["Content-Encoding"] = "gzip"

            url = f"{self._relay_url}/events"
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        if resp.status < 300:
                            log.debug("sidecar.batch_sent",
                                     event_type=event_type, count=len(events))
                            return True
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt, 60)
                        log.warning("sidecar.retry", attempt=attempt + 1, wait=wait, error=str(e))
                        time.sleep(wait)
                    else:
                        log.error("sidecar.send_failed", event_type=event_type, error=str(e))

        except Exception:
            log.exception("sidecar.batch_error", event_type=event_type)

        return False

    def _load_watermarks(self) -> None:
        """Load watermarks from disk."""
        if self._watermark_file.exists():
            try:
                with open(self._watermark_file, "r", encoding="utf-8") as f:
                    self._watermarks = json.load(f)
            except Exception:
                log.warning("sidecar.watermark_load_failed")
                self._watermarks = {}

    def _save_watermarks(self) -> None:
        """Persist watermarks atomically for crash recovery."""
        tmp = self._watermark_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._watermarks, f)
            os.replace(tmp, self._watermark_file)
        except Exception:
            log.exception("sidecar.watermark_save_failed")
            if tmp.exists():
                tmp.unlink()
