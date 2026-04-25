"""PostgreSQL event sink for instrumentation output."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import structlog

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)

log = structlog.get_logger()


class PostgresSink:
    """Writes instrumentation events to PostgreSQL.

    Implements the Sink protocol (6 methods) for auto-dispatch via EventEmitter,
    plus 2 direct methods (write_equity, upsert_positions) called from engine.

    All methods swallow exceptions — never blocks the engine.
    """

    def __init__(self, dsn: str) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(dsn, min_size=1, max_size=3)

    # ------------------------------------------------------------------
    # Sink protocol methods (called via EventEmitter.add_sink)
    # ------------------------------------------------------------------

    def write_trade(self, event: InstrumentedTradeEvent) -> None:
        """INSERT trade, idempotent via ON CONFLICT DO NOTHING."""
        try:
            net_pnl = event.pnl - event.funding_paid
            confluences = json.dumps(event.confluences) if event.confluences else "[]"
            market_ctx = (
                json.dumps(event.market_context.to_dict())
                if event.market_context
                else None
            )
            strategy_id = event.metadata.strategy_id if event.metadata else "unknown"
            confirmation_type = event.entry_signal or None

            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO trades (
                        trade_id, strategy_id, symbol, direction,
                        entry_time, exit_time, entry_price, exit_price,
                        position_size, pnl, net_pnl, r_multiple,
                        commission, funding_paid, setup_grade, exit_reason,
                        confirmation_type, entry_method, confluences,
                        mae_r, mfe_r, exit_efficiency, market_context
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s::jsonb,
                        %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (trade_id) DO NOTHING
                    """,
                    (
                        event.trade_id,
                        strategy_id,
                        event.pair,
                        event.side,
                        event.entry_time,
                        event.exit_time,
                        event.entry_price,
                        event.exit_price,
                        event.position_size,
                        event.pnl,
                        net_pnl,
                        event.r_multiple,
                        event.commission,
                        event.funding_paid,
                        event.setup_grade,
                        event.exit_reason,
                        confirmation_type,
                        event.entry_method,
                        confluences,
                        event.mae_r,
                        event.mfe_r,
                        event.exit_efficiency,
                        market_ctx,
                    ),
                )
        except Exception:
            log.exception("postgres_sink.write_trade_failed")

    def write_daily(self, event: DailySnapshot) -> None:
        """UPSERT daily snapshot."""
        try:
            per_strategy = json.dumps(
                event.per_strategy_summary if event.per_strategy_summary else {}
            )
            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO daily_snapshots (
                        trade_date, total_trades, win_count, loss_count,
                        gross_pnl, net_pnl, max_drawdown_pct,
                        sharpe_rolling_30d, sortino_rolling_30d,
                        per_strategy_summary
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (trade_date) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        win_count = EXCLUDED.win_count,
                        loss_count = EXCLUDED.loss_count,
                        gross_pnl = EXCLUDED.gross_pnl,
                        net_pnl = EXCLUDED.net_pnl,
                        max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                        sharpe_rolling_30d = EXCLUDED.sharpe_rolling_30d,
                        sortino_rolling_30d = EXCLUDED.sortino_rolling_30d,
                        per_strategy_summary = EXCLUDED.per_strategy_summary
                    """,
                    (
                        event.date,
                        event.total_trades,
                        event.win_count,
                        event.loss_count,
                        event.gross_pnl,
                        event.net_pnl,
                        event.max_drawdown_pct,
                        event.sharpe_rolling_30d,
                        event.sortino_rolling_30d,
                        per_strategy,
                    ),
                )
        except Exception:
            log.exception("postgres_sink.write_daily_failed")

    def write_health_report(self, event: HealthReportSnapshot) -> None:
        """INSERT health snapshot."""
        try:
            report = event.report or {}
            assessment = report.get("assessment", "unknown")
            uptime_sec = report.get("uptime_sec")
            alerts = json.dumps(report.get("alerts", []))
            report_json = json.dumps(report, default=str)

            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO health_snapshots (
                        timestamp, assessment, uptime_sec, alerts, report
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        event.timestamp,
                        assessment,
                        uptime_sec,
                        alerts,
                        report_json,
                    ),
                )
        except Exception:
            log.exception("postgres_sink.write_health_report_failed")

    def write_missed(self, event: MissedOpportunityEvent) -> None:
        """No-op: missed opportunities stay in JSONL only."""
        pass

    def write_error(self, event: ErrorEvent) -> None:
        """No-op: errors stay in JSONL only."""
        pass

    def write_funnel(self, event: PipelineFunnelSnapshot) -> None:
        """No-op: funnels stay in JSONL only."""
        pass

    # ------------------------------------------------------------------
    # Direct methods (called from engine, NOT part of Sink protocol)
    # ------------------------------------------------------------------

    def write_equity(self, equity: float, timestamp: datetime) -> None:
        """Insert equity snapshot. Called every 5 min (~288/day)."""
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO equity_snapshots (timestamp, equity) VALUES (%s, %s)",
                    (timestamp, equity),
                )
        except Exception:
            log.exception("postgres_sink.write_equity_failed")

    def upsert_positions(self, positions: list[dict[str, Any]]) -> None:
        """Full-sync open positions: DELETE all then INSERT current.

        Max ~9 rows (PortfolioConfig.max_total_positions), so this is fast.
        """
        try:
            with self._pool.connection() as conn:
                with conn.transaction():
                    conn.execute("DELETE FROM positions")
                    for pos in positions:
                        conn.execute(
                            """
                            INSERT INTO positions (
                                strategy_id, symbol, direction, qty, avg_entry,
                                unrealized_pnl, risk_r, stop_price, entry_time
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                pos.get("strategy_id", "unknown"),
                                pos["symbol"],
                                pos["direction"],
                                pos["qty"],
                                pos["avg_entry"],
                                pos.get("unrealized_pnl", 0.0),
                                pos.get("risk_r", 0.0),
                                pos.get("stop_price"),
                                pos.get("entry_time"),
                            ),
                        )
        except Exception:
            log.exception("postgres_sink.upsert_positions_failed")

    def close(self) -> None:
        """Close connection pool."""
        try:
            self._pool.close()
        except Exception:
            log.exception("postgres_sink.close_failed")
