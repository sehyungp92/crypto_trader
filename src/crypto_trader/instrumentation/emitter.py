"""EventEmitter — dispatches events to registered sinks."""

from __future__ import annotations

import structlog

from crypto_trader.instrumentation.sinks import Sink
from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)

log = structlog.get_logger()


class EventEmitter:
    """Dispatches instrumentation events to all registered sinks."""

    def __init__(self) -> None:
        self._sinks: list[Sink] = []

    def add_sink(self, sink: Sink) -> None:
        self._sinks.append(sink)

    def emit_trade(self, event: InstrumentedTradeEvent) -> None:
        for sink in self._sinks:
            try:
                sink.write_trade(event)
            except Exception:
                log.exception("emitter.trade_failed", sink=type(sink).__name__)

    def emit_missed(self, event: MissedOpportunityEvent) -> None:
        for sink in self._sinks:
            try:
                sink.write_missed(event)
            except Exception:
                log.exception("emitter.missed_failed", sink=type(sink).__name__)

    def emit_daily(self, event: DailySnapshot) -> None:
        for sink in self._sinks:
            try:
                sink.write_daily(event)
            except Exception:
                log.exception("emitter.daily_failed", sink=type(sink).__name__)

    def emit_error(self, event: ErrorEvent) -> None:
        for sink in self._sinks:
            try:
                sink.write_error(event)
            except Exception:
                log.exception("emitter.error_failed", sink=type(sink).__name__)

    def emit_funnel(self, event: PipelineFunnelSnapshot) -> None:
        for sink in self._sinks:
            try:
                sink.write_funnel(event)
            except Exception:
                log.exception("emitter.funnel_failed", sink=type(sink).__name__)

    def emit_health_report(self, event: HealthReportSnapshot) -> None:
        for sink in self._sinks:
            try:
                sink.write_health_report(event)
            except Exception:
                log.exception("emitter.health_report_failed", sink=type(sink).__name__)
