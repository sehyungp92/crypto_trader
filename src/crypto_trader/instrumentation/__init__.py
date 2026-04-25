"""Instrumentation package for trading assistant integration."""

from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.emitter import EventEmitter
from crypto_trader.instrumentation.quality import ProcessQualityScorer
from crypto_trader.instrumentation.sinks import InMemorySink, JsonlSink, Sink
from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    EventMetadata,
    FilterDecision,
    InstrumentedTradeEvent,
    MarketContext,
    MissedOpportunityEvent,
    ROOT_CAUSE_TAXONOMY,
    SignalFactor,
)

__all__ = [
    "DailySnapshot",
    "ErrorEvent",
    "EventEmitter",
    "EventMetadata",
    "FilterDecision",
    "InMemorySink",
    "InstrumentationCollector",
    "InstrumentedTradeEvent",
    "JsonlSink",
    "MarketContext",
    "MissedOpportunityEvent",
    "ProcessQualityScorer",
    "ROOT_CAUSE_TAXONOMY",
    "SignalFactor",
    "Sink",
]
