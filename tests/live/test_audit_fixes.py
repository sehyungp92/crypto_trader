"""Behavioral regressions for remaining live audit fixes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crypto_trader.core.engine import MultiTimeFrameBars, StrategyContext
from crypto_trader.core.events import EventBus
from crypto_trader.core.models import Bar, Order, OrderType, SetupGrade, Side, TimeFrame
from crypto_trader.instrumentation.collector import InstrumentationCollector
from crypto_trader.instrumentation.types import EventMetadata, MissedOpportunityEvent
from crypto_trader.live.config import LiveConfig


def _indicator_snapshot():
    from crypto_trader.strategy.momentum.indicators import IndicatorSnapshot

    return IndicatorSnapshot(
        ema_fast=100.0,
        ema_mid=99.0,
        ema_slow=98.0,
        ema_fast_arr=np.array([100.0]),
        ema_mid_arr=np.array([99.0]),
        ema_slow_arr=np.array([98.0]),
        adx=25.0,
        di_plus=20.0,
        di_minus=10.0,
        adx_rising=True,
        atr=5.0,
        atr_avg=5.0,
        rsi=55.0,
        volume_ma=1000.0,
    )


class _WarmupGateStrategy:
    def __init__(self) -> None:
        self.symbols = ["BTC"]
        self.window_checks: list[bool] = []
        self.ctx_config = None
        self._collector = MagicMock()
        self._collector.pipeline = MagicMock()

    def on_init(self, ctx) -> None:
        self.ctx_config = ctx.config

    def on_bar(self, bar, ctx) -> None:
        start_date = getattr(ctx.config, "start_date", None)
        window_open = start_date is None or bar.timestamp >= start_date
        self.window_checks.append(window_open)
        if window_open:
            ctx.broker.submit_order(
                Order(
                    order_id="warmup_entry",
                    symbol=bar.symbol,
                    side=Side.LONG,
                    order_type=OrderType.MARKET,
                    qty=1.0,
                    tag="entry",
                )
            )

    def on_fill(self, fill, ctx) -> None:
        return None

    def on_shutdown(self, ctx) -> None:
        return None


class TestLiveWarmupBehavior:
    @pytest.mark.asyncio
    async def test_start_keeps_warmup_bars_outside_entry_window(self, tmp_path):
        from crypto_trader.live.engine import LiveEngine

        broker = MagicMock()
        broker.get_equity.return_value = 10_000.0
        broker.get_positions.return_value = []

        warmup_bars = [
            Bar(
                timestamp=datetime(2026, 4, 25, 8, 0, tzinfo=timezone.utc),
                symbol="BTC",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000.0,
                timeframe=TimeFrame.M15,
            ),
            Bar(
                timestamp=datetime(2026, 4, 25, 8, 15, tzinfo=timezone.utc),
                symbol="BTC",
                open=100.5,
                high=101.5,
                low=100.0,
                close=101.0,
                volume=1_100.0,
                timeframe=TimeFrame.M15,
            ),
        ]

        feed = MagicMock()
        feed.load_warmup_bars.return_value = warmup_bars

        strategy = _WarmupGateStrategy()
        strategy_cfg = SimpleNamespace(symbols=["BTC"])
        config = LiveConfig(
            wallet_address="0xabc",
            private_key="0xdef",
            symbols=["BTC"],
            state_dir=tmp_path,
            strategy_configs={"momentum": tmp_path / "momentum.json"},
        )

        with (
            patch("crypto_trader.live.engine.HyperliquidBroker", return_value=broker),
            patch("crypto_trader.live.engine.LiveFeed", return_value=feed),
            patch("crypto_trader.live.engine._create_strategy", return_value=(strategy, [TimeFrame.M15], TimeFrame.M15)),
            patch("hyperliquid.info.Info", return_value=MagicMock()),
            patch.object(LiveEngine, "_load_strategy_config", return_value=strategy_cfg),
        ):
            engine = LiveEngine(config)
            await engine.start()

        assert strategy.window_checks == [False, False]
        broker.submit_order.assert_not_called()
        assert not hasattr(strategy.ctx_config, "start_date")
        strategy._collector.flush_missed.assert_called_once_with()
        strategy._collector.pipeline.snapshot_and_reset.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_start_passes_asset_meta_cache_to_broker(self, tmp_path):
        from crypto_trader.live.engine import LiveEngine

        asset_meta_path = tmp_path / "asset_meta.json"
        asset_meta_path.write_text(
            json.dumps({
                "asset_index": {"BTC": 0},
                "tick_sizes": {"BTC": 0.5},
                "lot_sizes": {"BTC": 0.001},
            }),
            encoding="utf-8",
        )
        broker = MagicMock()
        broker.get_equity.return_value = 10_000.0
        broker.get_positions.return_value = []
        broker.get_open_orders.return_value = []
        feed = MagicMock()
        feed.load_warmup_bars.return_value = []
        strategy = _WarmupGateStrategy()
        strategy_cfg = SimpleNamespace(symbols=["BTC"])
        config = LiveConfig(
            wallet_address="0xabc",
            private_key="0xdef",
            symbols=["BTC"],
            state_dir=tmp_path / "state",
            strategy_configs={"momentum": tmp_path / "momentum.json"},
            asset_meta_path=asset_meta_path,
        )

        with (
            patch("crypto_trader.live.engine.HyperliquidBroker", return_value=broker) as broker_cls,
            patch("crypto_trader.live.engine.LiveFeed", return_value=feed),
            patch("crypto_trader.live.engine._create_strategy", return_value=(strategy, [TimeFrame.M15], TimeFrame.M15)),
            patch("hyperliquid.info.Info", return_value=MagicMock()),
            patch.object(LiveEngine, "_load_strategy_config", return_value=strategy_cfg),
        ):
            engine = LiveEngine(config)
            await engine.start()

        kwargs = broker_cls.call_args.kwargs
        assert kwargs["tick_sizes"] == {"BTC": 0.5}
        assert kwargs["lot_sizes"] == {"BTC": 0.001}


class TestLiveConfigRelayPlumbing:
    def test_round_trips_optional_bot_and_relay_fields(self, tmp_path):
        cfg = LiveConfig.from_dict({
            "wallet_address": "0xabc",
            "private_key": "0xdef",
            "symbols": ["BTC"],
            "state_dir": str(tmp_path),
            "strategy_configs": {"momentum": "config/strategies/momentum.json"},
            "bot_id": "paper_bot_01",
            "relay_url": "https://relay.example.com",
            "relay_secret": "secret",
        })

        data = cfg.to_dict()
        assert data["bot_id"] == "paper_bot_01"
        assert data["relay_url"] == "https://relay.example.com"
        assert data["relay_secret"] == "secret"

    def test_validate_requires_complete_relay_configuration(self):
        cfg = LiveConfig(
            wallet_address="0xabc",
            private_key="0xdef",
            symbols=["BTC"],
            relay_url="https://relay.example.com",
        )

        errors = cfg.validate()
        assert "bot_id is required when relay is configured" in errors
        assert "relay_secret is required when relay is configured" in errors

    def test_validate_allows_bot_id_without_relay(self):
        cfg = LiveConfig(
            wallet_address="0x" + "1" * 40,
            private_key="0x" + "2" * 64,
            symbols=["BTC"],
            bot_id="paper_bot_01",
        )

        assert cfg.validate() == []


class TestLiveHealthRelayStatus:
    def test_report_intervals_apply_minimum_floor(self):
        from crypto_trader.live.engine import LiveEngine

        engine = object.__new__(LiveEngine)
        engine._config = LiveConfig(
            health_report_interval_sec=30.0,
            funnel_report_interval_sec=30.0,
        )

        assert engine._health_report_interval() == 60.0
        assert engine._funnel_report_interval() == 60.0

        engine._config.health_report_interval_sec = 300.0
        engine._config.funnel_report_interval_sec = 3600.0
        assert engine._health_report_interval() == 300.0
        assert engine._funnel_report_interval() == 3600.0

    def test_relay_health_status_reports_disabled_without_sidecar(self):
        from crypto_trader.live.engine import LiveEngine

        engine = object.__new__(LiveEngine)
        engine._sidecar = None

        assert engine._relay_health_status() == {
            "enabled": False,
            "sidecar_running": False,
            "event_files": [],
        }

    def test_relay_health_status_maps_sidecar_fields(self):
        from crypto_trader.live.engine import LiveEngine

        engine = object.__new__(LiveEngine)
        sidecar = MagicMock()
        sidecar.status.return_value = {
            "enabled": True,
            "running": True,
            "event_files": ["pipeline_funnels", "health_reports"],
            "watermarks": {"pipeline_funnels": 123},
            "watermark_file": "/state/.sidecar_watermarks.json",
            "last_successful_send_at": "2026-05-10T00:00:00+00:00",
            "consecutive_send_failures": 0,
            "last_error": None,
        }
        engine._sidecar = sidecar

        status = engine._relay_health_status()

        assert status["enabled"] is True
        assert status["sidecar_running"] is True
        assert status["last_successful_send_at"] == "2026-05-10T00:00:00+00:00"
        assert status["consecutive_send_failures"] == 0
        assert status["watermarks"]["pipeline_funnels"] == 123

    def test_relay_health_status_survives_sidecar_status_error(self):
        from crypto_trader.live.engine import LiveEngine

        engine = object.__new__(LiveEngine)
        sidecar = MagicMock()
        sidecar.status.side_effect = RuntimeError("status unavailable")
        engine._sidecar = sidecar

        status = engine._relay_health_status()

        assert status["enabled"] is True
        assert status["sidecar_running"] is False
        assert "status unavailable" in status["status_error"]


class TestScaledRiskUnits:
    def test_momentum_full_risk_maps_to_one_r_unit(self):
        from crypto_trader.strategy.momentum.config import MomentumConfig
        from crypto_trader.strategy.momentum.strategy import MomentumStrategy

        cfg = MomentumConfig(symbols=["BTC"])
        strategy = MomentumStrategy(cfg)
        sizing, reason = strategy._sizer.compute(
            equity=10_000.0,
            entry_price=100.0,
            stop_distance=10.0,
            setup_grade=SetupGrade.A,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
        )

        assert reason == ""
        assert sizing is not None
        assert strategy._scaled_risk_units(sizing.risk_pct_actual, cfg.risk.risk_pct_a) == pytest.approx(1.0)

    def test_trend_reentry_risk_scale_reduces_r_units(self):
        from crypto_trader.strategy.trend.config import TrendConfig
        from crypto_trader.strategy.trend.strategy import TrendStrategy

        cfg = TrendConfig(symbols=["BTC"])
        cfg.reentry.risk_scale = 0.5
        strategy = TrendStrategy(cfg)
        sizing, reason = strategy._sizer.compute(
            equity=10_000.0,
            entry_price=100.0,
            stop_distance=5.0,
            grade=SetupGrade.A,
            symbol="BTC",
            open_positions=[],
            direction=Side.LONG,
            risk_scale=cfg.reentry.risk_scale,
        )

        assert reason == ""
        assert sizing is not None
        assert strategy._scaled_risk_units(sizing.risk_pct_actual, cfg.risk.risk_pct_a) == pytest.approx(0.5)

    def test_breakout_relaxed_body_entry_records_snapshot_and_scaled_risk(self):
        from crypto_trader.strategy.breakout.balance import BalanceZone
        from crypto_trader.strategy.breakout.confirmation import BreakoutConfirmation
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        from crypto_trader.strategy.breakout.setup import BreakoutSetupResult
        from crypto_trader.strategy.breakout.strategy import BreakoutStrategy

        cfg = BreakoutConfig(symbols=["BTC"])
        strategy = BreakoutStrategy(cfg)

        broker = MagicMock()
        broker.get_equity.return_value = 10_000.0
        broker.get_position.return_value = None
        broker.get_portfolio_snapshot.return_value = {
            "heat_R": 1.0,
            "heat_cap_R": 5.0,
            "open_risk_count": 1,
            "directional_risk_R": 0.5,
            "symbol_risk_R": 0.25,
            "portfolio_daily_pnl_R": 0.1,
            "strategy_daily_pnl_R": 0.05,
        }

        submitted_orders: list[Order] = []

        def _submit(order: Order) -> str:
            submitted_orders.append(order)
            return order.order_id

        broker.submit_order.side_effect = _submit

        ctx = StrategyContext(
            broker=broker,
            clock=MagicMock(),
            bars=MultiTimeFrameBars(),
            events=EventBus(),
        )
        strategy.on_init(ctx)
        strategy._stop_placer.compute = MagicMock(return_value=95.0)
        strategy._context_analyzer.evaluate = MagicMock(
            return_value=SimpleNamespace(reasons=[], direction=None)
        )

        bar = Bar(
            timestamp=datetime(2026, 4, 25, 9, 0, tzinfo=timezone.utc),
            symbol="BTC",
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=1_200.0,
            timeframe=TimeFrame.M30,
        )
        setup = BreakoutSetupResult(
            grade=SetupGrade.B,
            is_a_plus=False,
            direction=Side.LONG,
            balance_zone=BalanceZone(
                center=99.0,
                upper=100.0,
                lower=95.0,
                bars_in_zone=8,
                touches=3,
                formation_bar_idx=10,
                volume_contracting=False,
                width_atr=1.0,
            ),
            breakout_price=100.0,
            lvn_runway_atr=2.0,
            confluences=("volume_surge", "lvn_runway", "balance_duration", "ema_support", "multi_hvn"),
            room_r=1.8,
            volume_mult=1.4,
            body_ratio=0.4,
            signal_variant="relaxed_body",
            risk_scale=0.5,
        )
        confirmation = BreakoutConfirmation(
            model="model1_close",
            trigger_price=bar.close,
            bar_index=0,
            volume_confirmed=True,
        )

        entered = strategy._execute_entry(
            bar=bar,
            sym="BTC",
            ctx=ctx,
            setup=setup,
            confirmation=confirmation,
            m30_ind=_indicator_snapshot(),
            retest_bar=None,
        )

        assert entered is True
        assert submitted_orders
        assert submitted_orders[0].metadata["risk_R"] == pytest.approx(0.5)
        assert strategy._collector._entry_portfolio_state["BTC"] == broker.get_portfolio_snapshot.return_value


class TestLiveMissedBackfill:
    def test_engine_backfills_pending_missed_and_reemits_updates(self):
        from crypto_trader.live.engine import LiveEngine, _StrategySlot

        ts = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
        event = MissedOpportunityEvent(
            metadata=EventMetadata.create("bot", "momentum", ts, "missed_opportunity", "BTC"),
            pair="BTC",
            hypothetical_entry=100.0,
        )

        collector = InstrumentationCollector(strategy_id="momentum", bot_id="bot")
        collector._missed_buffer = [event]

        bars = MultiTimeFrameBars()
        for offset, close in ((1, 101.0), (4, 103.0), (24, 105.0)):
            bars.append(
                Bar(
                    timestamp=ts + timedelta(hours=offset),
                    symbol="BTC",
                    open=100.0,
                    high=close,
                    low=99.0,
                    close=close,
                    volume=1_000.0,
                    timeframe=TimeFrame.H1,
                )
            )

        slot = _StrategySlot(
            strategy_id="momentum",
            strategy=SimpleNamespace(symbols=["BTC"], _collector=collector),
            ctx=MagicMock(),
            bars=bars,
            subscribed_tfs={TimeFrame.H1},
            primary_tf=TimeFrame.H1,
        )

        engine = object.__new__(LiveEngine)
        engine._slots = [slot]
        engine._emitter = MagicMock()

        engine._drain_and_backfill_missed()

        assert event.backfill_status == "complete"
        assert event.outcome_24h == pytest.approx(5.0, abs=0.01)
        engine._emitter.emit_missed.assert_called_once_with(event)
        assert getattr(engine, "_pending_missed", {}) == {}
