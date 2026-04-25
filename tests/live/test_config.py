"""Tests for live trading configuration."""

from pathlib import Path

import pytest

from crypto_trader.live.config import LiveConfig


class TestLiveConfig:
    def test_defaults(self):
        cfg = LiveConfig()
        assert cfg.is_testnet is True
        assert cfg.poll_interval_sec == 15.0
        assert cfg.symbols == ["BTC", "ETH", "SOL"]
        assert cfg.max_slippage_pct == 0.005

    def test_validate_empty(self):
        cfg = LiveConfig()
        errors = cfg.validate()
        assert len(errors) >= 1
        assert any("wallet_address" in e for e in errors)

    def test_validate_valid(self):
        cfg = LiveConfig(wallet_address="0x123", private_key="0xabc")
        errors = cfg.validate()
        assert len(errors) == 0

    def test_validate_read_only(self):
        cfg = LiveConfig(wallet_address="0x123", private_key=None)
        errors = cfg.validate()
        assert any("read-only" in e for e in errors)

    def test_base_url_testnet(self):
        cfg = LiveConfig(is_testnet=True)
        assert "testnet" in cfg.base_url

    def test_base_url_mainnet(self):
        cfg = LiveConfig(is_testnet=False)
        assert "testnet" not in cfg.base_url

    def test_from_dict(self):
        d = {
            "wallet_address": "0x123",
            "is_testnet": False,
            "symbols": ["BTC"],
            "poll_interval_sec": 30.0,
            "strategy_configs": {"momentum": "configs/momentum.json"},
        }
        cfg = LiveConfig.from_dict(d)
        assert cfg.wallet_address == "0x123"
        assert cfg.is_testnet is False
        assert cfg.symbols == ["BTC"]
        assert cfg.poll_interval_sec == 30.0
        assert cfg.strategy_configs["momentum"] == Path("configs/momentum.json")

    def test_to_dict_excludes_private_key(self):
        cfg = LiveConfig(wallet_address="0x123", private_key="secret")
        d = cfg.to_dict()
        assert "private_key" not in d
        assert d["wallet_address"] == "0x123"

    def test_roundtrip(self):
        cfg = LiveConfig(
            wallet_address="0x123",
            is_testnet=True,
            symbols=["BTC", "ETH"],
            poll_interval_sec=20.0,
        )
        d = cfg.to_dict()
        cfg2 = LiveConfig.from_dict(d)
        assert cfg2.wallet_address == cfg.wallet_address
        assert cfg2.symbols == cfg.symbols
        assert cfg2.poll_interval_sec == cfg.poll_interval_sec
