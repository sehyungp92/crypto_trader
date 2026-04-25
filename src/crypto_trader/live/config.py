"""Live trading configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LiveConfig:
    """Configuration for live/paper trading on Hyperliquid."""

    wallet_address: str = ""
    private_key: str | None = None  # None = read-only mode
    is_testnet: bool = True
    poll_interval_sec: float = 15.0  # candle poll frequency
    fill_poll_interval_sec: float = 30.0
    equity_snapshot_interval_sec: float = 300.0  # 5 minutes
    health_check_interval_sec: float = 60.0
    rate_limit_per_sec: float = 5.0
    max_slippage_pct: float = 0.005  # 0.5% for market orders

    # Strategy configs
    strategy_configs: dict[str, Path] = field(default_factory=dict)
    portfolio_config_path: Path | None = None

    # Trading universe
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    state_dir: Path = field(default_factory=lambda: Path("data/live_state"))

    # Instrumentation / relay (optional)
    bot_id: str = ""
    relay_url: str = ""
    relay_secret: str = ""

    # PostgreSQL (optional; empty string = disabled)
    postgres_dsn: str = ""

    @property
    def base_url(self) -> str:
        from hyperliquid.utils import constants

        return constants.TESTNET_API_URL if self.is_testnet else constants.MAINNET_API_URL

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.wallet_address:
            errors.append("wallet_address is required")
        if self.private_key is None:
            errors.append("private_key is required for trading (None = read-only)")
        if not self.symbols:
            errors.append("at least one symbol required")
        if any((self.relay_url, self.relay_secret)):
            if not self.bot_id:
                errors.append("bot_id is required when relay is configured")
            if not self.relay_url:
                errors.append("relay_url is required when relay is configured")
            if not self.relay_secret:
                errors.append("relay_secret is required when relay is configured")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> LiveConfig:
        """Deserialize from dict."""
        strategy_configs = {
            k: Path(v) for k, v in d.get("strategy_configs", {}).items()
        }
        return cls(
            wallet_address=d.get("wallet_address", ""),
            private_key=d.get("private_key"),
            is_testnet=d.get("is_testnet", True),
            poll_interval_sec=d.get("poll_interval_sec", 15.0),
            fill_poll_interval_sec=d.get("fill_poll_interval_sec", 30.0),
            equity_snapshot_interval_sec=d.get("equity_snapshot_interval_sec", 300.0),
            health_check_interval_sec=d.get("health_check_interval_sec", 60.0),
            rate_limit_per_sec=d.get("rate_limit_per_sec", 5.0),
            max_slippage_pct=d.get("max_slippage_pct", 0.005),
            strategy_configs=strategy_configs,
            portfolio_config_path=Path(d["portfolio_config_path"]) if d.get("portfolio_config_path") else None,
            symbols=d.get("symbols", ["BTC", "ETH", "SOL"]),
            data_dir=Path(d.get("data_dir", "data")),
            state_dir=Path(d.get("state_dir", "data/live_state")),
            bot_id=d.get("bot_id", ""),
            relay_url=d.get("relay_url", ""),
            relay_secret=d.get("relay_secret", ""),
            postgres_dsn=os.environ.get("POSTGRES_DSN") or d.get("postgres_dsn", ""),
        )

    def to_dict(self) -> dict:
        """Serialize to dict (excludes private_key for safety)."""
        return {
            "wallet_address": self.wallet_address,
            "is_testnet": self.is_testnet,
            "poll_interval_sec": self.poll_interval_sec,
            "fill_poll_interval_sec": self.fill_poll_interval_sec,
            "equity_snapshot_interval_sec": self.equity_snapshot_interval_sec,
            "health_check_interval_sec": self.health_check_interval_sec,
            "rate_limit_per_sec": self.rate_limit_per_sec,
            "max_slippage_pct": self.max_slippage_pct,
            "strategy_configs": {k: str(v) for k, v in self.strategy_configs.items()},
            "portfolio_config_path": str(self.portfolio_config_path) if self.portfolio_config_path else None,
            "symbols": self.symbols,
            "data_dir": str(self.data_dir),
            "state_dir": str(self.state_dir),
            "bot_id": self.bot_id,
            "relay_url": self.relay_url,
            "relay_secret": self.relay_secret,
            "postgres_dsn": self.postgres_dsn,
        }
