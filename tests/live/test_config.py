"""Tests for live trading configuration."""

import json
from pathlib import Path

from crypto_trader.cli import _live_config_path_errors
from crypto_trader.live.config import LiveConfig

VALID_WALLET = "0x" + "1" * 40
VALID_PRIVATE_KEY = "0x" + "2" * 64
STRATEGY_IDS = ("momentum", "trend", "breakout")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_deployment_bundle(
    tmp_path: Path,
    *,
    root: Path = Path("output/portfolio/round_3"),
    strategy_ids: tuple[str, ...] = STRATEGY_IDS,
) -> Path:
    portfolio_ref = root / "recommended_portfolio_config.json"
    strategy_refs = {
        strategy_id: root / "recommended_strategy_configs" / f"{strategy_id}.json"
        for strategy_id in strategy_ids
    }
    _write_json(tmp_path / portfolio_ref, {"portfolio": "promoted", "strategies": list(strategy_ids)})
    for strategy_id, path in strategy_refs.items():
        _write_json(tmp_path / path, {"strategy": {"name": strategy_id, "version": "promoted"}})
    _write_json(
        tmp_path / "output" / "portfolio" / "rounds_manifest.json",
        {"rounds": [{"round": 1}, {"round": 2}, {"round": 3}]},
    )
    parity_ref = root / "parity_alignment.json"
    _write_json(
        tmp_path / parity_ref,
        {"portfolio_metric_replay": {"status": "matched", "max_abs_delta": 0.0, "tolerance": 1e-9}},
    )
    manifest_ref = root / "deployment_manifest.json"
    _write_json(
        tmp_path / manifest_ref,
        {
            "schema_version": 1,
            "required_strategy_ids": list(strategy_ids),
            "portfolio_config_path": str(portfolio_ref).replace("\\", "/"),
            "strategy_configs": {
                strategy_id: str(path).replace("\\", "/")
                for strategy_id, path in strategy_refs.items()
            },
            "portfolio_rounds_manifest_path": "output/portfolio/rounds_manifest.json",
            "required_portfolio_rounds": [1, 2, 3],
            "parity_alignment_path": str(parity_ref).replace("\\", "/"),
        },
    )
    return manifest_ref


class TestLiveConfig:
    def test_defaults(self):
        cfg = LiveConfig()
        assert cfg.is_testnet is True
        assert cfg.poll_interval_sec == 15.0
        assert cfg.symbols == ["BTC", "ETH", "SOL"]
        assert cfg.max_slippage_pct == 0.005
        assert cfg.health_report_interval_sec == 300.0
        assert cfg.funnel_report_interval_sec == 3600.0

    def test_validate_empty(self):
        cfg = LiveConfig()
        errors = cfg.validate()
        assert len(errors) >= 1
        assert any("wallet_address" in e for e in errors)

    def test_validate_valid(self):
        cfg = LiveConfig(wallet_address=VALID_WALLET, private_key=VALID_PRIVATE_KEY)
        errors = cfg.validate()
        assert len(errors) == 0

    def test_validate_read_only(self):
        cfg = LiveConfig(wallet_address=VALID_WALLET, private_key=None)
        errors = cfg.validate()
        assert any("read-only" in e for e in errors)

    def test_validate_rejects_placeholder_credentials(self):
        cfg = LiveConfig(
            wallet_address="0xYOUR_WALLET_ADDRESS_HERE",
            private_key="0xYOUR_PRIVATE_KEY_HERE",
        )
        errors = cfg.validate()
        assert any("wallet_address must be replaced" in e for e in errors)
        assert any("private_key must be replaced" in e for e in errors)

    def test_validate_rejects_malformed_hex_credentials(self):
        cfg = LiveConfig(wallet_address="0x123", private_key="0xabc")
        errors = cfg.validate()
        assert any("wallet_address must be 0x followed by 40 hex characters" in e for e in errors)
        assert any("private_key must be 0x followed by 64 hex characters" in e for e in errors)

    def test_live_config_example_credentials_do_not_validate(self):
        payload = json.loads(Path("config/live_config.example.json").read_text(encoding="utf-8"))
        errors = LiveConfig.from_dict(payload).validate()
        assert any("wallet_address" in e for e in errors)
        assert any("private_key" in e for e in errors)

    def test_live_config_example_leaves_local_postgres_disabled(self):
        payload = json.loads(Path("config/live_config.example.json").read_text(encoding="utf-8"))
        assert payload["postgres_dsn"] == ""

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
            "health_report_interval_sec": 120.0,
            "funnel_report_interval_sec": 1800.0,
            "strategy_configs": {"momentum": "configs/momentum.json"},
            "deployment_manifest_path": "deployments/manifest.json",
        }
        cfg = LiveConfig.from_dict(d)
        assert cfg.wallet_address == "0x123"
        assert cfg.is_testnet is False
        assert cfg.symbols == ["BTC"]
        assert cfg.poll_interval_sec == 30.0
        assert cfg.health_report_interval_sec == 120.0
        assert cfg.funnel_report_interval_sec == 1800.0
        assert cfg.strategy_configs["momentum"] == Path("configs/momentum.json")
        assert cfg.deployment_manifest_path == Path("deployments/manifest.json")

    def test_to_dict_excludes_private_key(self):
        cfg = LiveConfig(wallet_address=VALID_WALLET, private_key=VALID_PRIVATE_KEY)
        d = cfg.to_dict()
        assert "private_key" not in d
        assert d["wallet_address"] == VALID_WALLET

    def test_to_dict_redacted_omits_secret_fields(self):
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            bot_id="paper_bot",
            relay_url="https://relay.example.com",
            relay_secret="secret",
            postgres_dsn="postgres://user:pass@host/db",
        )

        d = cfg.to_dict(redacted=True)

        assert "private_key" not in d
        assert "wallet_address" not in d
        assert "relay_secret" not in d
        assert "postgres_dsn" not in d
        assert d["bot_id"] == "paper_bot"
        assert d["relay_url"] == "https://relay.example.com"

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

    def test_live_config_path_errors_require_explicit_runtime_files(self, tmp_path):
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=tmp_path / "missing_portfolio.json",
            strategy_configs={"trend": tmp_path / "missing_trend.json"},
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio_config_path" in error for error in errors)
        assert any("strategy_configs.trend" in error for error in errors)

    def test_live_config_path_errors_accept_asset_meta_covering_symbols(self, tmp_path):
        _write_json(tmp_path / "portfolio.json", {})
        _write_json(tmp_path / "trend.json", {})
        _write_json(
            tmp_path / "asset_meta.json",
            {
                "asset_index": {"BTC": 0, "ETH": 1},
                "tick_sizes": {"BTC": 0.5, "ETH": 0.05},
                "lot_sizes": {"BTC": 0.001, "ETH": 0.01},
            },
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            symbols=["BTC", "ETH"],
            portfolio_config_path=Path("portfolio.json"),
            strategy_configs={"trend": Path("trend.json")},
            asset_meta_path=Path("asset_meta.json"),
        )

        assert _live_config_path_errors(
            cfg,
            runtime_root=tmp_path,
            require_deployment_manifest=False,
        ) == []

    def test_live_config_path_errors_reject_incomplete_asset_meta(self, tmp_path):
        _write_json(tmp_path / "portfolio.json", {})
        _write_json(tmp_path / "trend.json", {})
        _write_json(
            tmp_path / "asset_meta.json",
            {
                "asset_index": {"BTC": 0},
                "tick_sizes": {"BTC": 0.5},
                "lot_sizes": {"BTC": 0.001},
            },
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            symbols=["BTC", "ETH"],
            portfolio_config_path=Path("portfolio.json"),
            strategy_configs={"trend": Path("trend.json")},
            asset_meta_path=Path("asset_meta.json"),
        )

        errors = _live_config_path_errors(
            cfg,
            runtime_root=tmp_path,
            require_deployment_manifest=False,
        )

        assert any("asset_meta_path.asset_index missing symbols: ETH" in error for error in errors)
        assert any("asset_meta_path.tick_sizes missing symbols: ETH" in error for error in errors)
        assert any("asset_meta_path.lot_sizes missing symbols: ETH" in error for error in errors)

    def test_live_config_path_errors_accept_exact_materialized_bundle_copies(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        expected_bundle = tmp_path / "output" / "portfolio" / "round_3"
        portfolio = tmp_path / "config" / "portfolio_config.json"
        portfolio.parent.mkdir(parents=True)
        portfolio.write_text(
            (expected_bundle / "recommended_portfolio_config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        strategy_configs = {}
        for strategy_id in STRATEGY_IDS:
            strategy = tmp_path / "config" / "strategies" / f"{strategy_id}.json"
            strategy.parent.mkdir(parents=True, exist_ok=True)
            strategy.write_text(
                (expected_bundle / "recommended_strategy_configs" / f"{strategy_id}.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            strategy_configs[strategy_id] = Path("config") / "strategies" / f"{strategy_id}.json"
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("config/portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs=strategy_configs,
        )

        assert _live_config_path_errors(cfg, runtime_root=tmp_path) == []

    def test_live_config_path_errors_reject_stale_existing_runtime_files(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        expected_bundle = tmp_path / "output" / "portfolio" / "round_3"
        stale_portfolio = tmp_path / "config" / "portfolio_config.json"
        stale_portfolio.parent.mkdir(parents=True)
        stale_portfolio.write_text('{"portfolio": "stale"}', encoding="utf-8")
        strategy_configs = {}
        for strategy_id in STRATEGY_IDS:
            strategy = tmp_path / "config" / "strategies" / f"{strategy_id}.json"
            strategy.parent.mkdir(parents=True, exist_ok=True)
            source = expected_bundle / "recommended_strategy_configs" / f"{strategy_id}.json"
            strategy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            strategy_configs[strategy_id] = Path("config") / "strategies" / f"{strategy_id}.json"
        (tmp_path / "config" / "strategies" / "trend.json").write_text(
            '{"strategy": {"name": "trend", "version": "stale"}}',
            encoding="utf-8",
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("config/portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs=strategy_configs,
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio_config_path does not match deployment manifest reference" in error for error in errors)
        assert any("strategy_configs.trend does not match deployment manifest reference" in error for error in errors)

    def test_live_config_path_errors_reject_missing_required_deployment_strategy(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                "momentum": Path("output/portfolio/round_3/recommended_strategy_configs/momentum.json"),
                "breakout": Path("output/portfolio/round_3/recommended_strategy_configs/breakout.json"),
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("missing required deployment strategies: trend" in error for error in errors)

    def test_live_config_path_errors_validate_portfolio_manifest_rounds(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        _write_json(tmp_path / "output" / "portfolio" / "rounds_manifest.json", {"rounds": [{"round": 1}, {"round": 2}]})
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                strategy_id: Path("output/portfolio/round_3/recommended_strategy_configs") / f"{strategy_id}.json"
                for strategy_id in STRATEGY_IDS
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio rounds manifest rounds [1, 2] do not match required [1, 2, 3]" in error for error in errors)

    def test_live_config_path_errors_require_complete_parity_numeric_fields(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        _write_json(
            tmp_path / "output" / "portfolio" / "round_3" / "parity_alignment.json",
            {"portfolio_metric_replay": {"status": "matched", "tolerance": 1e-9}},
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                strategy_id: Path("output/portfolio/round_3/recommended_strategy_configs") / f"{strategy_id}.json"
                for strategy_id in STRATEGY_IDS
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio parity evidence missing numeric fields: max_abs_delta" in error for error in errors)

    def test_live_config_path_errors_reject_negative_parity_abs_delta(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        _write_json(
            tmp_path / "output" / "portfolio" / "round_3" / "parity_alignment.json",
            {"portfolio_metric_replay": {"status": "matched", "max_abs_delta": -1.0, "tolerance": 1e-9}},
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                strategy_id: Path("output/portfolio/round_3/recommended_strategy_configs") / f"{strategy_id}.json"
                for strategy_id in STRATEGY_IDS
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio parity evidence has non-finite or negative numeric fields" in error for error in errors)

    def test_live_config_path_errors_are_manifest_driven_not_round3_hardcoded(self, tmp_path):
        manifest_path = _write_deployment_bundle(
            tmp_path,
            root=Path("deployments/reduced_risk"),
            strategy_ids=("trend",),
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("deployments/reduced_risk/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                "trend": Path("deployments/reduced_risk/recommended_strategy_configs/trend.json"),
            },
        )

        assert _live_config_path_errors(cfg, runtime_root=tmp_path) == []
