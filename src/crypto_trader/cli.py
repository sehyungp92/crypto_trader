"""CLI entry point for crypto_trader."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import click
import structlog


def _configure_logging() -> None:
    """Configure structlog with JSON output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )


@click.group()
def cli() -> None:
    """Crypto Trader — Hyperliquid perpetual futures trading system."""
    _configure_logging()


@cli.command()
@click.option("--coin", required=True, help="Comma-separated coin symbols (e.g. BTC,ETH)")
@click.option("--interval", default="15m", help="Comma-separated intervals (e.g. 15m,1h,4h)")
@click.option("--days", default=90, type=int, help="Number of days of history to download")
@click.option("--data-dir", default="data", type=click.Path(), help="Base data directory")
@click.option("--include-funding/--no-funding", default=True, help="Download funding rates")
def download(coin: str, interval: str, days: int, data_dir: str, include_funding: bool) -> None:
    """Download historical candle and funding data from Hyperliquid."""
    from crypto_trader.data.downloader import HyperliquidDownloader
    from crypto_trader.data.store import ParquetStore

    log = structlog.get_logger()

    store = ParquetStore(base_dir=Path(data_dir))
    downloader = HyperliquidDownloader(store=store)

    coins = [c.strip().upper() for c in coin.split(",")]
    intervals = [i.strip() for i in interval.split(",")]

    log.info(
        "download.start",
        coins=coins,
        intervals=intervals,
        days=days,
        include_funding=include_funding,
    )

    for c in coins:
        for iv in intervals:
            log.info("download.candles", coin=c, interval=iv)
            try:
                downloader.download_and_store(c, iv, days=days)
            except Exception:
                log.exception("download.candles.failed", coin=c, interval=iv)

        if include_funding:
            log.info("download.funding", coin=c)
            try:
                downloader.download_and_store_funding(c, days=days)
            except Exception:
                log.exception("download.funding.failed", coin=c)

    log.info("download.complete", coins=len(coins), intervals=len(intervals))


def _build_strategy_config(strategy: str, config_path: str | None, raw: dict | None = None):
    """Build strategy config from type and optional raw dict."""
    if strategy == "trend":
        from crypto_trader.strategy.trend.config import TrendConfig
        if raw:
            return TrendConfig.from_dict(raw.get("strategy", {}))
        return TrendConfig()
    elif strategy == "breakout":
        from crypto_trader.strategy.breakout.config import BreakoutConfig
        if raw:
            return BreakoutConfig.from_dict(raw.get("strategy", {}))
        return BreakoutConfig()
    else:
        from crypto_trader.strategy.momentum.config import MomentumConfig
        if raw:
            return MomentumConfig.from_dict(raw.get("strategy", {}))
        return MomentumConfig()


@cli.command()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbols")
@click.option("--output-dir", default="output", type=click.Path(), help="Output directory")
@click.option("--data-dir", default="data", type=click.Path(), help="Data directory")
@click.option("--equity", default=10000.0, type=float, help="Initial equity")
@click.option("--walk-forward", is_flag=True, help="Run walk-forward analysis")
@click.option("--strategy", default="momentum", type=click.Choice(["momentum", "trend", "breakout"]),
              help="Strategy type")
@click.option("--warmup-days", default=0, type=int, help="Extra days before start for indicator warmup")
def backtest(
    config_path: str | None,
    start_date: str,
    end_date: str,
    symbols: str,
    output_dir: str,
    data_dir: str,
    equity: float,
    walk_forward: bool,
    strategy: str,
    warmup_days: int,
) -> None:
    """Run a backtest with the specified strategy."""
    from datetime import date

    import yaml

    from crypto_trader.backtest.analysis import (
        export_equity_curve,
        export_trade_journal,
        generate_report,
        print_summary,
    )
    from crypto_trader.backtest.config import BacktestConfig
    from crypto_trader.backtest.runner import run, run_walk_forward

    log = structlog.get_logger()

    # Build strategy config
    raw = None
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    strategy_cfg = _build_strategy_config(strategy, config_path, raw)

    sym_list = [s.strip().upper() for s in symbols.split(",")]
    strategy_cfg.symbols = sym_list

    bt_cfg = BacktestConfig(
        symbols=sym_list,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        initial_equity=equity,
        warmup_days=warmup_days,
    )

    out = Path(output_dir)

    if walk_forward:
        log.info("backtest.walk_forward", train_pct=bt_cfg.train_pct)
        wf_result = run_walk_forward(strategy_cfg, bt_cfg, data_dir=Path(data_dir),
                                     strategy_type=strategy)
        print("\n--- TRAIN ---")
        print_summary(wf_result.train)
        generate_report(wf_result.train, out / "train")
        print("\n--- TEST ---")
        print_summary(wf_result.test)
        generate_report(wf_result.test, out / "test")
    else:
        result = run(strategy_cfg, bt_cfg, data_dir=Path(data_dir), strategy_type=strategy)
        print_summary(result)
        generate_report(result, out)
        export_equity_curve(result, out)
        export_trade_journal(result, out)
        log.info("backtest.complete", output_dir=str(out))


def _detect_next_round(base_dir: Path) -> int:
    """Scan for round_N/ dirs and return max+1 (or 1 if none)."""
    max_round = 0
    if base_dir.is_dir():
        for child in base_dir.iterdir():
            if child.is_dir():
                m = re.match(r"^round_(\d+)$", child.name)
                if m:
                    max_round = max(max_round, int(m.group(1)))
    return max_round + 1


def _update_rounds_manifest(
    base_dir: Path,
    round_num: int,
    mutations: dict,
    metrics: dict | None,
) -> None:
    """Append round entry to rounds_manifest.json."""
    from crypto_trader.optimize.phase_state import _atomic_write_json

    manifest_path = base_dir / "rounds_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"rounds": []}

    # Replace entry for same round_num if re-running
    manifest["rounds"] = [
        r for r in manifest["rounds"] if r.get("round") != round_num
    ]

    entry: dict = {
        "round": round_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mutations_count": len(mutations),
        "mutations": mutations,
    }
    if metrics:
        for k in ["total_trades", "win_rate", "profit_factor",
                   "max_drawdown_pct", "sharpe_ratio", "calmar_ratio",
                   "net_return_pct"]:
            if k in metrics:
                entry[k] = metrics[k]

    manifest["rounds"].append(entry)
    manifest["rounds"].sort(key=lambda r: r["round"])

    _atomic_write_json(manifest, manifest_path)


@cli.command()
@click.option("--config", "config_path", default=None, type=click.Path(exists=True), help="YAML config file")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default="BTC,ETH,SOL", help="Comma-separated symbols")
@click.option("--output-dir", default=None, type=click.Path(), help="Output directory (default: output/{strategy})")
@click.option("--data-dir", default="data", type=click.Path(), help="Data directory")
@click.option("--equity", default=10000.0, type=float, help="Initial equity")
@click.option("--phase", "phase_num", default=None, type=int, help="Run only phase N (default: all)")
@click.option("--resume", is_flag=True, help="Resume from phase_state.json checkpoint")
@click.option("--workers", default=None, type=int, help="Parallel workers (default: cpu_count - 1)")
@click.option("--round", "round_num", default=None, type=int,
              help="Load round N's optimized config as baseline and start round N+1")
@click.option("--strategy", default="momentum", type=click.Choice(["momentum", "trend", "breakout"]),
              help="Strategy type")
@click.option("--warmup-days", default=0, type=int, help="Extra days before start for indicator warmup")
def optimize(
    config_path: str | None,
    start_date: str,
    end_date: str,
    symbols: str,
    output_dir: str,
    data_dir: str,
    equity: float,
    phase_num: int | None,
    resume: bool,
    workers: int | None,
    round_num: int | None,
    strategy: str,
    warmup_days: int,
) -> None:
    """Run phased auto-optimization of the specified strategy."""
    from datetime import date

    import yaml

    from crypto_trader.backtest.config import BacktestConfig
    from crypto_trader.optimize.phase_runner import PhaseRunner
    from crypto_trader.optimize.phase_state import PhaseState

    log = structlog.get_logger()

    # Default output dir is strategy-specific
    if output_dir is None:
        output_dir = f"output/{strategy}"
    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Build strategy config — --round overrides --config
    if round_num is not None and config_path:
        log.warning("optimize.config_ignored", reason="--round overrides --config")

    if config_path and round_num is None:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        strategy_cfg = _build_strategy_config(strategy, config_path, raw)
    else:
        strategy_cfg = _build_strategy_config(strategy, None)

    # Load previous round's optimized config if --round specified
    if round_num is not None:
        prev_config_path = base_out / f"round_{round_num}" / "optimized_config.json"
        if not prev_config_path.exists():
            raise click.ClickException(
                f"No optimized config found at {prev_config_path}. "
                f"Round {round_num} must complete before starting round {round_num + 1}."
            )
        with open(prev_config_path, encoding="utf-8") as f:
            prev_raw = json.load(f)
        strategy_cfg = _build_strategy_config(strategy, None, prev_raw)
        next_round = round_num + 1
        log.info("optimize.loaded_round", source_round=round_num, next_round=next_round)
    else:
        next_round = _detect_next_round(base_out)

    # Round output directory
    round_dir = base_out / f"round_{next_round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    sym_list = [s.strip().upper() for s in symbols.split(",")]
    strategy_cfg.symbols = sym_list

    bt_cfg = BacktestConfig(
        symbols=sym_list,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        initial_equity=equity,
        warmup_days=warmup_days,
    )

    # Create strategy-specific plugin
    if strategy == "trend":
        from crypto_trader.optimize.trend_plugin import TrendPlugin
        plugin = TrendPlugin(bt_cfg, strategy_cfg, data_dir=Path(data_dir), max_workers=workers)
    elif strategy == "breakout":
        from crypto_trader.optimize.breakout_plugin import BreakoutPlugin
        plugin = BreakoutPlugin(bt_cfg, strategy_cfg, data_dir=Path(data_dir), max_workers=workers)
    else:
        from crypto_trader.optimize.momentum_plugin import MomentumPlugin
        plugin = MomentumPlugin(bt_cfg, strategy_cfg, data_dir=Path(data_dir), max_workers=workers)
    runner = PhaseRunner(plugin, round_dir)

    # Load or create state
    state_path = round_dir / "phase_state.json"
    if resume and state_path.exists():
        state = PhaseState.load(state_path)
        log.info("optimize.resumed", current_phase=state.current_phase, round=next_round)
    else:
        state = PhaseState(_path=state_path)

    # Run
    log.info("optimize.start", round=next_round, output_dir=str(round_dir))
    if phase_num is not None:
        log.info("optimize.single_phase", phase=phase_num)
        runner.run_phase(phase_num, state)
    else:
        log.info("optimize.all_phases")
        runner.run_all_phases(state)

    # Update manifest
    final_metrics = None
    if state.phase_metrics:
        last_phase = max(state.phase_metrics.keys())
        final_metrics = state.phase_metrics[last_phase]

    _update_rounds_manifest(
        base_out, next_round, state.cumulative_mutations, final_metrics,
    )

    # Summary
    print(f"\n=== Optimization Complete (Round {next_round}) ===")
    print(f"Output: {round_dir}")
    print(f"Completed phases: {state.completed_phases}")
    print(f"Total mutations: {len(state.cumulative_mutations)}")
    for key, val in state.cumulative_mutations.items():
        print(f"  {key} = {val}")

    if final_metrics:
        print(f"\nFinal metrics:")
        for k in ["total_trades", "win_rate", "profit_factor", "max_drawdown_pct",
                   "sharpe_ratio", "calmar_ratio"]:
            print(f"  {k}: {final_metrics.get(k, 0):.2f}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True),
              help="Live config JSON file")
def paper(config_path: str) -> None:
    """Run paper trading on Hyperliquid testnet."""
    import asyncio

    from crypto_trader.live.config import LiveConfig
    from crypto_trader.live.engine import LiveEngine

    log = structlog.get_logger()

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = LiveConfig.from_dict(data)

    errors = config.validate()
    if errors:
        for e in errors:
            log.error("config.validation", error=e)
        raise click.ClickException("Invalid configuration. See errors above.")

    engine = LiveEngine(config)

    log.info("paper.starting", testnet=config.is_testnet)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        log.info("paper.interrupted")


@cli.command()
@click.option("--state-dir", default="state", type=click.Path(), help="State directory with JSONL files")
def status(state_dir: str) -> None:
    """Show live system status from the latest health report and pipeline funnels."""
    from crypto_trader.live.health_report import HealthReport

    state = Path(state_dir)

    # --- Helper: read last line of a JSONL file ---
    def _read_last_jsonl(filename: str) -> dict | None:
        path = state / filename
        if not path.exists():
            return None
        last_line = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
        except Exception:
            return None
        if not last_line:
            return None
        try:
            return json.loads(last_line)
        except json.JSONDecodeError:
            return None

    # --- Health report ---
    health_data = _read_last_jsonl("health_reports.jsonl")
    if health_data:
        # HealthReportSnapshot wraps the report in a "report" key
        report_dict = health_data.get("report", health_data)
        report = HealthReport(
            timestamp=report_dict.get("timestamp", "?"),
            uptime_sec=report_dict.get("uptime_sec", 0),
            data_flow=report_dict.get("data_flow", {}),
            signal_funnels=report_dict.get("signal_funnels", {}),
            gate_breakdown=report_dict.get("gate_breakdown", {}),
            positions=report_dict.get("positions", []),
            portfolio=report_dict.get("portfolio", {}),
            system=report_dict.get("system", {}),
            alerts=report_dict.get("alerts", []),
            assessment=report_dict.get("assessment", "unknown"),
        )
        print(report.to_text())
        print(f"\n  Last report: {report.timestamp}")
    else:
        print("=== System Health: NO DATA ===")
        print(f"  No health reports found in {state / 'health_reports.jsonl'}")

    # --- Pipeline funnels ---
    funnel_data = _read_last_jsonl("pipeline_funnels.jsonl")
    if funnel_data:
        print(f"\n--- Latest Pipeline Funnel ---")
        sid = funnel_data.get("strategy_id", "?")
        ts = funnel_data.get("timestamp", "?")
        assessment = funnel_data.get("assessment", "?")
        print(f"  Strategy: {sid} | Assessment: {assessment} | At: {ts}")
        funnel = funnel_data.get("funnel", {})
        if funnel:
            bars = sum(funnel.get("bars_received", {}).values()) if isinstance(funnel.get("bars_received"), dict) else funnel.get("bars_received", 0)
            ind = sum(funnel.get("indicators_ready", {}).values()) if isinstance(funnel.get("indicators_ready"), dict) else funnel.get("indicators_ready", 0)
            setups = sum(funnel.get("setups_detected", {}).values()) if isinstance(funnel.get("setups_detected"), dict) else funnel.get("setups_detected", 0)
            confirms = sum(funnel.get("confirmations", {}).values()) if isinstance(funnel.get("confirmations"), dict) else funnel.get("confirmations", 0)
            entries = sum(funnel.get("entries_attempted", {}).values()) if isinstance(funnel.get("entries_attempted"), dict) else funnel.get("entries_attempted", 0)
            fills = sum(funnel.get("fills", {}).values()) if isinstance(funnel.get("fills"), dict) else funnel.get("fills", 0)
            print(
                f"  bars={bars} -> indicators={ind} -> setups={setups} "
                f"-> confirms={confirms} -> entries={entries} -> fills={fills}"
            )
            # Gate rejections
            gates = funnel.get("gate_rejections", {})
            if gates:
                print("  Gate rejections:")
                for gate, count in gates.items():
                    total = sum(count.values()) if isinstance(count, dict) else count
                    print(f"    {gate}: {total}")
    else:
        print(f"\n--- Latest Pipeline Funnel ---")
        print(f"  No funnel data found in {state / 'pipeline_funnels.jsonl'}")

    print()


@cli.command("paper-status")
@click.option("--address", required=True, help="Wallet address")
@click.option("--testnet/--mainnet", default=True, help="Use testnet or mainnet")
def paper_status(address: str, testnet: bool) -> None:
    """Show current positions, orders, and equity from Hyperliquid."""
    from crypto_trader.live.broker import HyperliquidBroker

    log = structlog.get_logger()

    broker = HyperliquidBroker(
        wallet_address=address,
        private_key=None,  # read-only
        is_testnet=testnet,
    )

    print(f"\n{'='*50}")
    print(f"Hyperliquid {'Testnet' if testnet else 'Mainnet'} Status")
    print(f"Address: {address[:8]}...{address[-4:]}")
    print(f"{'='*50}")

    # Equity
    equity = broker.get_equity()
    print(f"\nEquity: ${equity:,.2f}")

    # Positions
    positions = broker.get_positions()
    if positions:
        print(f"\nOpen Positions ({len(positions)}):")
        for pos in positions:
            pnl_str = f"${pos.unrealized_pnl:+,.2f}" if pos.unrealized_pnl else "$0.00"
            print(f"  {pos.symbol:>5} {pos.direction.value:>5} qty={pos.qty:.4f} "
                  f"entry=${pos.avg_entry:,.2f} uPnL={pnl_str} "
                  f"lev={pos.leverage:.0f}x")
    else:
        print("\nNo open positions.")

    # Open orders
    orders = broker.get_open_orders()
    if orders:
        print(f"\nOpen Orders ({len(orders)}):")
        for o in orders:
            px = o.limit_price or o.stop_price or 0
            print(f"  {o.symbol:>5} {o.side.value:>5} {o.order_type.value:>8} "
                  f"qty={o.qty:.4f} px=${px:,.2f}")
    else:
        print("\nNo open orders.")

    print()
