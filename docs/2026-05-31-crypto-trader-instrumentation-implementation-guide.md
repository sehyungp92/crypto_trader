# Crypto Trader Instrumentation Implementation Guide

Date: 2026-05-31

Audience: Codex or another coding agent implementing the optimal
instrumentation target in `_references/crypto_trader/`.

Primary source guide:
`docs/2026-05-30-optimal-trading-repo-instrumentation-guide.md`

Reference repo audited:
`_references/crypto_trader/`

## 1. Purpose

This guide translates the optimal trading-repo instrumentation target into a
concrete implementation plan for the current `crypto_trader` repo.

The repo is not blank. It already has a useful instrumentation foundation:

- three live strategy families in one crypto perpetuals portfolio:
  `momentum`, `trend`, and `breakout`;
- per-strategy `InstrumentationCollector` hooks for gates, context snapshots,
  signal factors, entries, completed trades, missed opportunities, process
  quality, and pipeline funnels;
- JSONL sinks, in-memory test sinks, a daily aggregator, a sidecar forwarder,
  and a relay buffer;
- a live `PortfolioManager`, `StrategyCoordinator`, `BrokerProxy`,
  `ExecutionGateway`, `OmsStore`, reconciliation loop, health report loop, and
  parity event stream;
- deployment bundle preflight checks around strategy configs, portfolio config,
  deployment manifest, portfolio rounds manifest, and parity alignment.

The goal is therefore not to bolt on an unrelated logging system. The goal is
to upgrade the current foundation into the optimal evidence contract needed by
`trading_assistant` for:

- daily and weekly diagnosis;
- monthly authoritative validation;
- structural strategy and portfolio change review;
- live/backtest decision parity;
- attribution across strategy config, portfolio config, risk config,
  allocation state, OMS state, exchange execution, and deployment identity.

The most important current-state finding is:

`crypto_trader` already observes many strategy-level facts, but it does not yet
emit complete monthly-authoritative evidence. Strategy events need lineage and
stable decision identity. Portfolio, risk, order, OMS, config, deployment, and
snapshot events need to become first-class assistant telemetry rather than
being implicit in logs, local SQLite, health summaries, or parity-only JSONL.

## 2. Implementation Principle

Use the existing instrumentation package and live-runtime surfaces. Do not
replace them.

Make the implementation surgical:

- preserve `src/crypto_trader/instrumentation/*`;
- preserve current strategy behavior in `strategy/momentum`, `strategy/trend`,
  and `strategy/breakout`;
- enrich existing events rather than renaming all current fields at once;
- keep compatibility aliases such as `pair`, `side`, `assistant_strategy_id`,
  `r_multiple`, and `portfolio_state_at_entry`;
- add canonical fields alongside existing fields where names differ;
- make all instrumentation fail-open and bounded;
- never let sidecar, relay, Postgres, or JSONL failures block trading;
- never write secrets, private keys, raw relay secrets, or raw Postgres DSNs.

The new target state should make this true:

For every live strategy intent, the assistant can answer:

- Which strategy, strategy family, portfolio, account alias, deployment, config,
  code SHA, parameter set, portfolio config, risk config, allocation version,
  and experiment variant produced it?
- Which market bar and decision callback produced it?
- Which filters, indicators, signal factors, sizing inputs, portfolio rules,
  risk limits, OMS state, and exchange execution reports transformed it?
- Did live behavior match replay behavior at the market, decision, order,
  execution, trade, and portfolio-rule levels?

## 3. Current Implementation Map

### 3.1 Strategy Surface

Current strategy files:

- `_references/crypto_trader/src/crypto_trader/strategy/momentum/strategy.py`
- `_references/crypto_trader/src/crypto_trader/strategy/trend/strategy.py`
- `_references/crypto_trader/src/crypto_trader/strategy/breakout/strategy.py`

Current instrumentation files:

- `_references/crypto_trader/src/crypto_trader/instrumentation/types.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/collector.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/quality.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/pipeline_tracker.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/backfill.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/strategy_ids.py`

Useful current behavior:

- each strategy creates `InstrumentationCollector(strategy_id=..., bot_id=...)`;
- each strategy calls `begin_bar()`, `record_gate()`,
  `snapshot_context()`, `record_signal_factor()`, `record_entry()`,
  `end_bar()`, and `on_trade_closed()`;
- completed trade events include entry and exit prices, net economics, fees,
  funding, R multiple, setup grade, exit reason, confluences, signal factors,
  filter decisions, market context, MFE, MAE, exit efficiency, process quality,
  root causes, strategy params, sizing inputs, and compact portfolio state;
- missed opportunity events are generated when the setup gate passes and a
  downstream gate blocks the entry;
- missed opportunities can be backfilled with 1h, 4h, and 24h outcomes;
- process quality uses a controlled root-cause taxonomy compatible with the
  assistant vocabulary;
- pipeline funnel snapshots summarize stage throughput and broken/stalled
  pipeline status.

Current strategy-level gaps:

- `EventMetadata` does not include event type, payload key, local timestamp,
  clock skew, data source, bar ID, schema version, lineage, portfolio ID,
  account alias, config version, deployment ID, or code SHA;
- events do not carry a stable `decision_id`, `bar_id`, `signal_id`, or
  `entry_signal_id` that can be joined to parity streams and order intents;
- missed opportunity IDs are based on `datetime.now()` and
  `symbol + blocker`, which is not stable enough for replay attribution;
- backfilled missed-opportunity updates reuse the same `event_id`, but the
  relay deduplicates by `event_id`, so forwarded updates can be dropped;
- filter decisions and indicator snapshots are embedded in trades/misses but
  are not emitted as first-class events;
- market snapshots are present in parity JSONL but are not forwarded through
  the assistant sidecar;
- trade events do not include order IDs, fill IDs, slippage, latency, exchange
  order IDs, or OMS lifecycle references.

### 3.2 Live Runtime Surface

Current live files:

- `_references/crypto_trader/src/crypto_trader/live/engine.py`
- `_references/crypto_trader/src/crypto_trader/live/config.py`
- `_references/crypto_trader/src/crypto_trader/live/broker.py`
- `_references/crypto_trader/src/crypto_trader/live/execution_adapter.py`
- `_references/crypto_trader/src/crypto_trader/live/oms_store.py`
- `_references/crypto_trader/src/crypto_trader/live/reconciler.py`
- `_references/crypto_trader/src/crypto_trader/live/state.py`
- `_references/crypto_trader/src/crypto_trader/live/health.py`
- `_references/crypto_trader/src/crypto_trader/live/health_report.py`

Useful current behavior:

- `LiveEngine` wires `EventEmitter`, `JsonlSink`, `DailyAggregator`, optional
  `PostgresSink`, and optional `SidecarForwarder`;
- warmup uses `_WarmupBrokerProxy`, and strategy emitters are wired only after
  warmup, which avoids stale warmup telemetry;
- `LiveEngine` emits daily snapshots, missed-opportunity updates, pipeline
  funnels, and health reports;
- `LiveEngine` writes `parity_events.jsonl` from `CanonicalRuntimeEvent`;
- health reports include sidecar status, positions, portfolio heat, daily PnL,
  stale feed status, funnels, and relay watermarks;
- `PersistentState` already has JSONL paths for instrumented trades, missed
  opportunities, daily snapshots, errors, equity snapshots, and rule events.

Current live-runtime gaps:

- startup does not emit deployment, config, strategy registry, portfolio config,
  risk config, allocation, account alias, or code SHA snapshots;
- `parity_events.jsonl` is local-only and not forwarded to the assistant;
- position snapshots and portfolio snapshots are written only partially through
  health reports and optional Postgres state, not as canonical assistant events;
- errors mostly update health state or logs and are not consistently emitted as
  structured `ErrorEvent` payloads with lineage;
- health reports have no shared `EventMetadata` or lineage envelope;
- equity snapshots are local/Postgres-only and not joined to portfolio
  snapshot lineage.

### 3.3 Order, OMS, And Parity Surface

Current core files:

- `_references/crypto_trader/src/crypto_trader/core/runtime_types.py`
- `_references/crypto_trader/src/crypto_trader/core/strategy_runtime.py`
- `_references/crypto_trader/src/crypto_trader/core/execution_gateway.py`
- `_references/crypto_trader/src/crypto_trader/core/events.py`
- `_references/crypto_trader/src/crypto_trader/parity/report.py`
- `_references/crypto_trader/src/crypto_trader/parity/shadow.py`

Useful current behavior:

- `StrategySlotRuntime` emits canonical `market`, `decision`, `execution`, and
  `trade` streams;
- `DecisionContext` creates deterministic decision IDs from strategy, symbol,
  timeframe, and market availability time;
- `OrderIntent` normalizes strategy order requests;
- `ExecutionReport` normalizes accepted, rejected, resting, partial fill, fill,
  cancelled, and expired broker outcomes;
- `ExecutionGateway` emits `order_intent` and `execution` parity events and
  records execution reports into `OmsStore`;
- `parity/report.py` can compare actual and expected decision/order-intent
  streams and gate promotion on drift, stale fill watermarks, unprotected entry
  fills, OMS discrepancies, and accounting mismatches.

Current order/OMS/parity gaps:

- order intent and execution facts are parity-local, not assistant telemetry;
- order lifecycle events are not emitted through `EventEmitter` or sidecar;
- parity events do not include full lineage;
- `DecisionEvent` does not carry filter outcomes, signal factors, market
  context, strategy state hash, portfolio state hash, or lineage;
- completed trade events are not explicitly linked back to the order intents,
  execution reports, fills, and decision IDs that created them;
- `OmsStore` has rich state, but the assistant cannot ingest most of it unless
  it is promoted to JSONL/sidecar events.

### 3.4 Portfolio Surface

Current portfolio files:

- `_references/crypto_trader/src/crypto_trader/portfolio/config.py`
- `_references/crypto_trader/src/crypto_trader/portfolio/state.py`
- `_references/crypto_trader/src/crypto_trader/portfolio/manager.py`
- `_references/crypto_trader/src/crypto_trader/portfolio/coordinator.py`
- `_references/crypto_trader/src/crypto_trader/portfolio/backtest_runner.py`
- `_references/crypto_trader/src/crypto_trader/portfolio/sweep.py`

Useful current behavior:

- `PortfolioConfig` includes strategy allocations, base risk, per-strategy
  max concurrent positions, per-strategy daily stops, priority, heat cap,
  directional cap, portfolio daily stop, max total positions, drawdown tiers,
  symbol collision mode, symbol exposure cap, and priority headroom;
- `PortfolioState` tracks equity, peak equity, open risks, strategy daily PnL,
  portfolio daily PnL, drawdown, heat, directional risk, symbol risk, and open
  risk count;
- `PortfolioManager.check_entry()` evaluates nine rule families before allowing
  entries;
- `BrokerProxy.submit_order()` stamps strategy ownership, calls
  `PortfolioManager.check_entry()` for entry orders, blocks rejected entries,
  and applies drawdown-tier size multipliers;
- `StrategyCoordinator` routes fills and updates portfolio state on entry and
  exit;
- `BrokerProxy.get_portfolio_snapshot()` already exposes compact pre-entry
  state to strategy trade instrumentation.

Current portfolio-level gaps:

- portfolio approvals, denials, and size adjustments are only logs and return
  values, not structured assistant events;
- `PortfolioRuleResult` only contains `approved`, `denial_reason`, and
  `size_multiplier`, so most rule evidence is discarded;
- `check_entry()` short-circuits and does not preserve the full evaluated rule
  trail, thresholds, before/after values, active allocation, priority, or
  blocking rule identity;
- entry denials are not linked to the decision ID, order intent, strategy
  signal, portfolio config version, risk config version, allocation version, or
  portfolio state snapshot;
- there is no canonical `portfolio_rule`, `risk_decision`,
  `portfolio_snapshot`, or `allocation_snapshot` event;
- backtest portfolio rule logging exists in `portfolio/backtest_runner.py`, but
  live and assistant-side telemetry need the same contract.

### 3.5 Sidecar, Relay, And Persistence Surface

Current instrumentation persistence files:

- `_references/crypto_trader/src/crypto_trader/instrumentation/emitter.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/sinks.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/sidecar.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/daily_aggregator.py`
- `_references/crypto_trader/src/crypto_trader/instrumentation/postgres_sink.py`
- `_references/crypto_trader/src/crypto_trader/relay/app.py`
- `_references/crypto_trader/src/crypto_trader/relay/store.py`
- `_references/crypto_trader/src/crypto_trader/relay/auth.py`

Useful current behavior:

- `EventEmitter` dispatches to multiple sinks and swallows sink exceptions;
- `JsonlSink` writes flat JSONL files in `state_dir`;
- `InMemorySink` supports tests;
- `DailyAggregator` implements the sink protocol and computes daily snapshots;
- `SidecarForwarder` reads JSONL files by byte watermark, batches events, signs
  payloads with HMAC-SHA256, gzips large payloads, retries, and persists
  watermarks;
- `RelayStore` persists events in SQLite WAL mode and deduplicates by
  `event_id`;
- relay health includes pending counts, per-bot counts, oldest pending age,
  database size, and uptime.

Current sidecar/relay gaps:

- sidecar sends grouped file names like `instrumented_trades` and
  `health_reports` as event types instead of canonical assistant event types
  like `trade`, `heartbeat`, `portfolio_rule`, and `decision_event`;
- sidecar does not wrap each payload in a canonical envelope with event type,
  payload hash, schema version, lineage, local timestamp, source file, or
  priority;
- sidecar does not forward order, filter, indicator, market, portfolio rule,
  position, portfolio, allocation, config, deployment, risk, decision, or
  parity events;
- flat JSONL files are workable, but date-partitioned files are better for
  high-volume long-running production operation;
- relay dedup by `event_id` means mutable/revised events require a separate
  logical ID and revision-specific event ID.

### 3.6 Config And Deployment Surface

Current config/deployment files:

- `_references/crypto_trader/config/live_config.example.json`
- `_references/crypto_trader/config/portfolio_config.json`
- `_references/crypto_trader/config/strategies/momentum.json`
- `_references/crypto_trader/config/strategies/trend.json`
- `_references/crypto_trader/config/strategies/breakout.json`
- `_references/crypto_trader/output/portfolio/round_3/deployment_manifest.json`
- `_references/crypto_trader/src/crypto_trader/live/config.py`
- `_references/crypto_trader/src/crypto_trader/cli.py`

Useful current behavior:

- `LiveConfig` already names `bot_id`, `symbols`, `state_dir`, `data_dir`,
  strategy config paths, portfolio config path, deployment manifest path,
  asset meta path, relay URL, and optional Postgres DSN;
- `cli.py` validates that live config paths exist and can require a deployment
  manifest;
- deployment manifest validation checks required strategy IDs, strategy config
  paths, portfolio config path, portfolio rounds manifest, and parity alignment;
- deployment manifests include candidate and portfolio round identifiers.

Current config/deployment gaps:

- live startup does not emit a deployment event;
- config hashes are not computed and stored on every event;
- risk config and allocation config are not versioned separately from the full
  portfolio config;
- `LiveConfig.to_dict()` includes sensitive values such as `relay_secret` and
  `postgres_dsn`, so config snapshots need explicit redaction;
- account identity should be a stable alias, not wallet address or private key;
- code SHA is not emitted.

## 4. Current Coverage Against The Optimal Event Surface

| Optimal event | Current state | Required action |
|---|---|---|
| `trade` | Partially present as `InstrumentedTradeEvent` | Add lineage, decision/order/fill links, stable bar IDs, slippage/latency fields, envelope |
| `missed_opportunity` | Partially present | Add stable logical ID, revision ID, simulation policy, lineage, decision/bar IDs |
| `filter_decision` | Embedded only | Emit first-class events from `record_gate()` or `end_bar()` |
| `order` | Present as parity `order_intent` and `execution` only | Convert to assistant order lifecycle events and forward |
| `daily_snapshot` | Present | Add lineage, portfolio/family summaries, version fields |
| `error` | Present as type and some health updates | Emit consistently from live exception paths with lineage |
| `heartbeat` | Present as health report | Add metadata, lineage, canonical event type, liveness fields |
| `config_snapshot` | Missing | Add startup and change snapshot events with redaction |
| `deployment` | Missing | Add startup event from deployment manifest, code SHA, config versions |
| `indicator_snapshot` | Embedded only | Emit first-class event at actionable decision points |
| `market_snapshot` | Present in parity only | Forward canonical market snapshots or summarize them for assistant ingestion |
| `portfolio_rule` | Missing | Add full approval/denial/scale events in `PortfolioManager`/`BrokerProxy` |
| `risk_decision` | Missing | Add summarized order-intent transformation event |
| `position_snapshot` | Partial in health/Postgres | Emit canonical snapshots at startup, fills, reconciliation, and closeout |
| `portfolio_snapshot` | Partial in health | Emit canonical snapshots at startup, fills, reconciliation, and closeout |
| `allocation_snapshot` | Missing | Emit startup/change snapshots from `PortfolioConfig.strategies` |
| `pipeline_funnel` | Present | Add metadata, lineage, canonical event type |
| `regime_transition` | Missing | Emit when strategy regime/bias changes materially |
| `decision_event` | Present in parity only | Add lineage, signal/filter context, sidecar forwarding |

## 5. Target Data Flow

The target flow for this repo should be:

```text
StrategySlotRuntime
  -> market event
  -> decision context
  -> strategy collector gates/context/signal factors
  -> order intent
  -> portfolio rule/risk decision
  -> execution reports/fills
  -> OMS state transitions
  -> position/trade lifecycle
  -> EventEmitter
  -> JSONL sinks + optional Postgres
  -> SidecarForwarder canonical envelope
  -> relay
  -> trading_assistant event queue
```

Keep two local records:

- canonical assistant events for ingestion;
- parity event streams for replay comparison.

They can share source data, but they serve different purposes. Assistant events
must be lineage-rich and suitable for daily/monthly learning. Parity events
must be deterministic and normalized enough to compare live against replay.

## 6. Common Event Contract

### 6.1 Canonical Sidecar Envelope

Every forwarded event should be wrapped into this shape before it leaves the
bot:

```json
{
  "schema_version": "assistant_event_v1",
  "event_id": "stable-or-revision-specific-id",
  "logical_event_id": "stable-logical-id-when-the-event-can-revise",
  "event_type": "trade",
  "bot_id": "paper_bot_01",
  "family_id": "crypto_perps",
  "portfolio_id": "crypto_perps_main",
  "account_alias": "paper_hyperliquid_01",
  "strategy_id": "momentum",
  "assistant_strategy_id": "MomentumPullback_M15",
  "exchange_timestamp": "2026-05-31T00:00:00+00:00",
  "local_timestamp": "2026-05-31T00:00:00.025000+00:00",
  "clock_skew_ms": 25,
  "data_source_id": "hyperliquid",
  "bar_id": "hyperliquid:BTC:15m:2026-05-31T00:00:00+00:00",
  "lineage": {
    "deployment_id": "sha256:...",
    "code_sha": "abc123",
    "strategy_version": "round_3:momentum:sha256:...",
    "config_version": "sha256:...",
    "portfolio_config_version": "sha256:...",
    "risk_config_version": "sha256:...",
    "allocation_version": "sha256:...",
    "strategy_registry_version": "sha256:...",
    "parameter_set_id": "sha256:...",
    "experiment_id": null,
    "variant_id": null,
    "proposal_id": null,
    "suggestion_id": null
  },
  "priority": "normal",
  "payload_hash": "sha256:...",
  "payload": {}
}
```

The relay can keep its current batch shape:

```json
{
  "bot_id": "paper_bot_01",
  "event_type": "trade",
  "events": [{ "...": "canonical envelope" }]
}
```

But each item inside `events` should already be a canonical envelope.

### 6.2 EventMetadata Upgrade

Extend `instrumentation/types.py::EventMetadata` instead of creating a second
metadata class that competes with it.

Required fields:

- `schema_version`;
- `event_id`;
- `logical_event_id`;
- `event_type`;
- `payload_key`;
- `bot_id`;
- `family_id`;
- `portfolio_id`;
- `account_alias`;
- `strategy_id`;
- `assistant_strategy_id`;
- `exchange_timestamp`;
- `local_timestamp`;
- `clock_skew_ms`;
- `data_source_id`;
- `bar_id`;
- `decision_id`;
- `trace_id`;
- `lineage`;
- `source`;
- `revision`;
- `is_backfill_update`.

Compatibility rule:

- keep existing constructor defaults so current tests and strategies do not
  fail immediately;
- update `EventMetadata.create()` to accept optional keyword-only fields;
- keep `metadata.to_dict()["assistant_strategy_id"]`;
- keep `event_id` length expectations only for legacy tests or update tests to
  accept 16-char legacy IDs and longer canonical IDs during transition.

Recommended event ID rules:

| Event | `logical_event_id` | `event_id` |
|---|---|---|
| Completed trade | Hash of bot, strategy, trade ID, entry time, exit time | Same as logical ID unless trade is revised |
| Missed opportunity | Hash of bot, strategy, symbol, timeframe, bar ID, blocker, simulation policy | Hash of logical ID plus revision/backfill status |
| Filter decision | Hash of bot, strategy, decision ID, filter name, filter index | Same as logical ID |
| Order intent | `intent_id` | Hash of `intent_id` plus event stage |
| Execution report | `report_id` or exchange report identity | Hash of report ID |
| Portfolio rule | Hash of bot, portfolio, decision ID, strategy, symbol, rule sequence | Same as logical ID |
| Position snapshot | Hash of account alias, strategy, symbol, timestamp bucket | Same as logical ID |
| Portfolio snapshot | Hash of account alias, portfolio, timestamp bucket | Same as logical ID |
| Config snapshot | Hash of config kind and config version | Same as logical ID |
| Deployment | `deployment_id` | Same as logical ID |
| Decision event | `decision_id` | Same as logical ID |

Important missed-opportunity correction:

Do not forward backfilled missed opportunity updates with the same `event_id`.
The current relay deduplicates by `event_id`, so it will drop a later update
that fills in 1h/4h/24h outcomes. Use:

- stable `opportunity_id` or `logical_event_id` for the missed opportunity;
- revision-specific `event_id` for each append, for example
  `hash(logical_event_id + revision)`;
- `revision`, `backfill_status`, and `supersedes_event_id` fields.

### 6.3 Lineage Context

Add a new file:

`_references/crypto_trader/src/crypto_trader/instrumentation/lineage.py`

It should expose a small immutable context object:

```python
@dataclass(frozen=True)
class LineageContext:
    bot_id: str
    family_id: str
    portfolio_id: str
    account_alias: str
    data_source_id: str
    deployment_id: str
    code_sha: str
    strategy_registry_version: str
    portfolio_config_version: str
    risk_config_version: str
    allocation_version: str
    live_config_version: str
    strategy_versions: dict[str, str]
    config_versions: dict[str, str]
    parameter_set_ids: dict[str, str]
    deployment_manifest_version: str | None = None
    portfolio_round: int | None = None
    candidate: str | None = None
```

It should provide:

- `LineageContext.from_live_config(...)`;
- `lineage.for_strategy(strategy_id) -> dict`;
- `lineage.for_portfolio() -> dict`;
- `lineage.event_metadata_defaults(strategy_id=None) -> dict`;
- `lineage.redacted_config_snapshot() -> dict`;
- deterministic JSON canonicalization helper;
- deterministic SHA-256 helper.

Recommended default IDs for this repo:

- `family_id`: `crypto_perps`;
- `portfolio_id`: from config if added, otherwise `crypto_perps_main`;
- `data_source_id`: `hyperliquid:testnet` or `hyperliquid:mainnet`;
- `account_alias`: from new live config field or environment variable, never
  from raw private key;
- `deployment_id`: deployment manifest hash plus code SHA;
- `strategy_version`: strategy config path/version plus config hash;
- `parameter_set_id`: strategy config hash for `momentum`, `trend`,
  `breakout`;
- `allocation_version`: hash of `PortfolioConfig.strategies`, including
  enabled flags, base risk, max concurrent, daily stops, and priority;
- `risk_config_version`: hash of portfolio-level risk fields:
  `heat_cap_R`, `directional_cap_R`, `portfolio_daily_stop_R`,
  `max_total_positions`, `dd_tiers`, `symbol_collision`,
  `symbol_exposure_cap_R`, `priority_headroom_R`,
  `priority_reserve_threshold`, and per-strategy stops;
- `portfolio_config_version`: hash of full `PortfolioConfig.to_dict()`;
- `strategy_registry_version`: hash of sorted enabled strategy IDs plus their
  config versions and assistant strategy IDs.

### 6.4 Config Version Rules

Use canonical JSON hashing:

- parse JSON/YAML into structured values;
- remove secrets;
- sort keys;
- use compact separators;
- preserve numeric values as JSON numbers;
- hash bytes with SHA-256;
- prefix stored values with `sha256:`.

Files to hash:

- `LiveConfig` after redaction;
- each strategy config loaded by `LiveEngine._load_strategy_config()`;
- `PortfolioConfig.to_dict()`;
- deployment manifest JSON;
- asset meta file if present;
- strategy registry map;
- risk subset;
- allocation subset.

Redact at minimum:

- `private_key`;
- `wallet_address`, unless explicitly configured as non-sensitive;
- `relay_secret`;
- `postgres_dsn`;
- any key containing `secret`, `token`, `password`, `private`, `key`, `dsn`,
  or `credential`.

### 6.5 Event Type Map

Add or preserve the following canonical event types:

| Canonical type | Suggested JSONL file |
|---|---|
| `trade` | `trades/trades_YYYY-MM-DD.jsonl` and legacy `instrumented_trades.jsonl` |
| `missed_opportunity` | `missed/missed_YYYY-MM-DD.jsonl` and legacy `missed_opportunities.jsonl` |
| `filter_decision` | `filter_decisions/filter_decisions_YYYY-MM-DD.jsonl` |
| `indicator_snapshot` | `indicators/indicators_YYYY-MM-DD.jsonl` |
| `market_snapshot` | `market/market_YYYY-MM-DD.jsonl` |
| `order` | `orders/orders_YYYY-MM-DD.jsonl` |
| `portfolio_rule` | `portfolio_rules/portfolio_rules_YYYY-MM-DD.jsonl` |
| `risk_decision` | `risk_decisions/risk_decisions_YYYY-MM-DD.jsonl` |
| `position_snapshot` | `positions/positions_YYYY-MM-DD.jsonl` |
| `portfolio_snapshot` | `portfolio/portfolio_snapshots_YYYY-MM-DD.jsonl` |
| `allocation_snapshot` | `allocations/allocations_YYYY-MM-DD.jsonl` |
| `daily_snapshot` | `daily/daily_snapshots_YYYY-MM-DD.jsonl` and legacy `daily_snapshots.jsonl` |
| `error` | `errors/errors_YYYY-MM-DD.jsonl` and legacy `errors.jsonl` |
| `heartbeat` | `heartbeats/heartbeat_YYYY-MM-DD.jsonl` and legacy `health_reports.jsonl` |
| `pipeline_funnel` | `funnels/pipeline_funnels_YYYY-MM-DD.jsonl` and legacy `pipeline_funnels.jsonl` |
| `regime_transition` | `regime/regime_transitions_YYYY-MM-DD.jsonl` |
| `config_snapshot` | `config_snapshots/config_snapshots_YYYY-MM-DD.jsonl` |
| `deployment` | `deployments/deployments_YYYY-MM-DD.jsonl` |
| `decision_event` | `decisions/decisions_YYYY-MM-DD.jsonl` |
| `parity_event` | `parity/parity_events_YYYY-MM-DD.jsonl` or local-only parity path |

During migration, write both:

- current flat legacy files, so existing tests and status commands continue to
  work;
- new date-partitioned canonical files, so production operation matches the
  optimal target.

## 7. Implementation Sequence

### Phase 0: Baseline And Safety

Before editing the reference repo, run or record these baselines from
`_references/crypto_trader/`:

```bash
pytest tests/test_instrumentation.py
pytest tests/live tests/portfolio tests/parity
pytest tests/
```

Also inspect generated sample files after a small paper/backtest run if the
environment allows it:

```bash
python -m crypto_trader.cli status --state-dir data/live_state
python -m crypto_trader.cli parity-report --state-dir data/live_state
```

Implementation constraints for every phase:

- all new writes must be best-effort and exception-safe;
- all sidecar and Postgres failures must be non-fatal;
- all new fields must have safe defaults;
- no strategy behavior, sizing, risk decision, order status, or broker action
  may depend on whether instrumentation succeeds.

### Phase 1: Add Shared Lineage Helpers

Add:

- `src/crypto_trader/instrumentation/lineage.py`;
- `src/crypto_trader/instrumentation/redaction.py`;
- `src/crypto_trader/instrumentation/config_snapshot.py`;
- tests in `tests/instrumentation/test_lineage.py`.

Implement helpers:

- `canonical_json(value) -> str`;
- `hash_payload(value) -> str`;
- `redact_secrets(value) -> Any`;
- `load_json_or_empty(path) -> dict`;
- `git_code_sha(repo_root) -> str`;
- `build_live_lineage(config, portfolio_config, strategy_configs, repo_root)`.

Thread the resulting `LineageContext` into `LiveEngine`:

1. In `LiveEngine.start()`, after loading `portfolio_config` and strategy
   configs but before wiring strategies, build one lineage context.
2. Store it as `self._lineage`.
3. Pass it into `EventEmitter` or into each collector.
4. Use it for startup deployment/config events.

Do not read the private key or relay secret into payloads. If account identity
is needed, add a field to `LiveConfig`:

```python
account_alias: str = ""
portfolio_id: str = "crypto_perps_main"
family_id: str = "crypto_perps"
```

If those fields are absent, derive safe defaults without using the raw wallet
address.

### Phase 2: Upgrade Event Metadata And Types

Edit:

- `src/crypto_trader/instrumentation/types.py`;
- `src/crypto_trader/instrumentation/__init__.py`;
- `tests/test_instrumentation.py`.

Extend `EventMetadata` with the fields in Section 6.2.

Add event dataclasses:

- `OrderLifecycleEvent`;
- `FilterDecisionEvent`;
- `IndicatorSnapshotEvent`;
- `MarketSnapshotEvent`;
- `PortfolioRuleEvent`;
- `RiskDecisionEvent`;
- `PositionSnapshotEvent`;
- `PortfolioSnapshotEvent`;
- `AllocationSnapshotEvent`;
- `FamilyDailySnapshot`;
- `CorrelationExposureSnapshot`;
- `ConfigSnapshotEvent`;
- `DeploymentEvent`;
- `DecisionTelemetryEvent`;
- optionally `HeartbeatEvent`, or upgrade `HealthReportSnapshot` to carry
  metadata and map to canonical `heartbeat`.

Keep current dataclasses:

- `InstrumentedTradeEvent`;
- `MissedOpportunityEvent`;
- `DailySnapshot`;
- `ErrorEvent`;
- `PipelineFunnelSnapshot`;
- `HealthReportSnapshot`.

Add common helpers:

```python
def with_common_payload(metadata: EventMetadata, payload: dict) -> dict:
    ...

def payload_hash(payload: dict) -> str:
    ...
```

For all event `to_dict()` methods, include:

- top-level `metadata`;
- top-level canonical aliases for frequently queried fields:
  `event_id`, `event_type`, `bot_id`, `strategy_id`, `portfolio_id`,
  `family_id`, `exchange_timestamp`, `lineage`;
- legacy fields exactly where tests and existing consumers expect them.

### Phase 3: Upgrade EventEmitter And Sinks

Edit:

- `src/crypto_trader/instrumentation/emitter.py`;
- `src/crypto_trader/instrumentation/sinks.py`;
- `src/crypto_trader/instrumentation/daily_aggregator.py`;
- `src/crypto_trader/instrumentation/postgres_sink.py`;
- `tests/test_instrumentation.py`.

Add `EventEmitter` methods:

- `emit_order(event)`;
- `emit_filter_decision(event)`;
- `emit_indicator_snapshot(event)`;
- `emit_market_snapshot(event)`;
- `emit_portfolio_rule(event)`;
- `emit_risk_decision(event)`;
- `emit_position_snapshot(event)`;
- `emit_portfolio_snapshot(event)`;
- `emit_allocation_snapshot(event)`;
- `emit_config_snapshot(event)`;
- `emit_deployment(event)`;
- `emit_decision_event(event)`;
- `emit_regime_transition(event)`.

Better implementation:

Use one internal method:

```python
def emit(self, event_type: str, event: Any) -> None:
    for sink in self._sinks:
        method = getattr(sink, f"write_{event_type}", None)
        if method is not None:
            safe_call(method, event)
        generic = getattr(sink, "write_event", None)
        if generic is not None:
            safe_call(generic, event_type, event)
```

Then keep compatibility wrappers like `emit_trade()`.

Add sink behavior:

- `JsonlSink.write_event(event_type, event)` writes canonical
  date-partitioned files;
- `JsonlSink.write_trade()` still writes `instrumented_trades.jsonl`;
- `JsonlSink.write_missed()` still writes `missed_opportunities.jsonl`;
- `JsonlSink.write_daily()` still writes `daily_snapshots.jsonl`;
- `JsonlSink.write_error()` still writes `errors.jsonl`;
- `JsonlSink.write_funnel()` still writes `pipeline_funnels.jsonl`;
- `JsonlSink.write_health_report()` still writes `health_reports.jsonl`;
- `InMemorySink` stores generic events by type;
- `DailyAggregator` ignores unknown generic events unless useful.

For `PostgresSink`, do not block the first implementation on typed tables for
every event. Add a generic `events` table migration first, then typed tables
for dashboards.

### Phase 4: Upgrade Sidecar Contract

Edit:

- `src/crypto_trader/instrumentation/sidecar.py`;
- `src/crypto_trader/relay/app.py` only if needed;
- `src/crypto_trader/relay/store.py` only if needed;
- `tests/test_instrumentation.py`.

Replace `_EVENT_FILES` with a richer map while preserving legacy tests:

```python
_EVENT_FILE_MAP = {
    "instrumented_trades": {"event_type": "trade", "path": "instrumented_trades.jsonl"},
    "missed_opportunities": {"event_type": "missed_opportunity", "path": "missed_opportunities.jsonl"},
    "daily_snapshots": {"event_type": "daily_snapshot", "path": "daily_snapshots.jsonl"},
    "errors": {"event_type": "error", "path": "errors.jsonl"},
    "pipeline_funnels": {"event_type": "pipeline_funnel", "path": "pipeline_funnels.jsonl"},
    "health_reports": {"event_type": "heartbeat", "path": "health_reports.jsonl"},
    "orders": {"event_type": "order", "path": "orders/orders_*.jsonl"},
    "portfolio_rules": {"event_type": "portfolio_rule", "path": "portfolio_rules/portfolio_rules_*.jsonl"},
    "risk_decisions": {"event_type": "risk_decision", "path": "risk_decisions/risk_decisions_*.jsonl"},
    "positions": {"event_type": "position_snapshot", "path": "positions/positions_*.jsonl"},
    "portfolio": {"event_type": "portfolio_snapshot", "path": "portfolio/portfolio_snapshots_*.jsonl"},
    "allocations": {"event_type": "allocation_snapshot", "path": "allocations/allocations_*.jsonl"},
    "config_snapshots": {"event_type": "config_snapshot", "path": "config_snapshots/config_snapshots_*.jsonl"},
    "deployments": {"event_type": "deployment", "path": "deployments/deployments_*.jsonl"},
    "decisions": {"event_type": "decision_event", "path": "decisions/decisions_*.jsonl"}
}
```

Keep a compatibility tuple:

```python
_EVENT_FILES = tuple(_EVENT_FILE_MAP)
```

Sidecar must:

- read flat files and date-partitioned files;
- wrap legacy event payloads into canonical envelopes if they are not already
  wrapped;
- preserve canonical envelopes if already present;
- send canonical `event_type` in the batch payload;
- use event priority for ordering where practical;
- keep byte watermarks per physical file path, not just event type;
- detect truncation and reset watermarks as it already does;
- expose sidecar file paths, canonical event types, watermarks, and send
  failures in health reports.

Relay can remain mostly unchanged if events contain `event_id` at the top
level. If the event ID is only in `metadata`, `RelayStore._extract_event_id()`
already supports that. Prefer top-level `event_id` in the canonical envelope.

Add tests:

- sidecar maps `instrumented_trades` to `trade`;
- sidecar maps `health_reports` to `heartbeat`;
- sidecar wraps legacy payloads into canonical envelopes;
- sidecar preserves already canonical envelopes;
- sidecar sends missed-opportunity backfill revision events with distinct
  `event_id` and shared `logical_event_id`;
- relay stores two revisions of the same logical missed opportunity.

### Phase 5: Strategy-Level Instrumentation Upgrade

Edit:

- `src/crypto_trader/instrumentation/collector.py`;
- `src/crypto_trader/strategy/momentum/strategy.py`;
- `src/crypto_trader/strategy/trend/strategy.py`;
- `src/crypto_trader/strategy/breakout/strategy.py`;
- `src/crypto_trader/core/strategy_runtime.py` if needed to pass decision
  context;
- `tests/test_instrumentation.py`;
- strategy-specific tests if present.

Upgrade `InstrumentationCollector.__init__()`:

```python
def __init__(
    self,
    strategy_id: str,
    bot_id: str = "",
    *,
    lineage: LineageContext | None = None,
    data_source_id: str = "hyperliquid",
) -> None:
    ...
```

Upgrade `begin_bar()` to accept context:

```python
def begin_bar(
    self,
    sym: str,
    bar_close: float = 0.0,
    *,
    bar: Bar | None = None,
    market_event: MarketEvent | None = None,
    decision_id: str | None = None,
    data_source_id: str | None = None,
) -> None:
    ...
```

Store per-symbol current values:

- `bar_id`;
- `timeframe`;
- `bar_open_time`;
- `bar_close_time`;
- `available_at`;
- `decision_id`;
- `exchange_timestamp`;
- `local_timestamp`;
- `data_source_id`;
- `signal_id`;
- `decision_sequence`.

Bar ID rule:

```text
{data_source_id}:{symbol}:{timeframe}:{bar_open_time_iso}
```

Signal ID rule:

```text
{strategy_id}:{symbol}:{timeframe}:{bar_id}:{setup_or_signal_name}
```

Decision ID:

- use `StrategySlotRuntime` decision ID when available;
- otherwise use the same deterministic components as `StrategySlotRuntime`:
  strategy, symbol, timeframe, and available time.

Upgrade `record_gate()`:

- keep current embedded `FilterDecision`;
- add filter index within the decision;
- include `bar_id`, `decision_id`, `timeframe`, and context;
- optionally emit a first-class `FilterDecisionEvent`.

Emission policy:

- emit all post-warmup gates for decisions that reach `setup`;
- emit failed pre-setup gates as aggregate funnel data unless debugging is
  enabled;
- always embed the full gate trail in trade and missed-opportunity events.

Upgrade `snapshot_context()`:

- keep current `MarketContext`;
- include symbol, timeframe, bar ID, decision ID, source, funding rate,
  strategy-specific regime/bias fields, and indicator source;
- emit first-class `IndicatorSnapshotEvent` at actionable decision points:
  setup passed, entry submitted, missed opportunity created, or regime changed.

Upgrade `record_entry()`:

- freeze lineage, `signal_id`, `decision_id`, `bar_id`, strategy config version,
  parameter set ID, portfolio/risk/allocation versions, current filter trail,
  current market context, sizing inputs, and portfolio snapshot;
- accept optional order intent/client order ID if the strategy can know it;
- store enough to link the later completed trade back to the entry decision.

Upgrade `on_trade_closed()`:

Add fields to `InstrumentedTradeEvent`:

- `event_type = "trade"`;
- `family_id`;
- `portfolio_id`;
- `account_alias`;
- `strategy_version`;
- `config_version`;
- `parameter_set_id`;
- `deployment_id`;
- `code_sha`;
- `portfolio_config_version`;
- `risk_config_version`;
- `allocation_version`;
- `strategy_registry_version`;
- `entry_decision_id`;
- `exit_decision_id` if known;
- `entry_signal_id`;
- `entry_bar_id`;
- `exit_bar_id`;
- `entry_order_ids`;
- `exit_order_ids`;
- `entry_fill_ids`;
- `exit_fill_ids`;
- `client_order_ids`;
- `exchange_order_ids`;
- `avg_entry_slippage_pct`;
- `avg_exit_slippage_pct`;
- `entry_latency_ms`;
- `exit_latency_ms`;
- `time_in_trade_seconds`;
- `notional_usd`;
- `initial_risk_amount`;
- `realized_r_net`;
- `geometric_r`;
- `price_pnl_after_funding`;
- `liquidation_distance_pct` when available;
- `leverage` or margin mode when available.

Keep current aliases:

- `pair`;
- `side`;
- `pnl`;
- `r_multiple`;
- `commission`;
- `funding_paid`;
- `entry_signal`;
- `market_context`;
- `portfolio_state_at_entry`.

Upgrade `end_bar()` missed opportunity behavior:

- do not use `datetime.now()` as the exchange timestamp when a bar timestamp is
  available;
- use stable `bar_id` and `decision_id`;
- create `opportunity_id`;
- set `logical_event_id = opportunity_id`;
- set revision `0` for first emission;
- include simulation policy:
  `policy_id`, `horizons`, `tp_model`, `sl_model`, `entry_price_source`,
  `fee_model`, `funding_model`, `version`;
- when backfill changes outcomes, append a new event with the same
  `logical_event_id`, incremented `revision`, and a distinct `event_id`;
- include `supersedes_event_id` for update events.

Crypto-specific missed-opportunity fields:

- `funding_rate`;
- `funding_direction_effect`;
- `perp_symbol`;
- `timeframe`;
- `hypothetical_liquidation_distance_pct` if leverage is available;
- `would_have_hit_tp`, `would_have_hit_sl`;
- `max_favorable_move_pct`, `max_adverse_move_pct` by horizon if available.

### Phase 6: Order And Execution Instrumentation

Edit:

- `src/crypto_trader/core/execution_gateway.py`;
- `src/crypto_trader/live/execution_adapter.py` only if source fields are
  missing;
- `src/crypto_trader/live/oms_store.py` only if additional joins are needed;
- `src/crypto_trader/live/engine.py`;
- `src/crypto_trader/instrumentation/types.py`;
- `tests/parity/*`;
- new `tests/instrumentation/test_order_events.py`.

Best hook:

- keep `ExecutionGateway._emit()` for parity;
- also call an instrumentation adapter that converts `order_intent` and
  `execution` payloads into assistant `OrderLifecycleEvent` payloads.

Do not put network or file I/O directly in `ExecutionGateway`. Instead:

- allow `ExecutionGateway` to accept an optional `instrumentation_emitter` or
  callback;
- use `EventEmitter.emit_order()` inside a safe wrapper;
- or handle conversion in `LiveEngine._record_canonical_event()` where
  `CanonicalRuntimeEvent` is already captured.

Order event stages:

- `intent_created`;
- `portfolio_approved`;
- `portfolio_denied`;
- `size_adjusted`;
- `submitted`;
- `accepted`;
- `rejected`;
- `resting`;
- `partial_fill`;
- `fill`;
- `cancel_requested`;
- `cancelled`;
- `expired`;
- `ttl_cancel_failed`;
- `reconciled`;
- `orphaned`;
- `manual_flatten_detected`.

Required order payload fields:

- `intent_id`;
- `decision_id`;
- `strategy_id`;
- `assistant_strategy_id`;
- `symbol`;
- `side`;
- `order_type`;
- `role` or `tag`;
- `client_order_id`;
- `exchange_order_id`;
- `fill_id`;
- `qty_requested`;
- `qty_filled`;
- `limit_price`;
- `stop_price`;
- `fill_price`;
- `reduce_only`;
- `time_in_force`;
- `ttl_bars`;
- `oca_group`;
- `bracket_group`;
- `risk_R`;
- `risk_metadata`;
- `commission`;
- `liquidity`;
- `reject_reason`;
- `slippage_pct`;
- `latency_ms`;
- `adapter_kind`;
- `exchange`;
- `oms_status`;
- `reconciliation_status`;
- lineage and metadata.

Slippage rule:

- for market entries, compare strategy-visible reference price from order
  metadata or bar close against fill price;
- for limit/stop orders, compare requested limit/stop trigger against fill
  price;
- store null if no fair reference exists rather than inventing a number.

Latency rule:

- if submit time and report timestamp are available, compute milliseconds;
- otherwise null.

### Phase 7: Portfolio Rule And Risk Decision Instrumentation

Edit:

- `src/crypto_trader/portfolio/manager.py`;
- `src/crypto_trader/portfolio/coordinator.py`;
- `src/crypto_trader/portfolio/backtest_runner.py`;
- `src/crypto_trader/instrumentation/types.py`;
- `src/crypto_trader/instrumentation/emitter.py`;
- `tests/portfolio/*`;
- new `tests/instrumentation/test_portfolio_rule_events.py`.

Upgrade `PortfolioRuleResult`:

```python
@dataclass(frozen=True)
class PortfolioRuleEvaluation:
    rule_name: str
    passed: bool
    action: str
    threshold: float | int | str | None = None
    actual_before: float | int | str | None = None
    proposed_delta: float | int | str | None = None
    actual_after: float | int | str | None = None
    margin: float | None = None
    reason: str = ""
    context: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PortfolioRuleResult:
    approved: bool
    denial_reason: str | None = None
    size_multiplier: float = 1.0
    blocking_rule: str | None = None
    evaluations: tuple[PortfolioRuleEvaluation, ...] = ()
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after_preview: dict[str, Any] = field(default_factory=dict)
    allocation: dict[str, Any] = field(default_factory=dict)
```

Modify `PortfolioManager.check_entry()`:

- gather `state_before` before evaluating rules;
- evaluate rules in the same order as today;
- record a `PortfolioRuleEvaluation` for each rule reached;
- preserve the same approval/denial behavior;
- return the richer result;
- do not emit directly from `PortfolioManager` unless an optional emitter is
  injected; a pure return value is easier to test.

Rules that must be represented:

- `entries_blocked_reason`;
- `strategy_registered`;
- `strategy_enabled`;
- `max_total_positions`;
- `strategy_max_concurrent`;
- `heat_cap_R`;
- `directional_cap_R`;
- `priority_headroom_R`;
- `symbol_collision`;
- `symbol_exposure_cap_R`;
- `portfolio_daily_stop_R`;
- `strategy_daily_stop_R`;
- `drawdown_tier`;
- `size_multiplier`.

Modify `BrokerProxy.submit_order()`:

- before `check_entry()`, capture `order_intent` identifiers from order
  metadata: `decision_id`, `intent_id`, `client_order_id`, `risk_R`, `tag`;
- after `check_entry()`, emit `PortfolioRuleEvent`;
- if approved with a multiplier, emit `RiskDecisionEvent` showing original and
  adjusted quantity/risk;
- if denied, emit both an `order` event with stage `portfolio_denied` and a
  `risk_decision` event with action `block`;
- if approved unchanged, emit `risk_decision` action `allow`;
- if approved with size multiplier, emit action `scale`;
- never let emission exceptions affect the returned order status.

Required `PortfolioRuleEvent` fields:

- `event_type = "portfolio_rule"`;
- `decision_id`;
- `intent_id`;
- `client_order_id`;
- `strategy_id`;
- `symbol`;
- `direction`;
- `requested_risk_R`;
- `approved`;
- `action`: `allow`, `block`, or `scale`;
- `denial_reason`;
- `blocking_rule`;
- `size_multiplier`;
- `adjusted_risk_R`;
- `state_before`;
- `state_after_preview`;
- `evaluations`;
- `allocation`;
- `portfolio_config`;
- `lineage`;
- `portfolio_config_version`;
- `risk_config_version`;
- `allocation_version`.

Required `RiskDecisionEvent` fields:

- `risk_decision_id`;
- `decision_id`;
- `intent_id`;
- `order_id`;
- `strategy_id`;
- `symbol`;
- `direction`;
- `requested_qty`;
- `approved_qty`;
- `requested_risk_R`;
- `approved_risk_R`;
- `action`;
- `reason`;
- `rule_event_id`;
- `portfolio_state_before`;
- `portfolio_state_after_preview`;
- `lineage`.

Backtest parity requirement:

- `portfolio/backtest_runner.py` should emit the same `PortfolioRuleEvent`
  schema in backtests, at least to local artifacts, so monthly validation can
  compare live and replay portfolio behavior.

### Phase 8: Position, Portfolio, Allocation, Family, And Exposure Snapshots

Edit:

- `src/crypto_trader/live/engine.py`;
- `src/crypto_trader/portfolio/state.py` if helper methods are useful;
- `src/crypto_trader/instrumentation/types.py`;
- `src/crypto_trader/instrumentation/emitter.py`;
- `src/crypto_trader/instrumentation/sinks.py`;
- `tests/live/*`;
- new snapshot tests.

Position snapshots:

Emit on:

- equity snapshot interval;
- entry fill;
- exit fill;
- reconciliation discrepancy;
- startup after OMS rehydration;
- shutdown if practical.

Fields:

- `position_instance_id`;
- `strategy_id`;
- `symbol`;
- `direction`;
- `qty`;
- `avg_entry`;
- `mark_price`;
- `notional_usd`;
- `unrealized_pnl`;
- `realized_pnl`;
- `risk_R`;
- `stop_price`;
- `liquidation_price` if available;
- `liquidation_distance_pct` if available;
- `fees_paid`;
- `funding_paid`;
- `mfe_r`;
- `mae_r`;
- `open_order_ids`;
- `entry_time`;
- `source`: `broker`, `oms`, `lifecycle`, or `reconciler`;
- lineage.

Portfolio snapshots:

Emit on:

- equity snapshot interval;
- entry registration;
- exit registration;
- daily reset;
- startup after portfolio state restore;
- reconciliation events.

Fields:

- `portfolio_id`;
- `account_alias`;
- `equity`;
- `peak_equity`;
- `drawdown_pct`;
- `total_heat_R`;
- `heat_cap_R`;
- `directional_risk_R` by side;
- `symbol_risk_R` by symbol and side;
- `open_risk_count`;
- `max_total_positions`;
- `portfolio_daily_pnl_R`;
- `portfolio_daily_stop_R`;
- `strategy_daily_pnl_R`;
- `open_risks`;
- `positions_count`;
- `pending_orders_count`;
- `timestamp`;
- lineage.

Allocation snapshots:

Emit on:

- startup;
- portfolio config change detection;
- deployment event;
- daily snapshot.

Fields:

- `allocation_version`;
- `portfolio_config_version`;
- `strategy_allocations`;
- enabled strategies;
- disabled strategies;
- base risk percentages;
- max concurrent limits;
- daily stop limits;
- priorities;
- heat/directional/symbol caps;
- drawdown tiers;
- priority headroom settings;
- lineage.

Final portfolio closeout reconciliation:

For this repo, reconcile the crypto portfolio as one family unless a future
split is introduced:

- `family_id = crypto_perps`;
- strategies: `momentum`, `trend`, `breakout`;
- symbols: `BTC`, `ETH`, `SOL` by default.

Fields:

- family-level trades;
- net PnL;
- gross PnL;
- fees;
- funding;
- realized R;
- win/loss count;
- max drawdown;
- exposure;
- missed opportunities;
- missed opportunities that would have won;
- process quality distribution;
- root-cause distribution;
- per-strategy summary;
- per-symbol summary;
- portfolio heat summary;
- risk-denial counts;
- lineage.

Approval-ready snapshot scope:

For the `approval_ready` path, keep snapshots tied to startup, fills,
reconciliation, risk-control changes, and closeout. Do not make
monitoring-only analytics part of the approval gate. Portfolio-rule and
risk-decision events should already carry the directional, symbol, and heat/cap
state that affected each live decision.

### Phase 9: Config Snapshot And Deployment Events

Edit:

- `src/crypto_trader/live/config.py`;
- `src/crypto_trader/live/engine.py`;
- `src/crypto_trader/instrumentation/config_snapshot.py`;
- `src/crypto_trader/instrumentation/types.py`;
- `tests/live/test_config.py`;
- new `tests/instrumentation/test_config_snapshot.py`.

At live startup, emit:

1. `deployment` event;
2. `config_snapshot` for live config;
3. `config_snapshot` for portfolio config;
4. `config_snapshot` for each strategy config;
5. `allocation_snapshot`;
6. initial `portfolio_snapshot`;
7. approval-grade `deployment_metadata.json`;
8. initial `heartbeat`.

Deployment event fields:

- `deployment_id`;
- `deployment_manifest_path`;
- `deployment_manifest_version`;
- `candidate`;
- `portfolio_round`;
- `required_strategy_ids`;
- `strategy_config_paths`;
- `portfolio_config_path`;
- `portfolio_rounds_manifest_path`;
- `parity_alignment_path`;
- `code_sha`;
- `started_at`;
- `is_testnet`;
- `symbols`;
- `bot_id`;
- `family_id`;
- `portfolio_id`;
- `account_alias`;
- all config version fields.

Approval-grade deployment metadata:

The `deployment` event is useful telemetry, but `approval_ready` requires a
dedicated runtime-emitted metadata artifact that can be imported into
`contracts/assistant_bridges/<bridge_id>/deployment_metadata.json`.
Generate it from the running live bot or VPS process, not from a local
assistant checkout.

Write one artifact per bridge:

- `crypto_trend_v1`;
- `crypto_momentum_v1`;
- `crypto_breakout_v1`.

Required fields:

```json
{
  "metadata_source": "live_bot_runtime_deployment_metadata_v1",
  "emission_environment": "production_vps",
  "repo_url": "https://github.com/<owner>/<repo>",
  "source_control_origin": "https://github.com/<owner>/<repo>",
  "deployed_commit_sha": "<git commit sha>",
  "source_control_commit_sha": "<same git commit sha>",
  "source_control_worktree_clean": true,
  "bot_id": "crypto_portfolio",
  "portfolio_id": "crypto_portfolio",
  "strategy_id": "crypto_trend_v1|crypto_momentum_v1|crypto_breakout_v1",
  "config_hash": "<effective strategy/portfolio config hash>",
  "strategy_version": "<deployed strategy/package version>",
  "config_version": "<effective strategy/portfolio config version>",
  "deployment_id": "<stable deployment id>",
  "telemetry_schema_version": "trade_event_v1",
  "strategy_plugin_contract_path": "contracts/assistant_bridges/<bridge_id>/strategy_plugin_contract.json",
  "strategy_plugin_contract_hash": "<sha256 of final strategy_plugin_contract.json>",
  "emitted_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "live_runtime_started_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "runtime_entrypoint": "<module:function or command>",
  "runtime_instance_id": "<stable process/session id>",
  "runtime_host_fingerprint": "<non-secret host fingerprint>",
  "dry_run": false
}
```

`metadata_source` must be `live_bot_runtime_deployment_metadata_v1` or
`vps_live_bot_runtime_deployment_metadata_v1`. `emission_environment` must be
`live_bot`, `vps`, `paper_vps`, or `production_vps`; prefer `paper_vps` or
`production_vps` when emitted on a VPS. `repo_url` and
`source_control_origin` must be the same real Git remote, never `local://`.
`source_control_worktree_clean` must be `true`. `source_control_commit_sha`
must match `deployed_commit_sha`.

Promotion rule:

Do not set any crypto bridge to `maturity = "approval_ready"` until this
metadata artifact exists, its contract hash matches the final contract file,
the deployed SHA matches `live_repo_commit_sha`, and the five validation tests
still pass.

Config snapshot event fields:

- `config_kind`: `live`, `strategy`, `portfolio`, `risk`, `allocation`,
  `asset_meta`, `deployment_manifest`;
- `config_path`;
- `config_version`;
- `redacted_config`;
- `hash_inputs`;
- `loaded_at`;
- `lineage`.

Config watcher:

- If this repo does not hot-reload configs, the first implementation can emit
  startup snapshots only.
- Add a lightweight watcher later that detects file hash changes and emits
  `config_snapshot` and `parameter_change` events without applying changes.
- If the live bot does apply a config change in the future, emit before/after
  snapshots and require human approval outside this repo.

### Phase 10: Decision Parity Upgrade

Edit:

- `src/crypto_trader/core/runtime_types.py`;
- `src/crypto_trader/core/strategy_runtime.py`;
- `src/crypto_trader/live/engine.py`;
- `src/crypto_trader/parity/report.py`;
- `src/crypto_trader/parity/shadow.py`;
- `src/crypto_trader/instrumentation/collector.py`;
- tests in `tests/parity`.

Current `StrategySlotRuntime` is a strong starting point. Keep it.

Upgrade `DecisionEvent` to include:

- `decision_id`;
- `strategy_id`;
- `assistant_strategy_id`;
- `symbol`;
- `timeframe`;
- `bar_id`;
- `decision_time`;
- `decision_key`;
- `action`;
- `order_count`;
- `signal_context`;
- `filter_summary`;
- `indicator_summary`;
- `market_context_hash`;
- `strategy_state_hash` if supported by strategy snapshots;
- `portfolio_state_hash` if available;
- `lineage`.

`InstrumentationCollector` should expose a compact decision summary:

```python
def decision_summary(self, sym: str) -> dict[str, Any]:
    return {
        "signal_id": ...,
        "gates": ...,
        "failed_gate": ...,
        "market_context_hash": ...,
        "signal_factors": ...,
    }
```

`StrategySlotRuntime._emit_decision_event()` should merge strategy collector
summary into the canonical decision event when available.

Parity normalization rules:

- do not compare wall-clock local timestamps;
- compare deterministic market availability time, decision ID, action,
  strategy, symbol, timeframe, order count, and normalized signal/filter
  outcomes;
- round floats before comparison or compare with tolerance;
- keep full raw payload for diagnosis but compare normalized fields.

Forwarding rule:

- continue writing `parity_events.jsonl`;
- also emit assistant `decision_event` payloads through `EventEmitter` or
  sidecar for monthly review;
- optionally keep raw parity events local-only if payload volume is high, but
  then emit summarized decision events with enough drift evidence.

### Phase 11: Error And Heartbeat Coverage

Edit:

- `src/crypto_trader/live/engine.py`;
- `src/crypto_trader/live/health.py`;
- `src/crypto_trader/live/health_report.py`;
- `src/crypto_trader/instrumentation/types.py`;
- tests for health/relay status.

Error event policy:

Emit structured `ErrorEvent` for:

- sidecar send failures after retry exhaustion;
- stale feed critical status;
- broker submit/cancel failures;
- execution report rejections;
- reconciliation discrepancies;
- fill processing failures;
- daily snapshot errors;
- health report errors;
- funnel report errors;
- Postgres sink initialization/write failures if enabled;
- config/deployment validation warnings at startup.

Error payload fields:

- `error_type`;
- `severity`;
- `message`;
- `stack_trace`;
- `component`;
- `strategy_id` if known;
- `symbol` if known;
- `order_id` or `fill_id` if known;
- `decision_id` if known;
- `recovery_action`;
- lineage.

Heartbeat event policy:

Map `HealthReportSnapshot` to canonical `heartbeat`.

Add:

- metadata;
- lineage;
- `uptime_sec`;
- `assessment`;
- `alerts`;
- `stale_feeds`;
- `relay`;
- `sidecar`;
- `positions_summary`;
- `portfolio_state`;
- `oms_summary`;
- `last_market_event_at`;
- `last_order_event_at`;
- `last_trade_event_at`;
- `last_successful_sidecar_send_at`.

### Phase 12: Postgres Optional Persistence

Edit:

- `infra/postgres/migrations/001_tables.sql`;
- add `003_instrumentation_events.sql` or similar;
- `src/crypto_trader/instrumentation/postgres_sink.py`;
- dashboard views if any.

The assistant should not depend on Postgres. JSONL plus sidecar is the primary
contract. Postgres is useful for local dashboards and operator queries.

Add a generic event table first:

```sql
CREATE TABLE IF NOT EXISTS instrumentation_events (
    event_id TEXT PRIMARY KEY,
    logical_event_id TEXT,
    event_type TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    family_id TEXT,
    portfolio_id TEXT,
    account_alias TEXT,
    strategy_id TEXT,
    symbol TEXT,
    exchange_timestamp TIMESTAMPTZ,
    local_timestamp TIMESTAMPTZ,
    payload JSONB NOT NULL,
    lineage JSONB,
    received_at TIMESTAMPTZ DEFAULT now()
);
```

Add indexes:

- `(event_type, exchange_timestamp DESC)`;
- `(strategy_id, exchange_timestamp DESC)`;
- `(portfolio_id, exchange_timestamp DESC)`;
- `(logical_event_id)`;
- `(payload->>'decision_id')`;
- `(payload->>'bar_id')`.

Typed tables can follow for:

- order events;
- portfolio rule events;
- position snapshots;
- portfolio snapshots;
- config snapshots;
- deployment events.

### Phase 13: Relay Compatibility

The current relay can keep its endpoints:

- `POST /events`;
- `GET /events`;
- `POST /ack`;
- `GET /health`;
- `POST /admin/purge`.

But after sidecar canonicalization:

- `event_type` in relay rows should be canonical;
- `payload` should be the canonical event envelope;
- `event_id` should be top-level;
- revised missed opportunities should insert as separate rows because their
  `event_id` differs;
- the stable opportunity ID should be available as `logical_event_id`.

Optional relay improvement:

- add columns for `logical_event_id`, `strategy_id`, `portfolio_id`, and
  `exchange_timestamp` extracted from the envelope for faster assistant polling
  and diagnostics.

Do not block the initial instrumentation upgrade on a relay schema migration if
the current JSON payload storage is sufficient.

## 8. File-By-File Edit Guide

### 8.1 `instrumentation/lineage.py` New File

Implement:

- `LineageContext`;
- canonical hash helpers;
- config subset extraction helpers;
- redaction helpers or import from `redaction.py`;
- `from_live_engine_inputs()`;
- `for_strategy()`;
- `for_portfolio()`;
- `metadata_defaults()`.

Test:

- hashes stable under key ordering changes;
- redaction removes secrets recursively;
- portfolio/risk/allocation versions differ when relevant fields change;
- strategy config version differs per strategy;
- deployment ID changes when manifest or code SHA changes.

### 8.2 `instrumentation/types.py`

Edit existing dataclasses and add new ones.

Specific changes:

- extend `EventMetadata`;
- add `lineage` dictionaries to all current events;
- add top-level aliases in `to_dict()`;
- add `event_type` to every event;
- add revision handling to `MissedOpportunityEvent`;
- add new event types listed in Phase 2;
- keep root-cause taxonomy unchanged unless the assistant taxonomy changes.

Test:

- legacy `to_dict()` fields still exist;
- metadata includes assistant strategy ID;
- lineage fields appear on trade, missed, daily, error, funnel, heartbeat;
- missed updates use distinct event IDs.

### 8.3 `instrumentation/collector.py`

Specific changes:

- accept lineage;
- store current bar/decision context;
- build stable bar IDs and signal IDs;
- emit optional standalone filter/indicator events;
- freeze entry lineage and decision IDs;
- enrich trade events;
- fix missed-opportunity identity and revision behavior.

Test:

- `begin_bar()` remains backward compatible;
- `record_gate()` still calculates margin;
- missed opportunity uses bar timestamp when available;
- two missed signals on different bars get different logical IDs;
- backfill update keeps logical ID and changes event ID;
- trade event contains decision/bar/lineage fields.

### 8.4 `instrumentation/emitter.py`

Specific changes:

- add generic `emit(event_type, event)`;
- keep compatibility wrappers;
- centralize sink exception handling;
- optionally count sink failures for heartbeat.

Test:

- one bad sink does not stop good sinks;
- generic event reaches sinks;
- compatibility methods still work.

### 8.5 `instrumentation/sinks.py`

Specific changes:

- add `write_event`;
- add canonical date-partitioned file map;
- keep legacy flat files;
- add `InMemorySink.events_by_type`;
- ensure writes are append-only and UTF-8.

Test:

- canonical and legacy files are both written for legacy events;
- new event types write to expected directories;
- JSON lines are valid.

### 8.6 `instrumentation/sidecar.py`

Specific changes:

- replace `_EVENT_FILES` internals with `_EVENT_FILE_MAP`;
- keep `_EVENT_FILES` compatibility;
- support globbed date-partitioned files;
- watermark per physical file;
- canonicalize event types;
- wrap legacy payloads;
- preserve canonical payloads;
- expose richer status.

Test:

- watermarks survive restart;
- truncation resets safely;
- canonical mapping works;
- missed revision events are both forwarded;
- health reports map to heartbeat.

### 8.7 `live/config.py`

Specific changes:

- add optional `family_id`, `portfolio_id`, `account_alias`;
- ensure `to_dict(redacted=True)` or a separate redaction function is used for
  snapshots;
- do not include secrets in snapshots.

Test:

- `from_dict()` supports new fields;
- defaults are safe;
- redacted config does not contain sensitive values.

### 8.8 `live/engine.py`

Specific changes:

- build `LineageContext` at startup;
- pass lineage into collectors or emitter;
- emit deployment/config/allocation/initial snapshots;
- convert `CanonicalRuntimeEvent` streams into assistant order, market, and
  decision events as needed;
- emit position and portfolio snapshots in the equity loop;
- emit portfolio snapshots on startup, fills, reconciliation, risk-control
  changes, and closeout;
- emit structured errors in exception handlers;
- include metadata and lineage in funnel and heartbeat events.

Be careful:

- `_record_canonical_event()` is currently synchronous file I/O. Keep any new
  work small and exception-safe;
- do not forward every raw market bar if volume becomes excessive. At minimum,
  emit decision-linked market snapshots and keep full parity local;
- startup snapshot failures must not stop live trading.

### 8.9 `core/strategy_runtime.py`

Specific changes:

- expose decision context to strategy collectors if needed;
- enrich `DecisionEvent`;
- include `bar_id`;
- keep parity deterministic.

Avoid:

- changing the strategy `on_bar()` API unless absolutely necessary;
- adding nondeterministic fields to parity comparison keys.

### 8.10 `core/execution_gateway.py`

Specific changes:

- either accept an optional assistant event callback or rely on
  `LiveEngine._record_canonical_event()` conversion;
- include submit timestamp in order intent metadata if needed for latency;
- preserve existing parity stream emission.

Avoid:

- direct sidecar/network writes in the gateway;
- changing returned visible order IDs.

### 8.11 `portfolio/manager.py`

Specific changes:

- enrich `PortfolioRuleResult`;
- collect rule evaluations;
- include state before and preview after;
- preserve exact approval and denial semantics.

Test with existing portfolio tests to ensure no behavior drift.

### 8.12 `portfolio/coordinator.py`

Specific changes:

- pass an optional event emitter or callback into `BrokerProxy`;
- emit `portfolio_rule`, `risk_decision`, and portfolio-denied order events;
- include order metadata and decision IDs;
- continue returning rejected order IDs exactly as before.

### 8.13 `portfolio/backtest_runner.py`

Specific changes:

- ensure backtest portfolio rule artifacts use the same schema as live;
- use the same lineage/version fields when available;
- keep existing portfolio sweep outputs intact.

### 8.14 `relay/store.py`

Optional changes:

- extract and store `logical_event_id`;
- extract and store `exchange_timestamp`;
- expose event type counts in health.

Required only if assistant polling needs faster filtering. The current JSON
payload store can carry canonical events without schema changes.

### 8.15 Tests

Extend:

- `tests/test_instrumentation.py`;
- `tests/live/test_config.py`;
- `tests/live/test_audit_fixes.py`;
- `tests/portfolio/*`;
- `tests/parity/*`.

Add:

- `tests/instrumentation/test_lineage.py`;
- `tests/instrumentation/test_canonical_envelope.py`;
- `tests/instrumentation/test_order_events.py`;
- `tests/instrumentation/test_portfolio_rule_events.py`;
- `tests/instrumentation/test_snapshots.py`;
- `tests/instrumentation/test_config_snapshot.py`;
- `tests/instrumentation/test_decision_events.py`.

## 9. Required Event Payload Details

### 9.1 Completed Trade

Required payload fields:

- identity: `trade_id`, `event_id`, `logical_event_id`, `strategy_id`,
  `assistant_strategy_id`, `symbol`, `pair`, `side`;
- lineage: all fields from `LineageContext.for_strategy()`;
- decision links: `entry_decision_id`, `exit_decision_id`,
  `entry_signal_id`, `entry_bar_id`, `exit_bar_id`;
- order links: `entry_order_ids`, `exit_order_ids`, `entry_fill_ids`,
  `exit_fill_ids`, `client_order_ids`, `exchange_order_ids`;
- timing: `entry_time`, `exit_time`, `time_in_trade_seconds`;
- economics: `entry_price`, `exit_price`, `position_size`, `notional_usd`,
  `price_pnl_gross`, `funding_paid`, `total_fees`,
  `price_pnl_after_funding`, `realized_pnl_net`, `pnl`, `pnl_pct`,
  `r_multiple`, `realized_r_net`, `geometric_r`;
- execution: slippage, latency, liquidity if available;
- signal: `entry_signal`, `entry_signal_strength`, `setup_grade`,
  `confluences`, `entry_method`, `signal_factors`;
- filters: full `filter_decisions`, `passed_filters`, `active_filters`;
- market: `market_context`, `funding_rate`, regime/bias fields;
- risk: `sizing_inputs`, `portfolio_state_at_entry`,
  `portfolio_rule_event_id`, `risk_decision_id`;
- quality: `mfe_r`, `mae_r`, `exit_efficiency`,
  `process_quality_score`, `root_causes`;
- post-exit: 1h/4h move fields when backfilled.

### 9.2 Missed Opportunity

Required payload fields:

- `opportunity_id`;
- `logical_event_id`;
- revision-specific `event_id`;
- `revision`;
- `supersedes_event_id`;
- `strategy_id`;
- `symbol`;
- `timeframe`;
- `bar_id`;
- `decision_id`;
- `signal_id`;
- `blocked_by`;
- `block_reason`;
- `blocking_rule_type`: `strategy_filter`, `portfolio_rule`, `risk_rule`,
  `order_reject`, or `data_gap`;
- `margin_pct`;
- `hypothetical_entry`;
- `simulation_policy`;
- `market_context`;
- `filter_decisions`;
- `portfolio_rule_event_id` if blocked by portfolio;
- `outcome_1h`, `outcome_4h`, `outcome_24h`;
- `would_have_hit_tp`, `would_have_hit_sl`;
- `backfill_status`;
- lineage.

### 9.3 Filter Decision

Required payload fields:

- `filter_event_id`;
- `decision_id`;
- `bar_id`;
- `signal_id`;
- `strategy_id`;
- `symbol`;
- `timeframe`;
- `filter_name`;
- `filter_index`;
- `passed`;
- `threshold`;
- `actual_value`;
- `margin_pct`;
- `reason`;
- `context`;
- lineage.

### 9.4 Order

Required payload fields are listed in Phase 6. The most important join keys are:

- `decision_id`;
- `intent_id`;
- `client_order_id`;
- `exchange_order_id`;
- `fill_id`;
- `trade_id` when known;
- `portfolio_rule_event_id`;
- `risk_decision_id`;
- `bar_id`;
- lineage.

### 9.5 Portfolio Rule

Required payload fields are listed in Phase 7. The most important requirement
is that an approval must be as observable as a denial. The assistant cannot
learn portfolio behavior if it only sees blocks.

### 9.6 Snapshots

Snapshot events should be versioned evidence, not vague status logs.

Each snapshot must include:

- timestamp;
- lineage;
- source;
- complete enough state to reconstruct the risk decision context;
- coverage/null explanations for fields not available from Hyperliquid.

## 10. Crypto-Specific Requirements

Because this repo trades crypto perpetual futures, include these fields wherever
available:

- exchange: `hyperliquid`;
- venue environment: `testnet` or `mainnet`;
- funding rate at signal/entry/exit;
- funding paid on closed trades;
- expected funding direction effect for long/short signals;
- leverage or margin mode if available;
- liquidation price or liquidation distance if available;
- tick size and lot size versions from `AssetMeta`;
- 24/7 UTC session/day boundary assumptions;
- symbol universe and symbol enablement;
- exchange order IDs and client order IDs;
- reduce-only flags;
- time-in-force and TTL emulation fields;
- OCA/bracket group IDs;
- manual flatten/reconciliation flags.

Do not emit:

- private key;
- raw relay secret;
- raw Postgres DSN;
- raw wallet address unless the user explicitly configures it as public-safe;
- exchange API tokens;
- full stack traces containing secrets without redaction.

## 11. Verification Plan

### 11.1 Unit Tests

Run:

```bash
pytest tests/test_instrumentation.py
pytest tests/instrumentation
pytest tests/portfolio
pytest tests/parity
pytest tests/live
```

Required assertions:

- metadata includes lineage on every event type;
- config hashes are stable and secret-safe;
- sidecar canonicalizes event types;
- sidecar preserves watermarks per file;
- relay accepts revised missed opportunities;
- portfolio rules emit allow, block, and scale events;
- order intent and execution events emit assistant order events;
- decision parity events remain deterministic;
- legacy files and fields still exist.

### 11.2 End-To-End Synthetic Day

Create a synthetic live/paper run or test fixture that produces:

- at least one approved entry;
- at least one portfolio-denied entry;
- at least one size-adjusted entry if drawdown tier can be simulated;
- at least one completed trade;
- at least one missed opportunity with backfill update;
- at least one order reject;
- one health report;
- one funnel report;
- one config snapshot;
- one deployment event;
- one position snapshot;
- one portfolio snapshot;
- one allocation snapshot;
- one decision event.

Then assert:

- every event has a canonical envelope;
- every monthly-critical event has lineage;
- related events join by `decision_id`, `bar_id`, `intent_id`, or
  `logical_event_id`;
- sidecar sends canonical event types;
- relay stores all expected events;
- no secrets are present in any JSONL or relay payload.

### 11.3 Manual Inspection Commands

From `_references/crypto_trader/`:

```bash
rg -n "\"private_key\"|\"relay_secret\"|postgres_dsn|0x" data/live_state instrumentation/data
rg -n "\"event_type\": \"portfolio_rule\"" data/live_state instrumentation/data
rg -n "\"event_type\": \"order\"" data/live_state instrumentation/data
rg -n "\"event_type\": \"deployment\"" data/live_state instrumentation/data
rg -n "\"logical_event_id\"" data/live_state instrumentation/data
```

Use the status commands:

```bash
python -m crypto_trader.cli status --state-dir data/live_state
python -m crypto_trader.cli parity-report --state-dir data/live_state
python -m crypto_trader.cli parity-gate path/to/parity_report.json
```

## 12. Acceptance Matrix And Finite Checklist

This section is the release gate. An implementation is not complete because the
tests pass or because events are present. It is complete only when every row in
the matrix passes and every checklist item is checked, or the item is marked
not applicable with a short written rationale and a test that proves the
exclusion is safe.

### 12.1 Acceptance Matrix

| Area | Acceptance criterion | Required evidence | Fails if |
|---|---|---|---|
| Canonical envelope | Every relay-bound JSONL row is wrapped in `assistant_event_v1` with top-level `event_id`, `logical_event_id`, `event_type`, timestamps, lineage, source, and payload. | Unit tests over all sink writers plus synthetic-day JSONL inspection. | Any relay-bound row is legacy-only, lacks top-level identity, or relies on nested metadata identity only. |
| Event type names | Relay-bound events use canonical assistant event types, not old file names or parity-only names. | Sidecar canonicalization tests and sidecar payload fixture. | Relay receives `instrumented_trades`, `missed_opportunities`, or other file-derived names as event types. |
| Lineage completeness | Every monthly-critical event includes bot, family, portfolio, account alias, strategy when applicable, config versions, deployment ID, code SHA, and symbol universe context. | Parametrized lineage tests for trade, missed, filter, order, fill, portfolio rule, risk decision, snapshots, config, deployment, heartbeat, error. | Any critical event needs another file to recover basic deployment/config identity. |
| Deployment event identity | Deployment `event_id` and `logical_event_id` both equal `deployment_id`. | Startup test asserting the canonical envelope identity, not just the payload field. | `_emit_assistant_payload()` derives deployment identity from timestamped payload content. |
| Config snapshot identity | Config snapshot `event_id` and `logical_event_id` are the same stable hash of config kind and config version. | Restart test proving same config version gives same identity and changed version gives a new identity. | Identity includes `loaded_at`, `runtime_instance_id`, deployment ID, path-only logical ID, or other restart-varying fields. |
| Secret redaction | Config/deployment snapshots include redacted effective config and hash inputs without private keys, wallet secrets, relay secrets, API tokens, or raw Postgres DSNs. | Redaction tests plus `rg` inspection commands from Section 11.3. | Any secret-bearing key or configured secret value appears in JSONL, relay payload, or metadata artifact. |
| Deployment metadata path | Runtime emits one importable `deployment_metadata.json` per bridge at `contracts/assistant_bridges/<bridge_id>/deployment_metadata.json`, or emits there and optionally copies to state dir. | Startup artifact test against the contract path for `crypto_trend_v1`, `crypto_momentum_v1`, and `crypto_breakout_v1`. | Metadata exists only under `state_dir/deployment_metadata` with no contract-tree artifact/import step. |
| Deployment metadata approval | Each bridge metadata artifact is approval-grade when run from the intended deployment checkout: real repo URL, clean worktree, commit SHA, contract hash, config hash, schema versions, host/runtime fields, and no blockers. | Test fixture with bridge contracts plus a production/paper runtime smoke test recording `approval_ready: true`. | Tests merely assert `approval_ready` is a boolean or accept missing contract hashes as complete. |
| Backtest/live portfolio schema | Backtest portfolio evidence uses the same rich schema as live for `portfolio_rule`, `risk_decision`, and portfolio-rejected `order` events: evaluations, before state, after preview, size multiplier, lineage, `rule_event_id`, `risk_decision_id`, decision/order join keys. | Backtest runner tests comparing required key sets with live `portfolio_rule`, `risk_decision`, and rejected `order` payloads. | Backtest stores only `RuleEvent`, drops first-class `risk_decision` or rejected-order events, or keeps a lossy approval-only legacy shape. |
| Portfolio approvals and denials | Approved, denied, and size-adjusted entries emit first-class `portfolio_rule` and `risk_decision` events. | Tests covering allow, deny, and scale paths. | Only denials emit evidence, or risk decisions are implied by portfolio rule text. |
| Portfolio join keys | Portfolio rule/risk events join to decision, strategy, symbol, order intent/client order ID, config, and deployment evidence. | Synthetic-day graph assertion from decision to portfolio rule to order. | Any portfolio rule cannot be joined back to its decision and intended order. |
| Decision event identity | Decision event `event_id` and `logical_event_id` equal stable `decision_id`. | Runtime and live engine tests checking the emitted canonical envelope. | `decision_event_id` exists in payload but is not promoted to the envelope identity. |
| Market snapshot identity | Market snapshot identity is the stable bar/market snapshot ID, not a hash of a timestamped payload. | Runtime/live tests checking `event_id` and `logical_event_id` for market snapshots. | `market_snapshot_id` or `bar_id` exists only inside payload. |
| Decision determinism | Replaying the same bar/config emits the same enriched decision IDs, market IDs, filter IDs, and order intent IDs, excluding allowed execution-only differences. | Parity replay test and known-differences manifest. | Decision evidence changes because of wall-clock time, UUIDs, random order IDs, or dictionary ordering. |
| Entry order IDs | Trend, breakout, and momentum entry order intent IDs are deterministic from decision context, strategy, symbol, side, and order sequence. | Strategy/runtime tests proving stable entry IDs across replay. | Entry orders use `uuid.uuid4()` or timestamp-only identity. |
| Protective order IDs | Protective stop, break-even, trailing, lock, TP, time-stop, quick-exit, invalidation, follow-through, MFE-retrace, cancel/resubmit, and replacement order IDs are deterministic from position/decision/order context and lifecycle sequence for trend, breakout, and momentum. | Strategy tests for trend, breakout, and momentum protective lifecycle replay, including exit-manager-created market exits, stop resubmits, and no-context management orders. | Management orders use blank IDs, `uuid.uuid4()`, or repeated fallback identities such as `<strategy>:<symbol>:manual:intent:1` outside a decision context. |
| Order lifecycle events | Order intents, execution reports, fills, cancellations, rejects, TTL expirations, and TTL cancel failures are first-class assistant events with lineage. | Execution gateway/live adapter tests plus synthetic-day order lifecycle graph. | Lifecycle changes are only logged, only persisted in OMS, or not forwarded through instrumentation. |
| Fill evidence | Each fill emits a first-class fill event joined to order, decision, strategy, position, portfolio/config/deployment evidence, and OMS status. | Fill routing tests plus synthetic day with at least one fill. | Fill is visible only as a trade side effect or broker log. |
| Completed trade evidence | Completed trades join to decision, order, fill, portfolio rule, risk decision, config, deployment, funding, and exchange/client order IDs where available. | Closed-trade test and synthetic-day join assertion. | A trade can be reviewed only by matching symbol/time heuristics. |
| Missed opportunity initial event | Missed opportunities use bar/decision identity, stable `logical_event_id`, blocker/gate context, lineage, and simulation policy. | Collector identity tests. | Initial missed opportunities use `datetime.now()` or unstable UUID identity. |
| Missed opportunity revisions | Backfilled missed opportunity updates emit revision-specific `event_id`, preserve stable `logical_event_id`, increment `revision`, and set `supersedes_event_id`. | Relay dedup test storing at least two revisions of one logical missed opportunity. | Backfilled updates reuse the original `event_id` or mutate an already-forwarded row without a new revision. |
| Startup sequencing | Live startup syncs durable/exchange open orders and seeds TTL trackers before startup portfolio/position/allocation snapshots. | Startup lifecycle test with pre-existing exchange/OMS open orders. | First startup snapshot can report no pending orders while exchange orders exist. |
| Snapshot triggers | Position, portfolio, allocation, family, and exposure snapshots exist at startup, fills, reconciliation, risk-control changes, and closeout. | Tests or synthetic-day assertions for each trigger. | Snapshots are only periodic health data or only emitted at startup. |
| Sidecar legacy rollout | Sidecar forwards unread legacy files even after canonical files exist for the same event type. | Migration test with unread legacy bytes plus canonical rows. | Legacy rows are skipped once any canonical file exists. |
| Sidecar watermarks | Sidecar watermarks are per file/source and survive restart without duplicating or dropping rows. | Sidecar restart test over legacy and canonical files. | Watermarks are keyed only by event type or reset on restart. |
| Sidecar health | Health reports expose canonical files, watermarks, relay status, last success, consecutive failures, and last error. | Health report test and CLI status output. | Operators cannot see which canonical files are pending or stuck. |
| Error events | Sidecar retry exhaustion, enabled Postgres init/write failures, broker submit/cancel failures, feed critical status, reconciliation discrepancies, fill errors, daily/funnel/health errors, and startup validation warnings emit structured `ErrorEvent`s with lineage or a documented startup fallback before lineage exists. | Fault-injection tests for each enabled path, including Postgres initialization before lineage is loaded. | Failures are only logged or counted, or pre-lineage startup failures silently skip assistant-visible error evidence. |
| Postgres optionality | Postgres is optional and fail-open; JSONL plus sidecar remains the primary contract. | Tests with Postgres disabled, missing, and failing. | Trading or JSONL instrumentation depends on Postgres availability. |
| Postgres JSON shape | Postgres stores canonical envelopes and indexes the actual JSON paths for decision ID, bar ID, logical event ID, strategy, portfolio, and event type. | Migration test explaining/querying indexes against inserted canonical rows. | Indexes target `payload->>'decision_id'` when the key lives under `payload->'payload'`. |
| Relay dedup | Relay deduplicates by top-level `event_id`, falls back to nested metadata only for legacy payloads, and allows multiple revisions for the same `logical_event_id`. | Relay store tests for duplicate top-level event ID, conflicting top-level-vs-nested metadata identity precedence, and revised missed opportunity. | Relay reads nested `metadata.event_id` before top-level `event_id`, drops valid revisions, or accepts duplicate top-level event IDs as new facts. |
| Runtime overhead | Instrumentation remains fail-open, bounded, and out of broker hot-path synchronous network writes; sidecar failure reporting cannot recursively grow the `errors` source while relay is down. | Fault and volume tests; code review of gateway/broker paths; sidecar failure test while forwarding `errors`. | Sidecar/relay/Postgres network writes happen directly in strategy/broker hot paths, or sidecar retry failures append unbounded new errors while failing to forward existing errors. |
| Crypto-specific evidence | Crypto perpetual fields are populated where available: Hyperliquid environment, funding, leverage/margin, reduce-only, tick/lot metadata, TTL, exchange/client order IDs, and UTC day boundary assumptions. | Payload tests and synthetic-day inspection. | Crypto-specific fields are omitted without a null/coverage explanation. |
| Synthetic-day graph | A synthetic paper day produces approved, denied, size-adjusted, filled, rejected, expired, completed-trade, missed/backfilled, snapshot, config, deployment, heartbeat, funnel, and health evidence. | One end-to-end test fixture plus graph join assertions. | Evidence exists as isolated rows but cannot be traversed across decision/order/fill/portfolio/config/deployment. |
| Coverage of prior review passes | Every Pass 1, Pass 2, and Pass 3 finding listed below has either a passing acceptance test or an explicit not-applicable rationale. | Checklist Section 12.2. | A previous finding is fixed only by code changes with no targeted acceptance assertion. |

### 12.2 Finite Checklist

Use this as the implementation checklist. Do not mark the implementation
complete until every item is checked.

#### Contract And Identity

- [ ] All relay-bound events are canonical `assistant_event_v1` envelopes.
- [ ] All canonical envelopes have top-level `event_id`.
- [ ] All canonical envelopes have top-level `logical_event_id`.
- [ ] All canonical envelopes have canonical `event_type`.
- [ ] All canonical envelopes include `schema_version`, timestamps, lineage,
  source, priority, payload hash, and payload.
- [ ] `deployment` envelope `event_id == deployment_id`.
- [ ] `deployment` envelope `logical_event_id == deployment_id`.
- [ ] `config_snapshot` envelope `event_id == logical_event_id`.
- [ ] `config_snapshot` identity is derived only from config kind and config
  version.
- [ ] `decision_event` envelope identity is promoted from `decision_id`.
- [ ] `market_snapshot` envelope identity is promoted from stable bar/market
  snapshot ID.
- [ ] `portfolio_rule` envelope identity is promoted from
  `portfolio_rule_event_id`.
- [ ] `risk_decision` envelope identity is promoted from `risk_decision_id`.
- [ ] `order` envelope identity is promoted from the order lifecycle event ID.
- [ ] `fill` envelope identity is promoted from `fill_event_id`.
- [ ] Revised missed-opportunity rows keep one stable `logical_event_id`.
- [ ] Revised missed-opportunity rows get a new `event_id` per revision.

#### Strategy And Decision Evidence

- [ ] Trend entry order IDs are deterministic.
- [ ] Breakout entry order IDs are deterministic.
- [ ] Momentum entry order IDs are deterministic or have a documented no-entry
  exception.
- [ ] Trend protective stop IDs are deterministic.
- [ ] Trend break-even and trailing replacement IDs are deterministic.
- [ ] Trend TP/cancel/resubmit management IDs are deterministic.
- [ ] Breakout protective stop IDs are deterministic.
- [ ] Breakout lock, break-even, and trailing replacement IDs are
  deterministic.
- [ ] Breakout TP, time-stop, quick-exit, invalidation, and cancel/resubmit
  management IDs are deterministic and non-blank.
- [ ] Momentum protective stop IDs are deterministic and non-blank.
- [ ] Momentum TP, time-stop, quick-exit, follow-through, MFE-retrace,
  cancel/resubmit, and exit-manager stop-resubmit IDs are deterministic and
  non-blank.
- [ ] Momentum trailing stop IDs are deterministic and non-blank.
- [ ] No strategy management order falls back to repeated manual intent IDs
  such as `<strategy>:<symbol>:manual:intent:1`.
- [ ] Strategy decisions use bar/exchange timestamps for identity, not
  wall-clock timestamps.
- [ ] Filter decisions include filter name, filter index, pass/fail state, and
  blocker reason.
- [ ] Indicator snapshots and regime transitions join to the same decision/bar
  context.
- [ ] Decision parity replay proves stable IDs and payload shape for repeated
  inputs.

#### Order, Fill, And Trade Lifecycle

- [ ] Order intent events emit for submitted entries and management orders.
- [ ] Execution report events emit for accepts/submits.
- [ ] Execution report events emit for rejections.
- [ ] Execution report events emit for cancellations.
- [ ] Execution report events emit for TTL expirations.
- [ ] Execution report events emit for TTL cancel failures.
- [ ] OMS order rows retain client ID, exchange ID, strategy ID, decision ID,
  position ID, status, and metadata.
- [ ] Fills emit first-class `fill` events.
- [ ] Fills update OMS state and TTL tracking without losing lineage.
- [ ] Completed trades join to decision, order, fill, portfolio rule, risk
  decision, config, deployment, funding, and exchange/client IDs.
- [ ] Closed-position trade events still emit from strategy-level handlers.
- [ ] Restored/reconciled positions can still produce joinable closeout
  evidence.

#### Portfolio And Backtest Evidence

- [ ] Live approved entries emit `portfolio_rule`.
- [ ] Live denied entries emit `portfolio_rule`.
- [ ] Live size-adjusted entries emit `portfolio_rule`.
- [ ] Live approved entries emit `risk_decision`.
- [ ] Live denied entries emit `risk_decision`.
- [ ] Live size-adjusted entries emit `risk_decision`.
- [ ] Portfolio events include full rule evaluations.
- [ ] Portfolio events include before state.
- [ ] Portfolio events include after-preview state.
- [ ] Portfolio events include lineage and config versions.
- [ ] Portfolio events include decision/order join keys.
- [ ] Backtest `RuleEvent` uses the same rich schema as live.
- [ ] Backtest artifacts preserve `rule_event_id`.
- [ ] Backtest artifacts preserve `risk_decision_id`.
- [ ] Backtest artifacts preserve first-class `risk_decision` events.
- [ ] Backtest artifacts preserve first-class portfolio-rejected `order`
  events.
- [ ] Backtest/live same-schema tests compare required key sets.

#### Startup, Snapshots, Config, And Deployment

- [ ] Startup syncs durable/exchange open orders before initial snapshots.
- [ ] Startup seeds live TTL trackers before initial snapshots.
- [ ] Startup emits `deployment`.
- [ ] Startup emits live config snapshot.
- [ ] Startup emits portfolio config snapshot.
- [ ] Startup emits risk config snapshot.
- [ ] Startup emits allocation config snapshot when applicable.
- [ ] Startup emits each strategy config snapshot.
- [ ] Startup emits asset metadata snapshot.
- [ ] Startup emits allocation snapshot.
- [ ] Startup emits initial portfolio snapshot with accurate pending order
  counts.
- [ ] Startup emits initial position snapshots.
- [ ] Startup emits family/portfolio-level heartbeat.
- [ ] Fill processing emits updated position and portfolio snapshots.
- [ ] Reconciliation emits position, portfolio, and reconciliation snapshots.
- [ ] Risk-control changes emit allocation/portfolio snapshots.
- [ ] Closeout emits terminal position and portfolio snapshots.
- [ ] Runtime emits `deployment_metadata.json` for `crypto_trend_v1` in the
  bridge contract directory.
- [ ] Runtime emits `deployment_metadata.json` for `crypto_momentum_v1` in the
  bridge contract directory.
- [ ] Runtime emits `deployment_metadata.json` for `crypto_breakout_v1` in the
  bridge contract directory.
- [ ] Each deployment metadata artifact is approval-ready in the intended
  deployment checkout.

#### Sidecar, Relay, Postgres, And Errors

- [ ] Sidecar maps legacy file names to canonical event types.
- [ ] Sidecar preserves already canonical envelopes.
- [ ] Sidecar wraps legacy payloads into canonical envelopes.
- [ ] Sidecar forwards unread legacy rows even when canonical rows exist.
- [ ] Sidecar watermarks are tracked per source file.
- [ ] Sidecar restart does not duplicate sent rows.
- [ ] Sidecar restart does not skip unread rows.
- [ ] Sidecar health reports expose canonical files and watermarks.
- [ ] Sidecar retry exhaustion emits structured `ErrorEvent`.
- [ ] Sidecar retry exhaustion while forwarding `errors` does not append
  unbounded recursive error events.
- [ ] Enabled Postgres initialization failure emits structured `ErrorEvent`.
- [ ] Enabled Postgres initialization failure is captured even before live
  lineage has been loaded.
- [ ] Enabled Postgres write failure emits structured `ErrorEvent`.
- [ ] Postgres remains optional and fail-open.
- [ ] Postgres migrations index the actual canonical JSON paths.
- [ ] Relay deduplicates exact duplicate `event_id`.
- [ ] Relay stores multiple missed-opportunity revisions for one
  `logical_event_id`.
- [ ] Relay extracts top-level `event_id`, `logical_event_id`, event type, and
  join fields before falling back to nested metadata.
- [ ] Relay has a regression test where top-level `event_id` conflicts with
  nested `metadata.event_id`, and the top-level ID wins.

#### Security, Performance, And Operations

- [ ] No private key appears in JSONL, relay payloads, Postgres payloads, or
  metadata artifacts.
- [ ] No relay secret appears in JSONL, relay payloads, Postgres payloads, or
  metadata artifacts.
- [ ] No raw Postgres DSN appears in JSONL, relay payloads, Postgres payloads,
  or metadata artifacts.
- [ ] No exchange API token appears in JSONL, relay payloads, Postgres payloads,
  or metadata artifacts.
- [ ] Instrumentation sink failures never block trading.
- [ ] Sidecar/relay/Postgres network writes do not run synchronously in broker
  hot paths.
- [ ] Sidecar error reporting is bounded while relay is unavailable.
- [ ] Event volume is bounded for mainnet operation.
- [ ] CLI status exposes instrumentation health, sidecar status, and latest
  watermarks.
- [ ] Parity report and parity gate still pass with enriched events.

#### Previous Review Finding Closure

- [ ] Pass 1/2 lossy backtest portfolio rule schema is fixed and covered by a
  same-schema backtest/live test.
- [ ] Pass 1/2 decision envelope identity promotion is fixed and covered by an
  envelope identity test.
- [ ] Pass 1/2 market snapshot envelope identity promotion is fixed and covered
  by an envelope identity test.
- [ ] Pass 1/2 trend entry order UUIDs are removed and covered by deterministic
  replay tests.
- [ ] Pass 1/2 breakout entry order UUIDs are removed and covered by
  deterministic replay tests.
- [ ] Pass 1/2 startup snapshot ordering is fixed and covered by an open-order
  startup test.
- [ ] Pass 1/2 approval-grade deployment metadata is fixed with bridge contract
  artifacts and `approval_ready: true` coverage.
- [ ] Pass 1/2 Postgres JSON indexes are fixed to target the stored canonical
  shape.
- [ ] Pass 1/2 sidecar legacy rollout skip is fixed and covered by unread
  legacy plus canonical file tests.
- [ ] Pass 3 deployment event identity is fixed and covered by startup envelope
  tests.
- [ ] Pass 3 config snapshot identity is fixed and covered by restart/version
  tests.
- [ ] Pass 3 deployment metadata contract-path integration is fixed and covered
  by artifact path tests.
- [ ] Pass 3 trend protective/management order UUIDs are removed and covered by
  deterministic lifecycle tests.
- [ ] Pass 3 breakout protective/management order UUIDs are removed and covered
  by deterministic lifecycle tests.
- [ ] Pass 3 sidecar/Postgres structured error events are implemented and
  covered by fault-injection tests.
- [ ] Current review momentum and breakout exit-manager blank management order
  IDs are replaced with deterministic IDs and covered by no-context lifecycle
  tests.
- [ ] Current review backtest `risk_decision` and portfolio-rejected `order`
  evidence is preserved and covered by same-schema tests.
- [ ] Current review pre-lineage Postgres initialization errors are emitted or
  captured through an explicit startup fallback test.
- [ ] Current review sidecar `errors` forwarding cannot recursively append
  unbounded sidecar failure events.

#### Final Release Gate

- [ ] `pytest tests/test_instrumentation.py` passes.
- [ ] `pytest tests/instrumentation` passes.
- [ ] `pytest tests/portfolio` passes.
- [ ] `pytest tests/parity` passes.
- [ ] `pytest tests/live` passes.
- [ ] `git diff --check` passes.
- [ ] Synthetic-day fixture produces the full evidence graph from Section 11.2.
- [ ] Manual secret inspection commands from Section 11.3 return no unsafe
  secrets.
- [ ] Manual JSONL inspection shows canonical event types and top-level
  identities.
- [ ] This guide and acceptance checklist are tracked or explicitly included in
  the review artifact despite any generated-doc ignore rules.
- [ ] All required new source files, tests, migrations, and bridge contract
  fixtures are tracked or explicitly included in the review artifact.
- [ ] A reviewer can join decision -> order -> fill -> trade -> portfolio rule
  -> risk decision -> config -> deployment without symbol/time heuristics.

## 13. Rollout Plan

Recommended rollout:

1. Add lineage helpers and metadata fields behind defaults.
2. Add generic emitter and sink support while preserving legacy JSONL files.
3. Add sidecar canonical wrapping and tests.
4. Enrich existing trade, missed, daily, funnel, and health events.
5. Add config/deployment/allocation startup events.
6. Add order lifecycle event conversion from parity streams.
7. Add portfolio rule/risk decision events.
8. Add snapshots.
9. Add decision-event enrichment and sidecar forwarding.
10. Add optional Postgres generic event table.
11. Run synthetic day and full tests.
12. Enable forwarding in paper/testnet first.
13. Enable on mainnet only after checking secret redaction and event volume.

Event volume controls:

- full filter decision emission can be enabled for paper/testnet and reduced to
  actionable decisions in mainnet if volume is too high;
- raw market bars can remain parity-local while decision-linked market
  snapshots are forwarded;
- snapshots should use startup, fill, reconciliation, risk-control, and
  closeout triggers, not periodic every-loop capture.

## 14. Definition Of Done

The `crypto_trader` repo is optimally instrumented when:

- all current strategy events still emit and existing behavior is unchanged;
- all events forwarded to relay use canonical assistant event types;
- every monthly-critical event includes complete lineage;
- trade events join to decision, order, fill, portfolio rule, risk decision,
  config, and deployment evidence;
- missed opportunity revisions are not dropped by relay deduplication;
- portfolio approvals, denials, and size adjustments are first-class events;
- order intents, execution reports, fills, cancellations, rejects, and TTL
  expirations are first-class events;
- position, portfolio, and allocation snapshots exist for startup, fills,
  reconciliation, risk-control changes, and closeout;
- startup emits deployment and config snapshots with secret redaction;
- startup emits approval-grade `deployment_metadata.json` for each crypto
  bridge;
- decision parity remains deterministic and is enriched enough for monthly
  replay comparison;
- sidecar health reports expose canonical event files and watermarks;
- Postgres, if enabled, is optional and fail-open;
- tests cover lineage, sidecar, relay revisions, portfolio rules, order events,
  snapshots, config snapshots, and parity;
- a synthetic day produces a complete joinable evidence graph with no secrets.

## 15. Non-Goals

Do not implement these while adding instrumentation:

- new trading rules;
- new entry/exit behavior;
- new optimizer behavior;
- automatic deployment or approval;
- assistant-to-bot command channels;
- strategy refactors unrelated to event capture;
- changing live portfolio rule semantics;
- changing order ID visibility;
- making Postgres required;
- making relay delivery synchronous in the trading hot path.
- monitoring-only analytics outside the lifecycle evidence listed above.

## 16. High-Risk Mistakes To Avoid

- Reusing the same `event_id` for missed-opportunity backfill revisions.
- Computing config hashes from unredacted secret-bearing dictionaries.
- Emitting only portfolio denials and not approvals.
- Treating health report portfolio state as a substitute for portfolio rule
  events.
- Forwarding raw parity events without lineage and calling the repo
  monthly-authoritative.
- Using wall-clock `datetime.now()` as the identity timestamp for bar-based
  strategy decisions.
- Adding sidecar or Postgres writes directly into broker hot paths without
  fail-open wrappers.
- Changing `PortfolioManager.check_entry()` behavior while enriching its return
  object.
- Comparing nondeterministic timestamps in parity gates.
- Emitting raw wallet/private key/relay/Postgres credentials in config
  snapshots.

## 17. Short Answer For Future Agents

The existing `crypto_trader` instrumentation is a strong partial foundation,
not the final optimal target. Keep it and enrich it.

Priority order:

1. Add shared lineage and secret-safe config/deployment snapshots.
2. Canonicalize sidecar envelopes and event types.
3. Fix missed-opportunity revision identity.
4. Enrich trade/missed/daily/funnel/heartbeat metadata.
5. Promote order intent/execution parity streams to assistant order events.
6. Add full portfolio rule and risk decision events.
7. Add position, portfolio, and allocation snapshots tied to startup, fills,
   reconciliation, risk-control changes, and closeout.
8. Emit approval-grade deployment metadata for the three crypto bridges.
9. Enrich and forward decision events for monthly parity review.

Once those are done, `crypto_trader` can serve as production-truth telemetry
for both strategy-level and portfolio-level learning in the new target state.
