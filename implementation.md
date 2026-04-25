# Deployment Guide — VPS with Docker

Step-by-step instructions for deploying the crypto_trader system on a Linux VPS using Docker.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [VPS Setup](#3-vps-setup)
4. [Project Transfer](#4-project-transfer)
5. [Docker Configuration](#5-docker-configuration)
6. [Configuration Files](#6-configuration-files)
7. [Data Pipeline](#7-data-pipeline)
8. [Paper Trading Deployment](#8-paper-trading-deployment)
9. [Monitoring & Logging](#9-monitoring--logging)
10. [Maintenance & Operations](#10-maintenance--operations)
11. [Security Hardening](#11-security-hardening)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  VPS (Ubuntu 22.04+)                                │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  docker compose                             │    │
│  │                                             │    │
│  │  ┌──────────────┐  ┌──────────────────────┐ │    │
│  │  │  trader      │  │   data-refresh       │ │    │
│  │  │  (paper/live)│  │  (cron every 3 days) │ │    │
│  │  │              │  │                      │ │    │
│  │  │  LiveEngine  │  │   refresh_data.py    │ │    │
│  │  │  3 strategies│  │   BTC/ETH/SOL        │ │    │
│  │  │  portfolio   │  │   7 timeframes       │ │    │
│  │  │  coordinator │  │   + funding rates    │ │    │
│  │  └──────┬───────┘  └──────────┬───────────┘ │    │
│  │         │                     │             │    │
│  │         ▼                     ▼             │    │
│  │  ┌──────────────────────────────────────┐   │    │
│  │  │  /app/data (Docker volume)           │   │    │
│  │  │  candles/{BTC,ETH,SOL}/*.parquet     │   │    │
│  │  │  funding/{BTC,ETH,SOL}.parquet       │   │    │
│  │  │  live_state/                         │   │    │
│  │  └──────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌──────────────────┐  ┌───────────────────────┐    │
│  │  Watchtower      │  │  Logs (stdout/json)   │    │
│  │  (auto-update)   │  │  → journald / file    │    │
│  └──────────────────┘  └───────────────────────┘    │
└─────────────────────────────────────────────────────┘
          │
          ▼
   Hyperliquid API
   (testnet / mainnet)
```

**Services:**
- **trader** — Long-running paper/live trading engine (async polling loop)
- **data-refresh** — Periodic cron job to keep candle data current
- **watchtower** (optional) — Auto-pulls updated Docker images

---

## 2. Prerequisites

### Local Machine
- Docker Desktop installed (for building images)
- Git
- SSH key pair for VPS access

### VPS Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 1 vCPU | 2 vCPU |
| RAM | 1 GB | 2 GB |
| Disk | 10 GB | 20 GB |
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| Network | Stable outbound HTTPS | Low-latency to Hyperliquid API |

> Paper trading is lightweight — 1 vCPU / 1 GB is sufficient. Optimization requires more (4+ vCPU for parallel workers).

### Accounts & Credentials
- [ ] VPS provider account (Hetzner, DigitalOcean, Vultr, etc.)
- [ ] Hyperliquid testnet wallet address
- [ ] Hyperliquid testnet private key (for order submission)
- [ ] (Optional) Docker Hub account for private image hosting

---

## 3. VPS Setup

### 3.1 — Provision the VPS

Choose a provider and create a VPS with Ubuntu 22.04+. Note the IP address.

### 3.2 — Initial Server Setup

```bash
# SSH into the VPS
ssh root@YOUR_VPS_IP

# Create a non-root user
adduser trader
usermod -aG sudo trader

# Set up SSH key auth for the new user
mkdir -p /home/trader/.ssh
cp ~/.ssh/authorized_keys /home/trader/.ssh/
chown -R trader:trader /home/trader/.ssh
chmod 700 /home/trader/.ssh
chmod 600 /home/trader/.ssh/authorized_keys

# Disable root login & password auth (edit sshd_config)
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd
```

### 3.3 — Install Docker

```bash
# As the trader user
ssh trader@YOUR_VPS_IP

# Install Docker (official method)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Log out and back in for group change to take effect
exit
ssh trader@YOUR_VPS_IP

# Verify
docker --version
docker compose version
```

### 3.4 — Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw enable

# No inbound ports needed — the trader only makes outbound HTTPS calls
```

---

## 4. Project Transfer

### 4.1 — Option A: Git Clone (Recommended)

Push your repo to a private Git remote (GitHub, GitLab, etc.), then clone on the VPS:

```bash
ssh trader@YOUR_VPS_IP
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:YOUR_USER/crypto_trader.git
cd crypto_trader
```

### 4.2 — Option B: Direct Upload

From your local machine:

```bash
# Exclude data/ and output/ (large, regeneratable)
rsync -avz --progress \
  --exclude 'data/' \
  --exclude 'output/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  --exclude '.git/' \
  . trader@YOUR_VPS_IP:~/projects/crypto_trader/
```

### 4.3 — Transfer Historical Data

The data directory contains Parquet files needed for warmup. Transfer the essential timeframes:

```bash
# From local machine — transfer only timeframes needed for live trading
# Momentum needs: 15m, 1h, 4h, 1d
# Trend needs: 1h, 1d
# Breakout needs: 30m, 4h
rsync -avz --progress \
  data/candles/ trader@YOUR_VPS_IP:~/projects/crypto_trader/data/candles/
rsync -avz --progress \
  data/funding/ trader@YOUR_VPS_IP:~/projects/crypto_trader/data/funding/
```

---

## 5. Docker Configuration

### 5.1 — Dockerfile

See `Dockerfile` in the project root. Multi-stage build: Python 3.12-slim builder (wheels) → slim runtime with non-root `app` user.

### 5.2 — .dockerignore

See `.dockerignore` in the project root. Excludes git, caches, data, output, tests, secrets.

### 5.3 — docker-compose.yml

See `docker-compose.yml` in the project root. Services: `trader` (paper/live), `data-refresh` (cron), `backtest` (on-demand, `--profile tools`), `optimize` (on-demand, `--profile tools`), `watchtower` (optional, `--profile monitoring`).

---

## 6. Configuration Files

### 6.1 — Live Trading Config

See `config/live_config.example.json` for the template. Copy to `config/live_config.json` and fill in wallet address and private key.

> **CRITICAL**: Never commit `live_config.json` to version control. The `private_key` field grants full trading authority over the wallet.

### 6.2 — Portfolio Config

See `config/portfolio_config.json` — weights: momentum 40%, trend 35%, breakout 25%.

### 6.3 — Strategy Configs

Latest optimized configs are in `config/strategies/` (copied from round 3 outputs). Each file has a `"strategy": { ... }` wrapper — the LiveEngine automatically unwraps it.

---

## 7. Data Pipeline

### 7.1 — Initial Data Seed

Before the trader can start, you need historical data for indicator warmup:

```bash
# On the VPS, inside the project directory
cd ~/projects/crypto_trader

# Build the image first
docker compose build

# Run initial data download (all timeframes, 90 days)
docker compose run --rm data-refresh crypto-trader download \
  --coin BTC,ETH,SOL \
  --interval 1m,5m,15m,30m,1h,4h,1d \
  --days 90 \
  --data-dir data

# OR if you transferred data from your local machine:
# Copy into the Docker volume
docker compose run --rm data-refresh sh -c "ls /app/data/candles/"
```

### 7.2 — Automated Data Refresh

Set up a systemd timer to refresh data every 3 days:

Create `/etc/systemd/system/crypto-data-refresh.service`:

```ini
[Unit]
Description=Crypto Trader Data Refresh
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
User=trader
WorkingDirectory=/home/trader/projects/crypto_trader
ExecStart=/usr/bin/docker compose run --rm data-refresh
TimeoutStartSec=1800
```

Create `/etc/systemd/system/crypto-data-refresh.timer`:

```ini
[Unit]
Description=Run crypto data refresh every 3 days

[Timer]
OnCalendar=*-*-01,04,07,10,13,16,19,22,25,28 06:00:00
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
```

Enable the timer:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now crypto-data-refresh.timer

# Verify
systemctl list-timers | grep crypto

# Test manually
sudo systemctl start crypto-data-refresh.service
journalctl -u crypto-data-refresh.service -f
```

---

## 8. Paper Trading Deployment

### 8.1 — Build & Start

```bash
cd ~/projects/crypto_trader

# Build the image
docker compose build

# Start paper trading (detached)
docker compose up -d trader

# Watch logs
docker compose logs -f trader
```

### 8.2 — Verify It's Running

```bash
# Check container status
docker compose ps

# Check health
docker inspect --format='{{.State.Health.Status}}' crypto-trader

# View recent logs
docker compose logs --tail 50 trader

# Check paper trading status (from inside container)
docker compose exec trader crypto-trader paper-status \
  --address 0xYOUR_WALLET_ADDRESS --testnet

# Check system health, signal funnels, and alerts
docker compose exec trader crypto-trader status --state-dir data/live_state
```

### 8.3 — Common Operations

```bash
# Stop the trader gracefully
docker compose stop trader

# Restart after config change
docker compose restart trader

# Rebuild after code changes
docker compose build trader
docker compose up -d trader

# View live logs
docker compose logs -f --tail 100 trader

# Run a one-off backtest
docker compose --profile tools run --rm backtest \
  --strategy momentum \
  --start-date 2026-02-25 \
  --end-date 2026-04-18 \
  --symbols BTC,ETH,SOL

# Run optimization (CPU-intensive)
docker compose --profile tools run --rm optimize \
  --strategy trend \
  --start-date 2026-02-25 \
  --end-date 2026-04-18 \
  --symbols BTC,ETH,SOL \
  --workers 4 \
  --round 3

# Shell into the container for debugging
docker compose exec trader bash
```

---

## 9. Monitoring & Logging

### 9.1 — Log Monitoring

The trader uses `structlog` with structured key-value output. All logs go to stdout, captured by Docker's json-file driver.

```bash
# Tail live logs
docker compose logs -f trader

# Search for errors
docker compose logs trader 2>&1 | grep "error"

# Search for fills (trades)
docker compose logs trader 2>&1 | grep "engine.fill"

# Export logs to file
docker compose logs --no-color trader > ~/trader_logs_$(date +%Y%m%d).txt
```

### 9.2 — Health Monitoring Script

See `scripts/check_health.sh` — checks container status, Docker health, and instrumentation assessment (healthy/degraded/critical). Includes placeholder for Telegram/Discord notifications.

Add to crontab on the VPS:

```bash
chmod +x scripts/check_health.sh

# Check every 5 minutes
crontab -e
# Add: */5 * * * * /home/trader/projects/crypto_trader/scripts/check_health.sh >> /home/trader/health.log 2>&1
```

### 9.3 — Equity Tracking

The LiveEngine writes equity snapshots to `data/live_state/`. Monitor from outside:

```bash
# Check latest portfolio state
docker compose exec trader cat data/live_state/portfolio_state.json | python3 -m json.tool

# Check equity history
docker compose exec trader cat data/live_state/equity_snapshots.jsonl | tail -20
```

### 9.4 — Instrumentation & Observability

The system includes a built-in observability layer (see `docs/implementation.md` §9 for architecture). LiveEngine emits structured events to JSONL files via the `JsonlSink`:

```bash
# Quick system health check (reads latest JSONL entries)
docker compose exec trader crypto-trader status --state-dir data/live_state

# View signal pipeline funnels (bars → setups → entries → fills per strategy)
docker compose exec trader cat data/live_state/pipeline_funnels.jsonl | tail -5 | python3 -m json.tool

# View hourly health reports (assessment: healthy/degraded/critical)
docker compose exec trader cat data/live_state/health_reports.jsonl | tail -1 | python3 -m json.tool

# Check for alerts in the latest health report
docker compose exec trader cat data/live_state/health_reports.jsonl | tail -1 | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
if r.get('alerts'): [print(f'{a[\"severity\"]}: {a[\"name\"]} — {a[\"message\"]}') for a in r['alerts']]
else: print('No alerts')
print(f'Assessment: {r.get(\"assessment\", \"unknown\")}')
"

# View trade events with enrichment data (r_multiple, MFE/MAE, root causes)
docker compose exec trader cat data/live_state/trades.jsonl | tail -5

# View gate rejections (which pipeline stage is blocking entries)
docker compose exec trader cat data/live_state/pipeline_funnels.jsonl | tail -1 | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
for strat, data in r.get('signal_funnels', {}).items():
    gates = data.get('gate_rejections', {})
    if gates: print(f'{strat}: {gates}')
"
```

**JSONL files produced by LiveEngine:**

| File | Content | Write interval |
|------|---------|----------------|
| `trades.jsonl` | InstrumentedTradeEvent (enriched with root causes, quality score) | On each trade close |
| `missed.jsonl` | MissedOpportunityEvent (signal detected but not taken) | On each missed signal |
| `daily_snapshots.jsonl` | DailySnapshot (PnL, drawdown, quality aggregates) | UTC midnight |
| `errors.jsonl` | ErrorEvent (structured errors with severity) | On each error |
| `pipeline_funnels.jsonl` | PipelineFunnelSnapshot (signal pipeline stage counts per strategy) | Every 60 min |
| `health_reports.jsonl` | HealthReportSnapshot (system-wide health assessment + alerts) | Every 60 min |
| `equity_snapshots.jsonl` | Equity history snapshots | Periodic |

**Alert conditions** (emitted in health reports):

| Alert | Severity | Trigger |
|-------|----------|---------|
| `no_bars` | error | Any (symbol, TF) stale > 2× expected interval |
| `heat_leak` | critical | Portfolio heat for symbols not on exchange |
| `pipeline_stalled` | error | Funnel assessment is "pipeline_broken" or "stalled" |
| `gate_blocked` | warning | Single gate blocks 100% of setups for >3h |
| `error_burst` | warning | >10 errors in last hour |
| `fill_gap` | warning | 0 fills in >2h despite open orders |

**Automated alerting:** Already integrated into `scripts/check_health.sh` — parses health reports and checks assessment level.

---

## 10. Maintenance & Operations

### 10.1 — Updating Code

```bash
cd ~/projects/crypto_trader

# Pull latest changes
git pull origin main

# Rebuild and restart
docker compose build trader
docker compose up -d trader

# Verify
docker compose logs --tail 20 trader
```

### 10.2 — Updating Strategy Configs

After running a new optimization round locally:

```bash
# From local machine — push updated config
scp output/momentum/round_4/optimized_config.json \
  trader@YOUR_VPS_IP:~/projects/crypto_trader/config/strategies/momentum.json

# On VPS — restart to pick up new config
ssh trader@YOUR_VPS_IP
cd ~/projects/crypto_trader
docker compose restart trader
```

### 10.3 — Backup

```bash
# Backup live state and data
docker run --rm \
  -v crypto-trader-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/trader-data-$(date +%Y%m%d).tar.gz /data

# Backup configs (already in git, but just in case)
cp config/live_config.json ~/backups/live_config.json.bak
```

### 10.4 — Migrating to Mainnet

When ready to go live:

1. **Update `config/live_config.json`:**
   ```json
   {
     "is_testnet": false,
     "wallet_address": "0xYOUR_MAINNET_WALLET",
     "private_key": "0xYOUR_MAINNET_KEY"
   }
   ```

2. **Start with conservative sizing** — reduce `risk_pct_a` and `risk_pct_b` in each strategy config by 50%.

3. **Monitor closely** for the first 48 hours:
   ```bash
   docker compose logs -f trader
   ```

4. **Scale up gradually** over 1-2 weeks once confirmed stable.

---

## 11. Security Hardening

### 11.1 — Secrets Management

**Never commit secrets to git.** Use one of these approaches:

**Option A: Docker secrets (recommended for Swarm/production)**

```yaml
# docker-compose.yml addition
services:
  trader:
    secrets:
      - wallet_private_key
    environment:
      - PRIVATE_KEY_FILE=/run/secrets/wallet_private_key

secrets:
  wallet_private_key:
    file: ./secrets/private_key.txt
```

**Option B: Environment variables**

```bash
# Set in .env file (git-ignored)
echo "WALLET_ADDRESS=0x..." > .env
echo "PRIVATE_KEY=0x..." >> .env
chmod 600 .env
```

Modify `live_config.json` to read from env:
```json
{
  "wallet_address": "${WALLET_ADDRESS}",
  "private_key": "${PRIVATE_KEY}"
}
```

**Option C: Bind-mount with restrictive permissions (simplest)**

```bash
chmod 600 config/live_config.json
chown trader:trader config/live_config.json
```

### 11.2 — Container Security

```yaml
# Add to trader service in docker-compose.yml
services:
  trader:
    read_only: true           # Read-only root filesystem
    tmpfs:
      - /tmp                  # Writable temp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

### 11.3 — Network Isolation

```yaml
# docker-compose.yml — restrict container networking
networks:
  trader-net:
    driver: bridge
    internal: false   # needs outbound for Hyperliquid API

services:
  trader:
    networks:
      - trader-net
```

### 11.4 — Automatic Updates

```bash
# Unattended OS security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## 12. Troubleshooting

### Container won't start

```bash
# Check build logs
docker compose build --no-cache trader

# Check startup logs
docker compose up trader   # foreground, see all output

# Common issue: missing data files
docker compose exec trader ls -la data/candles/BTC/
```

### No trades being placed

```bash
# Check if strategies are loaded
docker compose logs trader | grep "engine.started"

# Check for config validation errors
docker compose logs trader | grep "config.validation"

# Check for exchange connectivity
docker compose logs trader | grep "error"

# Verify warmup completed
docker compose logs trader | grep "engine.warmup_complete"
```

### Stale data warnings

```bash
# Check data freshness
docker compose logs trader | grep "stale_data"

# Force data refresh
docker compose run --rm data-refresh

# Verify data files updated
docker compose exec trader ls -lt data/candles/BTC/
```

### Out of memory

```bash
# Check container resources
docker stats crypto-trader

# Limit memory in docker-compose.yml
services:
  trader:
    deploy:
      resources:
        limits:
          memory: 1G
```

### Permission denied on volumes

```bash
# Fix volume ownership
docker compose run --rm --user root trader chown -R app:app /app/data
```

---

## Quick Reference — Command Cheat Sheet

| Action | Command |
|--------|---------|
| Start trading | `docker compose up -d trader` |
| Stop trading | `docker compose stop trader` |
| View logs | `docker compose logs -f trader` |
| Rebuild | `docker compose build trader && docker compose up -d trader` |
| Run backtest | `docker compose --profile tools run --rm backtest --strategy momentum --start-date 2026-02-25 --end-date 2026-04-18` |
| Run optimization | `docker compose --profile tools run --rm optimize --strategy trend --start-date 2026-02-25 --end-date 2026-04-18 --workers 4` |
| Refresh data | `docker compose run --rm data-refresh` |
| Check status | `docker compose ps && docker compose logs --tail 5 trader` |
| Paper status | `docker compose exec trader crypto-trader paper-status --address 0x... --testnet` |
| Shell access | `docker compose exec trader bash` |
| Backup data | `docker run --rm -v crypto-trader-data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/data-$(date +%Y%m%d).tar.gz /data` |
