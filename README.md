# WEADGE

**Weather resolution-mispricing trader for prediction markets** (Polymarket).

## Lane status

```text
core       stable infrastructure
research   🧊 FROZEN (v1) — no new strategies, ever
recorder   independent sidecar (research support)
resolver   🔥 only active production lane — docs/resolver.md
```

Research v1 (price-bin, stale-book, lead/lag, coherence arb, cheap YES,
multi-city forecast) is frozen. The **only** strategy: `Daily High →
Resolution/Nowcast Mispricing` (`weadge.resolver`). `resolver` never imports
`research`; both share `domain/` primitives. Legacy research CLI (kalshi /
dataset / noaa / research / backtest / live) is preserved under
`docs/archive/research-v1.md` and loads its heavy deps per-command.

## What resolver does (V0: Observation-Locked Scanner)

PM Daily High markets + airport-station observations → find outcomes that are
mathematically impossible by settlement rules → check whether the executable
CLOB book has reacted → record everything for the kill test.

```text
PM daily-high event (gamma)
      + station METAR (aviationweather, LFPB)
      ↓
already happened? → outcome impossible? (observed_max ≥ bucket cap + buffer)
      ↓
executable NO ask (CLOB /book) — never gamma display price
      ↓
shadow log (heartbeat + lock rows) → alert (Telegram, optional)
```

**Kill test** (10 trading days): every time a bucket first enters LOCKED,
record the executable NO ask + depth, and measure how long the CLOB takes to
reach 0.97/0.99. If no executable opportunity ever appears → V0 dead, move to
V1 (NEAR_LOCKED, nowcast). Full spec: [`docs/resolver.md`](docs/resolver.md).

## Run

```bash
uv sync                          # resolver runtime only (small)
uv sync --extra research --dev   # + frozen research stack + dev tools

uv run weadge scan --city paris          # one scan (shadow)
uv run weadge serve --city paris         # loop every 30s (scan window only)
uv run weadge stats --city paris         # kill-test summary from shadow logs
uv run weadge scan --mode alert          # + Telegram (needs TELEGRAM_BOT_TOKEN/CHAT_ID)
```

Shadow logs land in `data/resolver/shadow-<city>-<date>.jsonl` (stdlib JSONL,
one heartbeat per scan + one row per LOCKED bucket with book snapshot).

## Current hypothesis & scope lock

- Paris LFPB first (lowest bot saturation among liquid stations; station is
  written into the market rules). Other cities = same-shape extension.
- **Not building**: NEAR_LOCKED probability, forecasts (ECMWF/GFS), trade
  mode, multi-city, dashboard, LLM, cross-exchange, market making.
- Next 90 days: Weadge accepts resolver code only.

## Constraints

- `no_ask` is the executable CLOB top-of-ask, never gamma `outcomePrices`
  (display pricing; measured 40¢ away from the book on thin buckets).
- Bucket semantics: truncation ("be 33°C" wins iff daily max ∈ [33, 34)).
- PM Weather taker fee 5%: `fee = 0.05 × p × (1-p)`; extreme prices ≈ 0.
- No credentials needed for public data; trading (v1+) needs an API key.
- Recorder/legacy pipeline is not a resolver dependency — the bot survives
  the research sidecar being down.

## Repo layout

```
src/weadge/
  domain/     shared primitives (as-of time invariant)
  resolver/   🔥 markets, observations, state, edge, log, execution, service
  research_cli.py  frozen v1 CLI (heavy deps per-command)
  live/       research sidecar recorder (not a resolver dependency)
  adapters/   kalshi / noaa / openmeteo (frozen)
tests/resolver/  offline pure-function tests (real API fixtures)
docs/         resolver.md (authoritative) + archive/research-v1.md
config/       resolver.yaml, cities.yaml, models.yaml, research.yaml
```

## Development

```bash
uv run pytest tests            # full suite (research extra required)
uv run ruff check src tests
```
