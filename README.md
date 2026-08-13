# WEADGE

**Automated weather mispricing trader** — PM Daily High resolution/nowcast edge.

## Lane status

```text
core       stable infrastructure
research   🧊 FROZEN (v1) — no new strategies, ever
recorder   independent sidecar (research support)
resolver   🔥 only active production lane — docs/resolver.md
```

Research v1 (price-bin, stale-book, lead/lag, coherence arb, cheap YES,
multi-city forecast) is frozen as of this commit. The **only** new strategy:
`Daily High → Resolution/Nowcast Mispricing` (`weadge.resolver`).
Future 90 days: Weadge accepts resolver code only. `resolver` never imports
`research`; both share `domain/` primitives.

---

## Legacy README (research v1, frozen)

**Weather prediction-market alpha research engine.**

> v0 is not a bot. v0 is an evidence engine that answers one question:
>
> **At time T, did the weather information available then know more than the
> Kalshi market price at T?**

The first target is a single series — **KXHIGHNY** (NY Central Park daily max
temperature) — and the entire pipeline is built to falsify the alpha
hypothesis before any trading code exists. Complexity is added only when the
previous layer of evidence demands it.

## Goal

Determine whether public weather information contains **incremental,
executable** information beyond market prices.

## Non-goals

- maximizing backtest PnL
- building a trading UI
- high-frequency execution
- strategy optimization before alpha validation
- multi-city coverage before NY passes every gate

## Rules

1. **No synthetic market data.** Historical prices come from Kalshi's official APIs only.
2. **No future forecast leakage.** A forecast may influence a decision at `T`
   only if `forecast.available_at <= T` — never `run_init_at`.
3. **Every forecast has `available_at`.** `run_init_at`, `available_at`, and
   `ingested_at` are three distinct timestamps and are never conflated.
4. **Execution uses executable quotes.** Fills use real bid/ask OHLC, delayed
   by at least one bar. No same-bar fills, no mid fills.
5. **Historical fees are replayed.** The fee multiplier in effect at execution
   time comes from the series fee history — `KALSHI_FEE` is never a constant.
6. **All tests are out-of-sample.** Walk-forward is chronological; shuffling
   is forbidden.
7. **Event/day is the statistical independence unit.** Minute snapshots of the
   same event are not independent samples; bootstrap clusters by event/date.
8. **Forecast skill is not trading alpha.** A better Brier score proves
   nothing until it survives costs and executability.
9. **Alpha must survive costs.** Net EV > 0 after fees, slippage, and
   execution delay.
10. **Failed hypotheses are first-class results.** "Market already priced it
    in" is a valid, valuable answer.

## Architecture

```
                 weadge
                    │
        ┌───────────┴───────────┐
        │                       │
 Research / Model           Adapters
        │                       │
      Python             Kalshi / NOAA
        │
   Alpha proven?
        │
       YES
        ↓
  (later) latency-sensitive collector/executor → Rust
```

Research logic stays in Python. Rust, Kafka, Redis, Kubernetes, TimescaleDB,
ClickHouse, FastAPI, React, microservices — none of these are in v0, because
**none of them can add alpha**.

### Package layout

```
src/weadge/
  domain/     canonical models; as-of-time invariant (time is a first-class citizen)
  adapters/   kalshi (client, markets, candles, forecasts, fees, websocket), noaa, openmeteo
  storage/    parquet data lake + duckdb query layer + canonical schemas
  dataset/    settlement oracle, as-of alignment, probability features, gold builder
  models/     stacking only in v0; EMOS is an explicit challenger, not a default
  research/   scoring, calibration, incremental alpha test, latency, walk-forward
  backtest/   fee replay, delayed execution, taker engine, event-cluster bootstrap
  live/       JSONL.zst recorder (v2) + paper trading (v2)
  resolver/   🔥 Observation-Locked Scanner — markets, observations, state, edge, execution, service
```

Resolver V0 spec: [`docs/resolver.md`](docs/resolver.md).

Notebooks are for exploration only — core logic lives in `src/weadge` and is
tested with pytest.

### Data lake

```
RAW    data/raw/...    untouched API payloads (JSONL.zst)
BRONZE data/bronze/    standardized parquet (markets, candles, percentiles, fees, forecasts)
SILVER data/silver/    aligned, time-consistent (availability applied)
GOLD   data/gold/      alpha_dataset.parquet — one row per (event, market, snapshot)
```

## Pipeline & alpha gates

```
 ① Kalshi adapter → ② Historical lake → ③ Settlement audit → ④ Kalshi forecast
 → ⑤ NBM → ⑥ Gold dataset → ⑦ Market vs NBM → ⑧ Incremental alpha test
       │                                        │
   NO EDGE (stop)                          EDGE → ⑨ Taker backtest
                                                     → ⑩ Walk-forward
                                                          │
                                                    PASS → ⑪ EMOS ... ⑭ small live
```

Gates are enforced in code and config (`config/research.yaml`):

| Gate | Check |
| ---- | ----- |
| G0 DATA | settlement mismatch == 0 |
| G1 FORECAST | weather OOS skill > baseline |
| G2 INCREMENTAL | market + weather beats market OOS |
| G3 ECONOMIC | net EV > 0 after fees |
| G4 ROBUSTNESS | edge bins monotonic; walk-forward survives |
| G5 LIVE | paper ≈ historical expectation |
| G6 CAPITAL | small live |

If G2 fails, development stops. No execution bot, no dashboard, no Kelly,
no Rust.

## CLI

```bash
# data
weadge kalshi sync-series KXHIGHNY
weadge kalshi backfill KXHIGHNY
weadge kalshi audit KXHIGHNY

# dataset
weadge dataset build --series KXHIGHNY --snapshot 24h,12h,6h,3h,1h

# research
weadge research compare --series KXHIGHNY
weadge research calibration --series KXHIGHNY
weadge research incremental --series KXHIGHNY
weadge research latency --series KXHIGHNY
weadge research walk-forward --series KXHIGHNY

# backtest
weadge backtest taker --series KXHIGHNY --model p_nbm --edge 0.06
```
The Kalshi adapter routes live vs historical data automatically via the
`/historical/cutoff` endpoint; callers never see the split. Rate limits,
retries, and exponential backoff are handled inside the client.

## Naming

The Kalshi event forecast percentile history is named **`kalshi_forecast`**,
never `p_nbm` — Kalshi does not document the weather model behind it. Only
when provenance is confirmed may it be relabeled.

## Development

```bash
uv sync --dev          # install deps + dev group
uv run weadge --help   # CLI
uv run pytest          # tests (as-of invariants, settlement, fees, execution, ...)
uv run ruff check .    # lint
```

Requires Python 3.12+. Kalshi public market data needs no credentials;
authenticated endpoints read `KALSHI_API_KEY` / `KALSHI_API_SECRET` from the
environment (never committed).

## Milestones

- **v0 — Evidence Engine**: Kalshi historical, settlement, NBM, calibration,
  incremental alpha, taker backtest. *(this repo today)*
- **v1 — Weather Alpha Lab**: GEFS / ECMWF / ICON / GEM, EMOS, METAR, station
  bias, stacking, latency analysis.
- **v2 — Trading Engine**: WebSocket, paper, risk, positions, taker execution.
- **v3 — Microstructure Engine**: L2, maker, queue model, fill probability,
  rebates, Rust executor (only if truly needed).
