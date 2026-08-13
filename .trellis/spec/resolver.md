# Resolver

Weadge 唯一活跃 production 主线（research v1 FROZEN）。

- 策略唯一权威文档: `docs/resolver.md`（与代码冲突时以文档为准）
- 代码在 `src/weadge/resolver/`，业务逻辑绝不 import `research/`；日志用
  自己的 `resolver/log.py`（stdlib JSONL），不依赖 live/recorder
- 复用 `domain/` primitives（time 的 as-of 不变式是硬规则）
- V0 只做冷端 LOCKED：`observed_max >= cap_high + locked_buffer_c`
- **可成交价只信 CLOB `/book`**：gamma `outcomePrices` 是展示价（实测差
  40¢）；NO token 用 `/markets/{condition_id}` 解析（clobTokenIds 顺序不可靠）
- 只对 LOCKED 桶拉 book；每次扫描必写 heartbeat，LOCKED 桶写 lock 行
- 禁止：概率模型（V1）、forecast（V2）、trade mode（v1+）、NEAR_LOCKED、°F/`between` parser
- 纯函数核心（evaluate/find_edges）必须离线可测，fixture 用真实 API 样本
- serve 城市 = 与 Paris 同构的 °C Daily High + 人工核对结算站（PM `resolutionSource`）
- 观测候选可以更宽（NYC/Chicago 进 `observation_extra` / race+audit，不进 `weadge serve`）
- production scan 保持 30s；10s 轮询只存在于 `tools/observation_race.py`

## Scenario: observation race + WU audit

### 1. Scope / Trigger
- Trigger: new CLI in `tools/observation_race.py`; JSONL + yaml contracts; WU/IEM/AWC env/keys
- Resolver LOCKED / `bucket_cap_high` / 30s `serve` 不在本场景

### 2. Signatures
```text
uv run python tools/observation_race.py serve [--interval 10] [--once]
uv run python tools/observation_race.py summary
uv run python tools/observation_race.py audit [--backfill N] [--date YYYY-MM-DD] [--city slug]
ResolverConfig.cities              # serve 候选
ResolverConfig.observation_extra   # race/audit only
ResolverConfig.by_slug(slug)       # cities only → KeyError for nyc
ResolverConfig.observation_stations()  # cities + extra
```

### 3. Contracts
- Race row identity: `(station_icao, observation_at UTC seconds, source)`
- Race fields: `report_id`, `first_seen_at`, `provider_receipt_at` (AWC `receiptTime` or null), `decoded_temp`, `temp_unit` (AWC=`C`, IEM=`F`), `raw_metar`, `source_grade`
- AWC grade = city `source_grade`; IEM grade always `B`
- Freshness: skip if `now - observation_at > 20min` (cold-start cache must not become first_seen)
- AWC fetch: one request, `ids=LFPB,EGLC,RJTT,RKSI,KLGA,KORD`
- IEM current: `iem_station or station_icao` + `iem_network`. US: `KLGA→LGA` / `KORD→ORD` (KLGA returns `{}`)
- Audit: METAR local-day max vs WU `historical.json` `observations[].temp` max. °C → `units=m`, °F → `units=e`
- US audit: store `metar_max_c` and `metar_max = round(C*9/5+32)`
- Env: `WUNDERGROUND_API_KEY` optional; default is WU frontend key (rotates)
- Files: `data/race/race-YYYY-MM-DD.jsonl`, `data/race/audit-YYYY-MM-DD.jsonl`

### 4. Validation & Error Matrix
- AWC/IEM/WU HTTP error → JSONL `event=error`, loop continues
- empty IEM `{}` / no `last_ob` → skip (not a new obs)
- missing `wu_location` → error row, skip city
- WU fail → `bucket=wu_missing`; never substitute IEM daily max
- `audit --city nyc` ok; `weadge serve --city nyc` → `KeyError`

### 5. Good/Base/Bad Cases
- Good: same METAR seen on AWC then IEM → two first_seen, summary shows lead seconds
- Base: `--once` records only obs younger than 20min
- Bad: treating IEM `max_dayairtemp[F]` or CLI/DSM as WU settlement

### 6. Tests Required
- `tests/test_observation_race.py`: identity dedupe, IEM mapping, WU max, audit buckets, `c_to_f`, freshness, `by_slug("nyc")` KeyError
- `tests/resolver/test_resolver.py` unchanged and green

### 7. Wrong vs Correct
#### Wrong
```python
cfg.by_slug("nyc")           # serve path
iem current station=KLGA     # returns {}
WU historical.json units=m   # for NYC °F settlement
```
#### Correct
```python
[s for s in cfg.observation_stations() if s.slug == "nyc"]
iem_station="LGA", iem_network="NY_ASOS"
wu_api_units("fahrenheit") == "e"
```

