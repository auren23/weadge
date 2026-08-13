# Design — observation race / resolution audit / shadow cities

## Boundaries

| In | Out |
|---|---|
| `tools/observation_race.py` | `state.py` LOCKED 规则 |
| `config/resolver.yaml` + `ResolverCityConfig` 新可选字段 | `markets.py` `bucket_cap_high` |
| `weadge.config.ResolverConfig.observation_extra` | `service.py` scan 策略 / interval 默认 30 |
| `docs/resolver.md`、`.trellis/spec/resolver.md` | heartbeat/lock JSONL schema |
| `tests/test_observation_race.py` | NEAR_LOCKED、trade、HFMETAR、NWS |

单向依赖：`tools/` → `weadge.resolver.observations.MetarClient`、`weadge.resolver.log.JsonlAppender`、`weadge.config`、`weadge.resolver.observations.evaluate_observed`。resolver **不** import tools。

## Config

`cities` = serve 候选（四城）。`observation_extra` = 只跑 race/audit 的站（NYC、Chicago）。同一 pydantic 模型，避免两套字段。

```yaml
mode: shadow
cities:
  - slug: paris
    city: Paris
    station_icao: LFPB
    timezone: Europe/Paris
    unit: celsius
    scan_hours: [12, 21]
    settlement_source: wunderground
    settlement_url: https://www.wunderground.com/history/daily/fr/bonneuil-en-france/LFPB
    wu_location: LFPB:9:FR
    source_grade: A          # AWC vs this settlement; starting hypothesis
    iem_network: FR__ASOS
    # iem_station omitted → station_icao
  # london EGLC Europe/London, EGLC:9:GB, GB__ASOS
  # tokyo  RJTT Asia/Tokyo,    RJTT:9:JP, JP__ASOS
  # seoul  RKSI Asia/Seoul,    RKSI:9:KR, KR__ASOS
observation_extra:
  - slug: nyc
    city: NYC
    station_icao: KLGA
    timezone: America/New_York
    unit: fahrenheit
    scan_hours: [12, 21]     # unused by serve; kept for schema unity
    settlement_source: wunderground
    settlement_url: https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA
    wu_location: KLGA:9:US
    source_grade: B          # C→F + WU vs METAR
    iem_network: NY_ASOS
    iem_station: LGA
  - slug: chicago
    station_icao: KORD
    timezone: America/Chicago
    unit: fahrenheit
    wu_location: KORD:9:US
    source_grade: B
    iem_network: IL_ASOS
    iem_station: ORD
edge:                        # unchanged
  locked_buffer_c: 0.5
  ...
```

`load_resolver().by_slug("nyc")` 不在 `cities` 里 → 现有 `KeyError` 保留。Race 用 `cities + observation_extra`。

新字段全部给默认值（`source_grade="A"`, `iem_network=""` 等），以免测试里手写的 `ResolverCityConfig` 炸掉。空 `wu_location` 时 audit 跳过该站并记 error。

## Data flow

```text
serve loop 10s
  ├─ AWC GET /api/data/metar?ids=LFPB,EGLC,RJTT,RKSI,KLGA,KORD&format=json&hours=2
  └─ IEM  GET /json/current.py?station=&network=   (per station)

decode → Observation
identity = (icao, observation_at_utc_seconds)
if (identity, source) unseen: append race JSONL

audit (T+1 / backfill)
  ├─ METAR rows for local day → max (evaluate_observed or equivalent)
  ├─ WU historical.json observations[].temp → max
  └─ compare in settlement unit; US also store metar_max_c
```

## Contracts

### Race row

见 PRD R1。`report_id` = `{ICAO}-{observation_at:%Y-%m-%dT%H:%M:%SZ}`。`decoded_temp` 为源解码值：AWC 为 °C；IEM current 的 `airtemp[F]` 保留 F 并设 `temp_unit=F`（IEM 国际站也是 F）。比较 first_seen 不比较温度。

`source_grade`：AWC 用城市 config；IEM 恒为 `B`。

`provider_receipt_at`：AWC `receiptTime`；IEM 无则 `null`。

文件：`data/race/race-YYYY-MM-DD.jsonl`（UTC 日切，复用 `JsonlAppender`）。

### Audit row

```json
{
  "event": "audit",
  "station": "LFPB",
  "slug": "paris",
  "local_date": "2026-08-12",
  "unit": "C",
  "metar_source": "aviationweather",
  "metar_max": 35.0,
  "metar_max_c": 35.0,
  "wu_max": 35.0,
  "delta": 0.0,
  "bucket": "exact"
}
```

`bucket`: `exact` | `pm1` | `larger` | `wu_missing` | `metar_missing`。`pm1` = `|delta| <= 1` 且非 0。US：`metar_max` 为换算到 °F 的值（记录换算：`round(c * 9/5 + 32)` 并在行内留 `metar_max_c`）。换算方式写死并测试；不在本任务里发明更“正确”的 ASOS averaging。

文件：`data/race/audit-YYYY-MM-DD.jsonl`。

### Error row

`{"event":"error","source":"iem","station":"KLGA","ts":"...","error":"..."}`。不抛死循环。

## Source adapters (in the one file)

三个小函数，不建 package：

- `fetch_awc(client, icaos) -> list[Observation]`
- `fetch_iem(station_cfg) -> Observation | None`
- `fetch_wu_daily_max(wu_location, local_date, unit) -> float | None`

WU key：工具内常量，注释标明来自 WU history 页前端 key、可能轮换；可用环境变量 `WUNDERGROUND_API_KEY` 覆盖。禁止把 IEM CLI/DSM 日高温当作 WU。

AWC 历史窗口约 15 天。`audit --backfill N`：N≤15 用 AWC；超出部分 METAR 改 IEM `asos.py` `report_type=3,4`（与 `kalshi_lock_probe.fetch_metar` 同口径，**复制最小请求参数，不 import probe 模块**）。

## Latency

只在 race 行拆：

```text
L_source ≈ first_seen_at - observation_at
L_provider ≈ provider_receipt_at - observation_at   # AWC only
L_poll    ≈ first_seen_at - provider_receipt_at
```

`summary` 打印这些分布。不改 resolver 日志。

## Compatibility

- `weadge serve` / `scan` / `stats` 行为不变；只是 yaml 多字段、多三城。
- `locked_buffer_c` 四城共用 0.5，不按城改。
- 现有 Paris fixture / `bucket_cap_high` 测试必须原样通过。

## Tradeoffs

| 选择 | 为何 |
|---|---|
| 一个 tools 文件，两个子命令 | 用户要求只新增一个工具 |
| `observation_extra` 而不是 `serve: false` 混进 `cities` | serve 候选与观测候选分家；nyc 不会被误 serve |
| AWC + IEM，不做 NWS | 六城都能 race；NWS 仅美国且 raw 常空 |
| IEM 美国去 K | 实测 `KLGA`/`KORD` 的 current.py 返回 `{}` |
| WU 按结算单位拉 | PM 美城结算 °F；用 `units=m` 会把 audit 做成假精确 |
| 不改 MetarClient 签名 | `ids` 已是透传；race 传逗号列表 |
| 不把 race 逻辑放进 `src/weadge/resolver/` | 避免 production 包膨胀；与 lock probe 同一层 |

## Rollback

删 `tools/observation_race.py`、`tests/test_observation_race.py`，把 yaml/config/docs 还原。无 DB migration。`data/race/` gitignore 已由 `data/` 覆盖则不必提交。
