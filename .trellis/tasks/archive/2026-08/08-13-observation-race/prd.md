# V0.2 observation race + resolution audit + multi-city shadow

## Goal

回答一个问题：**哪个公开 feed 最早给出“足够可信、且与结算口径一致”的 threshold crossing？**

不是追 lowest latency，而是同时追 settlement fidelity。本任务只加观测层工具和 shadow 城市配置，不改 LOCKED 策略，不上 NEAR_LOCKED，不改 production 30s scan。

## User value

当前数据支持的 hypothesis：

> 当 authoritative-enough observation 比市场价格更新早 0–2 分钟时，Daily High 市场存在可交易的 stale liquidity；该 edge 在约 5 分钟内快速衰减。

没有按 `station + observation_at` 的多源 first_seen，就无法把 L_source 从 L_poll / L_market 里拆出来；没有 METAR-derived max vs WU Daily Observations max 的逐日对照，0.5°C buffer 就仍是猜测。多城市是为了加快估计 `P(stale | source_age, city, station)`，不是为了加策略。

Race/audit 与 serve 回答不同问题，城市集合可以不一致：

```text
race/audit: 这个城市的数据链值不值得交易？
serve:      这个城市的市场语义是否已验证到可以安全扫描？
```

**NYC / Chicago 是 observation candidates，不是本任务的 trading candidates。**

## Confirmed facts

- Production observation 源是 AviationWeather METAR JSON（`MetarClient`）。字段含 `obsTime` / `reportTime` / `receiptTime` / `temp` / `rawOb`。官方上限约 100 req/min；resolver `serve` 保持 30s。
- IEM：进入美国 ASOS 日高温统计的是 2-minute average、whole °F；近实时整数 °C / 不同 averaging 不能忠实预测 settlement（[IEM news 1469](https://mesonet.agron.iastate.edu/onsite/news.phtml?id=1469)）。
- 现有 Kalshi probe 只用 IEM `report_type=3+4`，没有 5-minute HFMETAR。那是保守的 edge 时间，不是 lookahead。HFMETAR 不当 settlement truth。
- PM Daily High 结算源是 Wunderground **Daily Observations 表**（不是 Day High & Low summary）。`api.weather.com/v1/location/{id}/observations/historical.json` 对六城均可用。
- AWC `temp` 一律 °C。WU 必须按城市结算单位取数：°C 城 `units=m`，°F 城 `units=e`。US 城的 METAR→°F 是 audit 要暴露的 roundtrip，不是要抹掉的细节。
- IEM `current.py`：国际站用 ICAO（`LFPB`+`FR__ASOS`）；美国站去掉 `K`（`LGA`+`NY_ASOS`，`ORD`+`IL_ASOS`）。日志里的 `station` 仍写 ICAO。
- 现有 `bucket_cap_high()` 只解析 `°c` + `or below` / `or above`。NYC/Chicago 为 `between 76-77°F`。PM NYC 结算站是 KLGA，不是 Kalshi KNYC。
- 工具惯例：`tools/` 独立脚本，resolver 不 import tools。纯逻辑离线测，无 network 测试打外部 API。

## Locked city split

| slug | ICAO | tz | 单位 | race | audit | resolver serve |
|---|---|---|---|---|---|---|
| paris | LFPB | Europe/Paris | °C | yes | yes | yes |
| london | EGLC | Europe/London | °C | yes | yes | yes |
| tokyo | RJTT | Asia/Tokyo | °C | yes | yes | yes |
| seoul | RKSI | Asia/Seoul | °C | yes | yes | yes |
| nyc | KLGA | America/New_York | °F | yes | yes | **no** |
| chicago | KORD | America/Chicago | °F | yes | yes | **no** |

Serve 四城与 Paris 同构（`be N°C` / `or below`，WU 机场站）。NYC/Chicago 的 °F / `between` parser、`buffer_f`、`weadge serve --city nyc|chicago` 明确留到下个极小任务。

## Requirements

### R1 — Observation race recorder

新增 `tools/observation_race.py`。不进 production CLI。不改 `evaluate()` / LOCKED / `bucket_cap_high`。

每次某个 source 第一次见到一条 observation，追加 JSONL：

```json
{
  "station": "LFPB",
  "report_id": "LFPB-2026-08-13T13:30:00Z",
  "observation_at": "2026-08-13T13:30:00+00:00",
  "source": "aviationweather",
  "source_grade": "A",
  "first_seen_at": "2026-08-13T13:31:08+00:00",
  "provider_receipt_at": "2026-08-13T13:30:41+00:00",
  "raw_temp": "37",
  "decoded_temp": 37.0,
  "temp_unit": "C",
  "raw_metar": "METAR LFPB 131330Z AUTO VRB04KT CAVOK 37/08 Q1019 NOSIG"
}
```

身份键：`station + observation_at`（UTC，秒级）。同一身份同一 source 只记第一次 `first_seen_at`。

必须实现的 source：

- `aviationweather`：10s 轮询（仅 race 进程）。一次请求带齐全部 ICAO（`MetarClient.fetch` 的 `ids` 原样透传）。`receiptTime` → `provider_receipt_at`。
- `iem`：同一循环拉 IEM current，否则工具只是单源迟到时钟。美国站映射写在 config，日志 `station` 仍为 ICAO。

`nws` 与 HFMETAR 不做。单源失败记 error 行，不中断其他源。自定义 User-Agent。

子命令：`serve [--interval 10] [--once]`、`summary`（按 observation 列出 per-source first_seen 与领先秒数）。

Race 覆盖全部六站，24h 跑（不受 `scan_hours` 限制）。`scan_hours` 只约束 resolver serve。

### R2 — Source grade（config 元数据，不驱动交易）

| Grade | 含义 |
|---|---|
| A | settlement-faithful |
| B | useful but lossy |
| C | directional only |

城市 `source_grade` = **AWC METAR 相对该城结算** 的起始假设（°C 四城 A；NYC/Chicago B）。IEM 行固定 B。`if grade A: buffer=0` 等交易规则本任务不实现。

### R3 — Resolution audit

同一文件的 `audit` 子命令。T+1（或 `--backfill N`）对六城各记一行：

```text
METAR-derived daily max  vs  WU Daily Observations max
```

- 日界：站本地自然日 `[00:00, 24:00)`，与 `evaluate_observed` 一致，不是 Kalshi local-standard。
- METAR max：优先 AWC（≤15 天）；更长 backfill 用 IEM routine+SPECI（`report_type=3,4`），行上标明 `metar_source`。不用 HFMETAR。
- WU max：`historical.json` 当日 `observations[].temp` 的 max。°C 城 `units=m`，°F 城 `units=e`。
- 对照在**结算单位**下进行。US 城额外记下 `metar_max_c` 与换算后的 °F，不把 C→F 藏进一个数字。
- WU 失败必须可见，禁止静默用 IEM 日高温冒充 WU。
- 输出 JSONL + 控制台：每城 `exact / ±1 / larger / n`。

### R4 — Config 与文档

- `config/resolver.yaml` `cities:` 只有四城 serve 候选，补齐 settlement / WU / IEM / grade 字段。
- NYC/Chicago 放在同文件的 `observation_extra:`，**不**进入 `ResolverConfig.cities`。`weadge serve --city nyc` 继续 `KeyError`（与今天一样），不要做半套 parser。
- `weadge serve --city` 保持单城；不写多城 orchestrator。
- `docs/resolver.md` 与 `.trellis/spec/resolver.md`：hypothesis 换成上文那句；「禁止多城市」改为「新 serve 城市 = 与 Paris 同构的市场语义 + 人工核对结算站；观测候选可以更宽」。

### R5 — Latency 字段仅 race 日志

Race 行：`observation_at` / `provider_receipt_at` / `first_seen_at`。不改 resolver heartbeat/lock schema。production scan 仍 30s。

## Acceptance criteria

- [ ] `uv run python tools/observation_race.py serve --interval 10 --once` 对六站拉 AWC+IEM；JSONL 在 `data/race/`；同一 `(station, observation_at, source)` 不重复 first_seen。
- [ ] `summary` 能打印每个 observation 的 per-source first_seen 表。
- [ ] `audit --backfill 7` 对六城产出 METAR max vs WU max；WU 失败不回退成 IEM 冒充；每城独立计数。
- [ ] 纯逻辑测试：identity 去重、IEM `KLGA→LGA` / `KORD→ORD`、WU observations → daily max、audit mismatch 分桶、°C/°F 单位选择。无 network 测试打外部 API。
- [ ] `resolver.yaml` `cities` 含 paris/london/tokyo/seoul；`observation_extra` 含 nyc/chicago。`weadge serve --city london` 能 load；`--city nyc` 仍失败。
- [ ] `docs/resolver.md` hypothesis 已更新；写明 NYC/Chicago 不是本轮 trading candidates。
- [ ] `tests/resolver/test_resolver.py` 全绿：LOCKED 规则、`bucket_cap_high`、30s 默认路径无行为变化。

## Out of scope

- NEAR_LOCKED、概率模型、forecast、trade mode
- grade → buffer 自动交易规则
- production 30s → 10s
- resolver lock/heartbeat 的 execution latency 链
- HFMETAR 回测、NWS racer
- °F parser、`between X-Y°F`、`buffer_f`、`serve --city nyc|chicago`
- Kalshi fee replay
- 多进程 serve orchestrator
- 假设 METAR Tmax == WU Tmax
