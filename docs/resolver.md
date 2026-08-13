# Resolver — Observation-Locked Scanner (V0)

> Weadge 唯一 production 主线。本文档是策略唯一权威文档；与代码冲突时以本文档为准。

## 一句话

PM Daily High 市场 + 机场站观测 → 找出"按结算规则已不可能"的 outcome → 看盘口有没有反应 → alert。

## Hypothesis（V0 要验证的东西）

**当 authoritative-enough observation 比市场价格更新早 0–2 分钟时，Daily High 市场存在可交易的 stale liquidity；该 edge 在约 5 分钟内快速衰减。**

这是 `latency × measurement fidelity` 的微结构 alpha，不是天气预报。支撑结构：

- 日高温通常在 local 14:00–17:00 达到，之后 losing buckets 应归零
- 结算延迟 2–4h（close → payout），规则锁定 ≠ 价格锁定
- 结算站是机场 ASOS + Wunderground Daily Observations 表（巴黎 = LFPB / Bonneuil-en-France）
- 更快的 feed 不一定忠实于结算口径（整数 °C、不同 averaging、C→F roundtrip）

**Kill test**：每次 bucket 首次进入 LOCKED 时，记录可执行 NO ask + depth，
测量 CLOB 达到 0.97/0.99 所需时间。10 个交易日后输出：

```text
LOCK EVENTS: 74
at lock: median NO ask 0.982 / p90 ...
< 0.97: 7  events    < 0.95: 2    < 0.90: 0
reaction to 0.97: median 41s / p90 126s
executable $5: 6   executable $20: 2
```

若 LOCKED 时 NO ask 通常已 ≥ 0.99 → V0 死，转 V1 (NEAR_LOCKED)。
`weadge stats` 直接输出该报告。

## 范围（V0 明确不做）

| 不做 | 理由 |
|------|------|
| NEAR_LOCKED / 概率模型 | 复杂度只能由已验证的问题购买 |
| Forecast (ECMWF/GFS) | V2，V0 失败才加 |
| 自动下单 (trade mode) | 先 shadow → alert → trade |
| Telegram 实际发送 | 无 token 不可测，留 stub |
| °F / `between X-Y°F` parser | NYC/Chicago 是 observation candidates，不是 trading candidates |
| production scan 10s | 10s 只在 `tools/observation_race.py`；resolver 仍 30s |
| Kalshi / 跨平台 | production 只接 PM |
| 热端锁定（or-above 桶） | 无上限，依赖时间+斜率 = V1 |

## 结算事实（已验证）

- PM 事件：`highest-temperature-in-paris-on-2026-08-13`，tag `daily-temperature`
- 每温度桶一个二元市场（YES/NO），negRisk，同一事件内互斥
- bucket 语义：**向下取整** —— 读数 23.4°C → 23°C 桶；"be 33°C" 赢区间 [33, 34)
- 巴黎结算源写死在市场规则：`wunderground.com/history/daily/fr/bonneuil-en-france/LFPB`
- 同构 °C serve 城市（规则已人工核对）：London EGLC、Tokyo RJTT、Seoul RKSI
- NYC KLGA / Chicago KORD 结算为 °F + `between` 桶，只进 observation race / resolution audit，**不** `weadge serve`
- Weather taker fee 5%：`fee = C × 0.05 × p(1-p)`，极端价格 fee ≈ 0
- 公共数据免凭证（gamma + aviationweather + CLOB /book），只有下单要 key
- **gamma `outcomePrices` 只是展示价，不是可成交价**（实测巴黎 37°C 桶
display 0.415/0.585 vs 真实 book 0.01/0.99）；`no_ask` 一律取 CLOB `/book`
- gamma `clobTokenIds` 数组顺序不可靠（py-clob-client issue #276）——NO token
  由 CLOB `/markets/{condition_id}` 的 tokens 节点权威解析，且只对 LOCKED
  桶拉 book（省请求）

## 状态机（V0 只实现 LOCKED）

```text
OPEN         → 信息不足，不动作
NEAR_LOCKED  → V1（当前温度+时间+斜率 → P(Tmax survives)）
LOCKED       → observed_max ≥ bucket cap + buffer → YES 数学上不可能
```

冷端锁定规则（state.py）：

```python
bucket 赢区间 [cap_low, cap_high)  # "or below" 桶 cap_high = 该值+1；"or above" 桶 = ∞
LOCKED ⟺ observed_max ≥ cap_high + locked_buffer_c   # buffer 防观测/结算源偏差
```

## Edge 计算（edge.py）

```text
理论 NO 价 = 1.0（LOCKED）
no_ask = CLOB /book 最优卖价（无 resting ask → 不可成交, 不产出 signal）
fee = 0.05 × p_no × (1 - p_no)          # taker
net_edge = 1.0 - no_ask - fee - exec_buffer
signal ⟺ net_edge ≥ min_net_edge
```

## 扫描与日志（service.py / log.py）

- `weadge serve`：scan window 内每 30s 循环（V0 不需要 WS；
  10 分钟粒度会直接测错反应延迟）
- 每次扫描必写 heartbeat 行（scanner 存活 + observed 状态）
- 每个 LOCKED 桶写 lock 行：`{ts, bucket, state, no_best_ask, no_ask_size, net_edge, signal}`
- 无 signal 也记录 —— kill test 需要区分"没机会"与"scanner 没跑"
- 日志是 stdlib JSONL（`resolver/log.py`），不依赖 live/recorder（recorder 挂了 bot 不能死）

## 模式（同一代码路径，最后一步不同）

```bash
weadge scan --mode shadow   # 记模拟成交（JSONL）
weadge scan --mode alert    # + Telegram（有 token 时）
weadge scan --mode trade    # v1+，当前 NotImplementedError
```

## 数据流

```text
gamma API (daily-temperature tag) ─┐
aviationweather METAR (serve 30s) ─┼→ evaluate() → find_edges() → shadow/alert
                                   └→ tools/observation_race.py（10s AWC+IEM；WU audit）
```

`evaluate()` 是纯函数 —— 实时和历史 replay 走同一代码路径，不建框架。

## 参数（config/resolver.yaml）

| 参数 | 默认 | 含义 |
|------|------|------|
| locked_buffer_c | 0.5 | 观测必须超过桶上界 ≥0.5°C 才算锁定 |
| min_net_edge | 0.02 | 净 edge 阈值（2¢） |
| exec_buffer | 0.01 | 滑点/成交率缓冲 |
| stale_after_min | 30 | 观测超过 30min 视为 stale，不发信号 |
| scan_hours | 12:00–21:00 local | 扫描时间窗（日高温时段前后） |

`cities` 是 serve 候选（paris/london/tokyo/seoul）。`observation_extra` 是 race/audit 专用（nyc/chicago），`weadge serve --city nyc` 会 KeyError。

观测 race / WU audit：

```bash
uv run python tools/observation_race.py serve --interval 10
uv run python tools/observation_race.py audit --backfill 7
```

## 冻结线

Research v1（price-bin / stale-book / lead-lag / coherence arb / cheap YES /
multi-city forecast）FROZEN。未来 90 天 Weadge 只允许增加 resolver 所需代码。
