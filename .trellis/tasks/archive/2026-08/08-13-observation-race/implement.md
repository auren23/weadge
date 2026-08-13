# Implement — observation race / resolution audit / shadow cities

## Order

1. **Config models** — `ResolverCityConfig` 增加可选字段（`settlement_source`, `settlement_url`, `wu_location`, `source_grade`, `iem_network`, `iem_station`）。`ResolverConfig.observation_extra: list[ResolverCityConfig] = []`。加 `observation_stations()` = cities + extra。默认值保证现有测试手写 config 仍能构造。
2. **yaml** — 四城写入 `cities`（含核对过的 WU/IEM/grade）。NYC/Chicago 只进 `observation_extra`。`edge` 不动。
3. **Tool core (pure)** — identity 键、first_seen 去重 set、IEM 站映射、WU max、audit `bucket`、°C/°F 单位与 `c_to_f`。先写 `tests/test_observation_race.py`（importlib 加载 tools 文件，抄 `tests/test_lock_probe.py`）。
4. **Tool I/O** — argparse：`serve` / `summary` / `audit`。AWC 批量 fetch、IEM current、WU historical、JSONL via `JsonlAppender`。`--once` 只跑一轮。User-Agent：`weadge-observation-race/0.1`。
5. **Audit METAR** — 短窗口走 AWC `hours=`；`--backfill` 超过 ~15 天的日期走 IEM asos.py `report_type=3,4`。本地日界复用 `evaluate_observed`（把 IEM 行适配成 `{obsTime, temp}` °C）。
6. **Docs** — `docs/resolver.md` hypothesis + 城市表 +「NYC/Chicago 非本轮 trading candidates」。`.trellis/spec/resolver.md` 同步：允许同构 °C 多城 shadow；禁止本任务改 LOCKED / 美城 parser。
7. **回归** — `tests/resolver/test_resolver.py` 必须全绿，证明没碰策略。

## Files expected to change

- `src/weadge/config.py`
- `config/resolver.yaml`
- `tools/observation_race.py` (new)
- `tests/test_observation_race.py` (new)
- `docs/resolver.md`
- `.trellis/spec/resolver.md`

Do not edit: `state.py`, `edge.py`, `markets.py`, `service.py`, `observations.py`（只 import，不改 MetarClient）, `tools/kalshi_lock_probe.py`.

## Validation

```bash
uv run pytest tests/test_observation_race.py tests/resolver/test_resolver.py
uv run ruff check src/weadge/config.py tests/test_observation_race.py
uv run ruff format src/weadge/config.py tests/test_observation_race.py tools/observation_race.py
uv run pyright
```

手动（实现后可选，非 CI）：

```bash
uv run python tools/observation_race.py serve --once
uv run python tools/observation_race.py summary
uv run python tools/observation_race.py audit --backfill 2 --city paris
```

`weadge serve --city nyc` 仍应 KeyError。`weadge serve --city london` 应能进入 scan（窗外则立刻 return）。

## Risks

- WU 前端 key 轮换 → audit 全 `wu_missing`；用 env 覆盖，失败要打印 HTTP 状态，不许吞掉。
- AWC 10s × 六站若改成逐站请求会逼近 100/min；必须 **一条 ids= 逗号列表**。
- IEM 美国站若传 `KLGA` 会静默 `{}` → 当成“没有新观测”而不是错误。映射缺失时写 error 行。
- `evaluate_observed` 的 temp 是 °C。IEM asos `tmpf` 必须先转 °C 再喂进去，否则 US audit 的 `metar_max_c` 错。
- 不要把 `observation_extra` 误加入 `by_slug`。

## Rollback point

yaml 与 config 模型是唯一可能影响 production serve 的改动。若 serve 因未知字段/校验失败起不来：还原 `ResolverCityConfig` 新字段为全 optional 默认，或还原 yaml。工具文件可单独删。
