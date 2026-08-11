"""weadge CLI.

    weadge kalshi sync-series KXHIGHNY
    weadge kalshi backfill KXHIGHNY
    weadge kalshi audit KXHIGHNY
    weadge dataset build --series KXHIGHNY --snapshot 12h,6h,3h,1h
    weadge research compare --series KXHIGHNY
    weadge research incremental --series KXHIGHNY
    weadge research latency --series KXHIGHNY
    weadge research walk-forward --series KXHIGHNY
    weadge backtest taker --series KXHIGHNY --model p_nbm --edge 0.06

v0 commands that need downloaded data raise a clear error when the lake is
empty — the pipeline order (data -> settlement -> dataset -> research) is
enforced by the gates, not by the CLI.
"""

from __future__ import annotations

from typing import Annotated

import polars as pl
import typer
from rich.console import Console

from weadge.config import data_root, load_cities, load_research
from weadge.storage.parquet import DataLake
from weadge.storage.schema import empty_frame

app = typer.Typer(help="weadge — weather prediction-market alpha research engine", no_args_is_help=True)
console = Console()

_series_arg = Annotated[str, typer.Argument(help="Kalshi series ticker, e.g. KXHIGHNY")]
_series_opt = Annotated[str, typer.Option("--series", help="Kalshi series ticker, e.g. KXHIGHNY")]


def _lake() -> DataLake:
    return DataLake(data_root())


def _require(table: str, layer: str = "bronze") -> None:
    if not _lake().exists(table, layer):
        raise SystemExit(
            f"table {layer}/{table} is empty — run the data pipeline first "
            f"(weadge kalshi backfill ...)"
        )


# ------------------------------------------------------------------- kalshi
kalshi_app = typer.Typer(help="Kalshi data pipeline.", no_args_is_help=True)
app.add_typer(kalshi_app, name="kalshi")


@kalshi_app.command("sync-series")
def sync_series(series: _series_arg) -> None:
    """Fetch series metadata (settlement_source, fee_type, fee_multiplier)."""
    from weadge.adapters.kalshi.client import KalshiClient
    from weadge.adapters.kalshi.markets import series_frame

    with KalshiClient() as client:
        frame = series_frame(client, series, save_raw=True)
    _lake().write_parquet("series", frame, layer="bronze")
    console.print(f"[green]series {series} synced[/green]")


@kalshi_app.command("backfill")
def backfill(series: _series_arg, start: str | None = None, end: str | None = None) -> None:
    """Backfill events, markets, candles, forecast history, and fee changes.

    start/end are ISO dates (YYYY-MM-DD). Without them, the adapter pulls
    from the earliest available history to today, routing live vs historical
    automatically by the /historical/cutoff endpoint.
    """
    from weadge.adapters.kalshi.client import KalshiClient
    from weadge.domain.time import parse_iso

    start_dt = parse_iso(start + "T00:00:00Z") if start else None
    end_dt = parse_iso(end + "T00:00:00Z") if end else None

    with KalshiClient() as client:
        client._lake = _lake()  # type: ignore[attr-defined]  # raw capture hook
        from weadge.adapters.kalshi.candles import candles_frame
        from weadge.adapters.kalshi.fees import fee_changes_frame
        from weadge.adapters.kalshi.forecasts import forecast_percentile_frame
        from weadge.adapters.kalshi.markets import events_frame, markets_frame

        lake = _lake()
        console.print(f"[cyan]backfilling {series}...[/cyan]")
        events = events_frame(client, series, start=start_dt, end=end_dt, save_raw=True)
        lake.write_parquet("events", events, layer="bronze", partition_by="series_ticker")
        console.print(f"  events: {events.height}")

        markets = markets_frame(client, series_ticker=series, start=start_dt, end=end_dt, save_raw=True)
        lake.write_parquet("markets", markets, layer="bronze", partition_by="series_ticker")
        console.print(f"  markets: {markets.height}")

        for ev in events.iter_rows(named=True):
            ev_ticker = ev["event_ticker"]
            ev_markets = markets.filter(pl_col_market_event(ev_ticker))

            for m in ev_markets.iter_rows(named=True):
                close_at = m["close_at"]
                if close_at is None:
                    continue
                open_at = m["open_at"]
                candle_start = open_at if open_at is not None else start_dt
                if candle_start is None:
                    continue
                candles = candles_frame(
                    client,
                    m["market_ticker"],
                    start=candle_start,
                    end=close_at,
                    series_ticker=series,
                    save_raw=True,
                )
                lake.write_parquet("quote_1m", candles, layer="bronze",
                                   partition_by="market_ticker")

            # forecast percentile window = the event's market open..close span
            f_start = ev_markets["open_at"].min()
            f_end = ev_markets["close_at"].max()
            if f_start is not None and f_end is not None:
                pcts = forecast_percentile_frame(
                    client, ev_ticker, series,
                    start=f_start, end=f_end, save_raw=True,
                )
            else:
                pcts = empty_frame("forecast_percentiles")
                console.print(f"  {ev_ticker}: no market window — forecast skipped")
            lake.write_parquet("forecast_percentiles", pcts, layer="bronze",
                               partition_by="event_ticker")
            console.print(f"  {ev_ticker}: {len(ev_markets)} markets, {pcts.height} pct rows")

        fees = fee_changes_frame(client, series, save_raw=True)
        lake.write_parquet("fee_changes", fees, layer="bronze", partition_by="series_ticker")
        console.print(f"  fee changes: {fees.height}")
    console.print("[green]backfill done[/green]")


def pl_col_market_event(event_ticker: str):
    """Return a polars expression filtering markets by event ticker."""
    return pl.col("event_ticker") == event_ticker


@kalshi_app.command("audit")
def audit(series: _series_arg) -> None:
    """Settlement audit: Kalshi result vs official observation.

    Research is FROZEN while mismatch > 0. This command is the G0 gate.
    """
    from weadge.dataset.settlement import SettlementOracle, SettlementSpec

    _require("events")
    _require("markets")
    _require("observations")
    city = load_cities().by_series(series)
    spec = SettlementSpec(
        series=series,
        location=city.location,
        station_id=city.station_id,
        timezone=city.timezone,
        source=city.settlement.source,
        day_window=city.settlement.day_window,  # type: ignore[arg-type]
        rounding=city.settlement.rounding,
    )
    lake = _lake()
    report = SettlementOracle(
        spec,
        events=lake.read("events").filter(pl_col("series_ticker") == series),
        markets=lake.read("markets").filter(pl_col("series_ticker") == series),
        observations=lake.read("observations"),
    ).audit()
    console.print(report)
    if not report.clean:
        raise SystemExit("settlement audit FAILED — fix data before research")
    console.print("[green]settlement audit passed (G0)[/green]")


# ------------------------------------------------------------------ dataset
dataset_app = typer.Typer(help="Alpha dataset construction.", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("build")
def dataset_build(
    series: _series_opt,
    snapshot: Annotated[str, typer.Option("--snapshot", help="comma-separated lead hours, e.g. 12h,6h,3h,1h")] = "24h,12h,6h,3h,1h",
) -> None:
    """Build gold/alpha_dataset.parquet for a series."""
    _require("events")
    _require("markets")
    _require("quote_1m")
    leads = [int(x.strip().rstrip("h")) for x in snapshot.split(",") if x.strip()]
    from weadge.dataset.builder import build_from_lake

    cfg = load_research()
    df, stats = build_from_lake(
        _lake(), series, snapshots_lead_hours=tuple(leads),
        fallback_fee_multiplier=cfg.fees.get("fallback_fee_multiplier"),
    )
    path = _lake().gold_path()
    df.write_parquet(path, compression="zstd")
    console.print(f"[green]alpha dataset: {df.height} rows -> {path}[/green]")
    console.print(f"cells (theoretical max)   {stats['cells_total']}")
    console.print(f"rows built                {stats['rows_built']}")
    console.print(f"rows dropped (all missing){stats['rows_dropped']}")
    console.print("drop reasons (cells)")
    console.print(f"  missing_market_quote      {stats['missing_market_quote']}")
    console.print(f"  missing_nbm               {stats['missing_nbm']}")
    console.print(f"  missing_kalshi_forecast   {stats['missing_kalshi_forecast']}")
    console.print("drop reasons (partitions)")
    console.print(f"  market_partition_incomplete {stats['market_partition_incomplete']}")
    console.print(f"  simplex_infeasible          {stats['simplex_infeasible']}")


# ------------------------------------------------------------------- noaa
noaa_app = typer.Typer(help="NOAA/NWS data pipeline (settlement truth).", no_args_is_help=True)
app.add_typer(noaa_app, name="noaa")


@noaa_app.command("backfill-dcr")
def noaa_backfill_dcr(
    series: _series_opt,
    start: Annotated[str, typer.Option("--start", help="ISO date YYYY-MM-DD (inclusive)")],
    end: Annotated[str, typer.Option("--end", help="ISO date YYYY-MM-DD (inclusive)")],
) -> None:
    """Backfill NWS Daily Climate Report (CLINYC) via the IEM text archive.

    Settlement truth only: preliminary bulletins are rejected, the latest
    complete daily per report date wins, and observations are written with
    source='NWS Daily Climate Report' (IEM is just the transport).
    """
    from datetime import date

    from weadge.adapters.noaa.dcr import backfill_dcr

    city = load_cities().by_series(series)
    station_id = city.station_id  # e.g. KNYC
    pil = f"CLI{station_id[1:]}"  # KNYC -> CLINYC
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)

    lake = _lake()
    summary = backfill_dcr(start_d, end_d, lake, pil=pil, station_id=station_id)

    console.print(f"[cyan]{pil} DCR BACKFILL[/cyan]  {start} -> {end}")
    console.print(f"products fetched      {summary['products_fetched']}")
    console.print(f"preliminary rejected  {summary['preliminary_rejected']}")
    console.print(f"complete daily        {summary['complete_daily']}")
    console.print(f"corrections           {summary['corrections']}")
    console.print(f"unique report days    {summary['unique_report_days']}")
    console.print(f"parsed maximum        {summary['parsed_maximum']}")
    console.print(f"missing maximum       {summary['missing_maximum']}")
    console.print(f"foreign rejected      {summary['foreign_rejected']}")


@noaa_app.command("backfill-nbm")
def noaa_backfill_nbm(
    series: _series_opt,
    start: Annotated[str, typer.Option("--start", help="ISO date YYYY-MM-DD (inclusive)")],
    end: Annotated[str, typer.Option("--end", help="ISO date YYYY-MM-DD (inclusive)")],
) -> None:
    """Backfill NBM MaxT QMD distributions (AWS archive) for each target date.

    Per target date D: the D-00Z run's f030 (day-1, window [D 12Z, D+1 06Z))
    and the (D-1)-00Z run's f054 (day-2 — the SAME window, knowable 24h
    earlier, which is what makes the T-24h snapshot covered). availability
    = observed S3 Last-Modified. The NBM window differs from the DCR
    settlement day — recorded, not aligned.
    """
    from datetime import date

    from weadge.adapters.noaa.nbm import backfill_nbm

    city = load_cities().by_series(series)
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    summary = backfill_nbm(
        start_d, end_d, _lake(),
        series=series, station_id=city.station_id,
        lat=city.lat, lon=city.lon,
    )
    console.print(f"[cyan]NBM MaxT QMD BACKFILL[/cyan]  {start} -> {end}")
    console.print(f"days requested       {summary['days_requested']}")
    console.print(f"f030 (day-1) fetched {summary['f030_fetched']}  missing {summary['f030_missing']}")
    console.print(f"f054 (day-2) fetched {summary['f054_fetched']}  missing {summary['f054_missing']}")
    console.print(f"model v5.0.x         {summary['v5_0_x']} runs")
    console.print(f"model v5.0.14        {summary['v5_0_14']} runs")


@noaa_app.command("audit-nbm")
def noaa_audit_nbm(series: _series_opt) -> None:
    """NBM smoke audit: snapshot coverage, distribution, ordering, as-of."""
    from weadge.adapters.noaa.nbm import nbm_smoke_audit

    _require("events")
    _require("markets")
    _require("forecasts")
    lake = _lake()
    a = nbm_smoke_audit(
        lake.read("events").filter(pl.col("series_ticker") == series),
        lake.read("markets").filter(pl.col("series_ticker") == series),
        lake.read("forecasts"),
        series=series,
    )
    console.print("[cyan]NBM SMOKE AUDIT[/cyan]")
    console.print(f"events                          {a['events']}")
    console.print(f"events with >=1 NBM forecast    {a['events_with_forecast']}")
    console.print("snapshot coverage (day-D MaxT window available)")
    for h, n in a["coverage"].items():
        console.print(f"  T-{h:>2}h                         {n}/{a['events']}")
    console.print("distribution (of forecast rows)")
    console.print(f"  mean present                   {a['pct_mean']:.0f}%")
    console.print(f"  std present                    {a['pct_std']:.0f}%")
    console.print(f"  p10..p90 present               {a['pct_p']:.0f}%")
    console.print(f"ordering violations              {a['ordering_violations']}")
    console.print(f"as-of violations                 {a['asof_violations']}")
    console.print(f"wrong target-window              {a['wrong_target_window']}")
    console.print(a["window_note"])
    console.print("NBM version distribution (rows / runs)")
    for v in a["versions"]:
        runs = next(
            (r["len"] for r in a["version_runs"] if r["model_version"] == v["model_version"]),
            0,
        )
        console.print(f"  {v['model_version']:<14} {v['len']:3d} rows  {runs:2d} runs")


# ---------------------------------------------------------------- research
research_app = typer.Typer(help="Forecast research (calibration, incremental alpha, latency).", no_args_is_help=True)
app.add_typer(research_app, name="research")


@research_app.command("compare")
def research_compare(series: _series_opt) -> None:
    """Brier / LogLoss table: Market (raw/normalized/simplex) vs KalshiForecast vs NBM."""
    from weadge.research.scoring import score_frame

    df = _load_gold(series)
    table = score_frame(
        df,
        ["p_market_raw", "p_market_normalized", "p_market_simplex",
         "p_kalshi_forecast", "p_nbm"],
    )
    console.print(table)


@research_app.command("calibration")
def research_calibration(series: _series_opt) -> None:
    """Reliability index per model (Market raw/normalized/simplex, KF, NBM)."""
    from weadge.research.calibration import calibration_table

    df = _load_gold(series)
    console.print(
        calibration_table(
            df,
            ["p_market_raw", "p_market_normalized", "p_market_simplex",
             "p_kalshi_forecast", "p_nbm"],
        )
    )


@research_app.command("incremental")
def research_incremental(series: _series_opt) -> None:
    """Alpha existence test: does weather add OOS info given the market?

    Walk-forward, paired samples, event/date clustered bootstrap, run for
    EACH market construction (raw / normalized / simplex). Weather alpha is
    only credible if it survives all three M0 baselines.
    """
    from weadge.research.edge import paired_incremental_gate
    from weadge.research.walk_forward import split_frame

    df = _load_gold(series)
    dates = sorted(df["event_date"].unique().to_list())
    baselines = ["p_market_raw", "p_market_normalized", "p_market_simplex"]
    wins = {b: 0 for b in baselines}
    total = 0
    for i in range(len(dates) - 3):
        train_start, test_start = dates[i], dates[i + 3]
        train, test = split_frame(df, train_start, test_start)
        if test.is_empty():
            continue
        total += 1
        for base in baselines:
            g = paired_incremental_gate(train, test, market_col=base)
            wins[base] += int(g.has_incremental_alpha)
            flag = "ALPHA" if g.has_incremental_alpha else "none"
            console.print(
                f"{test_start.date()} M0={base:<22} delta_ll={g.delta_ll:+.4f} "
                f"95% CI=[{g.ci_lower:+.4f}, {g.ci_upper:+.4f}] "
                f"gamma={g.gamma if g.gamma is None else f'{g.gamma:+.3f}'} "
                f"n={g.test_n} clusters={g.n_clusters} -> {flag}"
            )
    for base in baselines:
        console.print(f"incremental alpha vs {base}: {wins[base]}/{total} windows")
    if all(wins[b] == total for b in baselines):
        console.print("[green]weather alpha robust to market construction[/green]")
    elif wins["p_market_raw"] == total and wins["p_market_simplex"] < total:
        console.print(
            "[red]weather alpha does NOT survive the simplex baseline — this is "
            "likely market-probability-normalization alpha, not weather alpha[/red]"
        )


@research_app.command("latency")
def research_latency(series: _series_opt) -> None:
    """Edge decay across +1/+2/+5/+10 minute execution delays."""
    from weadge.research.latency import delayed_execution_edges, edge_by_delay_summary

    df = _load_gold(series)
    signals = df.rename({"p_nbm": "p_model"}).select(
        ["market_ticker", "decision_at", "p_model"]
    ).drop_nulls()
    quotes = _lake().read("quote_1m")
    delayed = delayed_execution_edges(signals, quotes)
    console.print(edge_by_delay_summary(delayed))


@research_app.command("walk-forward")
def research_walk_forward(series: _series_opt) -> None:
    """Chronological walk-forward: log loss of M0 vs M2 per expanding window.

    All models are scored on the same (paired) rows; the significance gate
    lives in `research incremental` (clustered bootstrap).
    """
    from weadge.research.edge import fit_incremental
    from weadge.research.walk_forward import split_frame

    df = _load_gold(series)
    dates = sorted(df["event_date"].unique().to_list())
    for i in range(0, len(dates) - 3):
        train_start, test_start = dates[i], dates[i + 3]
        train, test = split_frame(df, train_start, test_start)
        if test.is_empty():
            continue
        for r in fit_incremental(train, test):
            console.print(
                f"{test_start.date()} M{r.model[1]}: ll={r.test_log_loss:.4f} "
                f"brier={r.test_brier:.4f} n={r.test_n}"
            )


# ---------------------------------------------------------------- backtest
backtest_app = typer.Typer(help="Taker backtest.", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("taker")
def backtest_taker(
    series: _series_opt,
    model: Annotated[str, typer.Option("--model", help="p_nbm | p_market_raw | p_market_normalized | p_market_simplex | p_kalshi_forecast")] = "p_nbm",
    edge: Annotated[float, typer.Option("--edge", help="minimum pre-fee edge")] = 0.06,
) -> None:
    """Taker backtest: BUY YES if p_model - ask >= edge; delayed fills; fee replay."""
    from weadge.backtest.engine import run_taker_backtest
    from weadge.backtest.fees import series_fee_schedule

    df = _load_gold(series)
    if model not in df.columns:
        raise SystemExit(f"model column {model} not in gold dataset")
    signals = df.rename({model: "p_model"}).select(
        ["market_ticker", "decision_at", "p_model", "result"]
    ).drop_nulls()
    quotes = _lake().read("quote_1m")
    fee_schedule = series_fee_schedule(
        {"fee_multiplier": 1.0, "fee_type": "taker"}, _lake().read("fee_changes")
    )
    report = run_taker_backtest(
        signals, quotes, fee_schedule, threshold=edge, delay_min=1
    )
    console.print(report.as_text())
    console.print("\n[bold]Edge calibration (predicted -> realized)[/bold]")
    console.print(report.edge_bins)


# ---------------------------------------------------------------- live
live_app = typer.Typer(help="Live recording (v2).", no_args_is_help=True)
app.add_typer(live_app, name="live")


@live_app.command("record")
def live_record(series: _series_opt) -> None:
    """Start the real-time recorder (v2 — not available in v0)."""
    import asyncio

    from weadge.adapters.kalshi.websocket import KalshiWebSocket

    ws = KalshiWebSocket()
    try:
        asyncio.run(ws.connect(series))
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from None


def _load_gold(series: str) -> pl.DataFrame:
    _require("quote_1m")
    path = _lake().gold_path()
    if not path.exists():
        raise SystemExit("gold dataset missing — run: weadge dataset build")
    return pl.read_parquet(path).filter(pl.col("city") == series)


def pl_col(name: str):
    return pl.col(name)


if __name__ == "__main__":
    app()
