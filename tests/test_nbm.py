"""NBM archived probabilistic guidance — probe tests.

Fixtures are REAL data from the AWS NBM archive (2026-07-15 00Z run,
Hawaii f030 qmd):
    hi_max2t_p50_2026071500_f030.grib2   max_2t 50% level record (93KB)
    hi_f030.idx                          .idx excerpt (records 225-266)
    nbp_knyc_2026071500.txt              NBP station card, KNYC section
    nbp_phnl_2026071500.txt              NBP station card, PHNL section

The KNYC value cross-check (96.0°F == TXNP5) was verified live against the
CONUS file; the Hawaii fixture proves the same chain offline (PHNL within
grid resolution of the station card).
"""

from __future__ import annotations

import itertools
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest

from weadge.adapters.noaa.nbm import (
    DecodedField,
    NbmArchive,
    _window_for,
    backfill_nbm,
    decode_blob,
    max_2t_records,
    model_version_for,
    nbm_smoke_audit,
    nbp_max_t_values,
    nearest_value,
    neighborhood_max,
    parse_idx,
    qmd_url,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nbm"
RUN_INIT = datetime(2026, 7, 15, 0, 0, tzinfo=UTC)
RUN_DATE = datetime(2026, 7, 15, tzinfo=UTC)


class TestIdx:
    def test_parse_idx_and_offsets(self) -> None:
        rows = parse_idx((FIXTURES / "hi_f030.idx").read_text())
        assert rows[0].record == 225
        assert rows[-1].record == 266
        # consecutive offsets chain correctly
        for a, b in itertools.pairwise(rows):
            assert a.next_offset == b.offset

    def test_max_2t_records_family(self) -> None:
        rows = max_2t_records(parse_idx((FIXTURES / "hi_f030.idx").read_text()))
        assert "mean" in rows and "std" in rows
        for p in (10, 25, 50, 75, 90):
            assert p in rows, f"missing max_2t percentile {p}"
        # only the MaxT window family — instantaneous 30h fcst excluded
        assert rows[50].desc == "50% level"
        assert all("max fcst" in r.fcst or "StdDev fcst" in r.fcst for r in rows.values())


class TestDecode:
    def test_decode_fixture_metadata(self) -> None:
        blob = (FIXTURES / "hi_max2t_p50_2026071500_f030.grib2").read_bytes()
        fields = decode_blob(blob, RUN_INIT)
        assert len(fields) == 1
        f = fields[0]
        assert f.short_name == "max_2t"
        assert f.name == "Time-maximum 2 metre temperature"
        assert f.percentile == 50
        assert f.stat_proc == 2  # max
        assert (f.start_step, f.end_step) == (12, 30)
        # window [init+12h, init+30h) ends 06Z next day
        assert f.valid_at == datetime(2026, 7, 16, 6, 0, tzinfo=UTC)
        assert f.values.shape == f.lats.shape

    def test_nearest_and_neighborhood_phnl(self) -> None:
        """PHNL: the nearest grid point falls near the coast; the land patch
        max must land within grid resolution of the NBP station card."""
        fields = decode_blob(
            (FIXTURES / "hi_max2t_p50_2026071500_f030.grib2").read_bytes(), RUN_INIT
        )
        f = fields[0]
        nearest = (nearest_value(f, 21.3187, -157.9225) - 273.15) * 9 / 5 + 32
        land = (neighborhood_max(f, 21.3187, -157.9225) - 273.15) * 9 / 5 + 32
        nbp = nbp_max_t_values((FIXTURES / "nbp_phnl_2026071500.txt").read_text())
        assert abs(land - nbp["TXNP5"]) <= 2.0, f"land {land} vs NBP {nbp['TXNP5']}"
        assert nearest <= land  # coastal point can be cooler than the island


class TestNbpCrossCheck:
    def test_knyc_station_card(self) -> None:
        """The exact live-probe value: KNYC max_2t p50 (FHR24) == 96°F."""
        nbp = nbp_max_t_values((FIXTURES / "nbp_knyc_2026071500.txt").read_text())
        assert nbp["TXNP5"] == 96
        assert nbp["TXNMN"] == 96
        assert nbp["TXNSD"] == 2
        assert nbp["TXNP9"] == 98


class TestVersion:
    def test_v5_0_14_schedule(self) -> None:
        assert model_version_for(datetime(2026, 7, 28, 11, 59, tzinfo=UTC)) == "nbm_v5.0.x"
        assert model_version_for(datetime(2026, 7, 28, 12, 0, tzinfo=UTC)) == "nbm_v5.0.14"


class TestUrls:
    def test_qmd_url(self) -> None:
        assert qmd_url(RUN_DATE, 0, 30) == (
            "https://noaa-nbm-grib2-pds.s3.amazonaws.com/"
            "blend.20260715/00/qmd/blend.t00z.qmd.f030.co.grib2"
        )


class TestArchiveLive:
    """End-to-end against the real archive (network; not run by default)."""

    @pytest.mark.network
    def test_max_2t_qmd_knyc_matches_nbp(self) -> None:
        from weadge.adapters.noaa.nbm import nearest_value

        with NbmArchive() as arch:
            qmd = arch.max_2t_qmd(RUN_DATE, 0, 30, domain="co")
        assert "p50" in qmd
        f = (nearest_value(qmd["p50"], 40.7790, -73.9692) - 273.15) * 9 / 5 + 32
        assert f == pytest.approx(96.0, abs=0.6)  # live-verified: exactly 96.0


# ------------------------------------------------------------------ backfill
class _FakeResp:
    def __init__(self, status_code: int, headers: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _FakeArchive:
    """NbmArchive stand-in: canned 1x1 max_2t fields, scripted HEAD."""

    def __init__(self, head_status: int = 200) -> None:
        self.head_status = head_status

    def max_2t_records_raw(self, run_date, cycle, fhour, domain="co"):
        run_init = datetime(run_date.year, run_date.month, run_date.day, cycle, tzinfo=UTC)

        def field(percentile=None, stat=2):
            return DecodedField(
                name="Time-maximum 2 metre temperature",
                short_name="max_2t",
                percentile=percentile,
                start_step=fhour - 18,
                end_step=fhour,
                valid_at=run_init + timedelta(hours=fhour),
                stat_proc=stat,
                values=np.array([[300.0]]),
                lats=np.array([[40.78]]),
                lons=np.array([[-73.97]]),
            )

        return b"fake-grib", {
            "mean": field(),
            "std": field(stat=6),
            **{f"p{p}": field(percentile=p) for p in (10, 25, 50, 75, 90)},
        }

    def _request(self, method, url, **kwargs):
        if method == "HEAD":
            if self.head_status == 200:
                m = re.search(r"blend\.(\d{8})/", url)
                assert m, url
                run_day = datetime.strptime(m.group(1), "%Y%m%d")
                lm = run_day.strftime("%a, %d %b %Y 07:15:00 GMT")
                return _FakeResp(200, {"last-modified": lm})
            return _FakeResp(404, {})
        raise AssertionError(f"unexpected request: {method} {url}")


class TestBackfill:
    def test_rows_windows_and_partition(self, tmp_path) -> None:
        from weadge.storage.parquet import DataLake

        lake = DataLake(tmp_path)
        summary = backfill_nbm(
            datetime(2026, 7, 15).date(),
            datetime(2026, 7, 16).date(),
            lake,
            series="KXHIGHNY",
            station_id="KNYC",
            lat=40.78,
            lon=-73.97,
            archive=_FakeArchive(),
        )
        assert summary["days_requested"] == 2
        assert summary["f030_fetched"] == 2 and summary["f054_fetched"] == 3
        assert summary["f030_missing"] == 0 and summary["f054_missing"] == 0
        # leading day (07-14) supplies f054 only; 07-15/07-16 both windows
        fc = lake.read("forecasts")
        assert fc.height == 5
        # the day-1 row of run 07-15...
        f030 = [
            r
            for r in fc.iter_rows(named=True)
            if r["run_id"] == "2026071500Z"
            and r["valid_start"] == datetime(2026, 7, 15, 12, tzinfo=UTC)
        ]
        assert len(f030) == 1
        assert f030[0]["valid_end"] == datetime(2026, 7, 16, 6, tzinfo=UTC)
        assert f030[0]["model_version"] == "nbm_v5.0.x"
        assert f030[0]["available_at"] == datetime(2026, 7, 15, 7, 15, tzinfo=UTC)
        # ...and its day-2 row (same calendar window as the next run's day-1)
        f054 = [
            r
            for r in fc.iter_rows(named=True)
            if r["run_id"] == "2026071500Z"
            and r["valid_start"] == datetime(2026, 7, 16, 12, tzinfo=UTC)
        ]
        assert len(f054) == 1
        assert f054[0]["valid_end"] == datetime(2026, 7, 17, 6, tzinfo=UTC)
        # the leading day's f054 has the SAME window as run 07-15's f030
        lead = [
            r
            for r in fc.iter_rows(named=True)
            if r["run_id"] == "2026071400Z"
            and r["valid_start"] == datetime(2026, 7, 15, 12, tzinfo=UTC)
        ]
        assert len(lead) == 1
        assert lead[0]["valid_end"] == datetime(2026, 7, 16, 6, tzinfo=UTC)
        # raw capture exists with the observed availability sidecar
        meta = json.loads(
            (tmp_path / "raw/noaa/nbm/KXHIGHNY/2026071500Z/f030/metadata.json").read_text()
        )
        assert meta["availability_source"] == "observed"
        assert meta["product_created_at"] == "2026-07-15T07:15:00+00:00"

    def test_rebackfill_is_idempotent(self, tmp_path) -> None:
        from weadge.storage.parquet import DataLake

        lake = DataLake(tmp_path)
        for _ in range(2):
            backfill_nbm(
                datetime(2026, 7, 15).date(),
                datetime(2026, 7, 15).date(),
                lake,
                series="KXHIGHNY",
                station_id="KNYC",
                lat=40.78,
                lon=-73.97,
                archive=_FakeArchive(),
            )
        assert lake.read("forecasts").height == 3  # 07-14 f054 + 07-15 f030/f054

    def test_head_failure_falls_back_to_conservative_offset(self, tmp_path) -> None:
        from weadge.storage.parquet import DataLake

        lake = DataLake(tmp_path)
        backfill_nbm(
            datetime(2026, 7, 15).date(),
            datetime(2026, 7, 15).date(),
            lake,
            series="KXHIGHNY",
            station_id="KNYC",
            lat=40.78,
            lon=-73.97,
            archive=_FakeArchive(head_status=404),
        )
        fc = lake.read("forecasts")
        r = next(r for r in fc.iter_rows(named=True) if r["run_id"] == "2026071500Z")
        assert r["available_at"] == datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
        meta = json.loads(
            (tmp_path / "raw/noaa/nbm/KXHIGHNY/2026071500Z/f030/metadata.json").read_text()
        )
        assert meta["availability_source"] == "conservative_offset"
        assert meta["product_created_at"] is None

    def test_missing_fhour_counts_as_missing(self, tmp_path) -> None:
        from weadge.storage.parquet import DataLake

        class _Partial(_FakeArchive):
            def max_2t_records_raw(self, run_date, cycle, fhour, domain="co"):
                if fhour == 54:
                    raise httpx.HTTPStatusError("404", request=None, response=None)  # type: ignore[arg-type]
                return super().max_2t_records_raw(run_date, cycle, fhour, domain)

        lake = DataLake(tmp_path)
        summary = backfill_nbm(
            datetime(2026, 7, 15).date(),
            datetime(2026, 7, 15).date(),
            lake,
            series="KXHIGHNY",
            station_id="KNYC",
            lat=40.78,
            lon=-73.97,
            archive=_Partial(),
        )
        assert summary["f030_fetched"] == 1 and summary["f054_missing"] == 2
        assert lake.read("forecasts").height == 1  # only 07-15's f030 (leading f054 failed too)


class TestSmokeAudit:
    def _frames(self):
        import polars as pl

        from weadge.storage.schema import FORECAST_SCHEMA

        ev = pl.DataFrame(
            {
                "event_ticker": ["KXHIGHNY-26JUL15", "KXHIGHNY-26JUL16"],
                "series_ticker": ["KXHIGHNY", "KXHIGHNY"],
                "target_date": [
                    datetime(2026, 7, 15, tzinfo=UTC),
                    datetime(2026, 7, 16, tzinfo=UTC),
                ],
                "location_id": ["KXHIGHNY", "KXHIGHNY"],
                "ingested_at": [datetime(2026, 7, 20, tzinfo=UTC)] * 2,
            }
        )
        mk = pl.DataFrame(
            {
                "market_ticker": ["m1", "m2"],
                "event_ticker": ["KXHIGHNY-26JUL15", "KXHIGHNY-26JUL16"],
                "series_ticker": ["KXHIGHNY", "KXHIGHNY"],
                "close_at": [
                    datetime(2026, 7, 16, 4, 59, tzinfo=UTC),
                    datetime(2026, 7, 17, 4, 59, tzinfo=UTC),
                ],
            }
        )

        def fc_row(run_day, fhour, available, p10=94.0, p25=95.0):
            run_init = datetime(run_day.year, run_day.month, run_day.day, tzinfo=UTC)
            vs, ve = _window_for(run_init, fhour)
            return {
                "source": "nbm",
                "model": "nbm",
                "model_version": "nbm_v5.0.x",
                "run_id": f"{run_init:%Y%m%d%H}Z",
                "run_init_at": run_init,
                "available_at": available,
                "ingested_at": datetime(2026, 7, 20, tzinfo=UTC),
                "valid_start": vs,
                "valid_end": ve,
                "location_id": "KXHIGHNY",
                "station_id": "KNYC",
                "lat": 40.78,
                "lon": -73.97,
                "mean": 95.0,
                "std": 2.0,
                "p10": p10,
                "p25": p25,
                "p50": 96.0,
                "p75": 97.0,
                "p90": 98.0,
                "raw_payload_path": "/tmp/x",
            }

        rows = [
            # run 07-14: day-1 window = 07-14 (wrong day for E1), day-2 = 07-15
            fc_row(datetime(2026, 7, 14), 30, datetime(2026, 7, 14, 7, 15, tzinfo=UTC)),
            fc_row(datetime(2026, 7, 14), 54, datetime(2026, 7, 14, 7, 15, tzinfo=UTC)),
            # run 07-15: day-1 = 07-15 (ordering violation on purpose), day-2 = 07-16
            fc_row(
                datetime(2026, 7, 15),
                30,
                datetime(2026, 7, 15, 7, 15, tzinfo=UTC),
                p10=99.0,
                p25=90.0,
            ),
            fc_row(datetime(2026, 7, 15), 54, datetime(2026, 7, 15, 7, 15, tzinfo=UTC)),
            # as-of violation: knowable before the run init
            fc_row(datetime(2026, 7, 14), 30, datetime(2026, 7, 13, 23, 0, tzinfo=UTC)),
            # wrong target-window candidate: same DATE as E1 but a different
            # window than [D 12Z, D+1 06Z)
            fc_row(datetime(2026, 7, 14), 30, datetime(2026, 7, 14, 7, 15, tzinfo=UTC)),
        ]
        # override the last row's window to [D 06Z, D 18Z)
        rows[-1]["valid_start"] = datetime(2026, 7, 15, 6, tzinfo=UTC)
        rows[-1]["valid_end"] = datetime(2026, 7, 15, 18, tzinfo=UTC)
        fc = pl.DataFrame(rows, schema=FORECAST_SCHEMA)
        return ev, mk, fc

    def test_coverage_is_event_day_matched(self) -> None:
        ev, mk, fc = self._frames()
        a = nbm_smoke_audit(ev, mk, fc, series="KXHIGHNY", snapshot_hours=(24, 12))
        assert a["events"] == 2
        assert a["events_with_forecast"] == 2
        # T-24h (D 04:59Z) is served by the D-1 run's day-2 window only
        assert a["coverage"][24] == 2
        assert a["coverage"][12] == 2
        # the wrong-day row (run 07-14 day-1, valid 07-14) never counts
        assert a["wrong_target_window"] == 2  # E1 at both leads: naive 2 vs matched 1
        assert a["ordering_violations"] == 1
        assert a["asof_violations"] == 1
        assert a["pct_mean"] == 100.0 and a["pct_p"] == 100.0
        assert "17/24h" in a["window_note"]

    def test_missing_day_window_shows_uncovered(self) -> None:
        ev, mk, fc = self._frames()
        # drop everything that could serve E2's day (run 07-15 day-2)
        fc = fc.filter(
            ~(
                (pl.col("run_id") == "2026071500Z")
                & (pl.col("valid_start") == datetime(2026, 7, 16, 12, tzinfo=UTC))
            )
        )
        a = nbm_smoke_audit(ev, mk, fc, series="KXHIGHNY", snapshot_hours=(24, 12))
        assert a["coverage"][24] == 1  # E2 uncovered at T-24h
        assert a["coverage"][12] == 1
        assert a["events_with_forecast"] == 1
