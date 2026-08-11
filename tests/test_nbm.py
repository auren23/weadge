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
from datetime import UTC, datetime
from pathlib import Path

import pytest

from weadge.adapters.noaa.nbm import (
    NbmArchive,
    decode_blob,
    max_2t_records,
    model_version_for,
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
        fields = decode_blob((FIXTURES / "hi_max2t_p50_2026071500_f030.grib2").read_bytes(), RUN_INIT)
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
