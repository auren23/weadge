"""NWS CLINYC DCR parser — content-based classification, fail-closed maximum.

Fixtures are REAL IEM-archived bulletins (July 2026):
    CLINYC_final_202607010620.txt   full daily for June 30 (issued Jul 1 06:20Z)
    CLINYC_prelim_202607012036.txt  preliminary for July 1 (issued Jul 1 20:36Z)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from weadge.adapters.noaa.dcr import (
    PIL_CLINYC,
    DCRRecord,
    backfill_dcr,
    parse_clinyc,
    select_final,
)

FIXTURES = Path(__file__).parent / "fixtures" / "dcr"
FINAL_PID = "202607010620-KOKX-CDUS41-CLINYC"
PRELIM_PID = "202607012036-KOKX-CDUS41-CLINYC"


def _final_text(maximum: str = "87") -> str:
    """Real final-daily bulletin with a replaced MAXIMUM value."""
    text = (FIXTURES / "CLINYC_final_202607010620.txt").read_text()
    return text.replace("  MAXIMUM         87", f"  MAXIMUM         {maximum}")


def _heading(date_str: str) -> str:
    return f"...THE CENTRAL PARK NY CLIMATE SUMMARY FOR {date_str}..."


class TestParseClinyc:
    def test_preliminary_rejected(self) -> None:
        r = parse_clinyc((FIXTURES / "CLINYC_prelim_202607012036.txt").read_text(), PRELIM_PID)
        assert r.is_preliminary is True
        assert r.is_complete is False
        assert r.report_date == date(2026, 7, 1)

    def test_full_daily_accepted(self) -> None:
        r = parse_clinyc((FIXTURES / "CLINYC_final_202607010620.txt").read_text(), FINAL_PID)
        assert r.is_preliminary is False
        assert r.is_complete is True
        assert r.maximum_f == 87

    def test_report_date_from_heading(self) -> None:
        r = parse_clinyc(_final_text(), FINAL_PID)
        assert r.report_date == date(2026, 6, 30)

    def test_report_date_differs_from_issue_date(self) -> None:
        """Regression: issued Jul 1 06:20Z, reports JUNE 30 — report_date must
        come from the heading, never from issued_at."""
        r = parse_clinyc((FIXTURES / "CLINYC_final_202607010620.txt").read_text(), FINAL_PID)
        assert r.issued_at == datetime(2026, 7, 1, 6, 20, tzinfo=UTC)
        assert r.report_date == date(2026, 6, 30)
        assert r.report_date != r.issued_at.date()

    def test_maximum_85(self) -> None:
        assert parse_clinyc(_final_text("85"), FINAL_PID).maximum_f == 85

    def test_maximum_negative(self) -> None:
        assert parse_clinyc(_final_text("-2"), FINAL_PID).maximum_f == -2

    def test_maximum_with_record_flag(self) -> None:
        """MAXIMUM 58R — the R is a record marker, the value is 58."""
        assert parse_clinyc(_final_text("58R"), FINAL_PID).maximum_f == 58

    def test_maximum_missing_fails_closed(self) -> None:
        for bad in ("M", "MM", "missing", "MMM"):
            assert parse_clinyc(_final_text(bad), FINAL_PID).maximum_f is None, bad

    def test_maximum_suspect_flag_fails_closed(self) -> None:
        """87S (suspect) is not R — never silently parsed as 87."""
        assert parse_clinyc(_final_text("87S"), FINAL_PID).maximum_f is None

    def test_identity_validation(self) -> None:
        """A foreign product (e.g. another station's CLI) must be rejected."""
        foreign = "THE CENTRAL PARK NY CLIMATE SUMMARY".replace("CENTRAL PARK NY", "BOSTON MA")
        text = _final_text().replace(
            "...THE CENTRAL PARK NY CLIMATE SUMMARY FOR JUNE 30 2026...", foreign
        )
        with pytest.raises(ValueError, match="not a CLINYC summary"):
            parse_clinyc(text, FINAL_PID)

    def test_completely_wrong_product_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a CLINYC summary"):
            parse_clinyc("AVIATION FORECAST... KVNY 1200Z", "202607011200-KOKX-FAUS41-FAXNY")

    def test_yesterday_outside_temperature_section_does_not_mark_complete(self) -> None:
        """Completeness is decided inside the TEMPERATURE section only: even
        if a preliminary carries a YESTERDAY row under PRECIPITATION, it must
        stay is_complete=False."""
        text = (FIXTURES / "CLINYC_prelim_202607012036.txt").read_text()
        text = text.replace("  TODAY            0.00", "  YESTERDAY        T\n  TODAY            0.00")
        r = parse_clinyc(text, PRELIM_PID)
        assert r.is_preliminary is True
        assert r.is_complete is False  # YESTERDAY sits under PRECIPITATION, not TEMPERATURE


class TestSelectFinal:
    def _rec(self, pid: str, report: str, maximum: int | None = 85) -> DCRRecord:
        return DCRRecord(
            product_id=pid, pil=PIL_CLINYC,
            issued_at=datetime(2026, int(pid[4:6]), int(pid[6:8]), int(pid[8:10]), int(pid[10:12]), tzinfo=UTC),
            report_date=date(2026, 7, 8),
            is_preliminary=False, is_complete=True, maximum_f=maximum,
        )

    def test_latest_issuance_wins(self) -> None:
        """A 03:40 correction supersedes the 02:15 original daily."""
        original = self._rec("202607090215-KOKX-CDUS41-CLINYC", "JULY 8 2026", 88)
        corrected = self._rec("202607090340-KOKX-CDUS41-CLINYC", "JULY 8 2026", 87)
        final = select_final([corrected, original])
        assert final[date(2026, 7, 8)] is corrected
        assert final[date(2026, 7, 8)].maximum_f == 87

    def test_preliminary_never_selected(self) -> None:
        prelim = DCRRecord(
            product_id="202607082035-KOKX-CDUS41-CLINYC", pil=PIL_CLINYC,
            issued_at=datetime(2026, 7, 8, 20, 35, tzinfo=UTC),
            report_date=date(2026, 7, 8), is_preliminary=True, is_complete=False,
            maximum_f=92,
        )
        assert select_final([prelim]) == {}

    def test_missing_maximum_kept_as_missing(self) -> None:
        r = self._rec("202607090215-KOKX-CDUS41-CLINYC", "JULY 8 2026", None)
        final = select_final([r])
        assert final[date(2026, 7, 8)].maximum_f is None  # fail closed, not dropped silently


class TestBackfillSummary:
    def test_summary_counts(self, tmp_path) -> None:
        """End-to-end against a fake IEM: 2 products (1 prelim, 1 final)
        -> final selection yields 1 report day with a parsed maximum."""
        import httpx
        import respx

        from weadge.adapters.noaa.dcr import IEMClient
        from weadge.storage.parquet import DataLake

        final_text = (FIXTURES / "CLINYC_final_202607010620.txt").read_text()
        prelim_text = (FIXTURES / "CLINYC_prelim_202607012036.txt").read_text()
        list_html = (
            f'<a href="https://mesonet.agron.iastate.edu/p.php?pid={FINAL_PID}">f</a>'
            f'<a href="https://mesonet.agron.iastate.edu/p.php?pid={PRELIM_PID}">p</a>'
        )

        with respx.mock:
            respx.get(url__startswith="https://mesonet.agron.iastate.edu/wx/afos/list.phtml").mock(
                return_value=httpx.Response(200, text=list_html)
            )
            respx.get(
                f"https://mesonet.agron.iastate.edu/api/1/nwstext/{FINAL_PID}"
            ).mock(return_value=httpx.Response(200, text="921 \n" + final_text))
            respx.get(
                f"https://mesonet.agron.iastate.edu/api/1/nwstext/{PRELIM_PID}"
            ).mock(return_value=httpx.Response(200, text="111 \n" + prelim_text))

            lake = DataLake(tmp_path)
            summary = backfill_dcr(
                date(2026, 6, 30), date(2026, 6, 30), lake, client=IEMClient()
            )

        assert summary["products_fetched"] == 2
        assert summary["preliminary_rejected"] == 1
        assert summary["complete_daily"] == 1
        assert summary["unique_report_days"] == 1
        assert summary["parsed_maximum"] == 1
        assert summary["missing_maximum"] == 0
        assert summary["foreign_rejected"] == 0

        obs = lake.read("observations")
        assert obs.height == 1
        assert obs["source"][0] == "NWS Daily Climate Report"
        assert obs["value"][0] == 87.0
        # raw bulletins captured for the audit trail
        raw = list((tmp_path / "raw" / "noaa" / "dcr" / PIL_CLINYC).glob("*.txt"))
        assert len(raw) == 2
