"""Recorder pure logic: appender round-trip/flush, trade dedupe, payloads."""

from __future__ import annotations

import io
import json
from pathlib import Path

import zstandard as zstd

from weadge.live.recorder import JsonlZstAppender
from weadge.live.settlement_recorder import cfb_auth_header, cfb_subscribe, new_trades


def _read_jsonl_zst(path: Path) -> list[dict]:
    with path.open("rb") as fh:
        stream = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=True)
        return [json.loads(line) for line in io.TextIOWrapper(stream) if line.strip()]


class TestAppender:
    def test_round_trip_and_hour_rotation(self, tmp_path: Path) -> None:
        app = JsonlZstAppender(tmp_path, "kalshi")
        app.append("2026-08-12T10:59:59+00:00", {"raw": {"a": 1}})
        app.append("2026-08-12T11:00:01+00:00", {"raw": {"a": 2}})
        app.close()
        first = tmp_path / "2026-08-12" / "kalshi" / "10.jsonl.zst"
        second = tmp_path / "2026-08-12" / "kalshi" / "11.jsonl.zst"
        assert _read_jsonl_zst(first) == [{"ts": "2026-08-12T10:59:59+00:00", "raw": {"a": 1}}]
        assert _read_jsonl_zst(second)[0]["raw"] == {"a": 2}

    def test_flush_makes_rows_readable_before_close(self, tmp_path: Path) -> None:
        """Crash-safety contract: after flush() the row must be recoverable
        from disk even though the writer is still open."""
        app = JsonlZstAppender(tmp_path, "kraken")
        app.append("2026-08-12T10:00:00+00:00", {"raw": "tick"})
        app.flush()
        path = tmp_path / "2026-08-12" / "kraken" / "10.jsonl.zst"
        rows = _read_jsonl_zst(path)
        app.close()
        assert rows == [{"ts": "2026-08-12T10:00:00+00:00", "raw": "tick"}]


class TestTradeDedupe:
    def test_new_trades_dedupes_and_restores_chronology(self) -> None:
        seen: set[str] = set()
        page = [{"trade_id": "b"}, {"trade_id": "a"}]  # API returns newest first
        assert new_trades(seen, page) == [{"trade_id": "a"}, {"trade_id": "b"}]
        # second poll overlaps: only the genuinely new print comes back
        page = [{"trade_id": "c"}, {"trade_id": "b"}, {"trade_id": "a"}]
        assert new_trades(seen, page) == [{"trade_id": "c"}]
        assert seen == {"a", "b", "c"}


class TestCfbPayloads:
    def test_subscribe_matches_documented_schema(self) -> None:
        assert cfb_subscribe() == {
            "type": "subscribe",
            "stream": "value",
            "id": "BRTI",
            "maxResolution": "PER_SECOND",
        }

    def test_basic_auth_header(self) -> None:
        assert cfb_auth_header("abc", "123") == {"Authorization": "Basic YWJjOjEyMw=="}
