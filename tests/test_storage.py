"""DataLake parquet round-trips: hive-style partitions must be readable."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from weadge.storage.parquet import DataLake


def _events_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_ticker": ["E1", "E2", "E3"],
            "series_ticker": ["KXHIGHNY"] * 3,
            "target_date": [datetime(2026, 7, 1, tzinfo=UTC)] * 3,
            "location_id": ["KXHIGHNY"] * 3,
            "ingested_at": [datetime(2026, 7, 1, tzinfo=UTC)] * 3,
        }
    )


class TestDataLake:
    def test_partitioned_roundtrip(self, tmp_path) -> None:
        lake = DataLake(tmp_path)
        lake.write_parquet("events", _events_df(), layer="bronze", partition_by="series_ticker")
        out = lake.read("events")
        assert out.height == 3
        assert set(out["event_ticker"].to_list()) == {"E1", "E2", "E3"}
        assert out["series_ticker"].to_list() == ["KXHIGHNY"] * 3  # partition col kept
        assert lake.exists("events", "bronze")

    def test_append_does_not_overwrite_existing_partitions(self, tmp_path) -> None:
        lake = DataLake(tmp_path)
        lake.write_parquet("events", _events_df().head(2), layer="bronze",
                           partition_by="series_ticker")
        lake.write_parquet("events", _events_df().tail(1), layer="bronze",
                           partition_by="series_ticker")
        assert lake.read("events").height == 3  # 2 + 1, not overwritten

    def test_multiple_partition_values(self, tmp_path) -> None:
        df = pl.DataFrame(
            {
                "event_ticker": ["E1", "E2"],
                "series_ticker": ["KXHIGHNY", "KXHIGHCHI"],
                "target_date": [datetime(2026, 7, 1, tzinfo=UTC)] * 2,
                "location_id": ["KXHIGHNY", "KXHIGHCHI"],
                "ingested_at": [datetime(2026, 7, 1, tzinfo=UTC)] * 2,
            }
        )
        lake = DataLake(tmp_path)
        lake.write_parquet("events", df, layer="bronze", partition_by="series_ticker")
        out = lake.read("events")
        assert set(out["series_ticker"].to_list()) == {"KXHIGHNY", "KXHIGHCHI"}

    def test_unpartitioned_roundtrip(self, tmp_path) -> None:
        lake = DataLake(tmp_path)
        lake.write_parquet("events", _events_df(), layer="bronze")
        assert lake.read("events").height == 3

    def test_empty_table_reads_as_empty_frame(self, tmp_path) -> None:
        lake = DataLake(tmp_path)
        out = lake.read("events")
        assert out.is_empty()
        assert out.columns == _events_df().columns

    def test_unknown_partition_column_raises(self, tmp_path) -> None:
        lake = DataLake(tmp_path)
        with pytest.raises(ValueError, match="partition column"):
            lake.write_parquet("events", _events_df(), layer="bronze", partition_by="nope")
