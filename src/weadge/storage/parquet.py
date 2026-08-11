"""Parquet data lake: raw -> bronze -> silver -> gold.

    data/raw/    provider/.../*.json(.zst)   — untouched API payloads
    data/bronze/ <domain>/<table>/*.parquet  — standardized frames
    data/silver/ <table>/*.parquet           — aligned, time-consistent
    data/gold/   alpha_dataset.parquet       — research-ready rows

All writes are append-with-partition: readers never rewrite old partitions,
so the lake is re-playable and safe to grow incrementally.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import polars as pl

from weadge.storage import schema as canon


class DataLake:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for layer in ("raw", "bronze", "silver", "gold"):
            (self.root / layer).mkdir(parents=True, exist_ok=True)

    # ---- raw JSONL capture -------------------------------------------------
    def save_raw_jsonl(self, provider: str, date_key: str, rows: list[dict]) -> Path:
        """Append API payload rows as JSONL.zst under data/raw/<provider>/<date_key>/."""
        out_dir = self.root / "raw" / provider / date_key
        out_dir.mkdir(parents=True, exist_ok=True)
        # deterministic filename: hash of first row + timestamp counter
        stamp = zlib.crc32(json.dumps(rows[:1], sort_keys=True).encode()) & 0xFFFFFFFF
        path = out_dir / f"{stamp:08x}.jsonl.zst"
        import zstandard as zstd

        compressor = zstd.ZstdCompressor(level=3)
        with path.open("ab") as fh, compressor.stream_writer(fh) as writer:
            for row in rows:
                writer.write((json.dumps(row, default=str) + "\n").encode())
        return path

    # ---- bronze / silver / gold --------------------------------------------
    def write_parquet(
        self,
        table: str,
        df: pl.DataFrame,
        layer: str = "bronze",
        partition_by: str | None = None,
    ) -> Path:
        """Write a canonical frame, partitioned by a column (dropped into dirs)."""
        if layer not in ("bronze", "silver", "gold"):
            raise ValueError(f"unknown layer {layer}")
        df = canon.cast_to_schema(df, table)
        out_dir = self.root / layer / table
        out_dir.mkdir(parents=True, exist_ok=True)

        if partition_by is not None:
            if partition_by not in df.columns:
                raise ValueError(f"partition column {partition_by} not in frame")
            df = df.sort(partition_by)
            out = out_dir / f"{partition_by}=*"
            df.write_parquet(out, compression="zstd")
            return out_dir
        path = out_dir / f"part-{len(list(out_dir.glob('*.parquet'))):05d}.parquet"
        df.write_parquet(path, compression="zstd")
        return path

    def read(self, table: str, layer: str = "bronze") -> pl.DataFrame:
        """Read all partitions of a table as one frame."""
        pattern = self.root / layer / table / "*.parquet"
        files = sorted(pattern.glob("*")) if pattern.parent.exists() else []
        if not files:
            # also handle partitioned dirs (a/b=*/part-*.parquet)
            files = sorted((pattern.parent).glob("*/*.parquet"))
        if not files:
            return canon.empty_frame(table)
        return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")

    def exists(self, table: str, layer: str = "bronze") -> bool:
        root = self.root / layer / table
        return root.exists() and any(root.rglob("*.parquet"))

    def gold_path(self, name: str = "alpha_dataset.parquet") -> Path:
        return self.root / "gold" / name
