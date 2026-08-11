"""DuckDB research database.

DuckDB is a query layer over the parquet lake, not a source of truth.
Views are registered per-session; data lives in parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


class ResearchDB:
    def __init__(self, lake_root: str | Path, db_path: str | Path | None = None) -> None:
        self.lake_root = Path(lake_root)
        if db_path is None:
            db_path = self.lake_root / "research.duckdb"
        self._conn = duckdb.connect(str(db_path), read_only=False)
        self._conn.execute("SET enable_progress_bar=false")

    def close(self) -> None:
        self._conn.close()

    def register_table(self, table: str, df: pl.DataFrame) -> None:
        """Register an in-memory polars frame as a DuckDB view."""
        self._conn.register(table, df)

    def register_parquet(self, table: str, layer: str = "bronze") -> None:
        """Register a lake table (all partitions) as a DuckDB view."""
        root = self.lake_root / layer / table
        files = sorted(root.rglob("*.parquet")) if root.exists() else []
        if not files:
            self._conn.register(table, pl.DataFrame())
            return
        glob = str(root / "**" / "*.parquet")
        self._conn.execute(f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM read_parquet(?)', [glob])

    def sql(self, query: str, params: list | tuple | None = None) -> pl.DataFrame:
        """Run SQL, return a polars frame."""
        result = self._conn.execute(query, params or []).fetch_arrow_table()
        return pl.DataFrame(result)  # cast from DataFrame|Series union

    def scalar(self, query: str, params: list | tuple | None = None):
        row = self._conn.execute(query, params or []).fetchone()
        return row[0] if row is not None else None
