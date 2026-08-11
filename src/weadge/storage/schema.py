"""Canonical table schemas for the bronze/silver layers.

These are the contracts between adapters (writers) and research (readers).
Columns intentionally mirror the Kalshi API so nothing is lost in translation.
"""

from __future__ import annotations

import polars as pl

# Polars accepts both DataType instances (pl.Datetime("us", ...)) and type
# classes (pl.Utf8) in schema dicts, so the value type is a union.
SchemaType = pl.DataType | type[pl.DataType]
Schema = dict[str, SchemaType]

# series: one row per Kalshi series (metadata, from /series/{ticker})
SERIES_SCHEMA: Schema = {
    "series_ticker": pl.Utf8,
    "title": pl.Utf8,
    "settlement_source": pl.Utf8,
    "fee_type": pl.Utf8,
    "fee_multiplier": pl.Float64,
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# events: one row per event (a target date for a series)
EVENT_SCHEMA: Schema = {
    "event_ticker": pl.Utf8,
    "series_ticker": pl.Utf8,
    "target_date": pl.Datetime("us", time_zone="UTC"),
    "location_id": pl.Utf8,
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# markets: one row per binary contract (bucket)
MARKET_SCHEMA: Schema = {
    "market_ticker": pl.Utf8,
    "event_ticker": pl.Utf8,
    "series_ticker": pl.Utf8,
    "floor_strike": pl.Float64,
    "cap_strike": pl.Float64,
    "open_at": pl.Datetime("us", time_zone="UTC"),
    "close_at": pl.Datetime("us", time_zone="UTC"),
    "settled_at": pl.Datetime("us", time_zone="UTC"),
    "result": pl.Utf8,
    "settlement_value": pl.Utf8,
    "rules_primary": pl.Utf8,
    "rules_secondary": pl.Utf8,
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# quote_1m: 1-minute YES bid/ask OHLC candlesticks
QUOTE_1M_SCHEMA: Schema = {
    "market_ticker": pl.Utf8,
    "ts": pl.Datetime("us", time_zone="UTC"),
    "yes_bid_open": pl.Float64,
    "yes_bid_high": pl.Float64,
    "yes_bid_low": pl.Float64,
    "yes_bid_close": pl.Float64,
    "yes_ask_open": pl.Float64,
    "yes_ask_high": pl.Float64,
    "yes_ask_low": pl.Float64,
    "yes_ask_close": pl.Float64,
    "volume": pl.Int64,
    "open_interest": pl.Int64,
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# forecast_percentiles: Kalshi event forecast percentile history
FORECAST_PERCENTILE_SCHEMA: Schema = {
    "event_ticker": pl.Utf8,
    "percentile": pl.Float64,          # e.g. 10.0 .. 90.0
    "numerical_forecast": pl.Float64,  # value at that percentile
    "end_period_ts": pl.Datetime("us", time_zone="UTC"),
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# fee_changes: series fee history (effective_at ascending)
FEE_CHANGES_SCHEMA: Schema = {
    "series_ticker": pl.Utf8,
    "effective_at": pl.Datetime("us", time_zone="UTC"),
    "fee_type": pl.Utf8,
    "fee_multiplier": pl.Float64,
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# observations: official settlement observations
OBSERVATION_SCHEMA: Schema = {
    "station_id": pl.Utf8,
    "observed_at": pl.Datetime("us", time_zone="UTC"),
    "value": pl.Float64,
    "unit": pl.Utf8,
    "source": pl.Utf8,
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
}

# forecasts: standardized weather model snapshots (silver)
FORECAST_SCHEMA: Schema = {
    "source": pl.Utf8,
    "model": pl.Utf8,
    "model_version": pl.Utf8,
    "run_id": pl.Utf8,
    "run_init_at": pl.Datetime("us", time_zone="UTC"),
    "available_at": pl.Datetime("us", time_zone="UTC"),
    "ingested_at": pl.Datetime("us", time_zone="UTC"),
    "valid_start": pl.Datetime("us", time_zone="UTC"),
    "valid_end": pl.Datetime("us", time_zone="UTC"),
    "location_id": pl.Utf8,
    "station_id": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "mean": pl.Float64,
    "std": pl.Float64,
    "p10": pl.Float64,
    "p25": pl.Float64,
    "p50": pl.Float64,
    "p75": pl.Float64,
    "p90": pl.Float64,
    "raw_payload_path": pl.Utf8,
}

# gold/alpha_dataset — one row per (event, market, snapshot)
ALPHA_DATASET_SCHEMA: Schema = {
    "event_date": pl.Datetime("us", time_zone="UTC"),
    "city": pl.Utf8,
    "decision_at": pl.Datetime("us", time_zone="UTC"),
    "lead_hours": pl.Float64,
    "market_ticker": pl.Utf8,
    "bucket_low": pl.Float64,
    "bucket_high": pl.Float64,
    "market_bid": pl.Float64,
    "market_ask": pl.Float64,
    "market_mid": pl.Float64,
    "p_market": pl.Float64,
    "p_kalshi_forecast": pl.Float64,
    "p_nbm": pl.Float64,
    "p_gefs": pl.Float64,
    "p_emos": pl.Float64,
    "p_stack": pl.Float64,
    "result": pl.Int64,                # 1 = bucket hit (YES settles $1), 0 = miss
    "fee": pl.Float64,
    "spread": pl.Float64,
}

SCHEMAS: dict[str, Schema] = {
    "series": SERIES_SCHEMA,
    "events": EVENT_SCHEMA,
    "markets": MARKET_SCHEMA,
    "quote_1m": QUOTE_1M_SCHEMA,
    "forecast_percentiles": FORECAST_PERCENTILE_SCHEMA,
    "fee_changes": FEE_CHANGES_SCHEMA,
    "observations": OBSERVATION_SCHEMA,
    "forecasts": FORECAST_SCHEMA,
    "alpha_dataset": ALPHA_DATASET_SCHEMA,
}


def empty_frame(table: str) -> pl.DataFrame:
    """Return an empty frame with the canonical schema for `table`."""
    if table not in SCHEMAS:
        raise KeyError(f"unknown canonical table: {table}")
    return pl.DataFrame(schema=SCHEMAS[table])


def cast_to_schema(df: pl.DataFrame, table: str) -> pl.DataFrame:
    """Best-effort cast of an adapter frame into the canonical schema."""
    schema = SCHEMAS[table]
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"frame missing canonical columns {missing} for table {table}")
    extra = [c for c in df.columns if c not in schema]
    df = df.drop(extra) if extra else df
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
