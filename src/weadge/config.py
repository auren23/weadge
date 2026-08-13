"""Configuration loading (YAML + env).

Config files live in ./config at the project root. Env vars (KALSHI_API_KEY,
KALSHI_API_SECRET) are read by adapters directly — never stored in YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class SettlementConfig(BaseModel):
    # day_window "local_standard": the NWS Daily Climate Report covers
    # midnight-to-midnight local STANDARD time (01:00 EDT -> 00:59 EDT in
    # DST), i.e. [05:00 UTC, 05:00 UTC) all year.
    day_window: str = "local_standard"
    rounding: float = 0.5
    source: str = "NWS Daily Climate Report"  # the ONLY settlement truth


class CityConfig(BaseModel):
    series_ticker: str
    city: str
    location: str
    station_id: str
    lat: float
    lon: float
    timezone: str
    target: str
    unit: str = "fahrenheit"
    settlement: SettlementConfig = SettlementConfig()


class CitiesConfig(BaseModel):
    cities: list[CityConfig]

    def by_series(self, series: str) -> CityConfig:
        for c in self.cities:
            if c.series_ticker == series:
                return c
        raise KeyError(f"series {series} not configured")


class ModelConfig(BaseModel):
    id: str
    description: str = ""
    availability_offset_min: int = 0
    grid: str = ""


class ModelsConfig(BaseModel):
    models: list[ModelConfig]

    def by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"model {model_id} not configured")


class AlphaGates(BaseModel):
    G0_data: dict = {}
    G1_forecast: dict = {}
    G2_incremental: dict = {}
    G3_economic: dict = {}
    G4_robustness: dict = {}


class BacktestConfig(BaseModel):
    taker: dict = {}
    bootstrap: dict = {}


class ResolverCityConfig(BaseModel):
    """PM daily-high station. `cities` are serve candidates; extras are race/audit only."""

    slug: str
    city: str
    station_icao: str
    timezone: str
    unit: str = "celsius"
    scan_hours: list[int] = [12, 21]
    settlement_source: str = "wunderground"
    settlement_url: str = ""
    wu_location: str = ""
    source_grade: str = "A"  # AWC vs this city's settlement; starting hypothesis
    iem_network: str = ""
    iem_station: str | None = None  # IEM current.py id; None → station_icao


class ResolverEdgeConfig(BaseModel):
    min_net_edge: float = 0.02
    exec_buffer: float = 0.01
    locked_buffer_c: float = 0.5
    stale_after_min: int = 30


class ResolverTelegramConfig(BaseModel):
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"


class ResolverConfig(BaseModel):
    mode: str = "shadow"
    cities: list[ResolverCityConfig]
    observation_extra: list[ResolverCityConfig] = Field(default_factory=list)
    edge: ResolverEdgeConfig = ResolverEdgeConfig()
    telegram: ResolverTelegramConfig = ResolverTelegramConfig()

    def by_slug(self, slug: str) -> ResolverCityConfig:
        for c in self.cities:
            if c.slug == slug:
                return c
        raise KeyError(f"resolver city {slug} not configured")

    def observation_stations(self) -> list[ResolverCityConfig]:
        """Serve cities plus race/audit-only extras (NYC/Chicago)."""
        return [*self.cities, *self.observation_extra]


class ResearchConfig(BaseModel):
    dataset: dict = {}
    research: dict = {}
    alpha_gates: AlphaGates = AlphaGates()
    backtest: BacktestConfig = BacktestConfig()
    fees: dict = {}


@lru_cache(maxsize=1)
def load_cities(path: Path | None = None) -> CitiesConfig:
    with open(path or CONFIG_DIR / "cities.yaml") as fh:
        return CitiesConfig.model_validate(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def load_models(path: Path | None = None) -> ModelsConfig:
    with open(path or CONFIG_DIR / "models.yaml") as fh:
        return ModelsConfig.model_validate(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def load_research(path: Path | None = None) -> ResearchConfig:
    with open(path or CONFIG_DIR / "research.yaml") as fh:
        return ResearchConfig.model_validate(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def load_resolver(path: Path | None = None) -> ResolverConfig:
    """Resolver config (docs/resolver.md). Independent of research config."""
    with open(path or CONFIG_DIR / "resolver.yaml") as fh:
        return ResolverConfig.model_validate(yaml.safe_load(fh))


def data_root() -> Path:
    return PROJECT_ROOT / "data"
