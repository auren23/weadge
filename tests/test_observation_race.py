"""Pure-logic checks for tools/observation_race.py (no network)."""

from __future__ import annotations

import importlib.util
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from weadge.config import ResolverCityConfig, ResolverConfig, load_resolver

spec = importlib.util.spec_from_file_location("race", "tools/observation_race.py")
race = importlib.util.module_from_spec(spec)
spec.loader.exec_module(race)

PARIS = ZoneInfo("Europe/Paris")


def test_c_to_f_whole_degree():
    assert race.c_to_f(0.0) == 32.0
    assert race.c_to_f(37.0) == 99.0
    assert race.c_to_f(25.6) == 78.0


def test_wu_api_units_follow_settlement():
    assert race.wu_api_units("celsius") == "m"
    assert race.wu_api_units("fahrenheit") == "e"


def test_iem_current_station_uses_mapping():
    nyc = ResolverCityConfig(
        slug="nyc",
        city="NYC",
        station_icao="KLGA",
        timezone="America/New_York",
        iem_station="LGA",
        iem_network="NY_ASOS",
    )
    paris = ResolverCityConfig(
        slug="paris",
        city="Paris",
        station_icao="LFPB",
        timezone="Europe/Paris",
    )
    assert race.iem_current_station(nyc) == "LGA"
    assert race.iem_current_station(paris) == "LFPB"


def test_wu_daily_max_ignores_nulls():
    obs = [{"temp": 24}, {"temp": None}, {"temp": 35}, {"temp": 18}]
    assert race.wu_daily_max(obs) == 35.0
    assert race.wu_daily_max([]) is None
    assert race.wu_daily_max([{"temp": None}]) is None


def test_audit_bucket():
    assert race.audit_bucket(35.0, 35.0) == "exact"
    assert race.audit_bucket(35.0, 34.0) == "pm1"
    assert race.audit_bucket(35.0, 33.0) == "larger"
    assert race.audit_bucket(None, 35.0) == "metar_missing"
    assert race.audit_bucket(35.0, None) == "wu_missing"


def test_fresh_enough_drops_cache_history():
    now = datetime(2026, 8, 13, 13, 31, tzinfo=UTC)
    assert race.fresh_enough(now - timedelta(minutes=5), now)
    assert not race.fresh_enough(now - timedelta(minutes=21), now)


def test_first_seen_dedupes_identity():
    seen = race.FirstSeen()
    t = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
    assert seen.offer("LFPB", t, "aviationweather") is True
    assert seen.offer("LFPB", t, "aviationweather") is False
    assert seen.offer("LFPB", t, "iem") is True
    assert seen.offer("LFPB", t + timedelta(minutes=30), "aviationweather") is True


def test_first_seen_hydrates_from_row():
    seen = race.FirstSeen()
    t = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
    seen.load_row(
        race.race_row(
            station="LFPB",
            observation_at=t,
            source="aviationweather",
            source_grade="A",
            first_seen_at=t + timedelta(seconds=40),
            decoded_temp=37.0,
            temp_unit="C",
            raw_temp="37",
        )
    )
    seen.load_row({"event": "error", "source": "iem", "station": "LFPB"})
    assert seen.offer("LFPB", t, "aviationweather") is False


def test_decode_awc_uses_receipt_and_obs_time():
    now = datetime(2026, 8, 13, 13, 31, 8, tzinfo=UTC)
    row = race.decode_awc(
        {
            "icaoId": "LFPB",
            "obsTime": int(datetime(2026, 8, 13, 13, 30, tzinfo=UTC).timestamp()),
            "receiptTime": "2026-08-13T13:30:41.203Z",
            "temp": 37,
            "rawOb": "METAR LFPB 131330Z AUTO 37/08",
        },
        "A",
        now,
    )
    assert row is not None
    assert row["report_id"] == "LFPB-2026-08-13T13:30:00Z"
    assert row["source"] == "aviationweather"
    assert row["source_grade"] == "A"
    assert row["decoded_temp"] == 37.0
    assert row["temp_unit"] == "C"
    assert row["provider_receipt_at"] == "2026-08-13T13:30:41.203000+00:00"
    assert row["first_seen_at"] == now.isoformat()


def test_decode_iem_keeps_fahrenheit_and_icao():
    now = datetime(2026, 8, 13, 13, 31, tzinfo=UTC)
    cfg = ResolverCityConfig(
        slug="nyc",
        city="NYC",
        station_icao="KLGA",
        timezone="America/New_York",
        iem_station="LGA",
    )
    row = race.decode_iem_current(
        {"last_ob": {"utc_valid": "2026-08-13T13:51:00Z", "airtemp[F]": 78.0}},
        cfg,
        now,
    )
    assert row is not None
    assert row["station"] == "KLGA"
    assert row["source"] == "iem"
    assert row["source_grade"] == "B"
    assert row["decoded_temp"] == 78.0
    assert row["temp_unit"] == "F"
    assert row["provider_receipt_at"] is None
    assert race.decode_iem_current({}, cfg, now) is None


def test_iem_asos_csv_converts_tmpf_to_c():
    csv = "station,valid,tmpf\nKLGA,2026-08-12 18:51,78.0\nKLGA,2026-08-12 19:51,M\n"
    rows = race.iem_asos_to_awc_rows(csv)
    assert len(rows) == 1
    assert rows[0]["temp"] == pytest.approx(25.555, rel=1e-3)


def test_metar_max_c_for_local_day():
    def row(hour_utc: int, temp: float) -> dict:
        ts = datetime(2026, 8, 12, hour_utc, tzinfo=UTC)
        return {"obsTime": int(ts.timestamp()), "temp": temp}

    rows = [row(8, 20.0), row(14, 35.0), row(22, 18.0)]  # 22Z = 13 Aug 00:00 Paris, next day
    mx = race.metar_max_c_for_day(rows, "LFPB", PARIS, date(2026, 8, 12))
    assert mx == 35.0


def test_group_first_seen_and_audit_dates():
    t = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
    rows = [
        race.race_row(
            station="LFPB",
            observation_at=t,
            source="iem",
            source_grade="B",
            first_seen_at=t + timedelta(seconds=20),
            decoded_temp=98.6,
            temp_unit="F",
            raw_temp="98.6",
        ),
        race.race_row(
            station="LFPB",
            observation_at=t,
            source="aviationweather",
            source_grade="A",
            first_seen_at=t + timedelta(seconds=50),
            decoded_temp=37.0,
            temp_unit="C",
            raw_temp="37",
        ),
    ]
    grouped = race.group_first_seen(rows)
    assert set(grouped[("LFPB", t.isoformat())]) == {"iem", "aviationweather"}
    assert race.audit_dates(2, today=date(2026, 8, 13)) == [date(2026, 8, 12), date(2026, 8, 11)]


def test_nyc_is_observation_not_serve():
    cfg = load_resolver()
    assert cfg.by_slug("london").station_icao == "EGLC"
    with pytest.raises(KeyError):
        cfg.by_slug("nyc")
    icaos = {c.station_icao for c in cfg.observation_stations()}
    assert icaos == {"LFPB", "EGLC", "RJTT", "RKSI", "KLGA", "KORD"}


def test_observation_extra_does_not_leak_into_empty_config():
    cfg = ResolverConfig(
        cities=[
            ResolverCityConfig(
                slug="paris", city="Paris", station_icao="LFPB", timezone="Europe/Paris"
            ),
        ]
    )
    assert cfg.observation_extra == []
    assert len(cfg.observation_stations()) == 1
