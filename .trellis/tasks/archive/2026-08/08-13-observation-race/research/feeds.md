# Feed and settlement research (2026-08-13)

## AviationWeather

- METAR JSON: `https://aviationweather.gov/api/data/metar?ids=ICAO&format=json`
- Fields used: `icaoId`, `obsTime`, `reportTime`, `receiptTime`, `temp` (°C), `rawOb`
- `ids` accepts comma-separated ICAOs (verified LFPB,EGLC,RJTT,RKSI,KLGA,KORD)
- Cache updates ~1/min; documented cap ~100 req/min
- US T-group example: KLGA `T02560200` → `temp: 25.6`; international LFPB often integer C, no T-group

## IEM

- News 1469: official ASOS daily max/min is 2-min average, whole °F. No known near-real-time public source with whole-°F fidelity and the same averaging.
- `current.py?station=&network=`: international ICAO (`LFPB`+`FR__ASOS`); US strips K (`LGA`+`NY_ASOS`, `ORD`+`IL_ASOS`). `KLGA`/`KORD` return `{}`.
- Temps in current.py are `airtemp[F]` even for LFPB.
- Historical METAR for audit backfill: `asos.py` `report_type=3,4` (routine+SPECI), same as `tools/kalshi_lock_probe.py`. Not HFMETAR.

## Wunderground (PM settlement)

- Rules: Daily Observations table, not Day High & Low summary.
- API used by the history page: `https://api.weather.com/v1/location/{id}/observations/historical.json?apiKey=&units=&startDate=&endDate=`
- Location ids: `LFPB:9:FR`, `EGLC:9:GB`, `RJTT:9:JP`, `RKSI:9:KR`, `KLGA:9:US`, `KORD:9:US`
- `units=m` → °C temps; US cities must use `units=e` for °F settlement comparison
- Frontend key lives in WU page JS (`API_KEY`); may rotate. Override via env.

## Polymarket Daily High (2026-08-13)

- Paris/London/Tokyo/Seoul: `be N°C` / `or below`, WU airport ICAO, isomorphic to current parser
- NYC: KLGA, °F, `between 76-77°F` — not this task
- Chicago: KORD, °F, same between syntax — not this task
- NYC ≠ Kalshi KXHIGHNY (KNYC Central Park)
