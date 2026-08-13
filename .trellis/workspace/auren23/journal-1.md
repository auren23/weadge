# Journal - auren23 (Part 1)

> AI development session journal
> Started: 2026-08-13

---



## Session 1: V0.2 observation race + WU audit + isomorphic °C shadow cities

**Date**: 2026-08-13
**Task**: V0.2 observation race + WU audit + isomorphic °C shadow cities
**Branch**: `main`

### Summary

Shipped observation_race (AWC+IEM first_seen, WU Daily Observations audit) and added London/Tokyo/Seoul to resolver shadow. NYC/Chicago stay race-only; LOCKED/30s scan unchanged.

### Main Changes

- Added tools/observation_race.py serve/summary/audit; resolver.yaml cities vs observation_extra split
- Hypothesis updated: latency × settlement fidelity, not weather prediction

### Git Commits

| Hash | Message |
|------|---------|
| `685a250` | (see git log) |

### Testing

- [OK] pytest tests/test_observation_race.py tests/resolver/test_resolver.py
- [OK] live serve --once recorded 6 first_seen rows

### Status

[OK] **Completed**

### Next Steps

- Run race serve 24h and audit --backfill 7 to start fidelity counts
- Later: tiny °F/between parser task before weadge serve --city nyc|chicago
