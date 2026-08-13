"""Resolver - Observation-Locked Scanner (V0).

PM Daily High markets + station observations -> outcomes that are impossible
by settlement rules -> compare against the book -> shadow/alert.
Authoritative spec: docs/resolver.md"""

from __future__ import annotations

from weadge.resolver.markets import DailyHighEvent  # noqa: F401  (re-export)
