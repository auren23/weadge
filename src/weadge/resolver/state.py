"""Resolution state - tag each bucket OPEN / NEAR_LOCKED / LOCKED.

V0 implements LOCKED only (cold-side lock):
    bucket win range [cap_low, cap_high)
    LOCKED iff observed_max >= cap_high + locked_buffer_c
    -> YES is mathematically impossible, NO fair value = 1.0

NEAR_LOCKED / OPEN are V1 (current temp + time + slope -> P(Tmax survives));
V0 returns OPEN for everything else, no probability model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from weadge.resolver.markets import Bucket
from weadge.resolver.observations import ObservedState


class ResolutionState(StrEnum):
    OPEN = "OPEN"                # 信息不足
    NEAR_LOCKED = "NEAR_LOCKED"  # V1
    LOCKED = "LOCKED"            # 已不可能


@dataclass(frozen=True)
class BucketState:
    bucket: Bucket
    state: ResolutionState


def evaluate_bucket(
    bucket: Bucket,
    obs: ObservedState,
    locked_buffer_c: float = 0.5,
) -> BucketState:
    """单桶状态判定。纯函数。V0 只产出 LOCKED / OPEN。"""
    if obs.observed_max_c is None or bucket.cap_high is None:
        return BucketState(bucket, ResolutionState.OPEN)
    if obs.observed_max_c >= bucket.cap_high + locked_buffer_c:
        return BucketState(bucket, ResolutionState.LOCKED)
    return BucketState(bucket, ResolutionState.OPEN)


def evaluate_event(
    buckets: list[Bucket],
    obs: ObservedState,
    locked_buffer_c: float = 0.5,
) -> list[BucketState]:
    return [evaluate_bucket(b, obs, locked_buffer_c) for b in buckets]
