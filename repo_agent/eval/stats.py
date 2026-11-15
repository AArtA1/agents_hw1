"""Small statistics helpers for eval reports."""

from __future__ import annotations

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion (None if n == 0)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def rate(values: list[bool]) -> dict:
    n = len(values)
    k = sum(bool(v) for v in values)
    return {"n": n, "k": k, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}
