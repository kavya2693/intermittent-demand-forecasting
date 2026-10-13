"""Forecast error measures, written so a missing actual is never scored as a zero."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    keep = ~np.isnan(y)
    if np.isnan(f[keep]).any():
        raise ValueError("forecast has gaps where actuals exist")
    return y[keep], f[keep]


def wape(y: np.ndarray, f: np.ndarray) -> float:
    """Sum of absolute errors over sum of actuals. Defined whenever any unit was sold."""
    y, f = _clean(y, f)
    total = y.sum()
    return float(np.abs(f - y).sum() / total) if total > 0 else float("nan")


def bias(y: np.ndarray, f: np.ndarray) -> float:
    """Signed total error over total actuals: +0.08 means 8% over-forecast in aggregate."""
    y, f = _clean(y, f)
    total = y.sum()
    return float((f - y).sum() / total) if total > 0 else float("nan")


def mape(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute percentage error over weeks with a sale; zero-sale weeks are undefined and skipped."""
    y, f = _clean(y, f)
    nz = y > 0
    return float(np.mean(np.abs(f[nz] - y[nz]) / y[nz])) if nz.any() else float("nan")


def mape_coverage(y: np.ndarray) -> float:
    """Share of scored weeks where MAPE is defined. Low coverage means MAPE describes few weeks."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    return float((y > 0).mean()) if len(y) else float("nan")


def window_wape(df: pd.DataFrame, actual: str = "y", forecast: str = "yhat") -> float:
    """WAPE on each SKU's eight-week total rather than on single weeks.

    Replenishment buys cover for a lead time, not a single week, so this is the error a
    planner pays for. It also stops rewarding a forecast of zero for slow movers, which
    weekly WAPE does: a SKU selling one unit every few weeks has a weekly median of zero.
    """
    seen = df[df[actual].notna()]
    totals = seen.groupby(["sku_id", "fold"], observed=True)[[actual, forecast]].sum()
    return wape(totals[actual].to_numpy(), totals[forecast].to_numpy())


def score(df: pd.DataFrame, by: list[str], actual: str = "y", forecast: str = "yhat") -> pd.DataFrame:
    """WAPE, MAPE (with its coverage) and bias for each group."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        y, f = g[actual].to_numpy(), g[forecast].to_numpy()
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "wape": wape(y, f),
                "wape_8wk": window_wape(g, actual, forecast) if {"sku_id", "fold"} <= set(g) else np.nan,
                "mape": mape(y, f),
                "mape_coverage": mape_coverage(y),
                "bias": bias(y, f),
                "actual_units": float(np.nansum(y)),
                "n": int((~np.isnan(y)).sum()),
            }
        )
    return pd.DataFrame(rows)
