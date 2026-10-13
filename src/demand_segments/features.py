"""Feature engineering for the global model, declared once and built for any forecast origin.

Every feature is a named function in one of two registries. Origin features may only read the
sales history before the origin; target-week features may read the calendar, the promotion
plan and the price list for the week being forecast, because a planner knows those in advance,
plus sales from a year earlier. Adding a feature is one entry in a registry: the training set,
the forecast frame and the leakage test all pick it up from there.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand_segments.calendar import calendar_frame

HORIZON = 8


@dataclass
class Panel:
    """Sales as SKU-by-week matrices. NaN marks a week with no record (pre-launch or feed gap)."""

    sku_ids: np.ndarray
    weeks: pd.DatetimeIndex
    y: np.ndarray
    promo: np.ndarray
    price: np.ndarray
    category: np.ndarray
    channel: np.ndarray
    cal: pd.DataFrame

    @classmethod
    def from_frames(cls, sales: pd.DataFrame, skus: pd.DataFrame) -> Panel:
        weeks = pd.DatetimeIndex(sorted(sales["week"].unique()))
        skus = skus.sort_values("sku_id").reset_index(drop=True)

        def wide(col: str) -> np.ndarray:
            m = sales.pivot(index="sku_id", columns="week", values=col)
            return np.array(m.reindex(index=skus["sku_id"], columns=weeks), dtype=float)

        price = wide("price")
        # Pre-launch weeks have no price row; carry the launch price backwards so ratios exist.
        price = np.array(pd.DataFrame(price).bfill(axis=1), dtype=float)
        return cls(
            sku_ids=skus["sku_id"].to_numpy(),
            weeks=weeks,
            y=wide("units"),
            promo=np.nan_to_num(wide("promo")),
            price=price,
            category=skus["category"].astype("category").cat.codes.to_numpy(),
            channel=(skus["channel"] == "B2B").astype(int).to_numpy(),
            cal=calendar_frame(weeks),
        )

    @property
    def n_weeks(self) -> int:
        return self.y.shape[1]


def _nanstat(fn: Callable[..., np.ndarray], a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return fn(a, axis=1)


def _window(p: Panel, origin: int, n: int) -> np.ndarray:
    return p.y[:, max(0, origin - n) : origin]


def scale(p: Panel, origin: int) -> np.ndarray:
    """Per-SKU level used to make the target unit-free: trailing 52-week mean, floored."""
    s = _nanstat(np.nanmean, _window(p, origin, 52))
    s = np.where(np.isnan(s) | (s <= 0), _nanstat(np.nanmean, p.y[:, :origin]), s)
    return np.where(np.isnan(s) | (s <= 0), 0.1, s)


def _weeks_since_sale(p: Panel, origin: int) -> np.ndarray:
    h = np.nan_to_num(p.y[:, :origin]) > 0
    last = np.where(h.any(axis=1), origin - 1 - np.argmax(h[:, ::-1], axis=1), -1)
    return np.where(last >= 0, origin - last, origin + 52).astype(float)


def _zero_share(p: Panel, origin: int, n: int) -> np.ndarray:
    w = _window(p, origin, n)
    obs = (~np.isnan(w)).sum(axis=1)
    zeros = (w == 0).sum(axis=1)
    return np.where(obs > 0, zeros / np.maximum(obs, 1), np.nan)


def _nonzero_mean(p: Panel, origin: int, n: int) -> np.ndarray:
    w = _window(p, origin, n)
    w = np.where(w > 0, w, np.nan)
    return _nanstat(np.nanmean, w)


def _cv2(p: Panel, origin: int, n: int) -> np.ndarray:
    w = np.where(_window(p, origin, n) > 0, _window(p, origin, n), np.nan)
    return _nanstat(np.nanvar, w) / _nanstat(np.nanmean, w) ** 2


# Origin features: functions of (panel, origin, scale). Level features are divided by the scale
# so one model can learn across SKUs selling 1 unit a month and 300 a week.
ORIGIN_FEATURES: dict[str, Callable[[Panel, int, np.ndarray], np.ndarray]] = {
    "log_scale": lambda p, o, s: np.log1p(s),
    "last_1": lambda p, o, s: p.y[:, o - 1] / s,
    "mean_4": lambda p, o, s: _nanstat(np.nanmean, _window(p, o, 4)) / s,
    "mean_13": lambda p, o, s: _nanstat(np.nanmean, _window(p, o, 13)) / s,
    "mean_26": lambda p, o, s: _nanstat(np.nanmean, _window(p, o, 26)) / s,
    "max_13": lambda p, o, s: _nanstat(np.nanmax, _window(p, o, 13)) / s,
    "std_13": lambda p, o, s: _nanstat(np.nanstd, _window(p, o, 13)) / s,
    "zero_share_13": lambda p, o, s: _zero_share(p, o, 13),
    "zero_share_52": lambda p, o, s: _zero_share(p, o, 52),
    "weeks_since_sale": lambda p, o, s: _weeks_since_sale(p, o),
    "nonzero_mean_52": lambda p, o, s: _nonzero_mean(p, o, 52) / s,
    "cv2_52": lambda p, o, s: _cv2(p, o, 52),
    "history_weeks": lambda p, o, s: (~np.isnan(p.y[:, :o])).sum(axis=1).astype(float),
    "trend_13_52": lambda p, o, s: _nanstat(np.nanmean, _window(p, o, 13)) / s - 1.0,
    "promo_share_13": lambda p, o, s: p.promo[:, max(0, o - 13) : o].mean(axis=1),
    "category": lambda p, o, s: p.category.astype(float),
    "channel_b2b": lambda p, o, s: p.channel.astype(float),
}


def _cal(col: str) -> Callable[[Panel, int, int, np.ndarray], np.ndarray]:
    return lambda p, o, t, s: np.full(len(p.sku_ids), float(p.cal[col].iloc[t]))


def _year_ago(p: Panel, t: int, s: np.ndarray, width: int) -> np.ndarray:
    lo, hi = t - 52 - width, t - 52 + width + 1
    if lo < 0:
        return np.full(len(s), np.nan)
    return _nanstat(np.nanmean, p.y[:, lo:hi]) / s


# Target-week features: functions of (panel, origin, target week index, scale).
TARGET_FEATURES: dict[str, Callable[[Panel, int, int, np.ndarray], np.ndarray]] = {
    "horizon": lambda p, o, t, s: np.full(len(s), float(t - o + 1)),
    "week_of_year": _cal("week_of_year"),
    "month": _cal("month"),
    "black_friday": _cal("black_friday"),
    "christmas": _cal("christmas"),
    "new_year": _cal("new_year"),
    "easter": _cal("easter"),
    "back_to_school": _cal("back_to_school"),
    "promo": lambda p, o, t, s: p.promo[:, t].astype(float),
    "promo_last_week": lambda p, o, t, s: p.promo[:, t - 1].astype(float),
    "price_ratio": lambda p, o, t, s: p.price[:, t] / _nanstat(np.nanmedian, p.price[:, max(0, o - 52) : o]),
    "year_ago": lambda p, o, t, s: _year_ago(p, t, s, 0),
    "year_ago_3wk": lambda p, o, t, s: _year_ago(p, t, s, 1),
}

FEATURES = list(ORIGIN_FEATURES) + list(TARGET_FEATURES)
CATEGORICAL = ["category", "channel_b2b"]


def frame_at(p: Panel, origin: int, horizon: int = HORIZON) -> pd.DataFrame:
    """Feature rows for every SKU and every week origin..origin+horizon-1, plus scale and target."""
    s = scale(p, origin)
    base = {name: fn(p, origin, s) for name, fn in ORIGIN_FEATURES.items()}
    parts = []
    for t in range(origin, min(origin + horizon, p.n_weeks)):
        cols = dict(base)
        cols.update({name: fn(p, origin, t, s) for name, fn in TARGET_FEATURES.items()})
        part = pd.DataFrame(cols)
        part["sku_idx"] = np.arange(len(s))
        part["t"] = t
        part["scale"] = s
        part["y"] = p.y[:, t]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def training_frame(p: Panel, before: int, first_origin: int = 52, step: int = 2) -> pd.DataFrame:
    """Rows from many past origins whose target weeks all fall before ``before``."""
    origins = range(first_origin, before - HORIZON + 1, step)
    df = pd.concat([frame_at(p, o) for o in origins], ignore_index=True)
    # A SKU needs a couple of months of history before its level means anything.
    return df[(df["history_weeks"] >= 8) & df["y"].notna()].reset_index(drop=True)
