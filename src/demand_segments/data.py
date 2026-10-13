"""Seeded generator for a weekly SKU sales history.

The catalogue mixes high-volume lines that sell every week with spare-part style lines that
sell a handful of units a few weeks a year, across B2B and B2C channels. Each SKU is given an
intended demand pattern; the segmentation in ``segments.py`` never sees that label, which lets
the tests check the classifier against an answer known by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand_segments.calendar import calendar_frame, week_starts

# Category: (seasonal amplitude, peak week of year, holiday sensitivity, typical unit price)
CATEGORIES: dict[str, tuple[float, int, float, float]] = {
    "garden": (0.65, 24, 0.3, 18.0),
    "heating": (0.70, 2, 0.2, 45.0),
    "toys": (0.35, 50, 1.6, 22.0),
    "stationery": (0.30, 34, 0.4, 4.0),
    "cleaning": (0.08, 1, 0.3, 6.0),
    "food_service": (0.15, 28, 0.2, 12.0),
    "electrical": (0.10, 48, 0.9, 30.0),
    "spare_parts": (0.05, 10, 0.0, 65.0),
}

# Intended pattern: share of the catalogue.
PATTERN_MIX = {"smooth": 0.30, "erratic": 0.15, "intermittent": 0.35, "lumpy": 0.20}


@dataclass(frozen=True)
class GeneratorConfig:
    n_skus: int = 1500
    n_weeks: int = 156
    seed: int = 7
    launch_share: float = 0.08
    phase_out_share: float = 0.03
    gap_share: float = 0.04


def _sku_table(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_skus
    patterns = rng.choice(list(PATTERN_MIX), size=n, p=list(PATTERN_MIX.values()))
    cats = np.array(list(CATEGORIES))
    # Spare parts are mostly slow movers; everything else draws categories evenly.
    category = np.where(
        np.isin(patterns, ["intermittent", "lumpy"]) & (rng.random(n) < 0.35),
        "spare_parts",
        rng.choice(cats[:-1], size=n),
    )
    channel = np.where(rng.random(n) < 0.45, "B2B", "B2C")
    launch = np.where(rng.random(n) < cfg.launch_share, rng.integers(4, 40, n), 0)
    phase_out = np.where(rng.random(n) < cfg.phase_out_share, rng.integers(100, 130, n), -1)
    return pd.DataFrame(
        {
            "sku_id": [f"SKU{i:05d}" for i in range(n)],
            "category": category,
            "channel": channel,
            "intended_pattern": patterns,
            "launch_week": launch,
            "phase_out_week": phase_out,
        }
    )


def _negbin(rng: np.random.Generator, mean: np.ndarray, r: float) -> np.ndarray:
    """Negative binomial draws with the given mean and shape r (smaller r = more dispersed)."""
    mean = np.maximum(mean, 1e-6)
    return rng.negative_binomial(r, r / (r + mean)).astype(float)


def generate(cfg: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales, skus). Sales is long format: sku_id, week, units, price, promo."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    skus = _sku_table(cfg, rng)
    weeks = week_starts(cfg.n_weeks)
    cal = calendar_frame(weeks)
    woy = cal["week_of_year"].to_numpy()
    t = np.arange(cfg.n_weeks)

    units = np.zeros((cfg.n_skus, cfg.n_weeks))
    price = np.zeros_like(units)
    promo = np.zeros_like(units, dtype=np.int8)

    for i, row in enumerate(skus.itertuples(index=False)):
        amp, peak, hol_sens, base_price = CATEGORIES[row.category]
        b2c = row.channel == "B2C"

        season = 1 + amp * rng.uniform(0.6, 1.3) * np.cos(2 * np.pi * (woy - peak - rng.integers(-3, 4)) / 52)
        trend = np.exp(rng.normal(0, 0.10) * t / 52)
        mult = np.clip(season, 0.15, None) * trend

        if b2c:
            mult *= 1 + cal["black_friday"].to_numpy() * hol_sens * rng.uniform(0.5, 1.2)
            mult *= 1 + cal["christmas"].to_numpy() * hol_sens * rng.uniform(0.3, 0.9)
            mult *= 1 + cal["easter"].to_numpy() * 0.25 * hol_sens
        else:
            # Trade customers close over the holidays and reopen with a restocking week.
            mult *= 1 - 0.55 * cal["christmas"].to_numpy()
            mult *= 1 - 0.35 * cal["new_year"].to_numpy()
        if row.category == "stationery":
            mult *= 1 + 0.8 * cal["back_to_school"].to_numpy()

        # Prices drift up in steps; promotions are planned weeks at a discount, B2C only.
        p = base_price * rng.lognormal(0, 0.4) * np.ones(cfg.n_weeks)
        for step in rng.integers(10, cfg.n_weeks, rng.integers(0, 3)):
            p[step:] *= rng.uniform(1.03, 1.10)
        pr = np.zeros(cfg.n_weeks, dtype=np.int8)
        if b2c and rng.random() < 0.7:
            starts = rng.choice(np.arange(4, cfg.n_weeks - 2), size=rng.integers(3, 10), replace=False)
            for s in starts:
                pr[s : s + rng.integers(1, 3)] = 1
        uplift = rng.lognormal(np.log(1.8), 0.25)
        mult *= np.where(pr == 1, uplift, 1.0)
        # Customers who bought on promotion do not buy the week after.
        after = np.roll(pr, 1) & (1 - pr)
        after[0] = 0
        mult *= np.where(after == 1, 0.75, 1.0)
        p = np.where(pr == 1, p * 0.75, p)

        if row.phase_out_week >= 0:
            k = np.clip(t - row.phase_out_week, 0, None)
            mult *= 0.92**k

        pattern = row.intended_pattern
        if pattern == "smooth":
            level = rng.lognormal(3.4, 0.8)
            y = _negbin(rng, level * mult, r=25.0)
        elif pattern == "erratic":
            level = rng.lognormal(2.8, 0.8)
            y = _negbin(rng, level * mult, r=1.3)
        elif pattern == "intermittent":
            occ = rng.uniform(0.12, 0.6)
            size = rng.lognormal(0.6, 0.5)
            fires = rng.random(cfg.n_weeks) < np.clip(occ * np.sqrt(mult), 0, 0.95)
            y = fires * (1 + rng.poisson(np.maximum(size * mult - 1, 0.05)))
        else:  # lumpy
            occ = rng.uniform(0.10, 0.55)
            size = rng.lognormal(2.0, 0.7)
            fires = rng.random(cfg.n_weeks) < np.clip(occ * np.sqrt(mult), 0, 0.95)
            y = fires * (1 + _negbin(rng, size * mult, r=0.9))

        # Occasional bulk orders, mostly from trade accounts.
        bulk = rng.random(cfg.n_weeks) < (0.006 if not b2c else 0.002)
        y = y + bulk * np.ceil(max(np.mean(y), 1) * rng.uniform(4, 12, cfg.n_weeks))

        # Stock-outs censor sales for one to three weeks.
        for s in np.flatnonzero(rng.random(cfg.n_weeks) < 0.008):
            span = slice(s, s + rng.integers(1, 4))
            y[span] = np.floor(y[span] * rng.uniform(0, 0.3))

        y = y.astype(float)
        y[: row.launch_week] = np.nan
        if rng.random() < cfg.gap_share:
            g = rng.integers(20, cfg.n_weeks - 6)
            y[g : g + rng.integers(2, 7)] = np.nan  # feed outage, not zero sales

        units[i], price[i], promo[i] = y, np.round(p, 2), pr

    sales = pd.DataFrame(
        {
            "sku_id": np.repeat(skus["sku_id"].to_numpy(), cfg.n_weeks),
            "week": np.tile(weeks.to_numpy(), cfg.n_skus),
            "units": units.ravel(),
            "price": price.ravel(),
            "promo": promo.ravel(),
        }
    )
    # Weeks before launch are not part of the history at all.
    launch_idx = np.repeat(skus["launch_week"].to_numpy(), cfg.n_weeks)
    sales = sales[np.tile(t, cfg.n_skus) >= launch_idx].reset_index(drop=True)
    return sales, skus


def write(out_dir: Path, cfg: GeneratorConfig | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sales, skus = generate(cfg)
    sales_path, skus_path = out_dir / "sales.parquet", out_dir / "skus.parquet"
    sales.to_parquet(sales_path, index=False)
    skus.to_parquet(skus_path, index=False)
    return sales_path, skus_path


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(data_dir / "sales.parquet"), pd.read_parquet(data_dir / "skus.parquet")
