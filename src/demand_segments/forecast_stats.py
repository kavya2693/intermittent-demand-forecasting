"""Per-SKU statistical models fitted with statsforecast, refitted at every backtest origin."""

from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    TSB,
    AutoARIMA,
    AutoETS,
    CrostonOptimized,
    SeasonalNaive,
    WindowAverage,
)

from demand_segments.features import HORIZON

BASELINE = "SeasonalNaive"


def models(include_arima: bool = True) -> list:
    # ETS and ARIMA are fitted without a 52-week season: ETS caps the season length far
    # below 52 and seasonal ARIMA at m=52 is too slow to refit 1,500 times per origin.
    # The seasonal naive baseline carries the yearly pattern instead.
    ms = [
        SeasonalNaive(season_length=52),
        WindowAverage(window_size=13),
        AutoETS(season_length=1),
        CrostonOptimized(),
        TSB(alpha_d=0.2, alpha_p=0.1),
    ]
    if include_arima:
        ms.append(AutoARIMA(season_length=1))
    return ms


def impute(sales: pd.DataFrame) -> pd.DataFrame:
    """Fill feed-gap weeks with the trailing 8-week mean so the per-SKU models can run.

    Only the model inputs are filled; scoring always uses the raw actuals, where gaps are skipped.
    """
    df = sales.sort_values(["sku_id", "week"]).copy()
    trailing = df.groupby("sku_id")["units"].transform(lambda s: s.rolling(8, min_periods=1).mean().ffill())
    df["units"] = df["units"].fillna(trailing).fillna(0.0)
    return df


def backtest(
    sales: pd.DataFrame, n_windows: int, include_arima: bool = True, n_jobs: int = 4
) -> pd.DataFrame:
    """Walk-forward forecasts: ``n_windows`` origins, eight weeks each, no overlap."""
    df = impute(sales).rename(columns={"sku_id": "unique_id", "week": "ds", "units": "y"})
    sf = StatsForecast(models=models(include_arima), freq="W-MON", n_jobs=n_jobs)
    cv = sf.cross_validation(
        df=df[["unique_id", "ds", "y"]], h=HORIZON, n_windows=n_windows, step_size=HORIZON
    )
    cv = cv.drop(columns="y").rename(columns={"unique_id": "sku_id", "ds": "week"})
    model_cols = [c for c in cv.columns if c not in ("sku_id", "week", "cutoff")]
    long = cv.melt(
        id_vars=["sku_id", "week", "cutoff"], value_vars=model_cols, var_name="model", value_name="yhat"
    )
    long["yhat"] = long["yhat"].clip(lower=0)
    return long
