"""Walk-forward backtest: every model forecasts the same eight-week windows from the same origins."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demand_segments import forecast_ml, forecast_stats
from demand_segments.features import HORIZON, Panel
from demand_segments.metrics import score, wape
from demand_segments.segments import segment_table

ML = "LightGBM"
ML_HISTORY_ONLY = "LightGBM-history-only"  # ablation: no calendar, promotion or price features
ZERO = "ZeroForecast"  # reference only: never a candidate, shows what weekly WAPE rewards


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame  # sku_id, week, fold, model, yhat, y, segment
    segments: pd.DataFrame
    importance: pd.DataFrame
    timings: dict[str, float] = field(default_factory=dict)


def origins(n_weeks: int, n_windows: int) -> list[int]:
    first = n_weeks - HORIZON * n_windows
    return [first + HORIZON * i for i in range(n_windows)]


def run(
    sales: pd.DataFrame,
    skus: pd.DataFrame,
    n_windows: int = 6,
    include_arima: bool = True,
    n_jobs: int = 1,
    ml_rounds: int = forecast_ml.ROUNDS,
    ablation: bool = False,
    log: bool = True,
) -> BacktestResult:
    panel = Panel.from_frames(sales, skus)
    starts = origins(panel.n_weeks, n_windows)
    week_of = dict(enumerate(panel.weeks))
    timings: dict[str, float] = {}

    # Segments are fixed from the history before the first origin, so a SKU is scored in one
    # segment across all windows.
    segs = segment_table(sales, panel.weeks[starts[0]])

    t0 = time.perf_counter()
    stats = forecast_stats.backtest(sales, n_windows, include_arima=include_arima, n_jobs=n_jobs)
    timings["statistical models"] = time.perf_counter() - t0
    cutoffs = sorted(stats["cutoff"].unique())
    expected = [panel.weeks[o - 1] for o in starts]
    if [pd.Timestamp(c) for c in cutoffs] != expected:
        raise RuntimeError("statsforecast windows do not line up with the backtest origins")
    stats["fold"] = stats["cutoff"].map({c: i for i, c in enumerate(cutoffs)})
    if log:
        print(f"statistical models: {timings['statistical models']:.0f}s", flush=True)

    variants = [(ML, False)] + ([(ML_HISTORY_ONLY, True)] if ablation else [])
    ml_parts, model = [], None
    for name, history_only in variants:
        t0 = time.perf_counter()
        for fold, origin in enumerate(starts):
            m = forecast_ml.fit(panel, before=origin, rounds=ml_rounds, history_only=history_only)
            pred = forecast_ml.predict(m, panel, origin)
            pred["sku_id"] = panel.sku_ids[pred["sku_idx"].to_numpy()]
            pred["week"] = pred["t"].map(week_of)
            pred["fold"] = fold
            pred["model"] = name
            ml_parts.append(pred[["sku_id", "week", "fold", "model", "yhat"]])
            if not history_only:
                model = m
            if log:
                print(f"{name} fold {fold + 1}/{n_windows} origin {panel.weeks[origin].date()}", flush=True)
        timings[name] = time.perf_counter() - t0
    ml = pd.concat(ml_parts, ignore_index=True)

    zero = (
        stats.loc[stats["model"] == forecast_stats.BASELINE]
        .drop(columns="cutoff")
        .assign(model=ZERO, yhat=0.0)
    )
    fc = pd.concat([stats.drop(columns="cutoff"), ml, zero], ignore_index=True)
    actual = sales[["sku_id", "week", "units"]].rename(columns={"units": "y"})
    fc = fc.merge(actual, on=["sku_id", "week"], how="left")
    fc = fc.merge(segs[["sku_id", "segment"]], on="sku_id", how="left")
    assert model is not None
    return BacktestResult(fc, segs, forecast_ml.importance(model), timings)


def by_segment(fc: pd.DataFrame) -> pd.DataFrame:
    seg = score(fc, ["segment", "model"])
    total = score(fc.assign(segment="all"), ["segment", "model"])
    return pd.concat([seg, total], ignore_index=True)


def champions(fc: pd.DataFrame, n_select: int) -> pd.DataFrame:
    """Pick each segment's model on the early windows, then score it on the later ones.

    Choosing and scoring on the same windows would flatter whichever model won by luck. The
    pick uses eight-week WAPE: on weekly WAPE the zero forecast wins the slow segments.
    """
    fc = fc[~fc["model"].isin([ZERO, ML_HISTORY_ONLY])]
    early, late = fc[fc["fold"] < n_select], fc[fc["fold"] >= n_select]
    pick = (
        score(early, ["segment", "model"]).sort_values("wape_8wk").groupby("segment", observed=True).head(1)
    )
    late_scores = score(late, ["segment", "model"]).set_index(["segment", "model"])
    rows = []
    for seg, chosen in pick[["segment", "model"]].itertuples(index=False):
        champ = late_scores.loc[(seg, chosen)]
        base = late_scores.loc[(seg, forecast_stats.BASELINE)]
        ml = late_scores.loc[(seg, ML)]
        rows.append(
            {
                "segment": seg,
                "champion": chosen,
                "champion_wape": champ["wape"],
                "champion_wape_8wk": champ["wape_8wk"],
                "champion_bias": champ["bias"],
                "seasonal_naive_wape": base["wape"],
                "seasonal_naive_wape_8wk": base["wape_8wk"],
                "lightgbm_wape": ml["wape"],
                "wape_reduction_vs_seasonal_naive": 1 - champ["wape"] / base["wape"],
            }
        )
    return pd.DataFrame(rows)


def ml_vs_best_stat(fc: pd.DataFrame) -> pd.DataFrame:
    """Where the global model loses: WAPE gap and per-SKU win share against the segment's best
    statistical model (best on weekly WAPE over all origins, which favours the statistical side)."""
    fc = fc[~fc["model"].isin([ZERO, ML_HISTORY_ONLY])]
    seg_scores = score(fc, ["segment", "model"])
    rows = []
    for seg, g in seg_scores.groupby("segment", observed=True):
        stat = g[g["model"] != ML].sort_values("wape").iloc[0]
        ml_wape = float(g.loc[g["model"] == ML, "wape"].iloc[0])
        sub = fc[(fc["segment"] == seg) & fc["model"].isin([ML, stat["model"]]) & fc["y"].notna()]
        per_sku = (
            sub.groupby(["sku_id", "model"])
            .apply(lambda d: wape(d["y"].to_numpy(), d["yhat"].to_numpy()), include_groups=False)
            .unstack()
            .dropna()
        )
        rows.append(
            {
                "segment": seg,
                "best_statistical": stat["model"],
                "best_statistical_wape": stat["wape"],
                "lightgbm_wape": ml_wape,
                "lightgbm_gap": ml_wape - stat["wape"],
                "best_statistical_wape_8wk": stat["wape_8wk"],
                "lightgbm_wape_8wk": float(g.loc[g["model"] == ML, "wape_8wk"].iloc[0]),
                "skus_scored": len(per_sku),
                "lightgbm_sku_win_share": float(np.mean(per_sku[ML] < per_sku[stat["model"]])),
            }
        )
    return pd.DataFrame(rows)
