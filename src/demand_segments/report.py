"""Write the backtest tables to CSV and a short markdown report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from demand_segments.backtest import BacktestResult, by_segment, champions, ml_vs_best_stat
from demand_segments.metrics import score


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def _md(df: pd.DataFrame, pct_cols: list[str]) -> str:
    out = df.copy()
    for c in pct_cols:
        out[c] = out[c].map(_pct)
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False)]
    return "\n".join(lines)


def write(res: BacktestResult, skus: pd.DataFrame, out: Path, n_windows: int) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fc = res.forecasts
    n_select = n_windows // 2

    seg_scores = by_segment(fc).sort_values(["segment", "wape"])
    folds = score(fc, ["segment", "model", "fold"])
    champ = champions(fc, n_select)
    gap = ml_vs_best_stat(fc)
    counts = res.segments["segment"].value_counts().rename_axis("segment").reset_index(name="skus")
    agree = pd.crosstab(
        res.segments.merge(skus, on="sku_id")["intended_pattern"],
        res.segments["segment"].to_numpy(),
    )

    seg_scores.to_csv(out / "scores_by_segment.csv", index=False)
    folds.to_csv(out / "scores_by_fold.csv", index=False)
    champ.to_csv(out / "champions.csv", index=False)
    gap.to_csv(out / "lightgbm_vs_best_statistical.csv", index=False)
    counts.to_csv(out / "segment_counts.csv", index=False)
    agree.to_csv(out / "segment_vs_generator_intent.csv")
    res.importance.to_csv(out / "feature_importance.csv", index=False)
    res.segments.to_csv(out / "sku_segments.csv", index=False)

    pct = ["wape", "wape_8wk", "mape", "mape_coverage", "bias"]
    show = seg_scores[["segment", "model", *pct]]
    lines = [
        "# Backtest report",
        "",
        f"{len(res.segments)} SKUs, {n_windows} walk-forward origins, 8-week horizon. "
        f"Champions are picked on the first {n_select} origins and scored on the last "
        f"{n_windows - n_select}.",
        "",
        "## SKUs per segment",
        "",
        _md(counts, []),
        "",
        "## All origins, by segment and model",
        "",
        _md(show, pct),
        "",
        "## Segment champion, chosen early and scored late",
        "",
        _md(
            champ,
            [
                "champion_wape",
                "champion_wape_8wk",
                "champion_bias",
                "seasonal_naive_wape",
                "seasonal_naive_wape_8wk",
                "lightgbm_wape",
                "wape_reduction_vs_seasonal_naive",
            ],
        ),
        "",
        "## LightGBM against the best statistical model in each segment",
        "",
        _md(
            gap,
            [
                "best_statistical_wape",
                "lightgbm_wape",
                "lightgbm_gap",
                "best_statistical_wape_8wk",
                "lightgbm_wape_8wk",
                "lightgbm_sku_win_share",
            ],
        ),
        "",
        "## Run time (seconds)",
        "",
        *[f"- {k}: {v:.0f}" for k, v in res.timings.items()],
        "",
    ]
    text = "\n".join(lines)
    (out / "run_report.md").write_text(text)
    return text
