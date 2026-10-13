"""Per-origin view of the backtest: how often LightGBM beats the best statistical model, and its bias.

Reads outputs/scores_by_fold.csv and writes outputs/fold_summary.csv. A segment-level average can
hide a model that wins big on two origins and loses on four; this table shows that case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STATS = ["AutoETS", "AutoARIMA", "CrostonOptimized", "TSB", "WindowAverage", "SeasonalNaive"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    folds = pd.read_csv(args.out / "scores_by_fold.csv")
    rows = []
    for metric in ["wape", "wape_8wk"]:
        p = folds.pivot_table(index=["segment", "fold"], columns="model", values=metric)
        best = p[[c for c in STATS if c in p.columns]].min(axis=1)
        wins = (p["LightGBM"] < best).groupby("segment").agg(["sum", "count"])
        for seg, r in wins.iterrows():
            rows.append(
                {"segment": seg, "metric": metric, "lightgbm_wins": int(r["sum"]), "origins": int(r["count"])}
            )
    out = pd.DataFrame(rows).pivot(index="segment", columns="metric", values="lightgbm_wins")
    out.columns = [f"origins_won_on_{c}" for c in out.columns]
    bias = folds[folds["model"] == "LightGBM"].groupby("segment")["bias"]
    out["origins"] = folds["fold"].nunique()
    out["lightgbm_bias_min"] = bias.min()
    out["lightgbm_bias_max"] = bias.max()
    out["origins_with_abs_bias_over_10pct"] = bias.apply(lambda b: int((b.abs() > 0.10).sum()))
    out.reset_index().to_csv(args.out / "fold_summary.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
