"""Export the planner's 8-week forecast sheet for the LAST backtest origin, plus the slide-06 data.

Uses the repo's own functions on the same data and settings as `make run`:
  - demand_segments.data.load            -> data/sales.parquet, data/skus.parquet (seed 7)
  - demand_segments.segments.segment_table -> ADI / CV2 / segment, history before the first origin
  - demand_segments.backtest.origins     -> the six walk-forward origins
  - demand_segments.metrics.window_wape / wape -> the repo's scorecards
and the forecasts that `make run` wrote to outputs/forecasts.parquet (nothing is re-fitted).

Sheet rows: for each segment, the SKU with the MEDIAN 8-week absolute % error of LightGBM at the
last origin (among SKUs with at least one unit sold in the window and no missing weeks), plus
the SKUs just above the median in the two slow segments. The pick is deterministic and is not
the best case.

Run:  cd <repo> && PYTHONPATH=src .venv/bin/python <this file> <out.json>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from demand_segments.backtest import ML, origins
from demand_segments.data import load
from demand_segments.features import (
    FEATURES,
    HORIZON,
    ORIGIN_FEATURES,
    TARGET_FEATURES,
    Panel,
    training_frame,
)
from demand_segments.forecast_ml import KNOWN_FUTURE
from demand_segments.metrics import wape, window_wape
from demand_segments.segments import SEGMENTS, segment_table

sales, skus = load(Path("data"))
weeks = pd.DatetimeIndex(sorted(sales["week"].unique()))
starts = origins(len(weeks), 6)
segs = segment_table(sales, weeks[starts[0]])

# the published segments must be what segment_table gives on this data
pub = pd.read_csv("outputs/sku_segments.csv")
assert (pub["segment"].to_numpy() == segs["segment"].astype(str).to_numpy()).all(), "segments differ"

fc = pd.read_parquet("outputs/forecasts.parquet")
last = len(starts) - 1
origin_week = weeks[starts[last]]
win = fc[(fc["fold"] == last) & (fc["model"] == ML)].copy()

# planner's columns, per SKU
tot = win.groupby("sku_id").agg(
    forecast=("yhat", "sum"), sold=("y", "sum"), missing=("y", lambda s: s.isna().sum())
)
hist = sales[(sales["week"] < origin_week) & (sales["week"] >= origin_week - pd.Timedelta(weeks=52))]
weeks_sold = hist.assign(sold=hist["units"] > 0).groupby("sku_id")["sold"].sum()
tot = (
    tot.join(weeks_sold.rename("weeks_sold_52"))
    .join(segs.set_index("sku_id")[["segment", "adi", "cv2"]])
    .join(skus.set_index("sku_id")[["category", "channel"]])
)
tot["abs_pct_err"] = (tot["forecast"] - tot["sold"]).abs() / tot["sold"]

pick = []
for seg in SEGMENTS:
    c = tot[(tot["segment"] == seg) & (tot["sold"] > 0) & (tot["missing"] == 0)].sort_values(
        ["abs_pct_err", "sold"], kind="mergesort"
    )
    mid = (len(c) - 1) // 2
    pick.append((seg, c.index[mid], True))
    if seg in ("intermittent", "lumpy"):
        pick.append((seg, c.index[mid + 1], False))
order = {s: i for i, s in enumerate(SEGMENTS)}
pick.sort(key=lambda t: (order[t[0]], not t[2]))

weekly = win.pivot(index="sku_id", columns="week", values=["yhat", "y"])
rows = []
anim = 0
for seg, sku, animate in pick:
    r = tot.loc[sku]
    rows.append(
        {
            "sku": sku,
            "category": r["category"],
            "channel": r["channel"],
            "segment": seg,
            "weeks_sold_52": int(r["weeks_sold_52"]),
            "forecast_8wk": round(float(r["forecast"]), 1),
            "sold_8wk": int(r["sold"]),
            "weekly_forecast": [round(float(v), 2) for v in weekly.loc[sku, "yhat"].to_numpy()],
            "weekly_sold": [int(v) for v in weekly.loc[sku, "y"].to_numpy()],
            "animate": anim if animate else -1,
        }
    )
    anim += int(animate)

scored = win[win["y"].notna()]
seg_8wk = {
    s: round(window_wape(scored[scored["sku_id"].isin(tot.index[tot["segment"] == s])]), 4) for s in SEGMENTS
}

# slide 06: ADI / CV2 of every SKU, and one SKU's backtest over all six origins
example = next(sku for seg, sku, a in pick if seg == "intermittent" and a)
ex = fc[(fc["sku_id"] == example) & fc["model"].isin([ML, "SeasonalNaive", "AutoARIMA"])]
ex_hist = sales[(sales["sku_id"] == example) & (sales["week"] < weeks[starts[0]])].tail(26)

# architecture annotations: LightGBM training rows per fit (first and last origin), feature counts
panel = Panel.from_frames(sales, skus)
arch = {
    "train_rows_first_origin": len(training_frame(panel, starts[0])),
    "train_rows_last_origin": len(training_frame(panel, starts[-1])),
    "features": len(FEATURES),
    "origin_features": len(ORIGIN_FEATURES),
    "target_features": len(TARGET_FEATURES),
    "known_future_features": len(KNOWN_FUTURE),
}

out = {
    "arch": arch,
    "origin_index": last + 1,
    "origin_week": str(origin_week.date()),
    "window_end_week": str(weeks[starts[last] + HORIZON - 1].date()),
    "skus": len(tot),
    "model": ML,
    "wape_8wk_all": round(window_wape(scored), 4),
    "wape_weekly_all": round(wape(scored["y"].to_numpy(), scored["yhat"].to_numpy()), 4),
    "wape_8wk_by_segment": seg_8wk,
    "forecast_units_8wk": round(float(win["yhat"].sum()), 1),
    "sold_units_8wk": int(scored["y"].sum()),
    "rows": rows,
    "forecast_rows_total": len(fc),
    "forecast_models": sorted(fc["model"].unique().tolist()),
    "scatter": [
        [round(float(a), 3), round(float(c), 3), str(s)]
        for a, c, s in segs[["adi", "cv2", "segment"]].itertuples(index=False)
    ],
    "example": {
        "sku": example,
        "category": str(tot.loc[example, "category"]),
        "origins": [str(weeks[o].date()) for o in starts],
        "totals": [
            [
                round(float(g.loc[g["model"] == ML, "yhat"].sum()), 1),
                int(g.loc[g["model"] == ML, "y"].sum()),
            ]
            for _, g in ex.groupby("fold")
        ],
        "history": [None if np.isnan(v) else int(v) for v in ex_hist["units"].to_numpy()],
        "weeks": [str(w.date()) for w in sorted(ex["week"].unique())],
        "actual": [
            None if np.isnan(v) else int(v) for v in ex[ex["model"] == ML].sort_values("week")["y"].to_numpy()
        ],
        "lightgbm": [round(float(v), 2) for v in ex[ex["model"] == ML].sort_values("week")["yhat"]],
        "seasonal_naive": [
            round(float(v), 2) for v in ex[ex["model"] == "SeasonalNaive"].sort_values("week")["yhat"]
        ],
        "arima": [round(float(v), 2) for v in ex[ex["model"] == "AutoARIMA"].sort_values("week")["yhat"]],
    },
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k not in ("rows", "scatter", "example")})
for r in rows:
    print(
        r["sku"],
        r["segment"],
        r["category"],
        r["weeks_sold_52"],
        r["forecast_8wk"],
        r["sold_8wk"],
        r["animate"],
    )
print("example", example)
