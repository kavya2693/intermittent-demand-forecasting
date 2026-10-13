"""Global LightGBM model: one model across all SKUs, direct multi-week horizon."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_segments.features import CATEGORICAL, FEATURES, Panel, frame_at, training_frame

PARAMS: dict[str, object] = {
    # Tweedie puts a point mass at zero and a long right tail, which is what weekly
    # unit sales look like once slow movers are in the mix.
    "objective": "tweedie",
    "tweedie_variance_power": 1.3,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
    "num_threads": 4,
}
ROUNDS = 400


def fit(p: Panel, before: int, rounds: int = ROUNDS) -> lgb.Booster:
    """Train on every origin whose eight target weeks end before week index ``before``."""
    train = training_frame(p, before)
    # The target is unit-free (sales over the SKU's own level); weighting by the level makes
    # the loss count units, so a 300-a-week SKU matters more than a 1-a-month one, as in WAPE.
    ds = lgb.Dataset(
        train[FEATURES],
        label=train["y"] / train["scale"],
        weight=train["scale"],
        categorical_feature=CATEGORICAL,
        free_raw_data=True,
    )
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


def predict(model: lgb.Booster, p: Panel, origin: int) -> pd.DataFrame:
    frame = frame_at(p, origin)
    pred = np.asarray(model.predict(frame[FEATURES]))
    frame["yhat"] = np.clip(pred, 0, None) * frame["scale"]
    return frame[["sku_idx", "t", "yhat"]]


def importance(model: lgb.Booster) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain")
    out = pd.DataFrame({"feature": model.feature_name(), "gain": gain})
    out["share"] = out["gain"] / out["gain"].sum()
    return out.sort_values("gain", ascending=False).reset_index(drop=True)
