import pandas as pd
import pytest

from demand_segments.backtest import (
    ML,
    ML_HISTORY_ONLY,
    ZERO,
    by_segment,
    champions,
    ml_vs_best_stat,
    origins,
    run,
)


@pytest.fixture(scope="module")
def result(small_data):
    sales, skus = small_data
    return run(sales, skus, n_windows=2, include_arima=False, ml_rounds=30, ablation=True, log=False)


def test_every_model_forecasts_the_same_sku_weeks(result):
    fc = result.forecasts
    keys = fc.groupby("model")[["sku_id", "week"]].apply(lambda d: frozenset(map(tuple, d.to_numpy())))
    assert keys.nunique() == 1
    assert {ML, ML_HISTORY_ONLY, ZERO, "SeasonalNaive", "CrostonOptimized", "TSB", "AutoETS"} <= set(
        fc["model"]
    )
    assert fc.loc[fc["y"].notna(), "yhat"].notna().all()
    assert (fc["yhat"] >= 0).all()


def test_windows_are_consecutive_and_do_not_overlap(result):
    fc = result.forecasts[result.forecasts["model"] == ML]
    weeks = fc.groupby("fold")["week"].agg(["min", "max", "nunique"])
    assert (weeks["nunique"] == 8).all()
    assert weeks["max"].iloc[0] < weeks["min"].iloc[1]
    assert origins(120, 2) == [104, 112]


def test_champion_is_never_the_zero_reference(result):
    champ = champions(result.forecasts, n_select=1)
    assert not {ZERO, ML_HISTORY_ONLY} & set(champ["champion"])
    assert ZERO not in set(ml_vs_best_stat(result.forecasts)["best_statistical"])


def test_zero_reference_scores_exactly_as_defined(result):
    s = by_segment(result.forecasts)
    z = s[(s["model"] == ZERO) & (s["segment"] == "all")].iloc[0]
    assert z["wape"] == pytest.approx(1.0)
    assert z["wape_8wk"] == pytest.approx(1.0)
    assert z["bias"] == pytest.approx(-1.0)
    assert isinstance(s, pd.DataFrame)
