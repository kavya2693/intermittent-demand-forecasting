import numpy as np
import pandas as pd

from demand_segments.features import FEATURES, HORIZON, ORIGIN_FEATURES, Panel, frame_at, training_frame


def test_features_ignore_everything_sold_on_or_after_the_origin(small_data):
    """Rewrite the future with junk; no feature built at the origin may move."""
    sales, skus = small_data
    panel = Panel.from_frames(sales, skus)
    origin = 90
    before = frame_at(panel, origin)

    rng = np.random.default_rng(0)
    panel.y[:, origin:] = rng.integers(0, 10_000, panel.y[:, origin:].shape)
    after = frame_at(panel, origin)

    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
    assert not before["y"].equals(after["y"])  # the target itself did change


def test_leak_check_would_catch_a_feature_that_peeks(small_data):
    """The check above is only worth something if it fails on a leaking feature."""
    sales, skus = small_data
    panel = Panel.from_frames(sales, skus)
    ORIGIN_FEATURES["peek"] = lambda p, o, s: p.y[:, o]
    try:
        before = frame_at(panel, 90)["peek"].copy()
        panel.y[:, 90:] += 1
        assert not before.equals(frame_at(panel, 90)["peek"])
    finally:
        del ORIGIN_FEATURES["peek"]


def test_training_rows_never_reach_the_forecast_window(small_data):
    sales, skus = small_data
    panel = Panel.from_frames(sales, skus)
    train = training_frame(panel, before=100)
    assert train["t"].max() < 100
    assert set(train["horizon"].unique()) == set(range(1, HORIZON + 1))
    assert train["y"].notna().all()


def test_panel_keeps_feed_gaps_as_missing_not_zero(small_data):
    sales, skus = small_data
    panel = Panel.from_frames(sales, skus)
    assert np.isnan(panel.y).sum() == len(skus) * panel.n_weeks - sales["units"].notna().sum()
