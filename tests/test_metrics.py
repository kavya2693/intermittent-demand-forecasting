import numpy as np
import pytest

from demand_segments.metrics import bias, mape, mape_coverage, wape
from demand_segments.segments import adi_cv2, classify


def test_wape_and_bias_on_hand_worked_numbers():
    y = np.array([10.0, 0.0, 5.0, 5.0])
    f = np.array([8.0, 2.0, 5.0, 7.0])
    # |errors| = 2 + 2 + 0 + 2 = 6 over 20 actual units
    assert wape(y, f) == pytest.approx(0.30)
    # signed = -2 + 2 + 0 + 2 = +2 over 20
    assert bias(y, f) == pytest.approx(0.10)


def test_bias_and_wape_disagree_when_errors_cancel():
    y = np.array([10.0, 10.0])
    f = np.array([0.0, 20.0])
    assert bias(y, f) == 0.0
    assert wape(y, f) == 1.0


def test_mape_skips_zero_weeks_and_reports_how_few_it_covers():
    y = np.array([0.0, 0.0, 0.0, 4.0])
    f = np.array([1.0, 1.0, 1.0, 2.0])
    assert mape(y, f) == pytest.approx(0.5)
    assert mape_coverage(y) == pytest.approx(0.25)


def test_missing_actuals_are_not_scored_as_zero_sales():
    y = np.array([10.0, np.nan])
    f = np.array([10.0, 50.0])
    assert wape(y, f) == 0.0
    assert bias(y, f) == 0.0


def test_forecast_gap_on_an_observed_week_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError):
        wape(np.array([1.0, 2.0]), np.array([1.0, np.nan]))


def test_zero_forecast_scores_exactly_one_wape_and_minus_one_bias():
    y = np.array([0.0, 3.0, 0.0, 1.0])
    f = np.zeros(4)
    assert wape(y, f) == 1.0
    assert bias(y, f) == -1.0


@pytest.mark.parametrize(
    "series, expected",
    [
        (np.full(52, 20.0), "smooth"),
        (np.tile([2.0, 40.0], 26), "erratic"),
        (np.tile([0.0, 0.0, 3.0, 0.0], 13), "intermittent"),
        (np.tile([0.0, 0.0, 1.0, 0.0, 0.0, 30.0], 9), "lumpy"),
    ],
)
def test_syntetos_boylan_quadrants(series, expected):
    assert classify(*adi_cv2(series)) == expected


def test_adi_ignores_missing_weeks():
    s = np.array([5.0, np.nan, np.nan, 5.0, 5.0, 5.0])
    adi, cv2 = adi_cv2(s)
    assert adi == 1.0
    assert cv2 == 0.0
