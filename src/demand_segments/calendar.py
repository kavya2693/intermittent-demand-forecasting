"""Weekly retail calendar: the events that move demand and are known in advance."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

START = pd.Timestamp("2023-01-02")  # a Monday; every week is labelled by its Monday


def week_starts(n_weeks: int, start: pd.Timestamp = START) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_weeks, freq="W-MON")


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def black_friday(year: int) -> dt.date:
    nov1 = dt.date(year, 11, 1)
    first_thu = nov1 + dt.timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + dt.timedelta(days=22)


def _contains(week_start: pd.Timestamp, day: dt.date) -> bool:
    d = week_start.date()
    return d <= day < d + dt.timedelta(days=7)


def calendar_frame(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per week with flags for the events a planner would put in a calendar."""
    rows = []
    for w in weeks:
        y = w.year
        years = {y - 1, y, y + 1}
        rows.append(
            {
                "week": w,
                "week_of_year": int(w.isocalendar().week),
                "month": w.month,
                "black_friday": any(_contains(w, black_friday(yy)) for yy in years),
                "christmas": any(_contains(w, dt.date(yy, 12, 25)) for yy in years)
                or any(_contains(w, dt.date(yy, 12, 18)) for yy in years),
                "new_year": any(_contains(w, dt.date(yy, 1, 1)) for yy in years),
                "easter": any(_contains(w, easter(yy) - dt.timedelta(days=2)) for yy in years),
                "back_to_school": w.month == 8 and w.day >= 7,
            }
        )
    cal = pd.DataFrame(rows)
    for col in ["black_friday", "christmas", "new_year", "easter", "back_to_school"]:
        cal[col] = cal[col].astype(np.int8)
    return cal
