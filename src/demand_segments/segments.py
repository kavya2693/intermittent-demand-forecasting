"""Syntetos-Boylan demand classification from ADI and CV squared.

ADI is the average number of weeks between sales; CV squared is the squared coefficient of
variation of the non-zero sale sizes. The 1.32 and 0.49 cut-offs come from Syntetos, Boylan
and Croston (2005), where they mark the regions in which Croston-type methods beat simple
exponential smoothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT = 1.32
CV2_CUT = 0.49
SEGMENTS = ("smooth", "erratic", "intermittent", "lumpy")


def adi_cv2(history: np.ndarray) -> tuple[float, float]:
    """ADI and CV squared of one series. Missing weeks are dropped, not treated as zero."""
    h = np.asarray(history, dtype=float)
    h = h[~np.isnan(h)]
    nz = h[h > 0]
    if len(nz) == 0:
        return float("inf"), 0.0
    adi = len(h) / len(nz)
    cv2 = float(np.var(nz) / np.mean(nz) ** 2) if len(nz) > 1 else 0.0
    return float(adi), cv2


def classify(adi: float, cv2: float) -> str:
    if adi < ADI_CUT:
        return "smooth" if cv2 < CV2_CUT else "erratic"
    return "intermittent" if cv2 < CV2_CUT else "lumpy"


def segment_table(sales: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Classify every SKU on the history strictly before ``before``."""
    hist = sales[sales["week"] < before]
    rows = []
    for sku, g in hist.groupby("sku_id", sort=True):
        adi, cv2 = adi_cv2(g["units"].to_numpy())
        rows.append({"sku_id": sku, "adi": adi, "cv2": cv2, "segment": classify(adi, cv2)})
    out = pd.DataFrame(rows)
    out["segment"] = pd.Categorical(out["segment"], categories=SEGMENTS)
    return out
