# ADR-0001: Segment by ADI/CV², compare every model on the same windows, pick champions on eight-week error

## Status
Accepted

## Context
A catalogue of about 1,500 SKUs sold weekly to trade and retail customers mixes lines that
sell every week with lines that sell a few units a few times a year. One model family rarely
wins everywhere, and the usual error measures behave differently across the two kinds of
series: MAPE is undefined on zero-sale weeks, and weekly WAPE rewards forecasting zero for a
SKU whose median week has no sale. The team needs to know, per demand pattern, which method
to run and how far it beats the seasonal naive forecast it would otherwise fall back on.

## Decision
- Classify each SKU by ADI and CV² (Syntetos–Boylan cut-offs 1.32 and 0.49) on history before
  the first backtest origin, and keep that label fixed for scoring.
- Run every model through the same six walk-forward origins, eight weeks each, refitting at
  each origin: seasonal naive (52), 13-week moving average, ETS, ARIMA, Croston (optimised),
  TSB, and a global LightGBM model.
- LightGBM is one model across all SKUs with a direct horizon: features are computed at the
  origin from a registry, and the target is sales divided by the SKU's trailing level, with a
  Tweedie loss weighted by that level.
- Report weekly WAPE, eight-week WAPE, MAPE with the share of weeks where it is defined, and
  bias, per segment. A zero forecast is scored as a reference row and never selected.
- Pick each segment's champion on the first three origins by eight-week WAPE, then score it
  on the last three, so the selection is not graded on the windows that chose it.

## Alternatives rejected
- **LSTM.** Adds a deep-learning dependency and minutes of CPU per fold for a series length
  (156 weeks) where gradient boosting with lag features is the stronger published baseline.
  Left as a possible extension, not built.
- **Seasonal ETS / seasonal ARIMA at m=52.** statsforecast ETS does not fit a 52-week season,
  and seasonal ARIMA at m=52 is too slow to refit 1,500 series six times. The seasonal naive
  baseline carries the yearly pattern; ETS and ARIMA are fitted without a season.
- **Recursive LightGBM forecasting.** Feeding predictions back as lags compounds error on
  intermittent series and makes the leakage test harder to state. Direct horizon was chosen.
- **Selecting on weekly WAPE.** On the intermittent and lumpy segments it would select the zero
  forecast, which is useless for replenishment.

## Consequences
- The stat models do not see promotions or holidays; LightGBM does. Part of its margin on
  smooth and erratic SKUs is that information, not the algorithm.
- The full run takes around twenty minutes on a laptop CPU, mostly ARIMA. CI runs the same
  pipeline on 120 SKUs without ARIMA.
