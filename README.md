# intermittent-demand-forecasting

[![ci](https://github.com/kavya2693/intermittent-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/kavya2693/intermittent-demand-forecasting/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Which forecasting method to run for which SKU, when nearly half the catalogue sells nothing in
most weeks: segment by demand pattern, backtest every method on the same windows, and report
where the machine-learning model does not win.

## The problem

A demand planner with 1,500 SKUs runs one forecasting method across the whole catalogue
because there is no time to tune 1,500 series. Half of that catalogue is not the kind of
series the method was built for. Floor cleaner sells 40 units every week; a replacement pump
seal sells two units in March, nothing until July, then six. A forecast that works for the
first produces nonsense for the second, and the planner finds out when the seal is out of
stock for a trade customer or when a pallet of it has sat in the warehouse for a year.

The usual scorecards make this worse. MAPE cannot be computed in a week with zero sales, so
it quietly scores only the weeks where something sold. Weekly WAPE is defined, but on a SKU
whose typical week has no sale, the forecast that minimises it is zero, which is the one
forecast nobody can replenish from. A team can pick a model on these numbers and ship
something that looks accurate and stocks nothing.

## The approach

Each SKU is classified by ADI (average weeks between sales) and CV² (variability of the sale
sizes) into the four Syntetos–Boylan segments: smooth, erratic, intermittent, lumpy. Every
method then forecasts the same eight-week windows from six walk-forward origins over the last
48 weeks: seasonal naive (the baseline), a 13-week moving average, ETS, Croston, TSB, ARIMA,
and one global LightGBM model trained across all SKUs on features built automatically from a
registry. Results are reported per segment on four measures: weekly WAPE, WAPE on each SKU's
eight-week total, MAPE with the share of weeks it actually covers, and bias.

The key decision was to choose each segment's model on eight-week WAPE, picked on the first
three origins and graded on the last three. Eight weeks is roughly what a replenishment order
has to cover, and on that measure a zero forecast scores 100%, so it can never win. On weekly
WAPE it beats every model on the intermittent and lumpy segments.

## The result

1,500 SKUs, 156 weeks, six origins, 8-week horizon (`outputs/run_report.md`). Lower WAPE is
better; bias is total forecast over total actual, minus one.

| Segment (SKUs) | Model | Weekly WAPE | 8-week WAPE | MAPE (weeks covered) | Bias |
|---|---|---|---|---|---|
| smooth (325) | seasonal naive | 36.8% | 19.4% | 67.6% (99%) | -3.0% |
| | best statistical: ARIMA | 35.5% | 25.0% | 61.1% | -1.2% |
| | LightGBM | **27.1%** | **15.2%** | 50.5% | -1.2% |
| erratic (346) | seasonal naive | 73.6% | 35.5% | 176.1% (95%) | +1.3% |
| | best statistical: ARIMA | 58.5% | 34.8% | 164.2% | +1.0% |
| | LightGBM | **51.4%** | **25.6%** | 138.3% | -4.0% |
| intermittent (397) | seasonal naive | 133.4% | 64.2% | 82.1% (35%) | -6.5% |
| | best statistical: ARIMA | 121.9% | 55.1% | 55.1% | -4.2% |
| | LightGBM | 121.1% | **51.4%** | 52.9% | +0.5% |
| | zero forecast | **100.0%** | 100.0% | 100.0% | -100.0% |
| lumpy (432) | seasonal naive | 156.8% | 80.3% | 123.4% (34%) | -1.5% |
| | best statistical: TSB | 136.2% | 72.3% | 76.9% | 0.0% |
| | LightGBM | 129.0% | **63.6%** | 70.4% | -8.9% |
| | zero forecast | **100.0%** | 100.0% | 100.0% | -100.0% |
| all | seasonal naive | 58.1% | 29.1% | 116.7% (62%) | -1.3% |
| all | ARIMA (best statistical overall) | 50.2% | 31.3% | 98.4% | -0.3% |
| all | LightGBM | **42.5%** | **21.9%** | 84.9% | -2.7% |

Champion picked on origins 1–3, scored on origins 4–6 (`outputs/champions.csv`): LightGBM was
picked in all four segments, and cut weekly WAPE against seasonal naive by 25.8% (smooth),
30.7% (erratic), 8.5% (intermittent) and 17.8% (lumpy).

**Where the ML model does not win.** On intermittent SKUs LightGBM is 0.8 points of weekly
WAPE better than ARIMA in aggregate, but it beats ARIMA on only 45.6% of those SKUs
individually, and on only 2 of the 6 origins (`outputs/lightgbm_vs_best_statistical.csv`,
`outputs/fold_summary.csv`). Its intermittent margin exists only on the eight-week total. On
lumpy SKUs it under-forecasts: bias is -8.9% across all origins and worse than -10% on two of
six origins (-15.7% at its worst), so the ±10% bias line is not held there.

**The experiment that could have embarrassed the result.** The statistical models see only
sales history; LightGBM also sees the calendar, the promotion plan and price. Refitting
LightGBM with those inputs removed (`LightGBM-history-only`) raises total weekly WAPE from
42.5% to 45.1%, and on the intermittent segment it then loses to ARIMA, ETS and TSB
(122.9% against 121.9%, 122.2% and 122.8%). It
still beats every statistical model on smooth, erratic and lumpy SKUs. So about a sixth of the
margin over seasonal naive is information the statistical models never had, and on
intermittent SKUs it is the whole weekly-WAPE margin (on the eight-week total the
history-only model still beats ARIMA, 53.7% vs 55.1%).

**The segmentation is noisier than the textbook picture.** The generator gives each SKU an
intended pattern and the classifier never sees it. 125 of 448 SKUs built as smooth are
classified erratic, because strong seasonality inflates CV² of the sale sizes, and 145 of 525
built as intermittent land in lumpy because of occasional bulk orders
(`outputs/segment_vs_generator_intent.csv`).

**The two WAPEs disagree on smooth SKUs too.** ARIMA beats seasonal naive week by week
(35.5% vs 36.8%) but loses on the eight-week total (25.0% vs 19.4%): without a yearly season
it misses the level of the coming two months even when its weekly shape is closer.

## How it works

```
data/sales.parquet ──► segments.py   ADI / CV² on history before the first origin ─► segment per SKU
        │
        ├────────────► forecast_stats.py   statsforecast cross-validation, refit at each origin
        │                 seasonal naive(52) · moving average(13) · ETS · Croston · TSB · ARIMA
        │
        └─► features.py  registry of origin features (levels, zero share, weeks since sale, CV²)
                         and target-week features (calendar, promotion plan, price, sales a year ago)
                 │
                 └─► forecast_ml.py   one LightGBM over all SKUs, direct 1–8 week horizon,
                                      target = sales / SKU level, Tweedie loss weighted by level
        ▼
backtest.py   same six origins for every model ─► metrics.py (WAPE, 8-week WAPE, MAPE + coverage, bias)
        ▼
report.py     per segment tables, champion picked early and scored late, LightGBM vs best statistical
```

The global model trains on every past origin whose eight target weeks end before the forecast
origin, stepping two weeks at a time, so at the first origin it has about 1,500 × 25 × 8 rows.
Feature functions only receive the history up to the origin; the calendar, promotion plan and
price for the target week are allowed because a planner has them in advance. Design decisions
and rejected alternatives (LSTM, seasonal ETS at m=52, recursive forecasting, weekly-WAPE
selection) are in [docs/adr/0001-segment-then-compare-on-shared-windows.md](docs/adr/0001-segment-then-compare-on-shared-windows.md).

## Running it

```bash
uv sync
make run      # 1,500 SKUs, six origins, every model, plus the history-only ablation (about 12 min)
make quick    # 120 SKUs, same pipeline, about a minute
make check    # ruff, format, mypy, pytest
docker build -t demand-segments . && docker run --rm demand-segments
```

`make data` writes the dataset alone. Tables land in `outputs/`: `scores_by_segment.csv`,
`scores_by_fold.csv`, `champions.csv`, `lightgbm_vs_best_statistical.csv`, `fold_summary.csv`,
`feature_importance.csv`, `sku_segments.csv` and `run_report.md`. Nothing needs an API key or
a model server.

## The data

The dataset is generated by the seeded script `scripts/generate_data.py` (`make data`) and is
sized to the production problem, 1,500 SKUs of weekly sales over three years; the production
data it stands in for is proprietary. It mixes trade (B2B) and retail (B2C) channels across
eight categories, with category-specific yearly seasonality, Black Friday, Christmas, Easter
and back-to-school effects, planned promotions with a post-promotion dip, step price
increases, bulk-order outliers, stock-out weeks that censor sales, product launches partway
through the history, a few SKUs being phased out, and short feed gaps recorded as missing
rather than zero. Its limit is that the patterns are the ones written into the generator:
real catalogues have substitution between SKUs, promotions that are not on the plan, and
demand shifts the generator does not produce.

## Tests

`tests/test_features.py` rewrites every sale on or after the forecast origin with random
numbers and checks that no feature changes, then proves the check has teeth by registering a
feature that peeks one week ahead and confirming it is caught. Any feature added to the
registry is covered automatically.

`tests/test_metrics.py` pins the measures to hand-worked numbers: missing weeks are skipped
rather than scored as zero sales, bias and WAPE disagree when errors cancel, MAPE reports how
few weeks it covers, and on a SKU selling four units every four weeks weekly WAPE prefers a
zero forecast while eight-week WAPE scores the steady forecast as perfect.
`tests/test_backtest.py` checks that every model forecasts exactly the same SKU-weeks over
non-overlapping windows and that neither the zero forecast nor the ablation can be picked as
a champion.

## Known limitations

- **The biggest gap: point forecasts only.** Replenishment for intermittent SKUs needs a
  quantile or a demand distribution over lead time; no service-level or inventory simulation
  is done, so the comparison is on error, not on fill rate or stock held.
- ETS and ARIMA are fitted without a 52-week season; only seasonal naive and LightGBM carry
  the yearly pattern, so the statistical side is weaker on seasonal smooth SKUs than a tuned
  seasonal model would be.
- LightGBM under-forecasts lumpy SKUs (bias -8.9%, -15.7% on its worst origin). A bias
  correction per segment was not tried.
- Segments are fixed at the first origin. SKUs that change pattern during the year are scored
  in their original segment.
- LSTM from the original brief was not built; see the ADR.
- The data is generated, so the numbers show the method behaving as expected on realistic
  patterns, not a measured business impact.

## Insights and learning

I picked this problem because the hardest SKUs in a forecasting portfolio are the ones that
almost never sell, and the metrics most teams report are least trustworthy exactly there. I
expected the interesting result to be which model wins on intermittent demand. The more
useful result was that the scorecard itself picks the wrong answer: weekly WAPE crowns a
forecast of zero on 829 of the 1,500 SKUs. Moving model selection to the eight-week total was
a bigger change than any model choice.

What surprised me was how much of the ML margin is information rather than algorithm. Taking
the calendar and promotion plan away from LightGBM costs it about a sixth of its gain over
seasonal naive, and its whole weekly-WAPE edge on intermittent SKUs. If I were presenting this to a
planning team, the honest message is that LightGBM is worth running for smooth and erratic
SKUs, while for intermittent SKUs ARIMA and ETS are level with it week by week and the case for
LightGBM rests on a four-point lead on the eight-week total (51.4% vs 55.1%), which is a smaller and cheaper
thing to replace with a tuned statistical model.

What I got wrong first: I sized the run for a dedicated CPU, and the first full run with
ARIMA starved on a shared machine and had to be stopped; it now runs the statistical models
across four worker processes. Next
I would add quantile forecasts and a lead-time inventory simulation, so the comparison is on
fill rate and stock held rather than on error alone, and try a per-segment bias correction
for lumpy SKUs.

## License

MIT
