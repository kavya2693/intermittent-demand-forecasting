# Backtest report

1500 SKUs, 6 walk-forward origins, 8-week horizon. Champions are picked on the first 3 origins and scored on the last 3.

## SKUs per segment

| segment | skus |
|---|---|
| lumpy | 432 |
| intermittent | 397 |
| erratic | 346 |
| smooth | 325 |

## All origins, by segment and model

| segment | model | wape | wape_8wk | mape | mape_coverage | bias |
|---|---|---|---|---|---|---|
| all | LightGBM | 42.5% | 21.9% | 84.9% | 62.5% | -2.7% |
| all | LightGBM-history-only | 45.1% | 24.8% | 87.7% | 62.5% | -3.8% |
| all | AutoARIMA | 50.2% | 31.3% | 98.4% | 62.5% | -0.3% |
| all | CrostonOptimized | 51.6% | 33.2% | 97.0% | 62.5% | -0.2% |
| all | TSB | 52.1% | 34.2% | 97.6% | 62.5% | -0.8% |
| all | AutoETS | 53.5% | 35.1% | 98.4% | 62.5% | 1.0% |
| all | WindowAverage | 54.7% | 37.7% | 103.0% | 62.5% | -0.5% |
| all | SeasonalNaive | 58.1% | 29.1% | 116.7% | 62.5% | -1.3% |
| all | ZeroForecast | 100.0% | 100.0% | 100.0% | 62.5% | -100.0% |
| erratic | LightGBM | 51.4% | 25.6% | 138.3% | 95.2% | -4.0% |
| erratic | LightGBM-history-only | 53.7% | 28.6% | 142.9% | 95.2% | -5.1% |
| erratic | AutoARIMA | 58.5% | 34.8% | 164.2% | 95.2% | 1.0% |
| erratic | CrostonOptimized | 59.1% | 36.5% | 157.8% | 95.2% | -0.2% |
| erratic | TSB | 59.5% | 37.4% | 156.0% | 95.2% | -0.5% |
| erratic | WindowAverage | 61.7% | 40.2% | 162.9% | 95.2% | -0.1% |
| erratic | AutoETS | 61.7% | 38.5% | 161.3% | 95.2% | 0.3% |
| erratic | SeasonalNaive | 73.6% | 35.5% | 176.1% | 95.2% | 1.3% |
| erratic | ZeroForecast | 100.0% | 100.0% | 100.0% | 95.2% | -100.0% |
| intermittent | ZeroForecast | 100.0% | 100.0% | 100.0% | 34.9% | -100.0% |
| intermittent | LightGBM | 121.1% | 51.4% | 52.9% | 34.9% | 0.5% |
| intermittent | AutoARIMA | 121.9% | 55.1% | 55.1% | 34.9% | -4.2% |
| intermittent | AutoETS | 122.2% | 56.5% | 56.2% | 34.9% | -3.6% |
| intermittent | TSB | 122.8% | 57.8% | 57.0% | 34.9% | -3.0% |
| intermittent | LightGBM-history-only | 122.9% | 53.7% | 53.0% | 34.9% | -0.1% |
| intermittent | WindowAverage | 123.9% | 62.9% | 58.9% | 34.9% | -2.3% |
| intermittent | CrostonOptimized | 124.4% | 58.1% | 55.3% | 34.9% | 1.2% |
| intermittent | SeasonalNaive | 133.4% | 64.2% | 82.1% | 34.9% | -6.5% |
| lumpy | ZeroForecast | 100.0% | 100.0% | 100.0% | 34.3% | -100.0% |
| lumpy | LightGBM | 129.0% | 63.6% | 70.4% | 34.3% | -8.9% |
| lumpy | LightGBM-history-only | 131.2% | 66.3% | 71.1% | 34.3% | -7.9% |
| lumpy | TSB | 136.2% | 72.3% | 76.9% | 34.3% | -0.0% |
| lumpy | AutoARIMA | 136.6% | 69.5% | 73.0% | 34.3% | 0.9% |
| lumpy | AutoETS | 137.2% | 71.6% | 75.0% | 34.3% | 1.8% |
| lumpy | WindowAverage | 137.2% | 78.0% | 81.2% | 34.3% | 0.0% |
| lumpy | CrostonOptimized | 140.1% | 73.8% | 76.2% | 34.3% | 6.7% |
| lumpy | SeasonalNaive | 156.8% | 80.3% | 123.4% | 34.3% | -1.5% |
| smooth | LightGBM | 27.1% | 15.2% | 50.5% | 98.8% | -1.2% |
| smooth | LightGBM-history-only | 30.0% | 18.0% | 53.8% | 98.8% | -2.6% |
| smooth | AutoARIMA | 35.5% | 25.0% | 61.1% | 98.8% | -1.2% |
| smooth | SeasonalNaive | 36.8% | 19.4% | 67.6% | 98.8% | -3.0% |
| smooth | CrostonOptimized | 37.2% | 27.0% | 62.3% | 98.8% | -0.8% |
| smooth | TSB | 38.2% | 28.2% | 64.7% | 98.8% | -1.1% |
| smooth | AutoETS | 39.2% | 29.1% | 62.8% | 98.8% | 1.7% |
| smooth | WindowAverage | 41.4% | 32.2% | 70.5% | 98.8% | -0.8% |
| smooth | ZeroForecast | 100.0% | 100.0% | 100.0% | 98.8% | -100.0% |

## Segment champion, chosen early and scored late

| segment | champion | champion_wape | champion_wape_8wk | champion_bias | seasonal_naive_wape | seasonal_naive_wape_8wk | lightgbm_wape | wape_reduction_vs_seasonal_naive |
|---|---|---|---|---|---|---|---|---|
| smooth | LightGBM | 27.9% | 16.4% | -2.5% | 37.6% | 20.7% | 27.9% | 25.8% |
| erratic | LightGBM | 51.9% | 25.5% | -4.2% | 74.8% | 36.0% | 51.9% | 30.7% |
| intermittent | LightGBM | 120.2% | 51.2% | 0.5% | 131.4% | 64.2% | 120.2% | 8.5% |
| lumpy | LightGBM | 129.1% | 62.4% | -7.5% | 157.1% | 82.2% | 129.1% | 17.8% |

## LightGBM against the best statistical model in each segment

| segment | best_statistical | best_statistical_wape | lightgbm_wape | lightgbm_gap | best_statistical_wape_8wk | lightgbm_wape_8wk | skus_scored | lightgbm_sku_win_share |
|---|---|---|---|---|---|---|---|---|
| erratic | AutoARIMA | 58.5% | 51.4% | -7.2% | 34.8% | 25.6% | 346 | 78.6% |
| intermittent | AutoARIMA | 121.9% | 121.1% | -0.8% | 55.1% | 51.4% | 397 | 45.6% |
| lumpy | TSB | 136.2% | 129.0% | -7.2% | 72.3% | 63.6% | 432 | 61.1% |
| smooth | AutoARIMA | 35.5% | 27.1% | -8.4% | 25.0% | 15.2% | 325 | 88.6% |

## Run time (seconds)

- statistical models: 329
- LightGBM: 256
- LightGBM-history-only: 144
