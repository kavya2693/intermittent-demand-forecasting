"""One command: generate data if needed, run the walk-forward backtest, write the report."""

from __future__ import annotations

import argparse
from pathlib import Path

from demand_segments import backtest, report
from demand_segments.data import GeneratorConfig, load, write


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demand-segments", description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--refresh", action="store_true", help="regenerate the dataset first")
    ap.add_argument("--skus", type=int, default=1500, help="catalogue size when generating")
    ap.add_argument("--windows", type=int, default=6, help="walk-forward origins, 8 weeks apart")
    ap.add_argument("--no-arima", action="store_true", help="skip AutoARIMA, the slowest model")
    ap.add_argument("--jobs", type=int, default=1, help="processes for the statistical models")
    ap.add_argument("--rounds", type=int, default=400, help="LightGBM boosting rounds")
    args = ap.parse_args(argv)

    if args.refresh or not (args.data / "sales.parquet").exists():
        write(args.data, GeneratorConfig(n_skus=args.skus))
    sales, skus = load(args.data)
    print(f"{skus.shape[0]} SKUs, {sales['week'].nunique()} weeks", flush=True)

    res = backtest.run(
        sales,
        skus,
        n_windows=args.windows,
        include_arima=not args.no_arima,
        n_jobs=args.jobs,
        ml_rounds=args.rounds,
    )
    print(report.write(res, skus, args.out, args.windows))
    res.forecasts.to_parquet(args.out / "forecasts.parquet", index=False)


if __name__ == "__main__":
    main()
