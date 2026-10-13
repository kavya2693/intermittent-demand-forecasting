"""Write the seeded weekly sales history to data/ (sales.parquet, skus.parquet)."""

from __future__ import annotations

import argparse
from pathlib import Path

from demand_segments.data import GeneratorConfig, write


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skus", type=int, default=1500)
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    paths = write(args.out, GeneratorConfig(n_skus=args.skus, n_weeks=args.weeks, seed=args.seed))
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
