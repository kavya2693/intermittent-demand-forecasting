.PHONY: data run quick lint test check

data:
	uv run python scripts/generate_data.py

# Full backtest: 1,500 SKUs, six origins, every model, plus the history-only ablation.
run:
	uv run demand-segments --refresh --jobs 4 --ablation 2>&1 | tee outputs/run_log.txt
	uv run python scripts/fold_summary.py | tee -a outputs/run_log.txt

# Same pipeline on 120 SKUs without ARIMA, for a quick check.
quick:
	uv run demand-segments --refresh --skus 120 --no-arima --data data/quick --out outputs/quick

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest -q

check: lint test
