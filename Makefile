.PHONY: setup test run-unsw run-cic run-all notebook clean-cache

setup:
	uv sync

test:
	uv run pytest tests/ -v

run-unsw:
	uv run python scripts/run_unsw.py

run-cic:
	uv run python scripts/run_cic.py

run-all: run-unsw run-cic

notebook:
	uv run jupyter notebook notebooks/analysis.ipynb

clean-cache:
	rm -f data/processed/*.npz
