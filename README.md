# adversarial-nids

NIDS adversarial robustness study: Random Forest vs MLP under FGSM attack + adversarial training defense.

Datasets: UNSW-NB15 + CIC-IDS2017. Stack: Python + uv + PyTorch + scikit-learn.

## Setup

After cloning, install all dependencies (exact versions from `uv.lock`):

```bash
make setup
```

## Usage

| Command | Description |
| ------- | ----------- |
| `make setup` | Install dependencies via `uv sync` |
| `make test` | Run test suite (68 tests) |
| `make run-unsw` | Run experiment on UNSW-NB15 → `results/results_unsw.csv` |
| `make run-cic` | Run experiment on CIC-IDS2017 → `results/results_cic.csv` |
| `make run-all` | Run both experiments sequentially |
| `make notebook` | Open analysis notebook |
| `make clean-cache` | Delete processed data cache (forces re-preprocessing) |

## Project structure

```text
config.py          — hyperparameters and paths (do not edit)
src/
  datasets.py      — data loading and preprocessing
  models.py        — RandomForest, MLP, TorchClassifier
  attacks.py       — FGSM attack
  defense.py       — adversarial training (static augmentation)
  metrics.py       — evaluate, ASR, timed_predict
  experiment.py    — full experiment matrix
scripts/
  run_unsw.py      — UNSW-NB15 experiment entry point
  run_cic.py       — CIC-IDS2017 experiment entry point
notebooks/
  analysis.ipynb   — results analysis and plots
tests/             — pytest test suite
```

## Data

Place raw datasets in:

- `data/raw/unsw/` — UNSW-NB15 CSV files (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`)
- `data/raw/cic/`  — CIC-IDS2017 CSV files (all daily `*.pcap_ISCX.csv` files)

Processed data is cached in `data/processed/` after first run.
