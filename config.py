SEED = 42
EPS_GRID = [0.0, 0.01, 0.05, 0.1, 0.15]   # 0.0 = clean baseline
MLP_HIDDEN = [128, 64]
MLP_EPOCHS = 25
MLP_BATCH = 256
MLP_LR = 1e-3
RF_TREES = 100
RF_CLASS_WEIGHT = "balanced"
PATHS = {
    "unsw_raw": "data/raw/unsw",
    "cic_raw": "data/raw/cic",
    "processed": "data/processed",
    "results": "results",
}
