import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from src.experiment import run_experiment
from config import PATHS

OUTPUT = os.path.join(PATHS["results"], "results_unsw.csv")

if __name__ == "__main__":
    df = run_experiment("unsw")
    os.makedirs(PATHS["results"], exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Zapisano {len(df)} wierszy do {OUTPUT}")
    print(df.to_string(index=False))
