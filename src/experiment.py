import numpy as np
import pandas as pd

from config import SEED, EPS_GRID
from src.datasets import load_dataset
from src.models import train_rf, train_mlp
from src.attacks import fgsm
from src.defense import make_adversarial_trainset
from src.metrics import evaluate, asr, timed_predict


def run_experiment(dataset_name: str) -> pd.DataFrame:
    """
    Pełna macierz eksperymentów dla jednego zbioru:
      {rf, mlp} × {none, advtrain} × EPS_GRID

    Returns
    -------
    pd.DataFrame z kolumnami:
      dataset, model, defense, eps,
      accuracy, precision, recall, f1, auc, asr,
      infer_time_ms, n_test, seed
    """
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, seed=SEED)
    n_test = len(y_test)
    rows = []

    # ---- trening modeli bazowych (bez obrony) ----
    print(f"[{dataset_name}] Trenuję RF...")
    rf_base  = train_rf(X_train, y_train, seed=SEED)
    print(f"[{dataset_name}] Trenuję MLP...")
    mlp_base = train_mlp(X_train, y_train, seed=SEED)

    # ---- adversarial training: powiększony zbiór ----
    # używamy epsilon z pierwszego niezerowego punktu siatki do augmentacji
    aug_eps = [e for e in EPS_GRID if e > 0][0]
    print(f"[{dataset_name}] Buduję augmented trainset (eps={aug_eps})...")
    X_aug, y_aug = make_adversarial_trainset(mlp_base, X_train, y_train, eps=aug_eps)

    print(f"[{dataset_name}] Trenuję RF+advtrain...")
    rf_adv  = train_rf(X_aug, y_aug, seed=SEED)
    print(f"[{dataset_name}] Trenuję MLP+advtrain...")
    mlp_adv = train_mlp(X_aug, y_aug, seed=SEED)

    model_pairs = [
        ("rf",  "none",     rf_base),
        ("mlp", "none",     mlp_base),
        ("rf",  "advtrain", rf_adv),
        ("mlp", "advtrain", mlp_adv),
    ]

    # jeden surrogate (mlp_base) dla wszystkich wariantów —
    # fair comparison: atakujemy tymi samymi przykładami, różni się tylko trening
    surrogates = {
        ("rf",  "none"):     mlp_base,
        ("mlp", "none"):     mlp_base,
        ("rf",  "advtrain"): mlp_base,
        ("mlp", "advtrain"): mlp_base,
    }

    for model_name, defense, model in model_pairs:
        surrogate = surrogates[(model_name, defense)]

        for eps in EPS_GRID:
            if eps == 0.0:
                X_eval = X_test
            else:
                X_eval = fgsm(surrogate, X_test, y_test, eps)

            metrics = evaluate(model, X_eval, y_test)
            _, infer_ms = timed_predict(model, X_eval)

            if eps == 0.0:
                attack_success = float("nan")
            else:
                X_clean_for_asr = X_test
                attack_success = asr(model, X_clean_for_asr, X_eval, y_test)

            rows.append({
                "dataset":       dataset_name,
                "model":         model_name,
                "defense":       defense,
                "eps":           eps,
                "accuracy":      round(metrics["accuracy"],  4),
                "precision":     round(metrics["precision"], 4),
                "recall":        round(metrics["recall"],    4),
                "f1":            round(metrics["f1"],        4),
                "auc":           round(metrics["auc"],       4),
                "asr":           round(attack_success, 4) if not np.isnan(attack_success) else float("nan"),
                "infer_time_ms": round(infer_ms, 2),
                "n_test":        n_test,
                "seed":          SEED,
            })

    return pd.DataFrame(rows)
