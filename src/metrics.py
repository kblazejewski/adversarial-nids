import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Oblicza accuracy, precision, recall, f1, auc dla dowolnego modelu
    z interfejsem .predict / .predict_proba.
    """
    preds = model.predict(X)
    proba = model.predict_proba(X)
    # RF zwraca (n, 2) — bierzemy kolumnę klasy 1
    if proba.ndim == 2:
        proba = proba[:, 1]

    return {
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "f1":        f1_score(y, preds, zero_division=0),
        "auc":       roc_auc_score(y, proba),
    }


def asr(model, X_clean: np.ndarray, X_adv: np.ndarray, y: np.ndarray) -> float:
    """
    Attack Success Rate.

    Z ataków (y == 1) poprawnie wykrytych na X_clean,
    jaki odsetek model przeoczył po zastąpieniu przez X_adv.

    Zwraca NaN jeśli nie ma żadnych poprawnie wykrytych ataków
    (np. eps == 0 albo model nie wykrywa nic).
    """
    attack_mask = (y == 1)
    preds_clean = model.predict(X_clean)
    correctly_detected = attack_mask & (preds_clean == 1)

    if correctly_detected.sum() == 0:
        return float("nan")

    preds_adv = model.predict(X_adv)
    evaded = correctly_detected & (preds_adv == 0)
    return evaded.sum() / correctly_detected.sum()


def timed_predict(model, X: np.ndarray):
    """
    Zwraca (predictions, time_ms) — czas wnioskowania dla całego X w ms.
    """
    t0 = time.perf_counter()
    preds = model.predict(X)
    t1 = time.perf_counter()
    return preds, (t1 - t0) * 1000.0
