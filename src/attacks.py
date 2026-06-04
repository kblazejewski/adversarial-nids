import numpy as np
from src.models import TorchClassifier


def fgsm(clf: TorchClassifier, X: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    """
    Fast Gradient Sign Method (jednostkowy krok).

    Parameters
    ----------
    clf : TorchClassifier
    X   : float32 (n, d), wszystkie wartości w [0, 1]
    y   : int {0, 1}, prawdziwe etykiety
    eps : siła perturbacji (float, np. 0.05)

    Returns
    -------
    X_adv : float32 (n, d), wartości w [0, 1]
    """
    if eps == 0.0:
        return X.copy()

    grad = clf.grad_wrt_input(X, y)          # kształt jak X
    X_adv = X + eps * np.sign(grad)
    return np.clip(X_adv, 0.0, 1.0).astype(np.float32)
