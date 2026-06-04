import numpy as np
from src.models import TorchClassifier
from src.attacks import fgsm


def make_adversarial_trainset(surrogate: TorchClassifier,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               eps: float):
    """
    Tworzy powiększony zbiór treningowy: oryginał + próbki po ataku FGSM.

    Parameters
    ----------
    surrogate : nauczony TorchClassifier (MLP) użyty do generowania perturbacji
    X_train   : float32 (n, d)
    y_train   : int {0, 1}
    eps       : siła perturbacji

    Returns
    -------
    X_aug, y_aug : oryginał + wygenerowane próbki adwersarialne (sklejone)
    """
    X_adv = fgsm(surrogate, X_train, y_train, eps)
    X_aug = np.concatenate([X_train, X_adv], axis=0)
    y_aug = np.concatenate([y_train, y_train], axis=0)
    return X_aug, y_aug
