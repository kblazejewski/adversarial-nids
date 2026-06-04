"""Testy dla src/defense.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.defense import make_adversarial_trainset


class TestMakeAdversarialTrainset:
    def test_double_size(self, trained_mlp, small_data):
        X, y = small_data
        X_aug, y_aug = make_adversarial_trainset(trained_mlp, X, y, eps=0.05)
        assert X_aug.shape[0] == 2 * len(X)
        assert y_aug.shape[0] == 2 * len(y)

    def test_feature_dim_preserved(self, trained_mlp, small_data):
        X, y = small_data
        X_aug, _ = make_adversarial_trainset(trained_mlp, X, y, eps=0.05)
        assert X_aug.shape[1] == X.shape[1]

    def test_first_half_is_original(self, trained_mlp, small_data):
        """Pierwsza połowa X_aug to oryginalny X_train bez zmian."""
        X, y = small_data
        X_aug, _ = make_adversarial_trainset(trained_mlp, X, y, eps=0.05)
        np.testing.assert_array_equal(X_aug[:len(X)], X)

    def test_labels_duplicated(self, trained_mlp, small_data):
        """y_aug = [y_train; y_train] — etykiety nie są zmieniane."""
        X, y = small_data
        _, y_aug = make_adversarial_trainset(trained_mlp, X, y, eps=0.05)
        np.testing.assert_array_equal(y_aug[:len(y)], y)
        np.testing.assert_array_equal(y_aug[len(y):], y)

    def test_values_in_unit_range(self, trained_mlp, small_data):
        X, y = small_data
        X_aug, _ = make_adversarial_trainset(trained_mlp, X, y, eps=0.1)
        assert X_aug.min() >= 0.0 - 1e-6
        assert X_aug.max() <= 1.0 + 1e-6

    def test_second_half_differs_from_first(self, trained_mlp, small_data):
        """Adwersarialna połowa powinna różnić się od oryginału (eps > 0)."""
        X, y = small_data
        X_aug, _ = make_adversarial_trainset(trained_mlp, X, y, eps=0.05)
        X_orig = X_aug[:len(X)]
        X_adv  = X_aug[len(X):]
        assert not np.array_equal(X_orig, X_adv)

    def test_eps_zero_second_half_equals_first(self, trained_mlp, small_data):
        """eps=0 → fgsm zwraca kopię, więc obie połowy są identyczne."""
        X, y = small_data
        X_aug, _ = make_adversarial_trainset(trained_mlp, X, y, eps=0.0)
        np.testing.assert_array_equal(X_aug[:len(X)], X_aug[len(X):])

    def test_dtype_float32(self, trained_mlp, small_data):
        X, y = small_data
        X_aug, _ = make_adversarial_trainset(trained_mlp, X, y, eps=0.05)
        assert X_aug.dtype == np.float32
