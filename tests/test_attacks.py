"""Testy dla src/attacks.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.attacks import fgsm


class TestFGSM:
    def test_eps_zero_returns_copy(self, trained_mlp, small_data):
        X, y = small_data
        X_adv = fgsm(trained_mlp, X, y, eps=0.0)
        np.testing.assert_array_equal(X_adv, X)
        # musi być kopia, nie ten sam obiekt
        assert X_adv is not X

    def test_output_shape(self, trained_mlp, small_data):
        X, y = small_data
        X_adv = fgsm(trained_mlp, X, y, eps=0.05)
        assert X_adv.shape == X.shape

    def test_values_in_unit_range(self, trained_mlp, small_data):
        X, y = small_data
        for eps in [0.01, 0.05, 0.1, 0.15]:
            X_adv = fgsm(trained_mlp, X, y, eps=eps)
            assert X_adv.min() >= 0.0 - 1e-6, f"eps={eps}: min={X_adv.min()}"
            assert X_adv.max() <= 1.0 + 1e-6, f"eps={eps}: max={X_adv.max()}"

    def test_dtype_float32(self, trained_mlp, small_data):
        X, y = small_data
        X_adv = fgsm(trained_mlp, X, y, eps=0.05)
        assert X_adv.dtype == np.float32

    def test_perturbs_input(self, trained_mlp, small_data):
        """eps > 0 powinno zmieniać przynajmniej część próbek."""
        X, y = small_data
        X_adv = fgsm(trained_mlp, X, y, eps=0.05)
        assert not np.array_equal(X_adv, X)

    def test_perturbation_bounded_by_eps(self, trained_mlp, small_data):
        """Każda cecha zmieniona o co najwyżej eps (norma L∞)."""
        X, y = small_data
        eps = 0.07
        X_adv = fgsm(trained_mlp, X, y, eps=eps)
        # L∞ odległość przed clippingiem może być = eps,
        # po clippingu może być mniejsza przy brzegach [0,1]
        diff = np.abs(X_adv - X)
        assert diff.max() <= eps + 1e-5

    def test_larger_eps_more_perturbation(self, trained_mlp, small_data):
        """Większe eps → większa średnia odległość L∞."""
        X, y = small_data
        d_small = np.abs(fgsm(trained_mlp, X, y, eps=0.01) - X).mean()
        d_large = np.abs(fgsm(trained_mlp, X, y, eps=0.10) - X).mean()
        assert d_large > d_small

    def test_clip_at_boundary(self, trained_mlp):
        """Próbki przy granicy 0/1 nie mogą wyjść poza zakres."""
        X_edge = np.zeros((10, 10), dtype=np.float32)   # wszystko = 0
        y = np.ones(10, dtype=int)
        X_adv = fgsm(trained_mlp, X_edge, y, eps=0.5)
        assert X_adv.min() >= 0.0 - 1e-6
        assert X_adv.max() <= 1.0 + 1e-6
