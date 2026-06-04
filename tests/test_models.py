"""Testy dla src/models.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from src.models import MLP, TorchClassifier, train_rf, train_mlp

_N, _D = 40, 10


def _xy(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (_N, _D)).astype(np.float32)
    y = rng.integers(0, 2, _N)
    return X, y


# ---------------------------------------------------------------------------
# MLP (nn.Module)
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self):
        model = MLP(input_dim=_D, hidden=[16, 8])
        x = torch.rand(20, _D)
        out = model(x)
        assert out.shape == (20,)

    def test_single_sample(self):
        model = MLP(input_dim=_D, hidden=[16])
        x = torch.rand(1, _D)
        out = model(x)
        assert out.shape == (1,)

    def test_custom_hidden(self):
        model = MLP(input_dim=5, hidden=[32, 16, 8])
        x = torch.rand(10, 5)
        assert model(x).shape == (10,)

    def test_parameters_exist(self):
        model = MLP(input_dim=_D, hidden=[16, 8])
        params = list(model.parameters())
        assert len(params) > 0


# ---------------------------------------------------------------------------
# TorchClassifier
# ---------------------------------------------------------------------------

class TestTorchClassifier:
    def test_predict_shape(self, trained_mlp, small_data):
        X, _ = small_data
        preds = trained_mlp.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_binary(self, trained_mlp, small_data):
        X, _ = small_data
        preds = trained_mlp.predict(X)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_proba_shape(self, trained_mlp, small_data):
        X, _ = small_data
        proba = trained_mlp.predict_proba(X)
        assert proba.shape == (len(X),)

    def test_predict_proba_range(self, trained_mlp, small_data):
        X, _ = small_data
        proba = trained_mlp.predict_proba(X)
        assert proba.min() >= 0.0 - 1e-6
        assert proba.max() <= 1.0 + 1e-6

    def test_predict_proba_dtype(self, trained_mlp, small_data):
        X, _ = small_data
        proba = trained_mlp.predict_proba(X)
        assert proba.dtype == np.float32

    def test_predict_consistent_with_proba(self, trained_mlp, small_data):
        """predict(X) == (predict_proba(X) >= 0.5)."""
        X, _ = small_data
        preds = trained_mlp.predict(X)
        proba = trained_mlp.predict_proba(X)
        np.testing.assert_array_equal(preds, (proba >= 0.5).astype(int))

    def test_grad_wrt_input_shape(self, trained_mlp, small_data):
        X, y = small_data
        grad = trained_mlp.grad_wrt_input(X, y)
        assert grad.shape == X.shape

    def test_grad_wrt_input_not_all_zero(self, trained_mlp, small_data):
        X, y = small_data
        grad = trained_mlp.grad_wrt_input(X, y)
        assert not np.all(grad == 0)

    def test_grad_wrt_input_dtype(self, trained_mlp, small_data):
        X, y = small_data
        grad = trained_mlp.grad_wrt_input(X, y)
        assert grad.dtype == np.float32


# ---------------------------------------------------------------------------
# train_rf
# ---------------------------------------------------------------------------

class TestTrainRF:
    def test_returns_rf(self, trained_rf):
        assert isinstance(trained_rf, RandomForestClassifier)

    def test_predict_shape(self, trained_rf, small_data):
        X, _ = small_data
        preds = trained_rf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_binary(self, trained_rf, small_data):
        X, _ = small_data
        preds = trained_rf.predict(X)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_proba_shape(self, trained_rf, small_data):
        """RF sklearn zwraca (n, 2) — kolumna 1 to P(atak)."""
        X, _ = small_data
        proba = trained_rf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, trained_rf, small_data):
        X, _ = small_data
        proba = trained_rf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)

    def test_seed_reproducible(self, small_data):
        X, y = small_data
        rf1 = train_rf(X, y, seed=0)
        rf2 = train_rf(X, y, seed=0)
        np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))


# ---------------------------------------------------------------------------
# train_mlp
# ---------------------------------------------------------------------------

class TestTrainMLP:
    def test_returns_torch_classifier(self, small_data):
        X, y = small_data
        clf = train_mlp(X, y, seed=42)
        assert isinstance(clf, TorchClassifier)

    def test_predict_works(self, small_data):
        X, y = small_data
        clf = train_mlp(X, y, seed=42)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds.tolist()).issubset({0, 1})

    def test_seed_reproducible(self, small_data):
        X, y = small_data
        clf1 = train_mlp(X, y, seed=7)
        clf2 = train_mlp(X, y, seed=7)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))

    def test_input_dim_matches_data(self, small_data):
        X, y = small_data
        clf = train_mlp(X, y, seed=42)
        # model powinien działać na danych tej samej szerokości
        X_new = np.random.rand(5, X.shape[1]).astype(np.float32)
        preds = clf.predict(X_new)
        assert preds.shape == (5,)
