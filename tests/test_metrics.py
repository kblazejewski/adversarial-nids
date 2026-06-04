"""Testy dla src/metrics.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest
from src.metrics import evaluate, asr, timed_predict


# ---------------------------------------------------------------------------
# Mock modelu — identyczny interfejs co RF i TorchClassifier
# ---------------------------------------------------------------------------

class _PerfectModel:
    """Zawsze przewiduje prawidłową etykietę."""
    def predict(self, X):
        return np.ones(len(X), dtype=int)  # zakładamy, że wszystko to ataki

    def predict_proba(self, X):
        return np.ones(len(X), dtype=np.float32)


class _ConstantModel:
    """Zawsze zwraca tę samą etykietę i to samo prawdopodobieństwo."""
    def __init__(self, label: int, proba: float):
        self._label = label
        self._proba = proba

    def predict(self, X):
        return np.full(len(X), self._label, dtype=int)

    def predict_proba(self, X):
        return np.full(len(X), self._proba, dtype=np.float32)


class _SklearnLikeModel:
    """Symuluje predict_proba z sklearn (zwraca macierz n×2)."""
    def __init__(self, proba_class1: float):
        self._p = proba_class1

    def predict(self, X):
        return (np.full(len(X), self._p) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._p
        return np.column_stack([np.full(len(X), 1 - p),
                                 np.full(len(X), p)]).astype(np.float32)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_all_keys(self):
        model = _PerfectModel()
        X = np.zeros((10, 3), dtype=np.float32)
        y = np.ones(10, dtype=int)
        result = evaluate(model, X, y)
        assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "auc"}

    def test_perfect_predictions(self):
        # AUC wymaga obu klas — model przewiduje 1, y zawiera 0 i 1
        class _PerfectMixed:
            def predict(self, X): return np.array([0]*10 + [1]*10, dtype=int)
            def predict_proba(self, X): return np.array([0.0]*10 + [1.0]*10, dtype=np.float32)

        X = np.zeros((20, 3), dtype=np.float32)
        y = np.array([0]*10 + [1]*10, dtype=int)
        result = evaluate(_PerfectMixed(), X, y)
        assert result["accuracy"]  == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"]    == pytest.approx(1.0)
        assert result["f1"]        == pytest.approx(1.0)
        assert result["auc"]       == pytest.approx(1.0)

    def test_all_wrong(self):
        # model mówi "atak" (1), a prawda to "normalny" (0)
        model = _ConstantModel(label=1, proba=1.0)
        X = np.zeros((10, 3), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        result = evaluate(model, X, y)
        assert result["accuracy"] == pytest.approx(0.0)
        assert result["recall"]   == pytest.approx(0.0, abs=1e-6)

    def test_sklearn_style_proba_2d(self):
        """predict_proba zwracający macierz n×2 powinien działać poprawnie."""
        model = _SklearnLikeModel(proba_class1=0.9)
        X = np.zeros((10, 3), dtype=np.float32)
        # AUC wymaga obu klas — połowa ataków, połowa normalnych
        y = np.array([0]*5 + [1]*5, dtype=int)
        result = evaluate(model, X, y)
        assert 0.0 <= result["auc"] <= 1.0

    def test_mixed_predictions(self):
        # 10 próbek: połowa ataków poprawnie wykryta
        preds = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        truth = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        class _MockMixed:
            def predict(self, X): return preds
            def predict_proba(self, X): return preds.astype(np.float32)

        result = evaluate(_MockMixed(), np.zeros((10, 2)), truth)
        assert result["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# asr
# ---------------------------------------------------------------------------

class TestASR:
    def _make_xy(self, n_normal=5, n_attack=5):
        y = np.array([0] * n_normal + [1] * n_attack, dtype=int)
        X = np.zeros((len(y), 3), dtype=np.float32)
        return X, y

    def test_full_evasion(self):
        """Atak powinien zwrócić ASR=1: model wykrywa wszystko na czystych,
        nic nie wykrywa po ataku."""
        X, y = self._make_xy()
        detects_clean = _ConstantModel(label=1, proba=1.0)

        class _FoolsAll:
            def predict(self, X): return np.zeros(len(X), dtype=int)
            def predict_proba(self, X): return np.zeros(len(X), dtype=np.float32)

        # potrzebujemy modelu, który wykrywa na clean i nie wykrywa na adv
        class _Switchable:
            def __init__(self):
                self._call = 0
            def predict(self, X):
                self._call += 1
                if self._call == 1:
                    return np.ones(len(X), dtype=int)   # clean: wykrywa wszystko
                return np.zeros(len(X), dtype=int)      # adv: nic nie wykrywa
            def predict_proba(self, X):
                return self.predict(X).astype(np.float32)

        model = _Switchable()
        X_adv = X.copy()
        result = asr(model, X, X_adv, y)
        assert result == pytest.approx(1.0)

    def test_no_evasion(self):
        """Model wykrywa ataki zarówno na clean jak i adv → ASR=0."""
        X, y = self._make_xy()
        model = _ConstantModel(label=1, proba=1.0)
        result = asr(model, X, X.copy(), y)
        assert result == pytest.approx(0.0)

    def test_no_detections_on_clean(self):
        """Gdy model nie wykrywa nic na czystych danych → ASR=NaN."""
        X, y = self._make_xy()
        model = _ConstantModel(label=0, proba=0.0)
        result = asr(model, X, X.copy(), y)
        assert math.isnan(result)

    def test_only_normal_traffic(self):
        """Brak ataków w zbiorze (wszystkie y==0) → ASR=NaN."""
        X = np.zeros((10, 3), dtype=np.float32)
        y = np.zeros(10, dtype=int)
        model = _ConstantModel(label=1, proba=1.0)
        result = asr(model, X, X.copy(), y)
        assert math.isnan(result)

    def test_partial_evasion(self):
        """4 ataki wykryte na clean, 2 przeoczone na adv → ASR ≈ 0.5."""
        y = np.array([0, 0, 1, 1, 1, 1], dtype=int)
        X_clean = np.zeros((6, 3), dtype=np.float32)
        X_adv   = np.ones((6, 3), dtype=np.float32)

        call_count = [0]

        class _PartialModel:
            def predict(self, X):
                call_count[0] += 1
                if call_count[0] == 1:
                    # clean: poprawnie wykrywa 4 ataki
                    return np.array([0, 0, 1, 1, 1, 1])
                else:
                    # adv: przeocza 2 z 4 ataków
                    return np.array([0, 0, 0, 0, 1, 1])
            def predict_proba(self, X):
                return self.predict(X).astype(np.float32)

        result = asr(_PartialModel(), X_clean, X_adv, y)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# timed_predict
# ---------------------------------------------------------------------------

class TestTimedPredict:
    def test_returns_tuple(self):
        model = _ConstantModel(label=1, proba=1.0)
        X = np.zeros((50, 4), dtype=np.float32)
        preds, ms = timed_predict(model, X)
        assert len(preds) == 50
        assert ms >= 0.0

    def test_predictions_match(self):
        model = _ConstantModel(label=0, proba=0.0)
        X = np.zeros((20, 4), dtype=np.float32)
        preds, _ = timed_predict(model, X)
        assert np.all(preds == 0)

    def test_time_is_float_ms(self):
        model = _ConstantModel(label=1, proba=1.0)
        X = np.zeros((100, 10), dtype=np.float32)
        _, ms = timed_predict(model, X)
        assert isinstance(ms, float)
