"""Testy dla src/datasets.py — nie wymagają surowych danych."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import numpy as np
import pandas as pd
import pytest
from src.datasets import _find_label_col, _preprocess, load_dataset


# ---------------------------------------------------------------------------
# _find_label_col
# ---------------------------------------------------------------------------

class TestFindLabelCol:
    def test_exact_match(self):
        df = pd.DataFrame({"label": [0, 1], "feat": [1.0, 2.0]})
        assert _find_label_col(df, "label") == "label"

    def test_case_insensitive(self):
        df = pd.DataFrame({"Label": [0, 1], "feat": [1.0, 2.0]})
        assert _find_label_col(df, "label") == "Label"

    def test_fallback_to_class(self):
        df = pd.DataFrame({"class": [0, 1], "feat": [1.0, 2.0]})
        assert _find_label_col(df, "label") == "class"

    def test_strips_whitespace(self):
        # _find_label_col stripuje whitespace z nazw kolumn, więc " label " pasuje
        df = pd.DataFrame({" label ": [0, 1], "feat": [1.0, 2.0]})
        result = _find_label_col(df, "label")
        assert result == " label "

    def test_missing_label_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(KeyError):
            _find_label_col(df, "label")


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------

def _make_df(n=50, d=5, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.uniform(0, 100, n).tolist() for i in range(d)}
    data["label"] = rng.integers(0, 2, n).tolist()
    return pd.DataFrame(data)


class TestPreprocess:
    def test_output_shapes(self):
        df = _make_df(n=50, d=5)
        X, y, scaler = _preprocess(df, "label", fit_scaler=True)
        assert X.shape == (50, 5)
        assert y.shape == (50,)

    def test_values_in_unit_range(self):
        df = _make_df(n=100, d=4)
        X, _, _ = _preprocess(df, "label", fit_scaler=True)
        assert X.min() >= 0.0 - 1e-6
        assert X.max() <= 1.0 + 1e-6

    def test_dtype_float32(self):
        df = _make_df(n=30, d=3)
        X, _, _ = _preprocess(df, "label", fit_scaler=True)
        assert X.dtype == np.float32

    def test_labels_binary(self):
        df = _make_df(n=40, d=3)
        _, y, _ = _preprocess(df, "label", fit_scaler=True)
        assert set(y.tolist()).issubset({0, 1})

    def test_scaler_reuse(self):
        """Scaler dopasowany na train powinien dać te same zakresy na test."""
        df_train = _make_df(n=100, d=4, seed=0)
        df_test  = _make_df(n=30,  d=4, seed=1)
        X_train, _, scaler = _preprocess(df_train, "label", fit_scaler=True)
        X_test,  _, _      = _preprocess(df_test,  "label", fit_scaler=False,
                                         scaler=scaler)
        # test może wyjść poza [0,1] bo pochodzi z innej dystrybucji,
        # ale po clipie w _preprocess nie przekroczy [0,1]
        assert X_test.min() >= 0.0 - 1e-6
        assert X_test.max() <= 1.0 + 1e-6

    def test_nan_handling(self):
        df = _make_df(n=20, d=3)
        df.loc[0, "f0"] = float("nan")
        X, _, _ = _preprocess(df, "label", fit_scaler=True)
        assert not np.isnan(X).any()

    def test_inf_handling(self):
        df = _make_df(n=20, d=3)
        df.loc[1, "f1"] = float("inf")
        df.loc[2, "f2"] = float("-inf")
        X, _, _ = _preprocess(df, "label", fit_scaler=True)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_categorical_feature_encoded(self):
        df = pd.DataFrame({
            "proto": ["tcp", "udp", "tcp", "icmp"] * 5,
            "val":   [1.0, 2.0, 3.0, 4.0] * 5,
            "label": [0, 1, 0, 1] * 5,
        })
        X, y, _ = _preprocess(df, "label", fit_scaler=True)
        assert X.shape[1] == 2
        assert not np.isnan(X).any()


# ---------------------------------------------------------------------------
# load_dataset — testy z cache'em (nie dotykają surowych danych)
# ---------------------------------------------------------------------------

class TestLoadDatasetCache:
    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        """Dane zapisane do cache powinny być odczytane identycznie."""
        import src.datasets as ds_mod

        # przestaw PATHS na tmpdir
        monkeypatch.setitem(ds_mod.PATHS, "processed", str(tmp_path))

        X_train_orig = np.random.rand(40, 5).astype(np.float32)
        y_train_orig = np.random.randint(0, 2, 40)
        X_test_orig  = np.random.rand(10, 5).astype(np.float32)
        y_test_orig  = np.random.randint(0, 2, 10)

        # zapisz cache ręcznie
        cache = tmp_path / "mydata.npz"
        np.savez(str(cache),
                 X_train=X_train_orig, y_train=y_train_orig,
                 X_test=X_test_orig,   y_test=y_test_orig)

        # load_dataset powinien go wczytać
        X_tr, y_tr, X_te, y_te = ds_mod.load_dataset("mydata")

        np.testing.assert_array_equal(X_tr, X_train_orig)
        np.testing.assert_array_equal(y_tr, y_train_orig)
        np.testing.assert_array_equal(X_te, X_test_orig)
        np.testing.assert_array_equal(y_te, y_test_orig)

    def test_unknown_dataset_raises(self, tmp_path, monkeypatch):
        import src.datasets as ds_mod
        monkeypatch.setitem(ds_mod.PATHS, "processed", str(tmp_path))
        with pytest.raises(ValueError, match="Nieznany zbiór"):
            ds_mod.load_dataset("unknown_xyz")
