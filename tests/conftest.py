import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from src.models import MLP, TorchClassifier, train_mlp, train_rf

# Mały zbiór: 200 próbek, 10 cech, zbalansowane klasy
_N, _D = 200, 10
_SEED = 42


@pytest.fixture(scope="session")
def small_data():
    rng = np.random.default_rng(_SEED)
    X = rng.uniform(0, 1, (_N, _D)).astype(np.float32)
    y = rng.integers(0, 2, _N)
    return X, y


@pytest.fixture(scope="session")
def trained_mlp(small_data):
    """TorchClassifier wytrenowany na małym zbiorze (1 epoka — szybko)."""
    X, y = small_data
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    mlp_raw = MLP(input_dim=_D, hidden=[16, 8])
    opt = torch.optim.Adam(mlp_raw.parameters(), lr=1e-3)
    import torch.nn as nn
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    mlp_raw.train()
    for _ in range(3):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(mlp_raw(X_t), y_t)
        loss.backward()
        opt.step()
    return TorchClassifier(mlp_raw)


@pytest.fixture(scope="session")
def trained_rf(small_data):
    X, y = small_data
    return train_rf(X, y, seed=_SEED)
