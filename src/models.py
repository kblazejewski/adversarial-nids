import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from config import SEED, RF_TREES, RF_CLASS_WEIGHT, MLP_HIDDEN, MLP_EPOCHS, MLP_BATCH, MLP_LR


# ---------------------------------------------------------------------------
# MLP (sieć neuronowa)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list = MLP_HIDDEN):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class TorchClassifier:
    """
    Wrapper MLP z interfejsem kompatybilnym z sklearn:
      .predict(X)        -> np.ndarray int {0,1}
      .predict_proba(X)  -> np.ndarray float [0,1]  (P(atak))
      .grad_wrt_input(X, y) -> np.ndarray kształt jak X  (potrzebne do FGSM)
    """

    def __init__(self, model: MLP):
        self.model = model
        self._device = next(model.parameters()).device

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32, device=self._device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(X))
            return (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_tensor(X))
            return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    def grad_wrt_input(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient BCE po wejściu — używane przez FGSM."""
        self.model.eval()
        x_t = self._to_tensor(X).requires_grad_(True)
        y_t = torch.tensor(y, dtype=torch.float32, device=self._device)
        logits = self.model(x_t)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        return x_t.grad.cpu().numpy()


# ---------------------------------------------------------------------------
# Funkcje treningowe
# ---------------------------------------------------------------------------

def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             seed: int = SEED) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=RF_TREES,
        class_weight=RF_CLASS_WEIGHT,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              seed: int = SEED) -> TorchClassifier:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    model = MLP(input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=MLP_BATCH, shuffle=True)

    model.train()
    for _ in range(MLP_EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return TorchClassifier(model)
