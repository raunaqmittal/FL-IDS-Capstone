import numpy as np
import torch
import torch.nn as nn


class _AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AEScorer:
    """
    Variant B — Autoencoder-based anomaly scorer.

    Trains a small AE on final-layer weight vectors from trusted clients.
    Scores each client by its reconstruction error: high error = anomalous.
    Integrates with RobustFLIDSStrategy as a drop-in MAD replacement.
    """

    def __init__(self, input_dim: int, hidden_factor: int = 4, train_epochs: int = 5, lr: float = 1e-3):
        self.input_dim = input_dim
        self.hidden_dim = max(1, input_dim // hidden_factor)
        self.train_epochs = train_epochs
        self.lr = lr
        self.model = None
        self.is_fitted = False

    def fit(self, vectors: np.ndarray) -> None:
        self.model = _AE(self.input_dim, self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X = torch.tensor(vectors, dtype=torch.float32)
        self.model.train()
        for _ in range(self.train_epochs):
            optimizer.zero_grad()
            recon = self.model(X)
            loss = loss_fn(recon, X)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def score(self, vectors: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            return np.zeros(len(vectors))

        self.model.eval()
        X = torch.tensor(vectors, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(X)
            errors = ((recon - X) ** 2).mean(dim=1).numpy()

        # Invert so high error = low score (consistent with MAD convention)
        return -errors
