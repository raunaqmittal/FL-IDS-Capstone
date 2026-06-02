import sys
from typing import Dict, List, Tuple, Optional, Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.components.model.model import (
    MLPClassifier,
    get_model_parameters,
    set_model_parameters,
)


class FLIDSClient(fl.client.NumPyClient):
    """
    Flower client representing one IoT Edge Gateway.

    The client:
    - receives global model parameters from the server
    - trains locally on private data
    - returns only model parameters, never raw data
    """

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: MLPClassifier,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.cid = str(cid)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.config = config or {}

        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.local_epochs = int(self.config.get("local_epochs", 1))
        self.lr = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))

        self.is_poisoned = bool(self.config.get("is_poisoned", False))

    def get_parameters(self, config) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config,
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        set_model_parameters(self.model, parameters)

        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_examples = 0
        correct = 0

        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                # Later: attacker.py label flipping can go here
                # if self.is_poisoned:
                #     y = flip_labels(y)

                optimizer.zero_grad()

                logits = self.model(x)
                loss = loss_fn(logits, y)

                loss.backward()
                optimizer.step()

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()

        avg_loss = total_loss / total_examples
        train_accuracy = correct / total_examples

        metrics = {
            "train_loss": float(avg_loss),
            "train_accuracy": float(train_accuracy),
            "cid": float(self.cid) if self.cid.isdigit() else -1.0,
        }

        return get_model_parameters(self.model), total_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config,
    ) -> Tuple[float, int, Dict[str, float]]:
        set_model_parameters(self.model, parameters)

        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_examples = 0
        correct = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                logits = self.model(x)
                loss = loss_fn(logits, y)

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()

        avg_loss = total_loss / total_examples
        accuracy = correct / total_examples

        metrics = {
            "val_accuracy": float(accuracy),
            "cid": float(self.cid) if self.cid.isdigit() else -1.0,
        }

        return float(avg_loss), total_examples, metrics