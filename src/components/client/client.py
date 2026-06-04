
import sys
from typing import Dict, List, Tuple, Optional, Any

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.logging.logger import logging
from src.exception.exception import FLIDSException
from src.components.model.model import (
    MLPClassifier,
    get_model_parameters,
    set_model_parameters,
)
from src.components.client.attacker import (
    flip_labels,
    inject_backdoor_trigger,
    scale_gradient_to_norm,
)


class FLIDSClient(fl.client.NumPyClient):
    """
    Flower client representing one IoT Edge Gateway.

    The client:
    - receives global model parameters from the server
    - trains locally on private data
    - returns only model parameters, never raw data

    Attack injection (if is_poisoned=True) is handled inside fit():
    - Label-flip: applied per-batch on tensor labels (no raw data needed)
    - Backdoor:   requires raw numpy arrays — pass X_train_raw + y_train_raw
                  to __init__ and a poisoned DataLoader is rebuilt in fit()
    """

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: MLPClassifier,
        config: Optional[Dict[str, Any]] = None,
        X_train_raw: Optional[np.ndarray] = None,
        y_train_raw: Optional[np.ndarray] = None,
    ):
        self.cid = str(cid)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.config = config or {}

        # Raw numpy arrays — required for backdoor injection (label-flip works per-batch)
        self.X_train_raw = X_train_raw
        self.y_train_raw = y_train_raw

        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.local_epochs = int(self.config.get("local_epochs", 1))
        self.lr = float(self.config.get("lr", 1e-3))
        self.weight_decay = float(self.config.get("weight_decay", 0.0))

        self.is_poisoned = bool(self.config.get("is_poisoned", False))
        self.weight_cap = float(self.config.get("weight_cap", 10.0))
        self.num_classes = int(self.config.get("num_classes", 27))

    def get_parameters(self, config) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config,
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        set_model_parameters(self.model, parameters)

        # ── Attack gate ───────────────────────────────────────────────────────
        server_round    = int(config.get("server_round", 0))
        attack_start    = int(self.config.get("attack_start_round", 11))
        attack_active   = self.is_poisoned and (server_round >= attack_start)
        attack_type     = self.config.get("attack_type", "label_flip")
        source_class    = int(self.config.get("source_class", 1))
        target_class    = int(self.config.get("target_class", 0))

        # ── Backdoor: rebuild DataLoader with poisoned raw data ───────────────
        # Requires X_train_raw / y_train_raw passed at construction time.
        active_loader = self.train_loader
        if attack_active and attack_type in ("backdoor", "both"):
            if self.X_train_raw is not None and self.y_train_raw is not None:
                trigger_idx    = self.config.get("trigger_feature_idx", [0, 5])
                trigger_vals   = self.config.get("trigger_values",       [999999, 1])
                inject_ratio   = float(self.config.get("inject_ratio",   0.1))
                X_p, y_p = inject_backdoor_trigger(
                    self.X_train_raw, self.y_train_raw,
                    trigger_idx, trigger_vals, inject_ratio,
                )
                dataset = TensorDataset(
                    torch.tensor(X_p, dtype=torch.float32),
                    torch.tensor(y_p, dtype=torch.long),
                )
                active_loader = DataLoader(
                    dataset,
                    batch_size=self.train_loader.batch_size or 64,
                    shuffle=True,
                    drop_last=True,
                )
                logging.info(f"[Client {self.cid}] Backdoor DataLoader rebuilt.")
            else:
                logging.warning(
                    f"[Client {self.cid}] Backdoor requested but raw arrays not provided — skipping."
                )

        # ── Training loop ─────────────────────────────────────────────────────
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Weighted loss: matches centralized baseline — boosts rare attack classes
        if self.y_train_raw is not None:
            y_for_weights = self.y_train_raw
        else:
            y_for_weights = np.concatenate(
                [y_batch.numpy() for _, y_batch in active_loader]
            )
        classes = np.arange(self.num_classes)
        try:
            raw_w = compute_class_weight(
                class_weight="balanced", classes=classes, y=y_for_weights
            )
        except Exception:
            raw_w = np.ones(self.num_classes)
        capped_w = np.clip(raw_w, None, self.weight_cap)
        weight_tensor = torch.tensor(capped_w, dtype=torch.float32).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        total_loss = 0.0
        total_examples = 0
        correct = 0

        for _ in range(self.local_epochs):
            for x, y in active_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()

                # ── Label-flip attack (per-batch, no raw data required) ───────
                if attack_active and attack_type in ("label_flip", "both"):
                    y_np = flip_labels(y.cpu().numpy(), source_class, target_class)
                    y = torch.tensor(y_np, dtype=torch.long, device=self.device)

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

        avg_loss = total_loss / max(total_examples, 1)
        train_accuracy = correct / max(total_examples, 1)

        params = get_model_parameters(self.model)

        # ── Gradient norm scaling (stealth bypass for backdoor) ───────────────
        if attack_active and attack_type in ("backdoor", "both"):
            scale_to_norm = self.config.get("scale_to_benign_norm", False)
            target_norm   = float(self.config.get("benign_norm_target", 1.0))
            if scale_to_norm:
                params = scale_gradient_to_norm(params, target_norm)

        metrics = {
            "train_loss":     float(avg_loss),
            "train_accuracy": float(train_accuracy),
            "cid":            float(self.cid) if self.cid.isdigit() else -1.0,
            "is_poisoned":    float(self.is_poisoned),
            "attack_active":  float(attack_active),
        }

        return params, total_examples, metrics

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

        avg_loss = total_loss / max(total_examples, 1)
        accuracy = correct / max(total_examples, 1)

        metrics = {
            "val_accuracy": float(accuracy),
            "cid": float(self.cid) if self.cid.isdigit() else -1.0,
        }

        return float(avg_loss), total_examples, metrics