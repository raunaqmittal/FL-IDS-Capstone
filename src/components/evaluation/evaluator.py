import csv
import os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from src.logging.logger import logging
from src.configs.paths import RESULTS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    fprs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + tp
        fpr_c = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr_c)
    macro_fpr = float(np.mean(fprs))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fpr": macro_fpr,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def compute_asr(model, trigger_loader, benign_class_idx: int = 0) -> float:
    import torch
    model.eval()
    total = 0
    classified_benign = 0

    with torch.no_grad():
        for x, _ in trigger_loader:
            logits = model(x.float())
            preds = torch.argmax(logits, dim=1)
            classified_benign += (preds == benign_class_idx).sum().item()
            total += len(preds)

    return classified_benign / total if total > 0 else 0.0


def log_round_results(server_round: int, metrics: dict, filename: str = "round_results.csv") -> None:
    path = RESULTS_DIR / filename
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["round", "accuracy", "macro_f1", "weighted_f1", "fpr"])
        writer.writerow([
            server_round,
            round(metrics.get("accuracy", 0), 6),
            round(metrics.get("macro_f1", 0), 6),
            round(metrics.get("weighted_f1", 0), 6),
            round(metrics.get("fpr", 0), 6),
        ])


def log_trust_scores(server_round: int, trust_scores: dict, filename: str = "trust_scores.csv") -> None:
    path = RESULTS_DIR / filename
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["round", "client_id", "trust_score"])
        for cid, score in trust_scores.items():
            writer.writerow([server_round, cid, round(float(score), 6)])
