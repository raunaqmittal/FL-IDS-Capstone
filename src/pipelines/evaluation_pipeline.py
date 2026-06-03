import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.configs.paths import RESULTS_DIR, PLOTS_DIR
from src.logging.logger import logging


def _load_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric_vs_rounds(dfs: dict, metric: str, ylabel: str, filename: str):
    plt.figure(figsize=(10, 5))
    for label, df in dfs.items():
        if df.empty or metric not in df.columns:
            continue
        plt.plot(df["round"], df[metric], marker="o", markersize=3, label=label)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs FL Round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    logging.info(f"[Eval] Saved {filename}")


def plot_trust_heatmap(filename: str = "trust_scores.csv"):
    df = _load_csv(filename)
    if df.empty:
        return

    pivot = df.pivot(index="client_id", columns="round", values="trust_score")
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.3)
    plt.title("Client Trust Scores per Round")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trust_heatmap.png", dpi=150)
    plt.close()
    logging.info("[Eval] Saved trust_heatmap.png")


def print_summary_table(dfs: dict):
    rows = []
    for label, df in dfs.items():
        if df.empty:
            continue
        last = df.iloc[-1]
        rows.append({
            "Strategy": label,
            "Final Macro F1": round(last.get("macro_f1", 0), 4),
            "Final Accuracy": round(last.get("accuracy", 0), 4),
            "Final FPR": round(last.get("fpr", 0), 4),
        })
    if rows:
        summary = pd.DataFrame(rows).set_index("Strategy")
        print("\n=== Strategy Comparison (Final Round) ===")
        print(summary.to_string())


def run_evaluation() -> None:
    _ensure_plots_dir()

    dfs = {
        "RobustFL (Ours)": _load_csv("round_results.csv"),
        "FedAvg": _load_csv("round_results_fedavg.csv"),
        "Trimmed Mean": _load_csv("round_results_trimmed_mean.csv"),
        "Krum": _load_csv("round_results_krum.csv"),
    }

    plot_metric_vs_rounds(dfs, "macro_f1", "Macro F1-Score", "macro_f1_vs_rounds.png")
    plot_metric_vs_rounds(dfs, "accuracy", "Accuracy", "accuracy_vs_rounds.png")
    plot_metric_vs_rounds(dfs, "fpr", "False Positive Rate", "fpr_vs_rounds.png")

    plot_trust_heatmap()
    print_summary_table(dfs)
    logging.info("[Eval] Evaluation pipeline complete.")


if __name__ == "__main__":
    run_evaluation()
