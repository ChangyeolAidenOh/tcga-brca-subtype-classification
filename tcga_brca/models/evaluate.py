"""
Stage 3-5: Model Comparison
============================
Aggregate results from all models and generate comparison table + visualization.

Usage:
    python -m tcga_brca.models.evaluate
"""

import os
import sys
import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_model_results():
    """Load results from all trained models."""
    results = {}

    # XGBoost baseline
    xgb_path = os.path.join(config.RESULTS_TABLES, "xgboost_summary.json")
    if os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xgb = json.load(f)
        results["XGBoost Baseline"] = {
            "accuracy": xgb["test_metrics"]["accuracy"],
            "macro_f1": xgb["test_metrics"]["macro_f1"],
            "auroc": xgb["test_metrics"]["auroc"],
            "cv_accuracy": f"{xgb['cv_accuracy_mean']:.4f} +/- {xgb['cv_accuracy_std']:.4f}",
            "cv_f1": f"{xgb['cv_f1_mean']:.4f} +/- {xgb['cv_f1_std']:.4f}",
            "imbalance_strategy": xgb["best_strategy"],
        }

    # Stacking ensemble
    stack_path = os.path.join(config.RESULTS_TABLES, "stacking_summary.json")
    if os.path.exists(stack_path):
        with open(stack_path) as f:
            stack = json.load(f)
        results["Stacking Ensemble"] = {
            "accuracy": stack["stacking_accuracy"],
            "macro_f1": stack["stacking_macro_f1"],
            "auroc": stack["stacking_auroc"],
        }
        for base_name, base_score in stack.get("base_model_scores", {}).items():
            results[f"  Base: {base_name}"] = {
                "accuracy": base_score["mean_acc"],
                "macro_f1": float("nan"),
                "auroc": float("nan"),
                "note": f"CV mean +/- {base_score['std_acc']:.4f}",
            }

    # TabNet (if exists)
    tabnet_path = os.path.join(config.RESULTS_TABLES, "tabnet_summary.json")
    if os.path.exists(tabnet_path):
        with open(tabnet_path) as f:
            tabnet = json.load(f)
        results["TabNet"] = {
            "accuracy": tabnet.get("accuracy", float("nan")),
            "macro_f1": tabnet.get("macro_f1", float("nan")),
            "auroc": tabnet.get("auroc", float("nan")),
        }

    return results


def generate_comparison_table(results):
    """Generate and save model comparison table."""
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row["Accuracy"] = f"{metrics.get('accuracy', float('nan')):.4f}"
        row["Macro F1"] = f"{metrics.get('macro_f1', float('nan')):.4f}"
        row["AUROC"] = f"{metrics.get('auroc', float('nan')):.4f}"
        if "cv_accuracy" in metrics:
            row["CV Accuracy"] = metrics["cv_accuracy"]
        if "cv_f1" in metrics:
            row["CV Macro F1"] = metrics["cv_f1"]
        if "imbalance_strategy" in metrics:
            row["Imbalance"] = metrics["imbalance_strategy"]
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comp_df.to_string(index=False))

    save_path = os.path.join(config.RESULTS_TABLES, "model_comparison.csv")
    comp_df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")
    return comp_df


def plot_comparison(results, save_path):
    """Bar chart comparing model performance."""
    models = [k for k in results if not k.startswith("  Base:")]
    accs = [results[m]["accuracy"] for m in models]
    f1s = [results[m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#378ADD", alpha=0.8)
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#1D9E75", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy vs Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" Baseline", "\nBaseline").replace(" Ensemble", "\nEnsemble") for m in models],
                       fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {save_path}")


def main():
    print("=" * 60)
    print("Stage 3-5: Model Comparison")
    print("=" * 60)

    results = load_model_results()

    if not results:
        print("No model results found. Run baseline_xgboost and stacking_ensemble first.")
        return

    comp_df = generate_comparison_table(results)

    plot_comparison(
        results,
        os.path.join(config.RESULTS_FIGURES, "model_comparison.png"),
    )

    print("\nStage 3-5 complete.")


if __name__ == "__main__":
    main()
