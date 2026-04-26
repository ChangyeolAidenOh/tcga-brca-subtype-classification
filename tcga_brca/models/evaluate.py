"""
Stage 3-5: Comprehensive Model Comparison
==========================================
Reads all metrics from all model JSON summaries.
Marks N/A for non-applicable metrics with explanation.

Usage:
    python -m tcga_brca.models.evaluate
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if isinstance(val, str):
        return val
    return f"{val:.{decimals}f}"


def fmt_cv(mean, std):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    return f"{mean:.4f} +/- {std:.4f}"


def load_all_results():
    results = []

    # 1. XGBoost
    path = os.path.join(config.RESULTS_TABLES, "xgboost_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        results.append({
            "Model": "XGBoost (class weights)",
            "Accuracy": fmt(d["test_metrics"]["accuracy"]),
            "Macro F1": fmt(d["test_metrics"]["macro_f1"]),
            "AUROC": fmt(d["test_metrics"]["auroc"]),
            "CV Accuracy": fmt_cv(d["cv_accuracy_mean"], d["cv_accuracy_std"]),
            "CV Macro F1": fmt_cv(d["cv_f1_mean"], d["cv_f1_std"]),
            "Imbalance": d["best_strategy"],
        })

    # 2. Stacking
    path = os.path.join(config.RESULTS_TABLES, "stacking_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        results.append({
            "Model": "Stacking Ensemble",
            "Accuracy": fmt(d["stacking_accuracy"]),
            "Macro F1": fmt(d["stacking_macro_f1"]),
            "AUROC": fmt(d.get("stacking_auroc")),
            "CV Accuracy": fmt_cv(d.get("cv_accuracy_mean"), d.get("cv_accuracy_std")),
            "CV Macro F1": fmt_cv(d.get("cv_f1_mean"), d.get("cv_f1_std")),
            "Imbalance": d.get("imbalance_strategy", "per-model (balanced)"),
        })

    # 3. TabNet
    path = os.path.join(config.RESULTS_TABLES, "tabnet_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        results.append({
            "Model": "TabNet (pretrained)",
            "Accuracy": fmt(d["tabnet_accuracy"]),
            "Macro F1": fmt(d["tabnet_macro_f1"]),
            "AUROC": fmt(d.get("tabnet_auroc")),
            "CV Accuracy": fmt_cv(d.get("cv_accuracy_mean"), d.get("cv_accuracy_std")),
            "CV Macro F1": fmt_cv(d.get("cv_f1_mean"), d.get("cv_f1_std")),
            "Imbalance": d.get("imbalance_strategy", "class_weights"),
        })

    # 4. Hierarchical
    path = os.path.join(config.RESULTS_TABLES, "hierarchical_results.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        results.append({
            "Model": "Hierarchical",
            "Accuracy": fmt(d["combined"]["accuracy"]),
            "Macro F1": fmt(d["combined"]["macro_f1"]),
            "AUROC": "N/A (multi-level)",
            "CV Accuracy": "N/A (multi-level)",
            "CV Macro F1": "N/A (multi-level)",
            "Imbalance": d.get("imbalance_strategy", "class_weights (per level)"),
        })

    # 5. Multi-task
    path = os.path.join(config.RESULTS_TABLES, "multitask_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        results.append({
            "Model": f"Multi-task (alpha={d.get('best_alpha', 0.1)})",
            "Accuracy": fmt(d["accuracy"]),
            "Macro F1": fmt(d["macro_f1"]),
            "AUROC": fmt(d.get("auroc")),
            "CV Accuracy": fmt_cv(d.get("cv_accuracy_mean"), d.get("cv_accuracy_std")),
            "CV Macro F1": fmt_cv(d.get("cv_f1_mean"), d.get("cv_f1_std")),
            "Imbalance": d.get("imbalance_strategy", "class_weights + Cox"),
        })

    # 6. METABRIC
    path = os.path.join(config.RESULTS_TABLES, "domain_adaptation_results.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        for strategy, metrics in d.items():
            results.append({
                "Model": f"METABRIC ({strategy})",
                "Accuracy": fmt(metrics["accuracy"]),
                "Macro F1": fmt(metrics["macro_f1"]),
                "AUROC": "N/A (external)",
                "CV Accuracy": "N/A (external)",
                "CV Macro F1": "N/A (external)",
                "Imbalance": "N/A (external)",
            })

    return results


def plot_comparison(results, save_path):
    internal = [r for r in results if "METABRIC" not in r["Model"]]
    models = [r["Model"] for r in internal]
    accs = []
    f1s = []
    for r in internal:
        try:
            accs.append(float(r["Accuracy"]))
        except (ValueError, TypeError):
            accs.append(0)
        try:
            f1s.append(float(r["Macro F1"]))
        except (ValueError, TypeError):
            f1s.append(0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), config.FIGURE_SIZE[1]), dpi=config.FIGURE_DPI)
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#378ADD", alpha=0.8)
    b2 = ax.bar(x + width/2, f1s, width, label="Macro F1", color="#1D9E75", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy vs Macro F1 (516 Consensus DEG Features)")
    ax.set_xticks(x)

    short_labels = []
    for m in models:
        m = m.replace(" (class weights)", "")
        m = m.replace(" Ensemble", "")
        m = m.replace(" (pretrained)", "\n(PT)")
        m = m.replace(" (alpha=0.1)", "")
        short_labels.append(m)

    ax.set_xticklabels(short_labels, fontsize=8, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in b1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {save_path}")


def main():
    print("=" * 60)
    print("Stage 3-5: Comprehensive Model Comparison")
    print("=" * 60)

    results = load_all_results()

    if not results:
        print("No model results found.")
        return

    comp_df = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("MODEL COMPARISON (all on 516 consensus DEG features)")
    print("=" * 100)
    print(comp_df.to_string(index=False))

    save_path = os.path.join(config.RESULTS_TABLES, "model_comparison.csv")
    comp_df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")

    plot_comparison(results, os.path.join(config.RESULTS_FIGURES, "model_comparison.png"))

    print("\nStage 3-5 complete.")


if __name__ == "__main__":
    main()
