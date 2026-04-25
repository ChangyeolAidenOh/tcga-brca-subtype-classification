"""
Prediction Confidence — Uncertainty Quantification
=====================================================
Compute prediction entropy to identify borderline patients
who need additional clinical review.

High confidence (entropy < 0.3): reliable classification
Borderline (entropy 0.3~1.0): additional review recommended
Low confidence (entropy > 1.0): unreliable, needs further testing

Usage:
    python -m tcga_brca.interpretation.prediction_confidence
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_model_and_data():
    """Load trained model and data."""
    model_path = os.path.join(config.RESULTS_MODELS, "xgboost_baseline.pkl")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]

    expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    common = expr.index.intersection(labels.index)
    expr, labels = expr.loc[common], labels.loc[common]

    X = scaler.transform(expr.values)
    y = le.transform(labels)

    return model, X, y, le, labels.index, labels


def compute_entropy(proba):
    """Compute Shannon entropy of prediction probabilities."""
    proba_clipped = np.clip(proba, 1e-10, 1.0)
    entropy = -np.sum(proba_clipped * np.log2(proba_clipped), axis=1)
    return entropy


def analyze_confidence(model, X, y, le, sample_ids, labels_raw):
    """Full confidence analysis."""
    proba = model.predict_proba(X)
    y_pred = model.predict(X)
    entropy = compute_entropy(proba)

    max_proba = np.max(proba, axis=1)
    class_names = le.classes_

    # Build results DataFrame
    results = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": labels_raw.values,
        "predicted_label": le.inverse_transform(y_pred),
        "correct": y == y_pred,
        "max_probability": max_proba,
        "entropy": entropy,
    })

    # Add per-class probabilities
    for i, cls in enumerate(class_names):
        results[f"prob_{cls.replace('BRCA_', '')}"] = proba[:, i]

    # Confidence categories
    results["confidence"] = pd.cut(
        entropy,
        bins=[-0.01, 0.3, 1.0, np.inf],
        labels=["High", "Borderline", "Low"],
    )

    return results


def print_confidence_summary(results):
    """Print detailed confidence analysis."""
    print(f"\n{'=' * 60}")
    print("PREDICTION CONFIDENCE SUMMARY")
    print(f"{'=' * 60}")

    # Overall distribution
    conf_dist = results["confidence"].value_counts()
    for conf in ["High", "Borderline", "Low"]:
        n = conf_dist.get(conf, 0)
        pct = n / len(results) * 100
        acc = results[results["confidence"] == conf]["correct"].mean() if n > 0 else 0
        print(f"  {conf:12s}: {n:4d} ({pct:5.1f}%) — accuracy {acc:.4f}")

    # Per-subtype confidence
    print(f"\nPer-subtype confidence distribution:")
    for subtype in sorted(results["true_label"].unique()):
        sub = results[results["true_label"] == subtype]
        mean_entropy = sub["entropy"].mean()
        pct_high = (sub["confidence"] == "High").mean() * 100
        pct_border = (sub["confidence"] == "Borderline").mean() * 100
        short = subtype.replace("BRCA_", "")
        print(f"  {short:10s}: mean entropy {mean_entropy:.3f} | "
              f"High {pct_high:.0f}% | Borderline {pct_border:.0f}%")

    # Misclassified samples analysis
    wrong = results[~results["correct"]]
    print(f"\nMisclassified samples ({len(wrong)}):")
    print(f"  Mean entropy: {wrong['entropy'].mean():.3f} (vs correct: {results[results['correct']]['entropy'].mean():.3f})")
    print(f"  Borderline+Low: {(wrong['confidence'].isin(['Borderline', 'Low'])).mean()*100:.1f}%")

    # Confusion pairs with highest uncertainty
    if len(wrong) > 0:
        wrong_pairs = wrong.groupby(["true_label", "predicted_label"]).agg(
            count=("correct", "size"),
            mean_entropy=("entropy", "mean"),
        ).sort_values("count", ascending=False)
        print(f"\n  Most common confusion pairs:")
        for (true, pred), row in wrong_pairs.head(5).iterrows():
            t = true.replace("BRCA_", "")
            p = pred.replace("BRCA_", "")
            print(f"    {t} → {p}: {int(row['count'])} cases, mean entropy {row['mean_entropy']:.3f}")


def plot_confidence(results, save_dir):
    """Generate confidence visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Entropy distribution by correctness
    fig, axes = plt.subplots(1, 2, figsize=(config.FIGURE_SIZE[0] * 1.4, config.FIGURE_SIZE[1]),
                              dpi=config.FIGURE_DPI)

    axes[0].hist(
        results[results["correct"]]["entropy"], bins=30, alpha=0.7,
        color="#1D9E75", label="Correct", density=True,
    )
    axes[0].hist(
        results[~results["correct"]]["entropy"], bins=30, alpha=0.7,
        color="#E24B4A", label="Incorrect", density=True,
    )
    axes[0].axvline(0.3, ls="--", color="#888780", lw=0.8)
    axes[0].axvline(1.0, ls="--", color="#888780", lw=0.8)
    axes[0].set_xlabel("Prediction Entropy")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Entropy Distribution")
    axes[0].legend()

    # 2. Accuracy by confidence bin
    conf_bins = pd.cut(results["entropy"], bins=10)
    acc_by_bin = results.groupby(conf_bins, observed=True)["correct"].mean()
    count_by_bin = results.groupby(conf_bins, observed=True)["correct"].count()

    midpoints = [(interval.left + interval.right) / 2 for interval in acc_by_bin.index]
    axes[1].bar(range(len(midpoints)), acc_by_bin.values, color="#378ADD", alpha=0.8)
    axes[1].set_xticks(range(len(midpoints)))
    axes[1].set_xticklabels([f"{m:.1f}" for m in midpoints], fontsize=8, rotation=45)
    axes[1].set_xlabel("Entropy Bin (midpoint)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Prediction Entropy")

    plt.suptitle("Prediction Confidence Analysis", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "prediction_confidence.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3. Per-subtype entropy boxplot
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    plot_data = results.copy()
    plot_data["subtype"] = plot_data["true_label"].str.replace("BRCA_", "")
    order = ["Basal", "Her2", "LumA", "LumB", "Normal"]
    order = [o for o in order if o in plot_data["subtype"].values]
    sns.boxplot(data=plot_data, x="subtype", y="entropy", order=order, ax=ax,
                palette={"Basal": "#E24B4A", "Her2": "#BA7517", "LumA": "#378ADD",
                         "LumB": "#1D9E75", "Normal": "#7F77DD"})
    ax.axhline(0.3, ls="--", color="#888780", lw=0.8)
    ax.axhline(1.0, ls="--", color="#888780", lw=0.8)
    ax.set_xlabel("PAM50 Subtype")
    ax.set_ylabel("Prediction Entropy")
    ax.set_title("Prediction Uncertainty by Subtype")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "confidence_by_subtype.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Confidence plots saved to {save_dir}")


def main():
    print("=" * 60)
    print("Prediction Confidence — Uncertainty Quantification")
    print("=" * 60)

    model, X, y, le, sample_ids, labels_raw = load_model_and_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

    results = analyze_confidence(model, X, y, le, sample_ids, labels_raw)
    print_confidence_summary(results)
    plot_confidence(results, config.RESULTS_FIGURES)

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    results.to_csv(
        os.path.join(config.RESULTS_TABLES, "prediction_confidence.csv"), index=False
    )

    # Borderline samples for clinical review
    borderline = results[results["confidence"].isin(["Borderline", "Low"])].sort_values("entropy", ascending=False)
    borderline.to_csv(
        os.path.join(config.RESULTS_TABLES, "borderline_patients.csv"), index=False
    )
    print(f"\nBorderline patients flagged: {len(borderline)}")
    print(f"Results saved to {config.RESULTS_TABLES}")

    print("\nPrediction confidence analysis complete.")


if __name__ == "__main__":
    main()
