"""
Domain Adaptation — CORAL for Cross-Platform Validation
=========================================================
Apply CORrelation ALignment (CORAL) to bridge the distribution
gap between TCGA (RNA-seq) and METABRIC (microarray) before
classification. Compare: no adaptation vs independent scaling
vs CORAL.

This is an ML-specific approach that bioinformatics researchers
rarely apply to cross-platform genomic data.

Usage:
    python -m tcga_brca.models.domain_adaptation
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_tcga_and_metabric():
    """Load both datasets with shared gene set."""
    # TCGA
    tcga_expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    tcga_labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()
    common_tcga = tcga_expr.index.intersection(tcga_labels.index)
    tcga_expr = tcga_expr.loc[common_tcga]
    tcga_labels = tcga_labels.loc[common_tcga]

    # METABRIC
    meta_expr = pd.read_csv(
        os.path.join(config.DATA_RAW, "metabric_expression.csv"), index_col=0
    )
    meta_clinical = pd.read_csv(
        os.path.join(config.DATA_RAW, "metabric_clinical.csv")
    )

    # Extract METABRIC PAM50 from CLAUDIN_SUBTYPE
    label_map = {
        "LumA": "BRCA_LumA", "LumB": "BRCA_LumB",
        "Basal": "BRCA_Basal", "Her2": "BRCA_Her2",
        "Normal": "BRCA_Normal",
    }
    meta_pam50 = meta_clinical.set_index("sampleId")["CLAUDIN_SUBTYPE"].map(label_map).dropna()

    common_meta = meta_expr.index.intersection(meta_pam50.index)
    meta_expr = meta_expr.loc[common_meta]
    meta_labels = meta_pam50.loc[common_meta]

    # Log transform METABRIC if needed
    if meta_expr.max().max() > 30:
        meta_expr = np.log2(meta_expr.clip(lower=0) + config.LOG2_PSEUDOCOUNT)

    # Shared genes
    shared = tcga_expr.columns.intersection(meta_expr.columns)
    tcga_expr = tcga_expr[shared]
    meta_expr = meta_expr[shared]

    print(f"TCGA:     {tcga_expr.shape[0]} samples x {tcga_expr.shape[1]} genes")
    print(f"METABRIC: {meta_expr.shape[0]} samples x {meta_expr.shape[1]} genes")
    print(f"Shared genes: {len(shared)}")

    return tcga_expr, tcga_labels, meta_expr, meta_labels


def coral_transform(source, target):
    """Apply CORAL (CORrelation ALignment) to align source to target distribution.

    Aligns the covariance of source features to match target features.
    CORAL paper: Sun et al., "Return of Frustratingly Easy Domain Adaptation" (2016)
    """
    # Center
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    # Covariance matrices + regularization
    n_feat = source.shape[1]
    reg = np.eye(n_feat) * 1e-6

    cov_source = np.cov(source_centered, rowvar=False) + reg
    cov_target = np.cov(target_centered, rowvar=False) + reg

    # Whitening source
    try:
        cov_source_sqrt_inv = np.linalg.inv(sqrtm(cov_source).real)
        cov_target_sqrt = sqrtm(cov_target).real

        source_aligned = source_centered @ cov_source_sqrt_inv @ cov_target_sqrt
        source_aligned += target_mean
    except np.linalg.LinAlgError:
        print("  CORAL matrix decomposition failed. Using fallback.")
        # Fallback: simple mean-std alignment
        source_aligned = (source - source_mean) / (source.std(axis=0) + 1e-8)
        source_aligned = source_aligned * (target.std(axis=0) + 1e-8) + target_mean

    return source_aligned


def train_and_evaluate(X_train, y_train, X_test, y_test, class_names, label):
    """Train XGBoost and evaluate."""
    sw = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sw)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n  {label}:")
    print(f"    Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    per_class = {cls.replace("BRCA_", ""): report[cls]["recall"] for cls in class_names if cls in report}
    for cls, recall in per_class.items():
        print(f"    {cls:10s} Recall: {recall:.4f}")

    return acc, f1, y_pred, per_class


def plot_domain_comparison(results, save_path):
    """Plot comparison of domain adaptation strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(config.FIGURE_SIZE[0] * 1.4, config.FIGURE_SIZE[1]),
                              dpi=config.FIGURE_DPI)

    strategies = list(results.keys())
    accs = [results[s]["accuracy"] for s in strategies]
    f1s = [results[s]["macro_f1"] for s in strategies]

    x = np.arange(len(strategies))
    w = 0.3
    axes[0].bar(x - w/2, accs, w, label="Accuracy", color="#378ADD", alpha=0.8)
    axes[0].bar(x + w/2, f1s, w, label="Macro F1", color="#1D9E75", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, fontsize=9)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Performance by Strategy")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.4, 0.9)

    for i, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        axes[0].text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=7)

    # Per-class recall comparison
    subtypes = list(results[strategies[0]]["per_class"].keys())
    x = np.arange(len(subtypes))
    width = 0.8 / len(strategies)
    colors = ["#E24B4A", "#BA7517", "#1D9E75"]

    for i, strategy in enumerate(strategies):
        recalls = [results[strategy]["per_class"].get(s, 0) for s in subtypes]
        axes[1].bar(x + i * width - 0.4 + width/2, recalls, width,
                    label=strategy, color=colors[i % len(colors)], alpha=0.8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subtypes, fontsize=9)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Per-Subtype Recall")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1.1)

    plt.suptitle("Domain Adaptation: TCGA → METABRIC Cross-Platform", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved: {save_path}")


def main():
    print("=" * 60)
    print("Domain Adaptation: CORAL for Cross-Platform Validation")
    print("=" * 60)

    tcga_expr, tcga_labels, meta_expr, meta_labels = load_tcga_and_metabric()

    le = LabelEncoder()
    le.fit(np.concatenate([tcga_labels.values, meta_labels.values]))
    y_train = le.transform(tcga_labels.values)
    y_test = le.transform(meta_labels.values)
    class_names = le.classes_

    X_source = tcga_expr.values
    X_target = meta_expr.values

    results = {}

    # Strategy 1: Independent standardization (current baseline)
    print("\n--- Strategy 1: Independent Standardization ---")
    scaler_s = StandardScaler()
    scaler_t = StandardScaler()
    X_s1 = scaler_s.fit_transform(X_source)
    X_t1 = scaler_t.fit_transform(X_target)
    acc, f1, _, per_class = train_and_evaluate(X_s1, y_train, X_t1, y_test, class_names, "Independent Scaling")
    results["Independent\nScaling"] = {"accuracy": acc, "macro_f1": f1, "per_class": per_class}

    # Strategy 2: CORAL domain adaptation
    print("\n--- Strategy 2: CORAL Domain Adaptation ---")
    # First standardize independently, then apply CORAL
    X_source_std = StandardScaler().fit_transform(X_source)
    X_target_std = StandardScaler().fit_transform(X_target)

    print("  Applying CORAL alignment...")
    X_source_coral = coral_transform(X_source_std, X_target_std)
    acc, f1, _, per_class = train_and_evaluate(
        X_source_coral, y_train, X_target_std, y_test, class_names, "CORAL"
    )
    results["CORAL"] = {"accuracy": acc, "macro_f1": f1, "per_class": per_class}

    # Strategy 3: Reverse CORAL (align target to source)
    print("\n--- Strategy 3: Reverse CORAL (align METABRIC to TCGA) ---")
    X_target_coral = coral_transform(X_target_std, X_source_std)
    # Train on source (std), test on CORAL-aligned target
    acc, f1, _, per_class = train_and_evaluate(
        X_source_std, y_train, X_target_coral, y_test, class_names, "Reverse CORAL"
    )
    results["Reverse\nCORAL"] = {"accuracy": acc, "macro_f1": f1, "per_class": per_class}

    # Summary
    print(f"\n{'=' * 60}")
    print("DOMAIN ADAPTATION SUMMARY")
    print(f"{'=' * 60}")
    for strategy, r in results.items():
        s_clean = strategy.replace("\n", " ")
        print(f"  {s_clean:25s}: Acc {r['accuracy']:.4f} | F1 {r['macro_f1']:.4f}")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\n  Best strategy: {best.replace(chr(10), ' ')}")

    plot_domain_comparison(
        results,
        os.path.join(config.RESULTS_FIGURES, "domain_adaptation_comparison.png"),
    )

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    save_results = {
        k.replace("\n", " "): {
            "accuracy": float(v["accuracy"]),
            "macro_f1": float(v["macro_f1"]),
            "per_class_recall": {kk: float(vv) for kk, vv in v["per_class"].items()},
        }
        for k, v in results.items()
    }
    with open(os.path.join(config.RESULTS_TABLES, "domain_adaptation_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    print("\nDomain adaptation complete.")


if __name__ == "__main__":
    main()
