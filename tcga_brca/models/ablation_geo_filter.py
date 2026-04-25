"""
Ablation Study: GEO DEG Filter ON vs OFF
==========================================
Compare classification performance with and without GEO DEG
feature filtering to validate the filtering strategy.

Usage:
    python -m tcga_brca.models.ablation_geo_filter
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_both_datasets():
    """Load GEO-filtered and unfiltered expression matrices."""
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    # GEO-filtered (925 genes)
    filtered_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if os.path.exists(filtered_path):
        filtered = pd.read_csv(filtered_path, index_col=0)
    else:
        filtered = pd.read_csv(
            os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv"), index_col=0
        )

    # Unfiltered — reload from full expression before GEO filter
    full_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_full.csv")
    if not os.path.exists(full_path):
        print("Full (unfiltered) expression not found. Generating...")
        generate_unfiltered_dataset(labels, full_path)

    unfiltered = pd.read_csv(full_path, index_col=0)

    common = filtered.index.intersection(unfiltered.index).intersection(labels.index)
    filtered = filtered.loc[common]
    unfiltered = unfiltered.loc[common]
    labels = labels.loc[common]

    print(f"GEO-filtered: {filtered.shape[0]} samples x {filtered.shape[1]} genes")
    print(f"Unfiltered:   {unfiltered.shape[0]} samples x {unfiltered.shape[1]} genes")

    return filtered, unfiltered, labels


def generate_unfiltered_dataset(labels, save_path):
    """Rebuild expression matrix without GEO DEG filter."""
    cache_path = os.path.join(config.DATA_RAW, "cbio_expression_long.csv")
    if not os.path.exists(cache_path):
        print("  ERROR: No cached expression data. Re-run Stage 2 first.")
        return

    print("  Loading cached long-format expression (this may take a moment)...")
    expr_long = pd.read_csv(cache_path)

    # Need gene symbols — check if already mapped
    if "gene" in expr_long.columns:
        expr = expr_long.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")
    elif "entrezGeneId" in expr_long.columns:
        # Re-map entrez to gene symbol
        import requests
        print("  Mapping entrez IDs to gene symbols...")
        unique_ids = expr_long["entrezGeneId"].unique().tolist()
        gene_map = {}
        headers = {"Content-Type": "application/json"}
        for i in range(0, len(unique_ids), 1000):
            batch = unique_ids[i:i + 1000]
            resp = requests.post(
                f"{config.CBIO_API_ENDPOINT}/genes/fetch",
                json=[str(g) for g in batch], headers=headers, timeout=60,
            )
            resp.raise_for_status()
            for g in resp.json():
                gene_map[g["entrezGeneId"]] = g["hugoGeneSymbol"]
        expr_long["gene"] = expr_long["entrezGeneId"].map(gene_map)
        expr_long = expr_long.dropna(subset=["gene"])
        expr = expr_long.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")
    else:
        print("  ERROR: Unexpected columns:", expr_long.columns.tolist())
        return

    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]

    if expr.max().max() > 30:
        expr = np.log2(expr + config.LOG2_PSEUDOCOUNT)

    medians = expr.median(axis=0)
    threshold = np.log2(config.LOW_EXPRESSION_THRESHOLD + config.LOG2_PSEUDOCOUNT)
    expr = expr.loc[:, medians >= threshold]

    expr.to_csv(save_path)
    print(f"  Unfiltered expression saved: {expr.shape}")


def run_cv_comparison(X, y, label, class_names):
    """Run 5-fold CV with XGBoost + class weights."""
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    fold_results = []
    per_class_recalls = {cls: [] for cls in class_names}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        sw = compute_sample_weight("balanced", y_tr)

        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        fold_results.append({"fold": fold + 1, "accuracy": acc, "macro_f1": f1})

        report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        for cls in class_names:
            per_class_recalls[cls].append(report[cls]["recall"])

    results_df = pd.DataFrame(fold_results)
    mean_acc = results_df["accuracy"].mean()
    std_acc = results_df["accuracy"].std()
    mean_f1 = results_df["macro_f1"].mean()
    std_f1 = results_df["macro_f1"].std()

    print(f"\n  {label}:")
    print(f"    Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    for cls in class_names:
        r = np.mean(per_class_recalls[cls])
        print(f"    {cls.replace('BRCA_', ''):10s} Recall: {r:.4f}")

    return {
        "condition": label,
        "n_features": X.shape[1],
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "per_class_recall": {cls: float(np.mean(per_class_recalls[cls])) for cls in class_names},
    }


def plot_ablation(result_filtered, result_unfiltered, save_path):
    """Plot side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(config.FIGURE_SIZE[0] * 1.4, config.FIGURE_SIZE[1]), dpi=config.FIGURE_DPI)

    # Bar chart: overall metrics
    labels_plot = ["GEO-filtered\n(925 genes)", "Unfiltered\n(~16K genes)"]
    accs = [result_filtered["accuracy_mean"], result_unfiltered["accuracy_mean"]]
    f1s = [result_filtered["macro_f1_mean"], result_unfiltered["macro_f1_mean"]]
    acc_errs = [result_filtered["accuracy_std"], result_unfiltered["accuracy_std"]]
    f1_errs = [result_filtered["macro_f1_std"], result_unfiltered["macro_f1_std"]]

    x = np.arange(2)
    w = 0.3
    axes[0].bar(x - w/2, accs, w, yerr=acc_errs, label="Accuracy", color="#378ADD", alpha=0.8, capsize=4)
    axes[0].bar(x + w/2, f1s, w, yerr=f1_errs, label="Macro F1", color="#1D9E75", alpha=0.8, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_plot, fontsize=10)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Performance")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0.7, 1.0)

    # Per-class recall comparison
    class_names = list(result_filtered["per_class_recall"].keys())
    short_names = [c.replace("BRCA_", "") for c in class_names]
    recalls_f = [result_filtered["per_class_recall"][c] for c in class_names]
    recalls_u = [result_unfiltered["per_class_recall"][c] for c in class_names]

    x = np.arange(len(class_names))
    axes[1].bar(x - w/2, recalls_f, w, label="GEO-filtered", color="#378ADD", alpha=0.8)
    axes[1].bar(x + w/2, recalls_u, w, label="Unfiltered", color="#BA7517", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=9)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Per-Class Recall")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0.3, 1.05)

    plt.suptitle("Ablation Study: GEO DEG Filter Effect", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nAblation plot saved: {save_path}")


def main():
    print("=" * 60)
    print("Ablation Study: GEO DEG Filter ON vs OFF")
    print("=" * 60)

    filtered, unfiltered, labels = load_both_datasets()

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    print(f"\nRunning {config.CV_FOLDS}-fold CV comparison...")
    result_f = run_cv_comparison(filtered.values, y, "GEO-filtered (925 genes)", class_names)
    result_u = run_cv_comparison(unfiltered.values, y, f"Unfiltered ({unfiltered.shape[1]} genes)", class_names)

    # Summary
    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 60}")
    f1_diff = result_f["macro_f1_mean"] - result_u["macro_f1_mean"]
    if f1_diff > 0:
        print(f"GEO filter IMPROVED Macro F1 by {f1_diff:.4f}")
    else:
        print(f"GEO filter DECREASED Macro F1 by {abs(f1_diff):.4f}")
    print(f"Feature reduction: {result_u['n_features']} → {result_f['n_features']} ({result_f['n_features']/result_u['n_features']*100:.1f}%)")

    plot_ablation(
        result_f, result_u,
        os.path.join(config.RESULTS_FIGURES, "ablation_geo_filter.png"),
    )

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    summary = {"filtered": result_f, "unfiltered": result_u}
    with open(os.path.join(config.RESULTS_TABLES, "ablation_geo_filter.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
