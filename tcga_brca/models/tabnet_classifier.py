"""
TabNet Classifier — DL Baseline + Attention vs SHAP Comparison
===============================================================
Train TabNet on PAM50 classification and compare its built-in
attention-based feature importance with XGBoost SHAP importance.

"Do two fundamentally different interpretation methods point
to the same genes?"

Usage:
    python -m tcga_brca.models.tabnet_classifier
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_data():
    """Load expression and labels."""
    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")

    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    common = expr.index.intersection(labels.index)
    expr, labels = expr.loc[common], labels.loc[common]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    return expr, y, le.classes_, le, expr.columns.tolist()


def train_tabnet(X_train, y_train, X_val, y_val, n_classes, feature_names):
    """Train TabNet with self-supervised pretraining + fine-tuning."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.pretraining import TabNetPretrainer
    import torch

    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (n_classes * class_counts)
    sample_weights = np.array([weights[c] for c in y_train])

    # Phase 1: Self-supervised pretraining (learns feature structure)
    print("\n  Phase 1: Self-supervised pretraining...")
    pretrainer = TabNetPretrainer(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        verbose=0,
    )
    pretrainer.fit(
        X_train=X_train,
        eval_set=[X_val],
        max_epochs=200,
        patience=30,
        batch_size=128,
        virtual_batch_size=32,
        pretraining_ratio=0.5,
    )

    # Phase 2: Fine-tuning with pretrained weights
    print("  Phase 2: Fine-tuning classifier...")
    model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        lambda_sparse=1e-4,
        momentum=0.3, clip_value=2.0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=5e-3, weight_decay=1e-5),
        scheduler_params={"step_size": 15, "gamma": 0.85},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=config.RANDOM_STATE,
        verbose=10,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["accuracy"],
        max_epochs=300,
        patience=40,
        batch_size=256,
        virtual_batch_size=64,
        weights=sample_weights,
        drop_last=False,
        from_unsupervised=pretrainer,
    )

    return model


def extract_tabnet_importance(model, feature_names):
    """Extract TabNet attention-based feature importance."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "gene": feature_names,
        "tabnet_importance": importance,
    }).sort_values("tabnet_importance", ascending=False)
    importance_df["tabnet_rank"] = range(1, len(importance_df) + 1)
    return importance_df


def compare_with_shap(tabnet_importance, top_n=20):
    """Compare TabNet attention importance with SHAP importance."""
    shap_path = os.path.join(config.RESULTS_TABLES, "shap_global_importance.csv")
    if not os.path.exists(shap_path):
        print("SHAP results not found. Run shap_analysis first.")
        return None

    shap_df = pd.read_csv(shap_path)

    merged = tabnet_importance.merge(
        shap_df[["gene", "mean_abs_shap", "rank"]].rename(columns={"rank": "shap_rank"}),
        on="gene", how="inner",
    )

    tabnet_top = set(merged.nsmallest(top_n, "tabnet_rank")["gene"])
    shap_top = set(merged.nsmallest(top_n, "shap_rank")["gene"])
    overlap = tabnet_top & shap_top

    print(f"\n{'=' * 60}")
    print(f"TABNET ATTENTION vs SHAP COMPARISON (Top {top_n})")
    print(f"{'=' * 60}")
    print(f"TabNet top {top_n}: {len(tabnet_top)} genes")
    print(f"SHAP top {top_n}:   {len(shap_top)} genes")
    print(f"Overlap:            {len(overlap)} genes ({len(overlap)/top_n*100:.0f}%)")

    if overlap:
        print(f"\nConsensus genes (both methods agree):")
        for gene in overlap:
            t_rank = merged[merged["gene"] == gene]["tabnet_rank"].values[0]
            s_rank = merged[merged["gene"] == gene]["shap_rank"].values[0]
            print(f"  {gene:15s} TabNet rank {t_rank:2d} | SHAP rank {s_rank:2d}")

    tabnet_only = tabnet_top - shap_top
    shap_only = shap_top - tabnet_top
    if tabnet_only:
        print(f"\nTabNet-only genes (attention captures, SHAP misses):")
        for gene in list(tabnet_only)[:5]:
            t_rank = merged[merged["gene"] == gene]["tabnet_rank"].values[0]
            s_rank = merged[merged["gene"] == gene]["shap_rank"].values[0]
            print(f"  {gene:15s} TabNet rank {t_rank:2d} | SHAP rank {s_rank:2d}")

    # Rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(merged["tabnet_rank"], merged["shap_rank"])
    print(f"\nSpearman rank correlation: rho={rho:.4f}, p={pval:.2e}")

    return merged, overlap, rho


def plot_comparison(merged, overlap, save_dir):
    """Visualize TabNet vs SHAP comparison."""
    os.makedirs(save_dir, exist_ok=True)

    top50 = merged.nsmallest(50, "tabnet_rank")

    fig, axes = plt.subplots(1, 2, figsize=(config.FIGURE_SIZE[0] * 1.4, config.FIGURE_SIZE[1]),
                              dpi=config.FIGURE_DPI)

    # Scatter: rank correlation
    colors = ["#1D9E75" if g in overlap else "#378ADD" for g in merged["gene"]]
    axes[0].scatter(merged["shap_rank"], merged["tabnet_rank"],
                    c=colors, alpha=0.3, s=8, edgecolors="none")
    for gene in overlap:
        row = merged[merged["gene"] == gene].iloc[0]
        axes[0].annotate(gene, (row["shap_rank"], row["tabnet_rank"]),
                         fontsize=6, alpha=0.8)
    axes[0].set_xlabel("SHAP Rank")
    axes[0].set_ylabel("TabNet Attention Rank")
    axes[0].set_title("Feature Rank Correlation")
    max_rank = min(200, len(merged))
    axes[0].set_xlim(0, max_rank)
    axes[0].set_ylim(0, max_rank)
    axes[0].plot([0, max_rank], [0, max_rank], "k--", alpha=0.3, lw=0.8)

    # Bar: top 20 comparison
    top20_t = merged.nsmallest(20, "tabnet_rank")[["gene", "tabnet_importance"]].set_index("gene")
    top20_s_genes = merged.nsmallest(20, "shap_rank")["gene"].tolist()

    importance_compare = pd.DataFrame(index=list(set(top20_t.index) | set(top20_s_genes)))
    importance_compare["TabNet"] = merged.set_index("gene").loc[importance_compare.index, "tabnet_importance"]
    importance_compare["SHAP"] = merged.set_index("gene").loc[importance_compare.index, "mean_abs_shap"]

    # Normalize for visual comparison
    importance_compare["TabNet_norm"] = importance_compare["TabNet"] / importance_compare["TabNet"].max()
    importance_compare["SHAP_norm"] = importance_compare["SHAP"] / importance_compare["SHAP"].max()
    importance_compare = importance_compare.sort_values("SHAP_norm", ascending=True).tail(20)

    y_pos = range(len(importance_compare))
    axes[1].barh(y_pos, importance_compare["SHAP_norm"], height=0.4, color="#378ADD",
                 alpha=0.8, label="SHAP (normalized)")
    axes[1].barh([y + 0.4 for y in y_pos], importance_compare["TabNet_norm"], height=0.4,
                 color="#1D9E75", alpha=0.8, label="TabNet (normalized)")
    axes[1].set_yticks([y + 0.2 for y in y_pos])
    axes[1].set_yticklabels(importance_compare.index, fontsize=7)
    axes[1].set_xlabel("Normalized Importance")
    axes[1].set_title("Top Genes: SHAP vs TabNet")
    axes[1].legend(fontsize=8)

    plt.suptitle("XGBoost SHAP vs TabNet Attention — Feature Importance Comparison", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "tabnet_vs_shap_comparison.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plots saved to {save_dir}")


def main():
    print("=" * 60)
    print("TabNet Classifier + Attention vs SHAP Comparison")
    print("=" * 60)

    expr, y, class_names, le, feature_names = load_data()
    n_classes = len(class_names)
    print(f"Data: {expr.shape[0]} samples x {expr.shape[1]} features, {n_classes} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        expr.values, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train TabNet
    print("\nTraining TabNet...")
    model = train_tabnet(X_train_s, y_train, X_test_s, y_test, n_classes, feature_names)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nTabNet Results:")
    print(f"  Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    short_names = [c.replace("BRCA_", "") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"TabNet — Confusion Matrix\nAccuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    plt.tight_layout()
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)
    fig.savefig(os.path.join(config.RESULTS_FIGURES, "tabnet_confusion_matrix.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # Extract TabNet attention importance
    tabnet_importance = extract_tabnet_importance(model, feature_names)

    # Compare with SHAP
    result = compare_with_shap(tabnet_importance)
    if result:
        merged, overlap, rho = result
        plot_comparison(merged, overlap, config.RESULTS_FIGURES)

        # Save
        os.makedirs(config.RESULTS_TABLES, exist_ok=True)
        tabnet_importance.to_csv(
            os.path.join(config.RESULTS_TABLES, "tabnet_feature_importance.csv"), index=False
        )
        merged.to_csv(
            os.path.join(config.RESULTS_TABLES, "tabnet_vs_shap_comparison.csv"), index=False
        )

        summary = {
            "tabnet_accuracy": float(acc),
            "tabnet_macro_f1": float(f1),
            "top20_overlap_with_shap": len(overlap),
            "spearman_rho": float(rho),
            "consensus_genes": list(overlap),
        }
        with open(os.path.join(config.RESULTS_TABLES, "tabnet_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\nTabNet classifier complete.")


if __name__ == "__main__":
    main()
