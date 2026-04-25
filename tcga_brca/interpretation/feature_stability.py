"""
Feature Stability Analysis — Bootstrap SHAP Robustness
========================================================
Run 100 bootstrap iterations, compute SHAP top-20 each time,
and measure how consistently each gene appears (stability score).

Stability 90%+ = robust biomarker
Stability <30% = split-dependent artifact

Usage:
    python -m tcga_brca.interpretation.feature_stability
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

N_BOOTSTRAP = 100
TOP_K = 20


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
    return expr, y, le


def get_shap_top_genes(model, X, feature_names, top_k):
    """Get top-k genes by mean |SHAP| using XGBoost native TreeSHAP."""
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    raw = booster.predict(dmatrix, pred_contribs=True)

    if raw.ndim == 3:
        # shape: (n_samples, n_classes, n_features+1)
        mean_abs = np.mean(np.abs(raw[:, :, :-1]), axis=(0, 1))
    else:
        mean_abs = np.mean(np.abs(raw[:, :-1]), axis=0)

    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    return [feature_names[i] for i in top_idx]


def run_bootstrap_stability(expr, y, n_bootstrap, top_k):
    """Run bootstrap iterations and track gene selection frequency."""
    feature_names = expr.columns.tolist()
    n_samples = len(expr)
    gene_counts = {gene: 0 for gene in feature_names}

    print(f"Running {n_bootstrap} bootstrap iterations (top {top_k} per iteration)...")
    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{n_bootstrap}")

        # Bootstrap sample (with replacement)
        boot_idx = np.random.RandomState(config.RANDOM_STATE + i).choice(
            n_samples, size=n_samples, replace=True
        )
        X_boot = expr.values[boot_idx]
        y_boot = y[boot_idx]

        scaler = StandardScaler()
        X_boot_s = scaler.fit_transform(X_boot)
        sw = compute_sample_weight("balanced", y_boot)

        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_STATE + i,
            use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
        )
        model.fit(X_boot_s, y_boot, sample_weight=sw)

        top_genes = get_shap_top_genes(model, X_boot_s, feature_names, top_k)
        for gene in top_genes:
            gene_counts[gene] += 1

    # Compute stability scores
    stability_df = pd.DataFrame([
        {"gene": gene, "count": count, "stability": count / n_bootstrap}
        for gene, count in gene_counts.items()
    ]).sort_values("stability", ascending=False)

    return stability_df


def merge_with_shap(stability_df):
    """Merge stability scores with original SHAP importance."""
    shap_path = os.path.join(config.RESULTS_TABLES, "shap_global_importance.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        merged = stability_df.merge(
            shap_df[["gene", "mean_abs_shap"]],
            on="gene", how="left"
        )
    else:
        merged = stability_df.copy()
        merged["mean_abs_shap"] = np.nan

    # Load survival info
    km_path = os.path.join(config.RESULTS_TABLES, "km_survival_results.csv")
    if os.path.exists(km_path):
        km_df = pd.read_csv(km_path)
        merged = merged.merge(
            km_df[["gene", "logrank_pvalue", "significant"]],
            on="gene", how="left"
        )

    return merged


def plot_stability(merged_df, save_dir):
    """Plot stability analysis results."""
    os.makedirs(save_dir, exist_ok=True)
    top = merged_df.head(30)

    # Bar chart: stability scores
    fig, ax = plt.subplots(figsize=(config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 1.3), dpi=config.FIGURE_DPI)
    colors = ["#1D9E75" if s >= 0.9 else "#BA7517" if s >= 0.5 else "#E24B4A"
              for s in top["stability"]]
    ax.barh(range(len(top)), top["stability"].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Stability Score (fraction of 100 bootstraps in top 20)")
    ax.set_title("Feature Stability Analysis — Top 30 Genes")
    ax.axvline(0.9, ls="--", color="#888780", lw=0.8, label="Robust (0.9)")
    ax.axvline(0.5, ls="--", color="#888780", lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "feature_stability_bar.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2D scatter: SHAP importance vs stability
    if "mean_abs_shap" in merged_df.columns and merged_df["mean_abs_shap"].notna().any():
        plot_df = merged_df[merged_df["mean_abs_shap"].notna()].head(50)
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)

        colors_scatter = []
        for _, row in plot_df.iterrows():
            if row.get("significant", False):
                colors_scatter.append("#1D9E75")
            else:
                colors_scatter.append("#378ADD")

        ax.scatter(
            plot_df["mean_abs_shap"], plot_df["stability"],
            c=colors_scatter, alpha=0.7, s=40, edgecolors="none"
        )

        # Label top genes
        for _, row in plot_df.head(15).iterrows():
            ax.annotate(
                row["gene"], (row["mean_abs_shap"], row["stability"]),
                fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points",
            )

        ax.set_xlabel("SHAP Importance (mean |SHAP|)")
        ax.set_ylabel("Stability Score (100 bootstraps)")
        ax.set_title("Biomarker Quality: Importance vs Stability\nGreen = prognostic (KM p<0.05)")
        ax.axhline(0.9, ls="--", color="#888780", lw=0.8)
        ax.axhline(0.5, ls="--", color="#888780", lw=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "feature_stability_2d.png"),
                    dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    print(f"Stability plots saved to {save_dir}")


def main():
    print("=" * 60)
    print("Feature Stability Analysis (100 Bootstrap Iterations)")
    print("=" * 60)

    expr, y, le = load_data()
    print(f"Data: {expr.shape[0]} samples x {expr.shape[1]} genes")

    stability_df = run_bootstrap_stability(expr, y, N_BOOTSTRAP, TOP_K)

    # Summary
    n_robust = (stability_df["stability"] >= 0.9).sum()
    n_moderate = ((stability_df["stability"] >= 0.5) & (stability_df["stability"] < 0.9)).sum()
    n_unstable = ((stability_df["stability"] > 0) & (stability_df["stability"] < 0.5)).sum()

    print(f"\n{'=' * 60}")
    print("STABILITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Robust (>=90%):   {n_robust} genes")
    print(f"Moderate (50-90%): {n_moderate} genes")
    print(f"Unstable (<50%):  {n_unstable} genes (appeared at least once)")
    print(f"Never selected:   {(stability_df['stability'] == 0).sum()} genes")

    print(f"\nTop 10 most stable genes:")
    for _, row in stability_df.head(10).iterrows():
        print(f"  {row['gene']:15s} stability: {row['stability']:.2f} ({row['count']}/{N_BOOTSTRAP})")

    # Merge with SHAP and survival data
    merged = merge_with_shap(stability_df)

    # Plot
    plot_stability(merged, config.RESULTS_FIGURES)

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    merged.to_csv(
        os.path.join(config.RESULTS_TABLES, "feature_stability_results.csv"), index=False
    )

    # Identify "gold standard" biomarkers: high SHAP + high stability + prognostic
    if "significant" in merged.columns:
        gold = merged[
            (merged["stability"] >= 0.7) &
            (merged["significant"] == True)
        ]
        if len(gold) > 0:
            print(f"\nGold standard biomarkers (stable + prognostic):")
            for _, row in gold.iterrows():
                print(f"  {row['gene']:15s} stability: {row['stability']:.2f} | KM p={row['logrank_pvalue']:.4f}")

    print("\nFeature stability analysis complete.")


if __name__ == "__main__":
    main()
