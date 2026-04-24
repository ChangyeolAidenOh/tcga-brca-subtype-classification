"""
Stage 3-1: Feature Selection + PCA Exploratory Analysis
========================================================
Statistical feature selection from GEO-filtered genes and
PCA visualization of subtype separation.

Usage:
    python -m tcga_brca.features.feature_selection
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_processed_data():
    """Load expression matrix and labels from Stage 2 output."""
    expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv"), index_col=0
    )
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()
    print(f"Loaded: {expr.shape[0]} samples x {expr.shape[1]} genes")
    print(f"Labels: {labels.value_counts().to_dict()}")
    return expr, labels


def statistical_selection(expr: pd.DataFrame, labels: pd.Series, p_threshold: float = 0.01) -> list:
    """Kruskal-Wallis test per gene across PAM50 subtypes.

    Non-parametric test chosen because gene expression distributions
    are often non-normal — same logic as GAM project's non-linearity
    diagnosis before model selection.
    """
    print(f"\nRunning Kruskal-Wallis test across {expr.shape[1]} genes...")
    groups = [expr.loc[labels == st] for st in labels.unique()]

    results = []
    for gene in expr.columns:
        gene_groups = [g[gene].dropna().values for g in groups if len(g[gene].dropna()) >= 3]
        if len(gene_groups) < 2:
            continue
        stat, pval = kruskal(*gene_groups)
        results.append({"gene": gene, "kw_stat": stat, "kw_pvalue": pval})

    kw_df = pd.DataFrame(results)

    from statsmodels.stats.multitest import multipletests
    kw_df = kw_df.dropna(subset=["kw_pvalue"]).reset_index(drop=True)
    _, adj_pvals, _, _ = multipletests(kw_df["kw_pvalue"].values, method="fdr_bh")
    kw_df["kw_adj_pvalue"] = adj_pvals

    sig_genes = kw_df[kw_df["kw_adj_pvalue"] < p_threshold].sort_values("kw_adj_pvalue")
    print(f"Genes significant at adj.p < {p_threshold}: {len(sig_genes)} / {len(kw_df)}")

    kw_df.to_csv(os.path.join(config.DATA_PROCESSED, "kruskal_wallis_results.csv"), index=False)
    return sig_genes["gene"].tolist()


def run_pca(expr: pd.DataFrame, labels: pd.Series, save_dir: str):
    """PCA visualization of subtype separation."""
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr)

    pca = PCA(n_components=min(50, expr.shape[1]))
    pca_result = pca.fit_transform(expr_scaled)

    # Variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cumvar >= 0.90) + 1
    n_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"\nPCA: 90% variance at {n_90} components, 95% at {n_95}")

    # Cumulative variance plot
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", markersize=3, color="#378ADD")
    ax.axhline(0.90, ls="--", color="#888780", lw=0.8)
    ax.axhline(0.95, ls="--", color="#888780", lw=0.8)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — Cumulative explained variance")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "pca_cumulative_variance.png"), dpi=config.FIGURE_DPI)
    plt.close(fig)

    # 2D scatter by subtype
    subtype_colors = {
        "BRCA_LumA": "#378ADD",
        "BRCA_LumB": "#1D9E75",
        "BRCA_Basal": "#E24B4A",
        "BRCA_Her2": "#BA7517",
        "BRCA_Normal": "#7F77DD",
    }

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    for st, color in subtype_colors.items():
        mask = labels.values == st
        if mask.sum() == 0:
            continue
        ax.scatter(
            pca_result[mask, 0], pca_result[mask, 1],
            c=color, label=st.replace("BRCA_", ""), alpha=0.6, s=20, edgecolors="none"
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — PAM50 subtype separation")
    ax.legend(fontsize=9, framealpha=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "pca_subtype_scatter.png"), dpi=config.FIGURE_DPI)
    plt.close(fig)

    print(f"PCA plots saved to {save_dir}")

    # Save PCA-transformed data
    pca_df = pd.DataFrame(
        pca_result[:, :n_95],
        index=expr.index,
        columns=[f"PC{i+1}" for i in range(n_95)],
    )
    pca_df.to_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_pca.csv"))
    return pca_df


def main():
    print("=" * 60)
    print("Stage 3-1: Feature Selection + PCA Analysis")
    print("=" * 60)

    expr, labels = load_processed_data()

    # Statistical selection
    sig_genes = statistical_selection(expr, labels, p_threshold=config.DEG_PVALUE_THRESHOLD)

    # Save statistically selected gene set
    sig_expr = expr[sig_genes]
    sig_expr.to_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"))
    print(f"\nStatistically selected expression saved: {sig_expr.shape}")

    # PCA on selected features
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)
    run_pca(sig_expr, labels, config.RESULTS_FIGURES)

    print("\nStage 3-1 complete.")


if __name__ == "__main__":
    main()
