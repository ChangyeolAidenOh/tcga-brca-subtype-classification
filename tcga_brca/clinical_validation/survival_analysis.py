"""
Stage 5: Kaplan-Meier Survival Analysis
========================================
Validate SHAP top genes for clinical significance by comparing
overall survival between high/low expression groups.

This stage answers: "Does the ML-identified biomarker actually
matter for patient outcomes?" — the core of translational research.

Usage:
    python -m tcga_brca.clinical_validation.survival_analysis
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_data():
    """Load expression, survival data, and SHAP top genes."""
    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")

    expr = pd.read_csv(expr_path, index_col=0)
    surv = pd.read_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv"))

    # Load SHAP top genes
    top_genes_path = os.path.join(config.DATA_PROCESSED, "shap_top_genes.txt")
    with open(top_genes_path) as f:
        top_genes = [line.strip() for line in f if line.strip()]

    # Align samples: expression index uses sample IDs, survival uses sampleId column
    # TCGA sample IDs might need trimming (e.g., TCGA-A2-A0T2-01)
    surv_indexed = surv.set_index("sampleId")
    common = expr.index.intersection(surv_indexed.index)

    expr_aligned = expr.loc[common]
    surv_aligned = surv_indexed.loc[common]

    # Filter top genes that exist in expression data
    available_genes = [g for g in top_genes if g in expr_aligned.columns]

    print(f"Expression: {expr_aligned.shape}")
    print(f"Survival data: {surv_aligned.shape}")
    print(f"SHAP top genes available: {len(available_genes)} / {len(top_genes)}")
    print(f"Events: {int(surv_aligned['os_event'].sum())} / {len(surv_aligned)}")

    return expr_aligned, surv_aligned, available_genes


def run_km_analysis(expr, surv, gene, split_method="median"):
    """Run KM analysis for a single gene.

    Split patients into high/low groups by median expression,
    compare survival curves with log-rank test.
    """
    gene_expr = expr[gene]

    if split_method == "median":
        threshold = gene_expr.median()
    elif split_method == "mean":
        threshold = gene_expr.mean()
    else:
        threshold = gene_expr.quantile(0.5)

    high_mask = gene_expr >= threshold
    low_mask = gene_expr < threshold

    high_surv = surv.loc[high_mask]
    low_surv = surv.loc[low_mask]

    # Log-rank test
    lr = logrank_test(
        high_surv["os_months"], low_surv["os_months"],
        event_observed_A=high_surv["os_event"],
        event_observed_B=low_surv["os_event"],
    )

    return {
        "gene": gene,
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
        "logrank_stat": lr.test_statistic,
        "logrank_pvalue": lr.p_value,
        "significant": lr.p_value < 0.05,
        "threshold": threshold,
    }


def plot_km_curve(expr, surv, gene, result, save_path):
    """Plot KM survival curves for high/low expression groups."""
    gene_expr = expr[gene]
    threshold = result["threshold"]

    high_mask = gene_expr >= threshold
    low_mask = gene_expr < threshold

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(
        surv.loc[high_mask, "os_months"],
        event_observed=surv.loc[high_mask, "os_event"],
        label=f"High (n={result['n_high']})",
    )
    kmf_high.plot_survival_function(ax=ax, color="#E24B4A", ci_show=True)

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(
        surv.loc[low_mask, "os_months"],
        event_observed=surv.loc[low_mask, "os_event"],
        label=f"Low (n={result['n_low']})",
    )
    kmf_low.plot_survival_function(ax=ax, color="#378ADD", ci_show=True)

    sig_marker = "*" if result["significant"] else "ns"
    ax.set_title(
        f"Overall Survival by {gene} Expression\n"
        f"Log-rank p = {result['logrank_pvalue']:.4f} ({sig_marker})",
        fontsize=12,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Stage 5: Kaplan-Meier Survival Analysis")
    print("=" * 60)

    expr, surv, top_genes = load_data()

    results = []
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)

    print(f"\nRunning KM analysis for {len(top_genes)} genes...")
    for gene in top_genes:
        result = run_km_analysis(expr, surv, gene, split_method=config.KM_SPLIT_METHOD)
        results.append(result)

        sig = "***" if result["logrank_pvalue"] < 0.001 else \
              "**" if result["logrank_pvalue"] < 0.01 else \
              "*" if result["logrank_pvalue"] < 0.05 else "ns"
        print(f"  {gene:15s} p={result['logrank_pvalue']:.4f} {sig}")

        # Plot KM curve for each gene
        plot_km_curve(
            expr, surv, gene, result,
            os.path.join(config.RESULTS_FIGURES, f"km_survival_{gene}.png"),
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("logrank_pvalue")

    # Summary
    n_sig = results_df["significant"].sum()
    n_total = len(results_df)

    print(f"\n{'=' * 60}")
    print(f"SURVIVAL ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Genes tested: {n_total}")
    print(f"Significant (p < 0.05): {n_sig} ({n_sig/n_total*100:.1f}%)")
    print(f"Not significant: {n_total - n_sig}")

    if n_sig > 0:
        print(f"\nSignificant genes (prognostic biomarkers):")
        for _, row in results_df[results_df["significant"]].iterrows():
            print(f"  {row['gene']:15s} p={row['logrank_pvalue']:.6f}")

    if n_sig < n_total:
        print(f"\nNon-significant genes (classification-only markers):")
        for _, row in results_df[~results_df["significant"]].head(5).iterrows():
            print(f"  {row['gene']:15s} p={row['logrank_pvalue']:.4f}")

    print(f"\nKey insight: ML classification importance != clinical prognostic value")
    print(f"  {n_sig}/{n_total} SHAP top genes are also prognostic biomarkers")
    print(f"  {n_total - n_sig}/{n_total} are classification-only (subtype separation)")

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    results_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "km_survival_results.csv"), index=False
    )
    print(f"\nResults saved to {config.RESULTS_TABLES}")

    print("\nStage 5 complete.")


if __name__ == "__main__":
    main()
