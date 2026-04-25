"""
Cox Proportional Hazards Regression — Multivariate Survival
==============================================================
Extends Stage 5 (KM univariate) with multivariate Cox regression.

Three analyses:
  1. Univariate Cox: each SHAP top gene individually
  2. Multivariate Cox: SHAP top genes + clinical covariates (age, stage)
  3. Multi-task gene validation: genes from multi-task model vs XGBoost SHAP

Connects to multi-task survival model by validating whether the
Cox partial likelihood loss learned biologically meaningful features.

Usage:
    python -m tcga_brca.clinical_validation.cox_regression
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_data():
    """Load expression, survival data, and SHAP/multi-task top genes."""
    expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    surv = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv")
    )
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    surv_indexed = surv.set_index("sampleId")
    common = expr.index.intersection(surv_indexed.index).intersection(labels.index)
    expr = expr.loc[common]
    surv_aligned = surv_indexed.loc[common]
    labels = labels.loc[common]

    # SHAP top genes
    shap_genes = []
    shap_path = os.path.join(config.DATA_PROCESSED, "shap_top_genes.txt")
    if os.path.exists(shap_path):
        with open(shap_path) as f:
            shap_genes = [line.strip() for line in f if line.strip()]

    # Multi-task top genes (if available)
    mt_genes = []
    mt_path = os.path.join(config.RESULTS_TABLES, "multitask_feature_importance.csv")
    if os.path.exists(mt_path):
        mt_df = pd.read_csv(mt_path)
        mt_genes = mt_df.head(20)["gene"].tolist()

    print(f"Data: {expr.shape[0]} samples, {expr.shape[1]} genes")
    print(f"Events: {int(surv_aligned['os_event'].sum())} / {len(surv_aligned)}")
    print(f"SHAP top genes: {len(shap_genes)}")
    print(f"Multi-task top genes: {len(mt_genes)}")

    return expr, surv_aligned, labels, shap_genes, mt_genes


def run_univariate_cox(expr, surv, genes):
    """Run univariate Cox regression for each gene.

    Compared with KM log-rank: Cox gives hazard ratios (HR),
    which quantify the magnitude of survival effect.
    HR > 1: higher expression → worse prognosis
    HR < 1: higher expression → better prognosis
    """
    print(f"\n{'=' * 60}")
    print("UNIVARIATE COX REGRESSION")
    print(f"{'=' * 60}")

    results = []
    for gene in genes:
        if gene not in expr.columns:
            continue

        df = pd.DataFrame({
            "T": surv["os_months"].values,
            "E": surv["os_event"].values,
            gene: expr[gene].values,
        }).dropna()

        if df["E"].sum() < 5:
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="T", event_col="E")

            summary = cph.summary
            hr = float(np.exp(summary["coef"].values[0]))
            pval = float(summary["p"].values[0])
            ci_lower = float(np.exp(summary["coef lower 95%"].values[0]))
            ci_upper = float(np.exp(summary["coef upper 95%"].values[0]))

            results.append({
                "gene": gene,
                "hazard_ratio": hr,
                "hr_ci_lower": ci_lower,
                "hr_ci_upper": ci_upper,
                "cox_pvalue": pval,
                "cox_significant": pval < 0.05,
                "direction": "risk" if hr > 1 else "protective",
            })

            sig = "*" if pval < 0.05 else "ns"
            direction = "risk" if hr > 1 else "protective"
            print(f"  {gene:15s} HR={hr:.3f} [{ci_lower:.3f}-{ci_upper:.3f}] p={pval:.4f} {sig} ({direction})")

        except Exception as e:
            print(f"  {gene:15s} Cox failed: {e}")

    return pd.DataFrame(results)


def run_multivariate_cox(expr, surv, labels, genes, top_n=5):
    """Multivariate Cox regression with top genes + clinical covariates.

    Answers: "Are these genes independently prognostic after
    controlling for known clinical factors?"
    """
    print(f"\n{'=' * 60}")
    print("MULTIVARIATE COX REGRESSION")
    print(f"{'=' * 60}")

    # Select top significant genes from univariate
    available_genes = [g for g in genes[:top_n] if g in expr.columns]

    df = pd.DataFrame({
        "T": surv["os_months"].values,
        "E": surv["os_event"].values,
    }, index=expr.index)

    # Add gene expression
    for gene in available_genes:
        df[gene] = expr[gene].values

    # Add subtype as covariate (one-hot, drop reference)
    subtype_dummies = pd.get_dummies(labels, prefix="subtype", drop_first=True)
    subtype_dummies.index = df.index
    df = pd.concat([df, subtype_dummies], axis=1)

    df = df.dropna()

    print(f"Covariates: {available_genes} + subtypes")
    print(f"Samples: {len(df)}, Events: {int(df['E'].sum())}")

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df, duration_col="T", event_col="E")

        print(f"\nMultivariate Cox Results:")
        print(cph.summary[["coef", "exp(coef)", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"]].to_string())

        # Which genes remain significant after controlling for subtype?
        gene_results = []
        for gene in available_genes:
            if gene in cph.summary.index:
                row = cph.summary.loc[gene]
                hr = float(np.exp(row["coef"]))
                pval = float(row["p"])
                sig = pval < 0.05
                print(f"\n  {gene}: HR={hr:.3f}, p={pval:.4f} {'*' if sig else 'ns'}")
                print(f"    → {'Independently prognostic' if sig else 'Not independent (confounded by subtype)'}")
                gene_results.append({
                    "gene": gene,
                    "multivariate_hr": hr,
                    "multivariate_pvalue": pval,
                    "independent": sig,
                })

        return pd.DataFrame(gene_results), cph

    except Exception as e:
        print(f"Multivariate Cox failed: {e}")
        return pd.DataFrame(), None


def compare_shap_vs_multitask_survival(univariate_results, shap_genes, mt_genes):
    """Compare survival relevance of XGBoost SHAP vs multi-task top genes."""
    print(f"\n{'=' * 60}")
    print("SHAP vs MULTI-TASK: SURVIVAL RELEVANCE COMPARISON")
    print(f"{'=' * 60}")

    if not mt_genes:
        print("Multi-task genes not available. Run multitask_survival first.")
        return

    # SHAP top 20: how many are Cox-significant?
    shap_set = set(shap_genes[:20])
    mt_set = set(mt_genes[:20])

    shap_sig = univariate_results[
        (univariate_results["gene"].isin(shap_set)) &
        (univariate_results["cox_significant"])
    ]
    mt_sig = univariate_results[
        (univariate_results["gene"].isin(mt_set)) &
        (univariate_results["cox_significant"])
    ]

    print(f"SHAP top 20:      {len(shap_sig)}/20 Cox-significant")
    print(f"Multi-task top 20: {len(mt_sig)}/20 Cox-significant")

    if len(mt_sig) > len(shap_sig):
        print("→ Multi-task model identifies MORE prognostic genes")
        print("  (Cox survival loss guides the model toward clinically relevant features)")
    elif len(mt_sig) < len(shap_sig):
        print("→ SHAP (classification-only) identifies more prognostic genes")
        print("  (classification importance partially correlates with prognosis)")
    else:
        print("→ Both methods identify similar number of prognostic genes")

    # Overlap
    overlap = shap_set & mt_set
    overlap_sig = univariate_results[
        (univariate_results["gene"].isin(overlap)) &
        (univariate_results["cox_significant"])
    ]
    if overlap:
        print(f"\nGenes in both top 20: {len(overlap)}")
        print(f"  Of which Cox-significant: {len(overlap_sig)}")
        for _, row in overlap_sig.iterrows():
            print(f"    {row['gene']:15s} HR={row['hazard_ratio']:.3f} p={row['cox_pvalue']:.4f}")


def plot_forest(univariate_results, save_path):
    """Forest plot of hazard ratios."""
    sig_results = univariate_results[univariate_results["cox_significant"]].sort_values("hazard_ratio")

    if len(sig_results) == 0:
        print("No significant genes for forest plot")
        return

    fig, ax = plt.subplots(figsize=(config.FIGURE_SIZE[0], max(config.FIGURE_SIZE[1], len(sig_results) * 0.4)),
                            dpi=config.FIGURE_DPI)

    y_pos = range(len(sig_results))

    colors = ["#E24B4A" if hr > 1 else "#1D9E75" for hr in sig_results["hazard_ratio"]]

    ax.barh(y_pos, sig_results["hazard_ratio"] - 1, left=1,
            color=colors, alpha=0.7, height=0.6)
    ax.errorbar(
        sig_results["hazard_ratio"], y_pos,
        xerr=[
            sig_results["hazard_ratio"] - sig_results["hr_ci_lower"],
            sig_results["hr_ci_upper"] - sig_results["hazard_ratio"],
        ],
        fmt="ko", capsize=3, markersize=4,
    )

    ax.axvline(1.0, color="black", linewidth=1, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_results["gene"], fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Cox Regression — Significant Genes\nRed: risk (HR>1) | Green: protective (HR<1)")

    for i, (_, row) in enumerate(sig_results.iterrows()):
        ax.text(
            max(row["hr_ci_upper"], row["hazard_ratio"]) + 0.02, i,
            f"p={row['cox_pvalue']:.3f}", va="center", fontsize=7,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Forest plot saved: {save_path}")


def main():
    print("=" * 60)
    print("Cox Proportional Hazards Regression")
    print("=" * 60)

    expr, surv, labels, shap_genes, mt_genes = load_data()

    # Combine all candidate genes
    all_genes = list(dict.fromkeys(shap_genes + mt_genes))
    print(f"Total unique candidate genes: {len(all_genes)}")

    # 1. Univariate Cox
    uni_results = run_univariate_cox(expr, surv, all_genes)

    n_sig = uni_results["cox_significant"].sum()
    print(f"\nUnivariate summary: {n_sig}/{len(uni_results)} genes Cox-significant")

    # 2. Multivariate Cox
    multi_results, cph = run_multivariate_cox(expr, surv, labels, shap_genes, top_n=5)

    # 3. SHAP vs Multi-task comparison
    compare_shap_vs_multitask_survival(uni_results, shap_genes, mt_genes)

    # Forest plot
    plot_forest(uni_results, os.path.join(config.RESULTS_FIGURES, "cox_forest_plot.png"))

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    uni_results.to_csv(
        os.path.join(config.RESULTS_TABLES, "cox_univariate_results.csv"), index=False
    )
    if not multi_results.empty:
        multi_results.to_csv(
            os.path.join(config.RESULTS_TABLES, "cox_multivariate_results.csv"), index=False
        )

    # KM vs Cox comparison
    km_path = os.path.join(config.RESULTS_TABLES, "km_survival_results.csv")
    if os.path.exists(km_path):
        km_df = pd.read_csv(km_path)
        comparison = km_df[["gene", "logrank_pvalue", "significant"]].merge(
            uni_results[["gene", "cox_pvalue", "hazard_ratio", "cox_significant"]],
            on="gene", how="outer",
        )
        comparison.to_csv(
            os.path.join(config.RESULTS_TABLES, "km_vs_cox_comparison.csv"), index=False
        )
        print(f"\nKM vs Cox comparison saved")

        # Agreement
        both = comparison.dropna(subset=["logrank_pvalue", "cox_pvalue"])
        agree = ((both["significant"] == True) & (both["cox_significant"] == True)).sum()
        print(f"KM and Cox agree on significance: {agree}/{len(both)} genes")

    print("\nCox regression complete.")


if __name__ == "__main__":
    main()
