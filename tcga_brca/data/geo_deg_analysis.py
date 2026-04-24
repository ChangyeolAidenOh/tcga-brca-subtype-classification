"""
Stage 1: GEO DEG Analysis (Feature Candidate Extraction)
=========================================================
Download GSE42568 (breast cancer, 104 samples: 17 normal + 87 tumor)
and perform differential expression analysis to identify feature candidates
for downstream TCGA classification.

Usage:
    python -m tcga_brca.data.geo_deg_analysis
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def download_geo_data(dataset_id: str, dest_dir: str) -> object:
    """Download GEO dataset using GEOparse."""
    import GEOparse

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {dataset_id}...")
    gse = GEOparse.get_GEO(geo=dataset_id, destdir=dest_dir, silent=True)
    print(f"Download complete. GSMs: {len(gse.gsms)}")
    return gse


def extract_expression_and_labels(gse) -> tuple:
    """Extract expression matrix and sample labels from GEO object.

    Returns:
        expression_df: DataFrame (genes x samples)
        labels: Series with 'Normal' or 'Tumor' per sample
    """
    sample_labels = {}
    for gsm_name, gsm in gse.gsms.items():
        source = gsm.metadata.get("source_name_ch1", [""])[0].lower()
        if "normal" in source:
            sample_labels[gsm_name] = "Normal"
        else:
            sample_labels[gsm_name] = "Tumor"

    labels = pd.Series(sample_labels)
    print(f"Label distribution: {labels.value_counts().to_dict()}")

    frames = {}
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if "ID_REF" in table.columns and "VALUE" in table.columns:
            frames[gsm_name] = table.set_index("ID_REF")["VALUE"]

    expression_df = pd.DataFrame(frames)
    expression_df = expression_df.apply(pd.to_numeric, errors="coerce")
    expression_df.dropna(how="all", inplace=True)
    print(f"Expression matrix shape: {expression_df.shape}")
    return expression_df, labels


def map_probe_to_gene(gse, expression_df: pd.DataFrame) -> pd.DataFrame:
    """Map Affymetrix probe IDs to gene symbols using GPL annotation."""
    gpl_name = list(gse.gpls.keys())[0]
    gpl = gse.gpls[gpl_name]
    annotation = gpl.table

    gene_col = None
    for col in annotation.columns:
        if "gene" in col.lower() and "symbol" in col.lower():
            gene_col = col
            break
    if gene_col is None:
        for col in annotation.columns:
            if col == "Gene Symbol":
                gene_col = col
                break
    if gene_col is None:
        print("Available annotation columns:", list(annotation.columns))
        gene_col = "Gene Symbol"

    probe_to_gene = annotation.set_index("ID")[gene_col].dropna()
    probe_to_gene = probe_to_gene[probe_to_gene.str.strip() != ""]
    probe_to_gene = probe_to_gene[~probe_to_gene.str.contains("///")]

    expression_mapped = expression_df.loc[
        expression_df.index.isin(probe_to_gene.index)
    ].copy()
    expression_mapped["gene_symbol"] = probe_to_gene.loc[expression_mapped.index]
    expression_mapped = expression_mapped.groupby("gene_symbol").mean()

    print(f"Mapped to {expression_mapped.shape[0]} unique genes")
    return expression_mapped


def run_deg_analysis(
    expression_df: pd.DataFrame,
    labels: pd.Series,
    pvalue_threshold: float,
    log2fc_threshold: float,
    fallback_log2fc: float,
    min_gene_count: int,
) -> pd.DataFrame:
    """Perform differential expression analysis (Normal vs Tumor)."""
    normal_samples = labels[labels == "Normal"].index
    tumor_samples = labels[labels == "Tumor"].index

    normal_expr = expression_df[normal_samples]
    tumor_expr = expression_df[tumor_samples]

    results = []
    for gene in expression_df.index:
        n_vals = normal_expr.loc[gene].dropna().values
        t_vals = tumor_expr.loc[gene].dropna().values

        if len(n_vals) < 3 or len(t_vals) < 3:
            continue

        log2fc = np.mean(t_vals) - np.mean(n_vals)
        stat, pvalue = stats.ttest_ind(t_vals, n_vals, equal_var=False)

        results.append({
            "gene": gene,
            "log2FC": log2fc,
            "pvalue": pvalue,
            "mean_normal": np.mean(n_vals),
            "mean_tumor": np.mean(t_vals),
        })

    deg_df = pd.DataFrame(results)
    deg_df = deg_df.dropna(subset=["pvalue"]).reset_index(drop=True)

    from statsmodels.stats.multitest import multipletests
    valid_mask = deg_df["pvalue"].notna()
    reject, adj_pvals, _, _ = multipletests(
        deg_df.loc[valid_mask, "pvalue"].values, method="fdr_bh"
    )
    deg_df.loc[valid_mask, "adj_pvalue"] = adj_pvals

    deg_df["significant"] = (
        (deg_df["adj_pvalue"] < pvalue_threshold)
        & (deg_df["log2FC"].abs() > log2fc_threshold)
    )
    n_sig = deg_df["significant"].sum()
    print(f"DEGs (|log2FC| > {log2fc_threshold}, adj.p < {pvalue_threshold}): {n_sig}")

    if n_sig < min_gene_count:
        print(f"Below minimum ({min_gene_count}). Relaxing to |log2FC| > {fallback_log2fc}")
        deg_df["significant"] = (
            (deg_df["adj_pvalue"] < pvalue_threshold)
            & (deg_df["log2FC"].abs() > fallback_log2fc)
        )
        n_sig = deg_df["significant"].sum()
        print(f"DEGs after relaxation: {n_sig}")

    deg_df = deg_df.sort_values("adj_pvalue")
    return deg_df


def plot_volcano(deg_df: pd.DataFrame, save_path: str):
    """Generate volcano plot of DEG results."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)

    neg_log10p = -np.log10(deg_df["adj_pvalue"].clip(lower=1e-300))

    colors = []
    for _, row in deg_df.iterrows():
        if row["significant"] and row["log2FC"] > 0:
            colors.append("#E24B4A")
        elif row["significant"] and row["log2FC"] < 0:
            colors.append("#378ADD")
        else:
            colors.append("#B4B2A9")

    ax.scatter(
        deg_df["log2FC"], neg_log10p,
        c=colors, alpha=0.5, s=8, edgecolors="none"
    )

    ax.axhline(-np.log10(config.DEG_PVALUE_THRESHOLD), ls="--", color="#888780", lw=0.8)
    ax.axvline(config.DEG_LOG2FC_THRESHOLD, ls="--", color="#888780", lw=0.8)
    ax.axvline(-config.DEG_LOG2FC_THRESHOLD, ls="--", color="#888780", lw=0.8)

    n_up = ((deg_df["significant"]) & (deg_df["log2FC"] > 0)).sum()
    n_down = ((deg_df["significant"]) & (deg_df["log2FC"] < 0)).sum()

    ax.set_xlabel("log2 Fold Change (Tumor vs Normal)", fontsize=12)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=12)
    ax.set_title(
        f"Volcano Plot — GSE42568 DEG Analysis\n"
        f"Up: {n_up} | Down: {n_down} | Total DEGs: {n_up + n_down}",
        fontsize=13,
    )

    top_genes = deg_df[deg_df["significant"]].head(10)
    for _, row in top_genes.iterrows():
        ax.annotate(
            row["gene"],
            (row["log2FC"], -np.log10(max(row["adj_pvalue"], 1e-300))),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords="offset points",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Volcano plot saved: {save_path}")


def save_deg_results(deg_df: pd.DataFrame, save_dir: str):
    """Save DEG results and significant gene list."""
    os.makedirs(save_dir, exist_ok=True)

    full_path = os.path.join(save_dir, "geo_deg_full_results.csv")
    deg_df.to_csv(full_path, index=False)

    sig_df = deg_df[deg_df["significant"]].copy()
    sig_path = os.path.join(save_dir, "geo_deg_significant_genes.csv")
    sig_df.to_csv(sig_path, index=False)

    gene_list_path = os.path.join(save_dir, "geo_feature_candidates.txt")
    sig_df["gene"].to_csv(gene_list_path, index=False, header=False)

    print(f"Full results: {full_path} ({len(deg_df)} genes)")
    print(f"Significant genes: {sig_path} ({len(sig_df)} genes)")
    print(f"Feature candidate list: {gene_list_path}")

    n_up = (sig_df["log2FC"] > 0).sum()
    n_down = (sig_df["log2FC"] < 0).sum()
    print(f"\nDEG Summary:")
    print(f"  Upregulated in tumor: {n_up}")
    print(f"  Downregulated in tumor: {n_down}")
    print(f"  Top 5 upregulated: {sig_df[sig_df['log2FC'] > 0].head(5)['gene'].tolist()}")
    print(f"  Top 5 downregulated: {sig_df[sig_df['log2FC'] < 0].nsmallest(5, 'log2FC')['gene'].tolist()}")

    return sig_df


def main():
    print("=" * 60)
    print("Stage 1: GEO DEG Analysis — Feature Candidate Extraction")
    print("=" * 60)

    gse = download_geo_data(config.GEO_DATASET_ID, config.DATA_RAW)
    expression_df, labels = extract_expression_and_labels(gse)
    expression_mapped = map_probe_to_gene(gse, expression_df)

    if expression_mapped.max().max() > 100:
        print("Applying log2 transformation")
        expression_mapped = np.log2(expression_mapped + config.LOG2_PSEUDOCOUNT)

    deg_df = run_deg_analysis(
        expression_mapped,
        labels,
        pvalue_threshold=config.DEG_PVALUE_THRESHOLD,
        log2fc_threshold=config.DEG_LOG2FC_THRESHOLD,
        fallback_log2fc=config.DEG_LOG2FC_FALLBACK,
        min_gene_count=config.DEG_MIN_GENE_COUNT,
    )

    plot_volcano(deg_df, os.path.join(config.RESULTS_FIGURES, "volcano_plot_GSE42568.png"))
    save_deg_results(deg_df, config.DATA_PROCESSED)

    print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
