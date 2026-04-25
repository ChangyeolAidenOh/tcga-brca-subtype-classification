"""
Stage 2: Load TCGAbiolinks-Exported Data + Preprocessing
==========================================================
Reads expression, labels, and survival data exported by the
R script (scripts/tcga_download_and_deg.R) which uses
TCGAbiolinks to download directly from GDC.

This replaces the cBioPortal API approach with the standard
bioinformatics workflow:
  TCGAbiolinks (R) → GDC download → DESeq2 → CSV export → Python ML

Usage:
    1. First run: Rscript scripts/tcga_download_and_deg.R
    2. Then run: python -m tcga_brca.data.tcga_data_loader

Falls back to cBioPortal if R-exported files are not found.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_r_exported_data():
    """Load data exported by TCGAbiolinks R script."""
    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")
    labels_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv")
    surv_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv")

    if not os.path.exists(expr_path):
        return None, None, None

    print("Loading TCGAbiolinks-exported data...")

    # Expression
    expr = pd.read_csv(expr_path, index_col=0)
    print(f"Expression: {expr.shape[0]} samples x {expr.shape[1]} genes")

    # Labels
    labels = None
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        if "PAM50" in labels_df.columns:
            labels = labels_df.set_index("sampleId")["PAM50"]
        elif labels_df.shape[1] == 2:
            labels = labels_df.set_index(labels_df.columns[0]).iloc[:, 0]
        else:
            labels = pd.read_csv(labels_path, index_col=0).squeeze()

        print(f"Labels loaded: {labels.value_counts().to_dict()}")

    # Survival
    surv = pd.DataFrame()
    if os.path.exists(surv_path):
        surv = pd.read_csv(surv_path)
        print(f"Survival: {len(surv)} patients, {int(surv['os_event'].sum())} events")

    return expr, labels, surv


def load_cbio_fallback():
    """Fallback: fetch from cBioPortal if R export not available."""
    print("R-exported data not found. Falling back to cBioPortal...")
    print("For the standard approach, run: Rscript scripts/tcga_download_and_deg.R")
    print("Proceeding with cBioPortal API...\n")

    # Import and run the cBioPortal-based loader
    import requests

    CBIO_STUDY_ID = "brca_tcga_pan_can_atlas_2018"
    headers = {"Content-Type": "application/json"}

    # Clinical data (sample + patient level)
    url = f"{config.CBIO_API_ENDPOINT}/studies/{CBIO_STUDY_ID}/clinical-data"

    print("Fetching sample clinical data...")
    resp = requests.get(url, params={"clinicalDataType": "SAMPLE", "projection": "DETAILED"}, timeout=60)
    resp.raise_for_status()
    sample_records = {}
    for entry in resp.json():
        sid = entry["sampleId"]
        if sid not in sample_records:
            sample_records[sid] = {"sampleId": sid, "patientId": entry["patientId"]}
        sample_records[sid][entry["clinicalAttributeId"]] = entry["value"]

    print("Fetching patient clinical data...")
    resp2 = requests.get(url, params={"clinicalDataType": "PATIENT", "projection": "DETAILED"}, timeout=60)
    resp2.raise_for_status()
    patient_records = {}
    for entry in resp2.json():
        pid = entry["patientId"]
        if pid not in patient_records:
            patient_records[pid] = {"patientId": pid}
        patient_records[pid][entry["clinicalAttributeId"]] = entry["value"]

    sample_df = pd.DataFrame(sample_records.values())
    patient_df = pd.DataFrame(patient_records.values())
    clinical_df = sample_df.merge(patient_df, on="patientId", how="left", suffixes=("", "_patient"))

    # PAM50 labels
    pam50_col = None
    for col in clinical_df.columns:
        if "claudin" in col.lower() or "pam50" in col.lower():
            pam50_col = col
            break
        if "subtype" in col.lower():
            pam50_col = col

    if pam50_col is None:
        raise ValueError("PAM50 column not found")

    label_map = {
        "LumA": "BRCA_LumA", "LumB": "BRCA_LumB",
        "Basal": "BRCA_Basal", "Her2": "BRCA_Her2",
        "Normal": "BRCA_Normal",
    }

    labels_raw = clinical_df.set_index("sampleId")[pam50_col]
    labels = labels_raw.map(lambda x: label_map.get(x, None)).dropna()

    # Survival
    surv_data = []
    for col in clinical_df.columns:
        cl = col.lower()
        if "os_months" in cl:
            os_m_col = col
        elif "os_status" in cl:
            os_s_col = col

    surv = clinical_df[["sampleId"]].copy()
    if "os_m_col" in dir() and "os_s_col" in dir():
        surv["os_months"] = pd.to_numeric(clinical_df[os_m_col], errors="coerce")
        surv["os_event"] = clinical_df[os_s_col].apply(
            lambda x: 1 if isinstance(x, str) and ("dead" in x.lower() or x == "1") else 0
        )
        surv = surv.dropna(subset=["os_months"])

    # Expression (batch fetch)
    # ... (same as existing cBioPortal fetch code)
    # For brevity, load from cache if available
    cache_path = os.path.join(config.DATA_RAW, "cbio_expression_raw.csv")
    if os.path.exists(cache_path):
        print("Loading cached cBioPortal expression...")
        expr = pd.read_csv(cache_path, index_col=0)
    else:
        raise FileNotFoundError(
            "No expression data available. Run R script or ensure cBioPortal cache exists."
        )

    return expr, labels, surv


def apply_geo_filter(expr, geo_candidates):
    """Apply GEO DEG feature filter."""
    overlap = expr.columns.intersection(geo_candidates)
    print(f"GEO DEG candidates: {len(geo_candidates)}, overlap: {len(overlap)}")
    if len(overlap) >= config.DEG_MIN_GENE_COUNT:
        expr = expr[overlap]
        print(f"After GEO DEG filter: {expr.shape[1]} genes")
    else:
        print(f"Overlap too small. Keeping all genes.")
    return expr


def apply_deseq2_filter(expr):
    """Apply DESeq2 DEG feature filter (if available from R script)."""
    deseq2_path = os.path.join(config.DATA_PROCESSED, "deseq2_feature_candidates.txt")
    consensus_path = os.path.join(config.DATA_PROCESSED, "consensus_deg_genes.txt")

    # Prefer consensus (GEO ∩ DESeq2) if available
    if os.path.exists(consensus_path):
        candidates = pd.read_csv(consensus_path, header=None)[0].tolist()
        print(f"Using consensus DEG genes (GEO ∩ DESeq2): {len(candidates)}")
    elif os.path.exists(deseq2_path):
        candidates = pd.read_csv(deseq2_path, header=None)[0].tolist()
        print(f"Using DESeq2 DEG genes: {len(candidates)}")
    else:
        return expr

    overlap = expr.columns.intersection(candidates)
    if len(overlap) >= config.DEG_MIN_GENE_COUNT:
        expr = expr[overlap]
        print(f"After DEG filter: {expr.shape[1]} genes")
    else:
        print(f"Overlap too small ({len(overlap)}). Keeping all genes.")

    return expr


def preprocess_expression(expr, labels):
    """Preprocess expression matrix."""
    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    print(f"Samples with both expression and labels: {len(common)}")

    # Log2 transform if needed (R script may have already done this)
    if expr.max().max() > 30:
        print("Applying log2(x + 1) transformation")
        expr = np.log2(expr + config.LOG2_PSEUDOCOUNT)

    # Low-expression filter
    medians = expr.median(axis=0)
    threshold = np.log2(config.LOW_EXPRESSION_THRESHOLD + config.LOG2_PSEUDOCOUNT)
    expr = expr.loc[:, medians >= threshold]
    print(f"After low-expression filter: {expr.shape[1]} genes")

    return expr


def save_processed(expr, labels, surv):
    """Save processed data for downstream stages."""
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    labels = labels.loc[common]

    expr.to_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv"))
    labels.to_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), header=True)

    if not surv.empty:
        surv.to_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv"), index=False)

    print(f"\nDataset saved:")
    print(f"  Expression: {expr.shape}")
    print(f"  Labels: {labels.value_counts().to_dict()}")


def main():
    print("=" * 60)
    print("Stage 2: TCGA-BRCA Data Loading & Preprocessing")
    print("=" * 60)

    # Try R-exported data first (standard approach)
    expr, labels, surv = load_r_exported_data()

    # Fallback to cBioPortal
    if expr is None:
        expr, labels, surv = load_cbio_fallback()

    # Preprocess
    expr = preprocess_expression(expr, labels)

    # Apply DEG filter
    # Priority: consensus (GEO ∩ DESeq2) > DESeq2 > GEO
    geo_path = os.path.join(config.DATA_PROCESSED, "geo_feature_candidates.txt")
    deseq2_path = os.path.join(config.DATA_PROCESSED, "deseq2_feature_candidates.txt")
    consensus_path = os.path.join(config.DATA_PROCESSED, "consensus_deg_genes.txt")

    if os.path.exists(consensus_path):
        expr = apply_deseq2_filter(expr)
    elif os.path.exists(deseq2_path):
        expr = apply_deseq2_filter(expr)
    elif os.path.exists(geo_path):
        geo_candidates = pd.read_csv(geo_path, header=None)[0].tolist()
        expr = apply_geo_filter(expr, geo_candidates)

    # Save
    save_processed(expr, labels, surv)

    print("\nStage 2 complete.")
    print("Next: python -m tcga_brca.features.feature_selection")


if __name__ == "__main__":
    main()
