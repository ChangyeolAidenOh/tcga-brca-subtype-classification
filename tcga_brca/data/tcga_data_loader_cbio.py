"""
Stage 2: TCGA-BRCA Data Collection and Preprocessing
=====================================================
Download RNA-seq expression data and clinical metadata from cBioPortal
(primary) or GDC Data Portal (fallback). Build integrated dataset for
downstream classification.

Usage:
    python -m tcga_brca.data.tcga_data_loader
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


CBIO_STUDY_ID = "brca_tcga_pan_can_atlas_2018"


def fetch_cbio_clinical(study_id: str) -> pd.DataFrame:
    """Fetch clinical data including PAM50 subtype from cBioPortal.

    PAM50 is in patient-level data, so both sample and patient
    clinical data are fetched and merged.
    """
    # Fetch sample-level clinical data
    url = f"{config.CBIO_API_ENDPOINT}/studies/{study_id}/clinical-data"
    params = {"clinicalDataType": "SAMPLE", "projection": "DETAILED"}

    print(f"Fetching sample clinical data from cBioPortal ({study_id})...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    records = {}
    for entry in data:
        sid = entry["sampleId"]
        pid = entry["patientId"]
        attr = entry["clinicalAttributeId"]
        val = entry["value"]
        if sid not in records:
            records[sid] = {"sampleId": sid, "patientId": pid}
        records[sid][attr] = val

    sample_df = pd.DataFrame(records.values())
    print(f"Sample clinical data: {sample_df.shape[0]} samples, {sample_df.shape[1]} attributes")

    # Fetch patient-level clinical data (contains PAM50, survival, etc.)
    params_patient = {"clinicalDataType": "PATIENT", "projection": "DETAILED"}
    print("Fetching patient clinical data...")
    resp2 = requests.get(url, params=params_patient, timeout=60)
    resp2.raise_for_status()
    data2 = resp2.json()

    patient_records = {}
    for entry in data2:
        pid = entry["patientId"]
        attr = entry["clinicalAttributeId"]
        val = entry["value"]
        if pid not in patient_records:
            patient_records[pid] = {"patientId": pid}
        patient_records[pid][attr] = val

    patient_df = pd.DataFrame(patient_records.values())
    print(f"Patient clinical data: {patient_df.shape[0]} patients, {patient_df.shape[1]} attributes")

    # Merge on patientId
    df = sample_df.merge(patient_df, on="patientId", how="left", suffixes=("", "_patient"))
    print(f"Merged clinical data: {df.shape[0]} samples, {df.shape[1]} attributes")

    return df


def fetch_cbio_expression(study_id: str) -> pd.DataFrame:
    """Fetch mRNA expression data from cBioPortal using POST fetch endpoint."""
    # Check cache
    cache_path = os.path.join(config.DATA_RAW, "cbio_expression_raw.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached expression data from {cache_path}")
        expr_wide = pd.read_csv(cache_path, index_col=0)
        print(f"Expression matrix: {expr_wide.shape[0]} samples x {expr_wide.shape[1]} genes")
        return expr_wide

    # Get molecular profile ID
    url = f"{config.CBIO_API_ENDPOINT}/studies/{study_id}/molecular-profiles"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    profiles = resp.json()

    mrna_profile = None
    for p in profiles:
        pid = p["molecularProfileId"].lower()
        if "rna_seq" in pid and "v2" in pid and "mrna" in pid:
            mrna_profile = p["molecularProfileId"]
            break
    if mrna_profile is None:
        for p in profiles:
            if "rna_seq" in p["molecularProfileId"].lower():
                mrna_profile = p["molecularProfileId"]
                break
    if mrna_profile is None:
        raise ValueError(
            f"No RNA-seq profile found. Available: "
            f"{[p['molecularProfileId'] for p in profiles]}"
        )

    print(f"Using molecular profile: {mrna_profile}")

    # Get sample list ID
    url = f"{config.CBIO_API_ENDPOINT}/studies/{study_id}/sample-lists"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    sample_lists = resp.json()
    rna_list = None
    for sl in sample_lists:
        sl_id = sl["sampleListId"].lower()
        if "rna_seq" in sl_id:
            rna_list = sl["sampleListId"]
            break
    if rna_list is None:
        for sl in sample_lists:
            if "all" in sl["sampleListId"].lower():
                rna_list = sl["sampleListId"]
                break

    print(f"Using sample list: {rna_list}")

    # Get sample IDs from the list
    url = f"{config.CBIO_API_ENDPOINT}/sample-lists/{rna_list}/sample-ids"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    sample_ids = resp.json()
    print(f"Samples in list: {len(sample_ids)}")

    # POST fetch endpoint — batch by 100 samples to avoid 502
    url = f"{config.CBIO_API_ENDPOINT}/molecular-profiles/{mrna_profile}/molecular-data/fetch?projection=SUMMARY"
    headers = {"Content-Type": "application/json"}

    batch_size = 100
    all_records = []
    total_batches = (len(sample_ids) + batch_size - 1) // batch_size

    print(f"Fetching expression data in {total_batches} batches...")
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} samples)...")

        payload = {"sampleIds": batch}
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        for entry in data:
            all_records.append({
                "sampleId": entry["sampleId"],
                "entrezGeneId": entry["entrezGeneId"],
                "value": entry["value"],
            })

    print(f"Received {len(all_records)} total data points")

    # Cache raw data
    expr_long = pd.DataFrame(all_records)
    expr_long.to_csv(os.path.join(config.DATA_RAW, "cbio_expression_long.csv"), index=False)
    print("Raw expression data cached")

    # Map entrezGeneId to gene symbol
    expr_long = pd.DataFrame(all_records)
    unique_ids = expr_long["entrezGeneId"].unique().tolist()
    print(f"Mapping {len(unique_ids)} entrez IDs to gene symbols...")

    gene_map = {}
    gene_batch_size = 1000
    for i in range(0, len(unique_ids), gene_batch_size):
        batch_ids = unique_ids[i:i + gene_batch_size]
        gene_url = f"{config.CBIO_API_ENDPOINT}/genes/fetch"
        resp = requests.post(
            gene_url,
            json=[str(g) for g in batch_ids],
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        for g in resp.json():
            gene_map[g["entrezGeneId"]] = g["hugoGeneSymbol"]

    expr_long["gene"] = expr_long["entrezGeneId"].map(gene_map)
    expr_long = expr_long.dropna(subset=["gene"])

    expr_wide = expr_long.pivot_table(
        index="sampleId", columns="gene", values="value", aggfunc="first"
    )
    print(f"Expression matrix: {expr_wide.shape[0]} samples x {expr_wide.shape[1]} genes")
    return expr_wide


def extract_pam50_labels(clinical_df: pd.DataFrame) -> pd.Series:
    """Extract PAM50 subtype labels from clinical data."""
    pam50_col = None
    # Priority search order
    search_terms = ["pam50", "subtype_pam50", "subtype", "molecular_subtype"]
    for term in search_terms:
        for col in clinical_df.columns:
            if term in col.lower():
                pam50_col = col
                break
        if pam50_col is not None:
            break

    if pam50_col is None:
        print("PAM50 column not found. Available columns:")
        for col in sorted(clinical_df.columns):
            print(f"  {col}")
        raise ValueError("PAM50 subtype column not found. Check column list above.")

    labels = clinical_df.set_index("sampleId")[pam50_col].dropna()
    labels = labels[labels != ""]

    print(f"\nPAM50 distribution:")
    for subtype, count in labels.value_counts().items():
        pct = count / len(labels) * 100
        print(f"  {subtype}: {count} ({pct:.1f}%)")

    return labels


def extract_survival_data(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Extract overall survival data for Stage 5 (KM analysis)."""
    surv_cols = {}
    for col in clinical_df.columns:
        cl = col.lower()
        if "os_months" in cl:
            surv_cols["os_months"] = col
        elif "os_status" in cl:
            surv_cols["os_status"] = col

    if len(surv_cols) < 2:
        for col in clinical_df.columns:
            cl = col.lower()
            if cl in ("os_months", "months_of_overall_survival"):
                surv_cols["os_months"] = col
            elif cl in ("os_status", "overall_survival_status"):
                surv_cols["os_status"] = col

    if len(surv_cols) < 2:
        print("Survival columns not found. Available:", list(clinical_df.columns))
        return pd.DataFrame()

    surv_df = clinical_df[
        ["sampleId", surv_cols["os_months"], surv_cols["os_status"]]
    ].copy()
    surv_df.columns = ["sampleId", "os_months", "os_status"]
    surv_df["os_months"] = pd.to_numeric(surv_df["os_months"], errors="coerce")

    surv_df["os_event"] = surv_df["os_status"].apply(
        lambda x: 1
        if isinstance(x, str)
        and ("dead" in x.lower() or "deceased" in x.lower() or x == "1")
        else 0
    )
    surv_df = surv_df.dropna(subset=["os_months"])
    print(f"Survival data: {len(surv_df)} patients, {surv_df['os_event'].sum()} events")
    return surv_df


def check_solid_tissue_normal(clinical_df: pd.DataFrame) -> dict:
    """Check for Solid Tissue Normal samples.

    Key decision for hierarchical classifier design.
    Record result in methodology_notes.md.
    """
    sample_type_col = None
    for col in clinical_df.columns:
        if "sample_type" in col.lower():
            sample_type_col = col
            break

    info = {"has_normal": False, "n_normal": 0, "n_tumor": 0}

    if sample_type_col:
        type_counts = clinical_df[sample_type_col].value_counts()
        print(f"\nSample type distribution:")
        for st, count in type_counts.items():
            print(f"  {st}: {count}")
            if "normal" in str(st).lower():
                info["has_normal"] = True
                info["n_normal"] += count
            else:
                info["n_tumor"] += count
    else:
        for sid in clinical_df["sampleId"]:
            parts = sid.split("-")
            if len(parts) >= 4:
                sample_code = parts[3][:2]
                if sample_code == "11":
                    info["has_normal"] = True
                    info["n_normal"] += 1
                else:
                    info["n_tumor"] += 1

    print(f"\nSolid Tissue Normal check:")
    print(f"  Normal samples: {info['n_normal']}")
    print(f"  Tumor samples: {info['n_tumor']}")
    print(f"  >> Record decision in docs/methodology_notes.md")
    return info


def preprocess_expression(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    geo_candidates: list = None,
) -> pd.DataFrame:
    """Preprocess expression matrix."""
    common = expr_df.index.intersection(labels.index)
    expr_aligned = expr_df.loc[common].copy()
    print(f"Samples with both expression and PAM50 labels: {len(common)}")

    if expr_aligned.max().max() > 30:
        print("Applying log2(x + 1) transformation")
        expr_aligned = np.log2(expr_aligned + config.LOG2_PSEUDOCOUNT)

    medians = expr_aligned.median(axis=0)
    threshold = np.log2(config.LOW_EXPRESSION_THRESHOLD + config.LOG2_PSEUDOCOUNT)
    mask = medians >= threshold
    expr_filtered = expr_aligned.loc[:, mask]
    print(f"After low-expression filter: {expr_filtered.shape[1]} genes")

    if geo_candidates is not None and len(geo_candidates) > 0:
        overlap = expr_filtered.columns.intersection(geo_candidates)
        print(f"GEO DEG candidates: {len(geo_candidates)}, overlap with TCGA: {len(overlap)}")
        if len(overlap) >= config.DEG_MIN_GENE_COUNT:
            expr_filtered = expr_filtered[overlap]
            print(f"After GEO DEG filter: {expr_filtered.shape[1]} genes")
        else:
            print(
                f"Overlap too small ({len(overlap)}). "
                f"Keeping all genes after low-expression filter."
            )

    return expr_filtered


def build_dataset(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    surv_df: pd.DataFrame,
    save_dir: str,
):
    """Save integrated dataset for downstream stages."""
    os.makedirs(save_dir, exist_ok=True)

    common = expr_df.index.intersection(labels.index)
    expr_final = expr_df.loc[common]
    labels_final = labels.loc[common]

    expr_path = os.path.join(save_dir, "tcga_brca_expression.csv")
    expr_final.to_csv(expr_path)

    labels_path = os.path.join(save_dir, "tcga_brca_labels.csv")
    labels_final.to_csv(labels_path, header=True)

    if not surv_df.empty:
        surv_path = os.path.join(save_dir, "tcga_brca_survival.csv")
        surv_df.to_csv(surv_path, index=False)

    print(f"\nDataset saved to {save_dir}:")
    print(f"  Expression: {expr_final.shape[0]} samples x {expr_final.shape[1]} genes")
    print(f"  Labels: {labels_final.value_counts().to_dict()}")
    if not surv_df.empty:
        print(f"  Survival: {len(surv_df)} patients")


def main():
    print("=" * 60)
    print("Stage 2: TCGA-BRCA Data Collection & Preprocessing")
    print("=" * 60)

    clinical_df = fetch_cbio_clinical(CBIO_STUDY_ID)
    labels = extract_pam50_labels(clinical_df)
    normal_info = check_solid_tissue_normal(clinical_df)
    surv_df = extract_survival_data(clinical_df)
    expr_df = fetch_cbio_expression(CBIO_STUDY_ID)

    geo_candidates = None
    geo_path = os.path.join(config.DATA_PROCESSED, "geo_feature_candidates.txt")
    if os.path.exists(geo_path):
        geo_candidates = pd.read_csv(geo_path, header=None)[0].tolist()
        print(f"\nLoaded {len(geo_candidates)} GEO DEG feature candidates")
    else:
        print("\nNo GEO DEG candidates found. Proceeding without GEO filter.")

    expr_processed = preprocess_expression(expr_df, labels, geo_candidates)
    build_dataset(expr_processed, labels, surv_df, config.DATA_PROCESSED)

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
