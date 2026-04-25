"""
GDC API Data Loader — Alternative TCGA Data Access
====================================================
Direct download from GDC Data Portal (Genomic Data Commons),
the official TCGA data distribution platform.

Provides an alternative to cBioPortal for:
  - RNA-seq HTSeq raw counts (for DESeq2 DEG analysis)
  - Clinical metadata with survival information
  - File manifest for reproducibility

Usage:
    python -m tcga_brca.data.gdc_data_loader
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


GDC_FILES_ENDPOINT = f"{config.GDC_API_ENDPOINT}/files"
GDC_CASES_ENDPOINT = f"{config.GDC_API_ENDPOINT}/cases"
GDC_DATA_ENDPOINT = f"{config.GDC_API_ENDPOINT}/data"


def query_gdc_file_ids(project="TCGA-BRCA", workflow="STAR - Counts", max_files=1200):
    """Query GDC for RNA-seq file UUIDs.

    This is the standard method for programmatic TCGA data access,
    as described in GDC documentation.
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
            {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow}},
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": str(max_files),
    }

    print(f"Querying GDC for {project} RNA-seq files...")
    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hits = data["data"]["hits"]
    print(f"Found {len(hits)} files")

    file_info = []
    for hit in hits:
        case = hit.get("cases", [{}])[0]
        samples = case.get("samples", [{}])
        sample_type = samples[0].get("sample_type", "Unknown") if samples else "Unknown"

        file_info.append({
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
            "case_id": case.get("case_id", ""),
            "submitter_id": case.get("submitter_id", ""),
            "sample_type": sample_type,
        })

    file_df = pd.DataFrame(file_info)
    print(f"Sample types: {file_df['sample_type'].value_counts().to_dict()}")
    return file_df


def query_gdc_clinical(project="TCGA-BRCA", max_cases=1200):
    """Query GDC for clinical metadata including survival."""
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": project},
    }

    fields = [
        "submitter_id",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.age_at_diagnosis",
        "diagnoses.primary_diagnosis",
        "diagnoses.tumor_stage",
        "demographic.gender",
        "demographic.race",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(max_cases),
    }

    print(f"Querying GDC clinical data for {project}...")
    resp = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hits = data["data"]["hits"]
    records = []
    for hit in hits:
        record = {"submitter_id": hit.get("submitter_id", "")}

        diag = hit.get("diagnoses", [{}])
        if diag:
            d = diag[0]
            record["vital_status"] = d.get("vital_status", "")
            record["days_to_death"] = d.get("days_to_death", None)
            record["days_to_last_follow_up"] = d.get("days_to_last_follow_up", None)
            record["age_at_diagnosis"] = d.get("age_at_diagnosis", None)
            record["primary_diagnosis"] = d.get("primary_diagnosis", "")
            record["tumor_stage"] = d.get("tumor_stage", "")

        demo = hit.get("demographic", {})
        record["gender"] = demo.get("gender", "")
        record["race"] = demo.get("race", "")

        records.append(record)

    clinical_df = pd.DataFrame(records)

    # Compute OS months
    clinical_df["os_months"] = np.where(
        clinical_df["vital_status"] == "Dead",
        clinical_df["days_to_death"],
        clinical_df["days_to_last_follow_up"],
    )
    clinical_df["os_months"] = pd.to_numeric(clinical_df["os_months"], errors="coerce") / 30.44
    clinical_df["os_event"] = (clinical_df["vital_status"] == "Dead").astype(int)

    print(f"Clinical data: {len(clinical_df)} cases")
    print(f"Events: {clinical_df['os_event'].sum()} / {clinical_df['os_months'].notna().sum()}")
    return clinical_df


def download_gdc_expression(file_ids, save_dir, max_files=5):
    """Download individual expression files from GDC.

    Note: For full dataset, use GDC Transfer Tool (gdc-client)
    which is faster for large downloads. This function demonstrates
    the API approach for a small number of files.
    """
    os.makedirs(save_dir, exist_ok=True)
    downloaded = []

    print(f"Downloading {min(len(file_ids), max_files)} expression files...")
    for i, fid in enumerate(file_ids[:max_files]):
        url = f"{GDC_DATA_ENDPOINT}/{fid}"
        resp = requests.get(url, timeout=120)

        if resp.status_code == 200:
            save_path = os.path.join(save_dir, f"{fid}.tsv")
            with open(save_path, "wb") as f:
                f.write(resp.content)
            downloaded.append(save_path)
            print(f"  [{i+1}/{min(len(file_ids), max_files)}] {fid[:8]}... downloaded")
        else:
            print(f"  [{i+1}] {fid[:8]}... failed ({resp.status_code})")

    print(f"Downloaded {len(downloaded)} files")
    return downloaded


def parse_star_counts(file_paths):
    """Parse STAR-Counts TSV files into expression matrix.

    STAR-Counts files have columns:
      gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded
    """
    frames = {}
    for path in file_paths:
        fid = os.path.basename(path).replace(".tsv", "")
        try:
            df = pd.read_csv(path, sep="\t", comment="#")
            if "gene_name" in df.columns and "unstranded" in df.columns:
                # Raw counts for DESeq2
                counts = df.set_index("gene_name")["unstranded"]
                # Remove version suffix from Ensembl IDs if gene_name is empty
                counts = counts[counts.index.notna()]
                frames[fid] = counts
        except Exception as e:
            print(f"  Error parsing {fid}: {e}")

    if frames:
        expr = pd.DataFrame(frames)
        # Remove non-gene rows (STAR metadata rows start with N_)
        expr = expr[~expr.index.str.startswith("N_")]
        return expr
    return pd.DataFrame()


def main():
    print("=" * 60)
    print("GDC API Data Loader — Alternative TCGA Access")
    print("=" * 60)

    # Step 1: Query file IDs
    file_df = query_gdc_file_ids()

    # Save file manifest
    os.makedirs(config.DATA_RAW, exist_ok=True)
    manifest_path = os.path.join(config.DATA_RAW, "gdc_file_manifest.csv")
    file_df.to_csv(manifest_path, index=False)
    print(f"File manifest saved: {manifest_path}")

    # Step 2: Query clinical data
    clinical_df = query_gdc_clinical()
    clinical_path = os.path.join(config.DATA_RAW, "gdc_clinical.csv")
    clinical_df.to_csv(clinical_path, index=False)
    print(f"Clinical data saved: {clinical_path}")

    # Step 3: Download sample expression files (demo — 5 files)
    # For full dataset: use gdc-client command line tool
    print("\nDownloading sample expression files (5 files for demo)...")
    print("For full dataset, use: gdc-client download -m gdc_manifest.txt")
    sample_files = download_gdc_expression(
        file_df["file_id"].tolist(),
        os.path.join(config.DATA_RAW, "gdc_expression"),
        max_files=5,
    )

    # Step 4: Parse sample files
    if sample_files:
        expr = parse_star_counts(sample_files)
        if not expr.empty:
            print(f"\nSample expression matrix: {expr.shape}")
            print(f"Column types available: raw counts (unstranded)")
            print("These raw counts can be used with DESeq2 for DEG analysis")
        else:
            print("No expression data parsed from sample files")

    print(f"\n{'=' * 60}")
    print("GDC ACCESS SUMMARY")
    print(f"{'=' * 60}")
    print(f"File manifest: {len(file_df)} files")
    print(f"Clinical data: {len(clinical_df)} cases")
    print(f"Sample downloads: {len(sample_files)} files")
    print(f"\nFor full analysis:")
    print(f"  1. Install gdc-client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool")
    print(f"  2. Run: gdc-client download -m {manifest_path}")
    print(f"  3. Parse all files with parse_star_counts()")
    print(f"  4. Use raw counts with DESeq2 for DEG analysis")

    print("\nGDC data loader complete.")


if __name__ == "__main__":
    main()
