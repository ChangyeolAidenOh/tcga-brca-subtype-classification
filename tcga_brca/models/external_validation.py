"""
External Validation: METABRIC Independent Cohort
==================================================
Validate TCGA-trained model on METABRIC (~2,000 patients).
Cross-platform validation (TCGA: RNA-seq, METABRIC: microarray).

Usage:
    python -m tcga_brca.models.external_validation
"""

import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


METABRIC_STUDY_ID = "brca_metabric"


def fetch_metabric_clinical():
    """Fetch METABRIC clinical data from cBioPortal."""
    cache_path = os.path.join(config.DATA_RAW, "metabric_clinical.csv")
    if os.path.exists(cache_path):
        print("Loading cached METABRIC clinical data...")
        return pd.read_csv(cache_path)

    url = f"{config.CBIO_API_ENDPOINT}/studies/{METABRIC_STUDY_ID}/clinical-data"
    headers = {"Content-Type": "application/json"}

    # Sample-level
    print("Fetching METABRIC sample clinical data...")
    resp = requests.get(url, params={"clinicalDataType": "SAMPLE", "projection": "DETAILED"}, timeout=60)
    resp.raise_for_status()
    sample_records = {}
    for entry in resp.json():
        sid = entry["sampleId"]
        if sid not in sample_records:
            sample_records[sid] = {"sampleId": sid, "patientId": entry["patientId"]}
        sample_records[sid][entry["clinicalAttributeId"]] = entry["value"]

    # Patient-level
    print("Fetching METABRIC patient clinical data...")
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
    df = sample_df.merge(patient_df, on="patientId", how="left", suffixes=("", "_patient"))

    df.to_csv(cache_path, index=False)
    print(f"METABRIC clinical data: {df.shape[0]} samples, {df.shape[1]} attributes")
    return df


def extract_metabric_pam50(clinical_df):
    """Extract PAM50 labels from METABRIC clinical data."""
    # METABRIC stores PAM50 subtypes in CLAUDIN_SUBTYPE column
    pam50_col = None
    for col in clinical_df.columns:
        if "claudin" in col.lower():
            pam50_col = col
            break

    if pam50_col is None:
        for col in clinical_df.columns:
            if "pam50" in col.lower():
                pam50_col = col
                break

    if pam50_col is None:
        print("Available columns with 'subtype':")
        for col in clinical_df.columns:
            if "subtype" in col.lower():
                print(f"  {col}: {clinical_df[col].unique()[:10]}")
        raise ValueError("PAM50 column not found in METABRIC")

    labels = clinical_df.set_index("sampleId")[pam50_col].dropna()
    labels = labels[labels != ""]

    # Standardize labels to match TCGA format
    label_map = {}
    for val in labels.unique():
        vl = str(val).lower().strip()
        if vl == "luma":
            label_map[val] = "BRCA_LumA"
        elif vl == "lumb":
            label_map[val] = "BRCA_LumB"
        elif vl == "basal":
            label_map[val] = "BRCA_Basal"
        elif vl == "her2":
            label_map[val] = "BRCA_Her2"
        elif vl == "normal":
            label_map[val] = "BRCA_Normal"
        else:
            label_map[val] = None  # claudin-low, NC, nan 제외

    labels = labels.map(label_map).dropna()

    print(f"\nMETABRIC PAM50 distribution:")
    for subtype, count in labels.value_counts().items():
        print(f"  {subtype}: {count} ({count/len(labels)*100:.1f}%)")

    return labels


def fetch_metabric_expression():
    """Fetch METABRIC expression data from cBioPortal."""
    cache_path = os.path.join(config.DATA_RAW, "metabric_expression.csv")
    if os.path.exists(cache_path):
        print("Loading cached METABRIC expression data...")
        return pd.read_csv(cache_path, index_col=0)

    # Get molecular profile
    url = f"{config.CBIO_API_ENDPOINT}/studies/{METABRIC_STUDY_ID}/molecular-profiles"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    profiles = resp.json()

    mrna_profile = None
    for p in profiles:
        pid = p["molecularProfileId"].lower()
        if "mrna" in pid and "microarray" not in pid:
            mrna_profile = p["molecularProfileId"]
        if "mrna" in pid:
            mrna_profile = mrna_profile or p["molecularProfileId"]

    if mrna_profile is None:
        print("Available profiles:")
        for p in profiles:
            print(f"  {p['molecularProfileId']}")
        raise ValueError("No mRNA profile found")

    print(f"Using profile: {mrna_profile}")

    # Get sample IDs
    url = f"{config.CBIO_API_ENDPOINT}/studies/{METABRIC_STUDY_ID}/sample-lists"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rna_list = None
    for sl in resp.json():
        if "mrna" in sl["sampleListId"].lower() or "all" in sl["sampleListId"].lower():
            rna_list = sl["sampleListId"]
            if "mrna" in sl["sampleListId"].lower():
                break

    url = f"{config.CBIO_API_ENDPOINT}/sample-lists/{rna_list}/sample-ids"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    sample_ids = resp.json()
    print(f"Samples: {len(sample_ids)}")

    # Fetch expression in batches
    url = f"{config.CBIO_API_ENDPOINT}/molecular-profiles/{mrna_profile}/molecular-data/fetch"
    headers = {"Content-Type": "application/json"}
    batch_size = 100
    all_records = []

    total_batches = (len(sample_ids) + batch_size - 1) // batch_size
    print(f"Fetching expression in {total_batches} batches...")
    for i in range(0, len(sample_ids), batch_size):
        batch = sample_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}...")

        payload = {"sampleIds": batch}
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()

        for entry in resp.json():
            all_records.append({
                "sampleId": entry["sampleId"],
                "entrezGeneId": entry["entrezGeneId"],
                "value": entry["value"],
            })

    print(f"Received {len(all_records)} data points")

    # Map entrez to gene symbol
    expr_long = pd.DataFrame(all_records)
    unique_ids = expr_long["entrezGeneId"].unique().tolist()
    print(f"Mapping {len(unique_ids)} entrez IDs...")

    gene_map = {}
    gene_url = f"{config.CBIO_API_ENDPOINT}/genes/fetch"
    for i in range(0, len(unique_ids), 1000):
        batch_ids = unique_ids[i:i + 1000]
        resp = requests.post(gene_url, json=[str(g) for g in batch_ids], headers=headers, timeout=60)
        resp.raise_for_status()
        for g in resp.json():
            gene_map[g["entrezGeneId"]] = g["hugoGeneSymbol"]

    expr_long["gene"] = expr_long["entrezGeneId"].map(gene_map)
    expr_long = expr_long.dropna(subset=["gene"])

    expr_wide = expr_long.pivot_table(index="sampleId", columns="gene", values="value", aggfunc="first")
    expr_wide.to_csv(cache_path)
    print(f"METABRIC expression: {expr_wide.shape}")
    return expr_wide


def train_and_validate(tcga_expr, tcga_labels, meta_expr, meta_labels, feature_genes):
    """Train on TCGA, validate on METABRIC using shared gene set."""
    shared_genes = tcga_expr.columns.intersection(meta_expr.columns).intersection(feature_genes)
    print(f"\nShared genes between TCGA and METABRIC: {len(shared_genes)}")

    if len(shared_genes) < 50:
        print("Too few shared genes. Trying without feature filter...")
        shared_genes = tcga_expr.columns.intersection(meta_expr.columns)
        print(f"Shared genes (no filter): {len(shared_genes)}")

    X_train = tcga_expr[shared_genes].values
    X_test = meta_expr[shared_genes].values

    le = LabelEncoder()
    le.fit(np.concatenate([tcga_labels.values, meta_labels.values]))
    y_train = le.transform(tcga_labels.values)
    y_test = le.transform(meta_labels.values)
    class_names = le.classes_

    # Cross-platform: standardize each dataset INDEPENDENTLY
    # RNA-seq (TCGA) and microarray (METABRIC) have different scales
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    X_train = scaler_train.fit_transform(X_train)
    X_test = scaler_test.fit_transform(X_test)

    sw = compute_sample_weight("balanced", y_train)

    print("Training XGBoost on TCGA...")
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sw)

    print("Validating on METABRIC...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'=' * 60}")
    print("EXTERNAL VALIDATION RESULTS (Train: TCGA → Test: METABRIC)")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Shared genes used: {len(shared_genes)}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return y_test, y_pred, class_names, acc, f1, len(shared_genes)


def plot_external_validation(y_test, y_pred, class_names, acc, f1, save_path):
    """Plot confusion matrix for external validation."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    short_names = [c.replace("BRCA_", "") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"External Validation (Train: TCGA → Test: METABRIC)\n"
        f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")


def main():
    print("=" * 60)
    print("External Validation: TCGA → METABRIC")
    print("=" * 60)

    # Load TCGA training data
    tcga_expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    tcga_labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()
    common = tcga_expr.index.intersection(tcga_labels.index)
    tcga_expr = tcga_expr.loc[common]
    tcga_labels = tcga_labels.loc[common]
    print(f"TCGA: {tcga_expr.shape[0]} samples x {tcga_expr.shape[1]} genes")

    # Fetch METABRIC
    meta_clinical = fetch_metabric_clinical()
    meta_labels = extract_metabric_pam50(meta_clinical)
    meta_expr = fetch_metabric_expression()

    # Align METABRIC
    common_meta = meta_expr.index.intersection(meta_labels.index)
    meta_expr = meta_expr.loc[common_meta]
    meta_labels = meta_labels.loc[common_meta]
    print(f"METABRIC (aligned): {meta_expr.shape[0]} samples x {meta_expr.shape[1]} genes")

    # Log transform METABRIC if needed
    if meta_expr.max().max() > 30:
        print("Applying log2 transform to METABRIC...")
        meta_expr = np.log2(meta_expr.clip(lower=0) + config.LOG2_PSEUDOCOUNT)

    # Feature genes from TCGA
    feature_genes = tcga_expr.columns.tolist()

    # Train on TCGA, validate on METABRIC
    y_test, y_pred, class_names, acc, f1, n_shared = train_and_validate(
        tcga_expr, tcga_labels, meta_expr, meta_labels, feature_genes
    )

    plot_external_validation(
        y_test, y_pred, class_names, acc, f1,
        os.path.join(config.RESULTS_FIGURES, "external_validation_metabric.png"),
    )

    # Save results
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    summary = {
        "train_dataset": "TCGA-BRCA",
        "test_dataset": "METABRIC",
        "train_samples": int(tcga_expr.shape[0]),
        "test_samples": int(meta_expr.shape[0]),
        "shared_genes": n_shared,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "report": classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
    }
    with open(os.path.join(config.RESULTS_TABLES, "external_validation_metabric.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExternal validation complete.")


if __name__ == "__main__":
    main()
