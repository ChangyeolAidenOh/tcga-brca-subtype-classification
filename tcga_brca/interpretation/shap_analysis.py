"""
Stage 4: SHAP Interpretation + Known Cancer Gene Validation
=============================================================
Apply SHAP TreeExplainer to best model (XGBoost with class weights),
extract top contributing genes, and validate against known breast
cancer markers.

Usage:
    python -m tcga_brca.interpretation.shap_analysis
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_model_and_data():
    """Load trained XGBoost model and expression data."""
    model_path = os.path.join(config.RESULTS_MODELS, "xgboost_baseline.pkl")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")

    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    labels = labels.loc[common]

    X_scaled = scaler.transform(expr.values)
    class_names = le.classes_

    print(f"Model loaded. Features: {len(feature_names)}")
    print(f"Classes: {list(class_names)}")
    return model, X_scaled, feature_names, labels, class_names, le


def compute_shap_values(model, X, feature_names):
    """Compute SHAP values using XGBoost's native implementation.

    Bypasses shap.TreeExplainer due to XGBoost 2.x multiclass
    base_score vector bug. XGBoost's predict(pred_contribs=True)
    uses the same TreeSHAP algorithm internally.
    """
    import xgboost as xgb

    print("\nComputing SHAP values (XGBoost native TreeSHAP)...")

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    # pred_contribs returns (n_samples, n_features + 1) per class
    # +1 is the base value (last column)
    raw_contribs = booster.predict(dmatrix, pred_contribs=True)

    # For multiclass: shape is (n_samples, n_features + 1, n_classes)
    # Rearrange to list of arrays per class (same format as shap.TreeExplainer)
    if raw_contribs.ndim == 3:
        # XGBoost multiclass: shape is (n_samples, n_classes, n_features + 1)
        n_classes = raw_contribs.shape[1]
        shap_values = []
        for c in range(n_classes):
            shap_values.append(raw_contribs[:, c, :-1])  # exclude base value
        print(f"SHAP values computed: {n_classes} classes x {shap_values[0].shape}")
    else:
        shap_values = raw_contribs[:, :-1]
        print(f"SHAP values shape: {shap_values.shape}")

    return shap_values, None


def get_global_importance(shap_values, feature_names, top_n):
    """Extract global feature importance by mean |SHAP| across all classes."""
    if isinstance(shap_values, list):
        # Average absolute SHAP across all classes
        all_shap = np.array(shap_values)  # (n_classes, n_samples, n_features)
        mean_abs = np.mean(np.abs(all_shap), axis=(0, 1))  # (n_features,)
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        "gene": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    print(f"\nTop {top_n} genes by global mean |SHAP|:")
    for _, row in importance_df.head(top_n).iterrows():
        print(f"  {row['rank']:2d}. {row['gene']:15s} {row['mean_abs_shap']:.4f}")

    return importance_df


def get_class_specific_importance(shap_values, feature_names, class_names, top_n=10):
    """Extract per-class top genes."""
    class_results = {}

    for i, cls in enumerate(class_names):
        if isinstance(shap_values, list):
            cls_shap = shap_values[i]
        else:
            cls_shap = shap_values[:, :, i] if shap_values.ndim == 3 else shap_values

        mean_abs = np.mean(np.abs(cls_shap), axis=0)
        cls_df = pd.DataFrame({
            "gene": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        class_results[cls] = cls_df.head(top_n)["gene"].tolist()

    print(f"\nClass-specific top {top_n} genes:")
    for cls, genes in class_results.items():
        short = cls.replace("BRCA_", "")
        print(f"  {short:10s}: {', '.join(genes[:5])}...")

    return class_results


def validate_known_genes(importance_df, top_n):
    """Compare SHAP top genes against known breast cancer markers."""
    top_genes = set(importance_df.head(top_n)["gene"].tolist())

    all_known = {}
    for subtype, genes in config.KNOWN_BREAST_CANCER_GENES.items():
        for g in genes:
            all_known[g] = subtype

    known_set = set(all_known.keys())
    overlap = top_genes & known_set
    novel = top_genes - known_set

    print(f"\n{'=' * 60}")
    print(f"KNOWN CANCER GENE VALIDATION (Top {top_n})")
    print(f"{'=' * 60}")
    print(f"SHAP top {top_n} genes: {len(top_genes)}")
    print(f"Known breast cancer genes in DB: {len(known_set)}")
    print(f"Overlap: {len(overlap)} ({len(overlap)/top_n*100:.1f}%)")

    if overlap:
        print(f"\nMatched genes:")
        for g in overlap:
            rank = importance_df[importance_df["gene"] == g]["rank"].values[0]
            print(f"  {g:15s} (rank {rank:2d}) — {all_known[g]} marker")

    if novel:
        print(f"\nNovel candidates (not in known list):")
        for g in list(novel)[:10]:
            rank = importance_df[importance_df["gene"] == g]["rank"].values[0]
            print(f"  {g:15s} (rank {rank:2d}) — potential novel biomarker")

    # Build validation table
    rows = []
    for _, row in importance_df.head(top_n).iterrows():
        gene = row["gene"]
        rows.append({
            "rank": row["rank"],
            "gene": gene,
            "mean_abs_shap": row["mean_abs_shap"],
            "known_marker": all_known.get(gene, ""),
            "status": "Known" if gene in known_set else "Novel",
        })

    validation_df = pd.DataFrame(rows)
    return validation_df, len(overlap), len(novel)


def plot_shap_summary(shap_values, X, feature_names, class_names, save_dir):
    """Generate SHAP summary plots."""
    os.makedirs(save_dir, exist_ok=True)

    # Global summary (bar)
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            class_names=[c.replace("BRCA_", "") for c in class_names],
            plot_type="bar", max_display=config.SHAP_TOP_N, show=False,
        )
    else:
        shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            plot_type="bar", max_display=config.SHAP_TOP_N, show=False,
        )
    plt.title("SHAP Feature Importance — Top 20 Genes")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "shap_summary_bar.png"),
        dpi=config.FIGURE_DPI, bbox_inches="tight",
    )
    plt.close("all")
    print(f"SHAP bar plot saved")

    # Per-class beeswarm (top class: Basal — most distinct)
    if isinstance(shap_values, list):
        for i, cls in enumerate(class_names):
            fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
            shap.summary_plot(
                shap_values[i], X, feature_names=feature_names,
                max_display=15, show=False,
            )
            short = cls.replace("BRCA_", "")
            plt.title(f"SHAP Beeswarm — {short}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"shap_beeswarm_{short}.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight",
            )
            plt.close("all")
        print(f"SHAP beeswarm plots saved (per class)")


def plot_validation_table(validation_df, save_path):
    """Visualize known vs novel genes."""
    fig, ax = plt.subplots(figsize=(config.FIGURE_SIZE[0], config.FIGURE_SIZE[1] * 1.2), dpi=config.FIGURE_DPI)
    ax.axis("off")

    colors = []
    for _, row in validation_df.iterrows():
        if row["status"] == "Known":
            colors.append("#E6F1FB")
        else:
            colors.append("#FFFFFF")

    table = ax.table(
        cellText=validation_df[["rank", "gene", "mean_abs_shap", "known_marker", "status"]].round(4).values,
        colLabels=["Rank", "Gene", "Mean |SHAP|", "Known Marker", "Status"],
        cellColours=[[c] * 5 for c in colors],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    ax.set_title(f"SHAP Top {len(validation_df)} Genes — Known Cancer Gene Validation", fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Validation table saved: {save_path}")


def save_results(importance_df, validation_df, class_results, n_overlap, n_novel):
    """Save all Stage 4 outputs."""
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)

    importance_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "shap_global_importance.csv"), index=False
    )
    validation_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "shap_known_gene_validation.csv"), index=False
    )

    # Top genes for Stage 5 (KM) and Stage 6 (KG)
    top_genes = importance_df.head(config.SHAP_TOP_N)["gene"].tolist()
    with open(os.path.join(config.DATA_PROCESSED, "shap_top_genes.txt"), "w") as f:
        for g in top_genes:
            f.write(g + "\n")

    # Class-specific results
    class_df = pd.DataFrame(
        {cls: genes + [""] * (10 - len(genes)) for cls, genes in class_results.items()}
    )
    class_df.to_csv(
        os.path.join(config.RESULTS_TABLES, "shap_class_specific_top_genes.csv"), index=False
    )

    print(f"\nResults saved. Top {config.SHAP_TOP_N} genes written for Stage 5/6.")
    print(f"  Known gene overlap: {n_overlap}/{config.SHAP_TOP_N}")
    print(f"  Novel candidates: {n_novel}/{config.SHAP_TOP_N}")


def main():
    print("=" * 60)
    print("Stage 4: SHAP Interpretation + Cancer Gene Validation")
    print("=" * 60)

    model, X, feature_names, labels, class_names, le = load_model_and_data()

    shap_values, explainer = compute_shap_values(model, X, feature_names)

    importance_df = get_global_importance(shap_values, feature_names, config.SHAP_TOP_N)

    class_results = get_class_specific_importance(
        shap_values, feature_names, class_names, top_n=10
    )

    validation_df, n_overlap, n_novel = validate_known_genes(importance_df, config.SHAP_TOP_N)

    plot_shap_summary(shap_values, X, feature_names, class_names, config.RESULTS_FIGURES)

    plot_validation_table(
        validation_df,
        os.path.join(config.RESULTS_FIGURES, "shap_known_gene_validation.png"),
    )

    save_results(importance_df, validation_df, class_results, n_overlap, n_novel)

    print("\nStage 4 complete.")


if __name__ == "__main__":
    main()
