"""
Streamlit Dashboard — Full Pipeline Explorer
==============================================
6-tab interactive dashboard covering all pipeline results.
Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# For Streamlit Cloud: use app/data/ for small processed files
APP_DATA = os.path.join(os.path.dirname(__file__), "data")

st.set_page_config(
    page_title="TCGA-BRCA Subtype Classification",
    layout="wide",
)

st.title("TCGA-BRCA Subtype Classification & Biomarker Clinical Validation")
st.caption(
    "GDC/TCGAbiolinks → DESeq2 → Consensus DEG → ML/DL → SHAP/TabNet/Multi-task → "
    "KM Survival → Cox Regression → Knowledge Graph"
)


def load_image(filename):
    path = os.path.join(config.RESULTS_FIGURES, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


def load_json(filename):
    path = os.path.join(config.RESULTS_TABLES, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Overview",
    "Model Comparison",
    "Biomarker Discovery",
    "Clinical Validation",
    "Cross-Platform",
    "Knowledge Graph",
])


# Tab 1: Data Overview
with tab1:
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", "1,099")
    with col2:
        st.metric("Consensus DEG Features", "516")
    with col3:
        st.metric("PAM50 Subtypes", "5")
    with col4:
        st.metric("Data Source", "GDC/TCGAbiolinks")

    st.subheader("Feature Selection Pipeline")
    st.markdown(
        "43,160 raw genes → 28,932 (low-expression) → 6,646 DESeq2 DEGs → "
        "1,103 GEO t-test DEGs → **526 Consensus** (GEO ∩ DESeq2) → **516** (Kruskal-Wallis)"
    )

    st.subheader("PAM50 Subtype Distribution")
    labels_path = os.path.join(APP_DATA, "tcga_brca_labels.csv")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv")
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path, index_col=0).squeeze()
        if hasattr(labels, 'iloc') and not isinstance(labels, str):
            dist = labels.value_counts()
            dist.index = [str(x).replace("BRCA_", "") for x in dist.index]
            col1, col2 = st.columns([1, 1])
            with col1:
                st.bar_chart(dist)
            with col2:
                for subtype, count in dist.items():
                    pct = count / len(labels) * 100
                    st.write(f"**{subtype}**: {count} ({pct:.1f}%)")

    st.subheader("PCA — Subtype Separation")
    pca_img = load_image("pca_subtype_scatter.png")
    if pca_img:
        st.image(pca_img, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("GEO DEG (t-test)")
        volcano_geo = load_image("volcano_plot_GSE42568.png")
        if volcano_geo:
            st.image(volcano_geo, use_container_width=True)
    with col2:
        st.subheader("TCGA DEG (DESeq2)")
        volcano_deseq = load_image("deseq2_volcano_plot.png")
        if volcano_deseq:
            st.image(volcano_deseq, use_container_width=True)

    st.subheader("Ablation: DEG Filter Effect")
    ablation_img = load_image("ablation_geo_filter.png")
    if ablation_img:
        st.image(ablation_img, use_container_width=True)
    ablation_data = load_json("ablation_geo_filter.json")
    if ablation_data:
        st.info(
            f"Consensus DEG filter: 98.8% feature reduction "
            f"(43,160 → 516) with only ~0.025 Macro F1 cost."
        )


# Tab 2: Model Comparison
with tab2:
    st.header("Model Comparison")

    comp_path = os.path.join(config.RESULTS_TABLES, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    comp_img = load_image("model_comparison.png")
    if comp_img:
        st.image(comp_img, use_container_width=True)

    st.subheader("Imbalance Handling Comparison")
    imb_path = os.path.join(config.RESULTS_TABLES, "xgboost_imbalance_comparison.csv")
    if os.path.exists(imb_path):
        imb_df = pd.read_csv(imb_path, index_col=0)
        st.dataframe(imb_df.round(4), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost (Best Single Model)")
        xgb_cm = load_image("xgboost_confusion_matrix.png")
        if xgb_cm:
            st.image(xgb_cm, use_container_width=True)
    with col2:
        st.subheader("Stacking Ensemble")
        stack_cm = load_image("stacking_confusion_matrix.png")
        if stack_cm:
            st.image(stack_cm, use_container_width=True)

    st.subheader("Hierarchical Classifier")
    st.markdown(
        "Clinical decision structure: "
        "Level 1 (ER+/ER-/Normal) → Level 2a (LumA vs LumB) → Level 2b (Basal vs HER2)"
    )
    hier_cm = load_image("hierarchical_confusion_matrix.png")
    if hier_cm:
        st.image(hier_cm, use_container_width=True)
    hier_data = load_json("hierarchical_results.json")
    if hier_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Level 1 Accuracy", f"{hier_data['level1']['accuracy']:.3f}")
        with col2:
            st.metric("Level 2a (LumA/B)", f"{hier_data['level2a_luminal']['accuracy']:.3f}")
        with col3:
            st.metric("Level 2b (Basal/HER2)", f"{hier_data['level2b_erneg']['accuracy']:.3f}")

    st.subheader("TabNet (Self-Supervised Pretrained)")
    tabnet_cm = load_image("tabnet_confusion_matrix.png")
    if tabnet_cm:
        st.image(tabnet_cm, use_container_width=True)
    tabnet_data = load_json("tabnet_summary.json")
    if tabnet_data:
        st.info(
            f"Pretraining improved TabNet from 83.6% to {tabnet_data.get('tabnet_accuracy', 0)*100:.1f}% accuracy. "
            f"Spearman correlation with SHAP: ρ = {tabnet_data.get('spearman_rho', 0):.3f}"
        )

    st.subheader("Multi-task (Classification + Cox Survival)")
    mt_data = load_json("multitask_summary.json")
    if mt_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{mt_data.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Macro F1", f"{mt_data.get('macro_f1', 0):.3f}")
        with col3:
            st.metric("Best α", f"{mt_data.get('best_alpha', 0)}")
    mt_hist = load_image("multitask_training_history.png")
    if mt_hist:
        st.image(mt_hist, use_container_width=True)


# Tab 3: Biomarker Discovery
with tab3:
    st.header("Biomarker Discovery")

    st.subheader("SHAP Feature Importance (XGBoost)")
    shap_bar = load_image("shap_summary_bar.png")
    if shap_bar:
        st.image(shap_bar, use_container_width=True)

    st.subheader("Known Cancer Gene Validation")
    val_path = os.path.join(config.RESULTS_TABLES, "shap_known_gene_validation.csv")
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        n_known = (val_df["status"] == "Known").sum()
        n_novel = (val_df["status"] == "Novel").sum()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SHAP Top Genes", len(val_df))
        with col2:
            st.metric("Known Markers", n_known)
        with col3:
            st.metric("Novel Candidates", n_novel)
        st.dataframe(
            val_df[["rank", "gene", "mean_abs_shap", "known_marker", "status"]].round(4),
            use_container_width=True, hide_index=True,
        )

    val_img = load_image("shap_known_gene_validation.png")
    if val_img:
        st.image(val_img, use_container_width=True)

    st.subheader("SHAP Beeswarm (per subtype)")
    subtypes = ["Basal", "Her2", "LumA", "LumB", "Normal"]
    selected = st.selectbox("Select subtype", subtypes, key="shap_subtype")
    bee_img = load_image(f"shap_beeswarm_{selected}.png")
    if bee_img:
        st.image(bee_img, use_container_width=True)

    st.subheader("TabNet Attention vs SHAP Comparison")
    tabnet_vs = load_image("tabnet_vs_shap_comparison.png")
    if tabnet_vs:
        st.image(tabnet_vs, use_container_width=True)
    tabnet_data = load_json("tabnet_summary.json")
    if tabnet_data:
        consensus = tabnet_data.get("consensus_genes", [])
        st.info(
            f"Top 20 overlap: {tabnet_data.get('top20_overlap_with_shap', 0)}/20. "
            f"Consensus genes: {', '.join(consensus) if consensus else 'N/A'}. "
            f"Different methods capture different aspects of the data."
        )

    st.subheader("Feature Stability (100 Bootstrap Iterations)")
    stability_bar = load_image("feature_stability_bar.png")
    if stability_bar:
        st.image(stability_bar, use_container_width=True)
    stability_2d = load_image("feature_stability_2d.png")
    if stability_2d:
        st.image(stability_2d, use_container_width=True)

    stability_path = os.path.join(config.RESULTS_TABLES, "feature_stability_results.csv")
    if os.path.exists(stability_path):
        stab_df = pd.read_csv(stability_path)
        robust = stab_df[stab_df["stability"] >= 0.9]
        st.metric("Robust Genes (≥90% stability)", len(robust))
        if len(robust) > 0:
            st.dataframe(
                robust[["gene", "stability", "mean_abs_shap"]].round(4).head(15),
                use_container_width=True, hide_index=True,
            )

    st.subheader("Prediction Confidence")
    conf_img = load_image("prediction_confidence.png")
    if conf_img:
        st.image(conf_img, use_container_width=True)
    conf_sub = load_image("confidence_by_subtype.png")
    if conf_sub:
        st.image(conf_sub, use_container_width=True)

    conf_path = os.path.join(config.RESULTS_TABLES, "prediction_confidence.csv")
    if os.path.exists(conf_path):
        conf_df = pd.read_csv(conf_path)
        if "confidence" in conf_df.columns:
            col1, col2, col3 = st.columns(3)
            high = (conf_df["confidence"] == "High").sum()
            border = (conf_df["confidence"] == "Borderline").sum()
            low = (conf_df["confidence"] == "Low").sum()
            with col1:
                st.metric("High Confidence", f"{high} ({high/len(conf_df)*100:.1f}%)")
            with col2:
                st.metric("Borderline", f"{border} ({border/len(conf_df)*100:.1f}%)")
            with col3:
                st.metric("Low", f"{low} ({low/len(conf_df)*100:.1f}%)")


# Tab 4: Clinical Validation
with tab4:
    st.header("Clinical Validation")

    st.subheader("Kaplan-Meier Survival Analysis")
    km_path = os.path.join(config.RESULTS_TABLES, "km_survival_results.csv")
    if os.path.exists(km_path):
        km_df = pd.read_csv(km_path)
        n_sig = km_df["significant"].sum()
        n_total = len(km_df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Genes Tested", n_total)
        with col2:
            st.metric("Prognostic (p < 0.05)", n_sig)
        with col3:
            st.metric("Classification-Only", n_total - n_sig)

        st.info(f"**Key insight**: {n_sig}/{n_total} SHAP top genes are also prognostic biomarkers. "
                f"ML classification importance ≠ clinical prognostic value.")

        st.dataframe(
            km_df[["gene", "n_high", "n_low", "logrank_pvalue", "significant"]]
            .sort_values("logrank_pvalue").round(6),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Survival Curves")
        gene_options = km_df.sort_values("logrank_pvalue")["gene"].tolist()
        selected_gene = st.selectbox("Select gene", gene_options, key="km_gene")
        km_img = load_image(f"km_survival_{selected_gene}.png")
        if km_img:
            st.image(km_img, use_container_width=True)

    st.subheader("Cox Proportional Hazards Regression")

    cox_uni_path = os.path.join(config.RESULTS_TABLES, "cox_univariate_results.csv")
    if os.path.exists(cox_uni_path):
        cox_df = pd.read_csv(cox_uni_path)
        st.markdown("**Univariate Cox (per gene)**")
        st.dataframe(
            cox_df[["gene", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "cox_pvalue", "direction", "cox_significant"]]
            .sort_values("cox_pvalue").round(4),
            use_container_width=True, hide_index=True,
        )

    forest_img = load_image("cox_forest_plot.png")
    if forest_img:
        st.image(forest_img, use_container_width=True)

    cox_multi_path = os.path.join(config.RESULTS_TABLES, "cox_multivariate_results.csv")
    if os.path.exists(cox_multi_path):
        cox_m = pd.read_csv(cox_multi_path)
        st.markdown("**Multivariate Cox (controlling for subtype)**")
        st.dataframe(cox_m.round(4), use_container_width=True, hide_index=True)
        independent = cox_m[cox_m.get("independent", False) == True] if "independent" in cox_m.columns else pd.DataFrame()
        if len(independent) > 0:
            st.success(f"Independently prognostic: {', '.join(independent['gene'].tolist())}")
        else:
            st.warning("No genes remained independently prognostic after controlling for subtype.")

    km_cox_path = os.path.join(config.RESULTS_TABLES, "km_vs_cox_comparison.csv")
    if os.path.exists(km_cox_path):
        st.subheader("KM vs Cox Agreement")
        km_cox = pd.read_csv(km_cox_path)
        st.dataframe(km_cox.round(4).head(20), use_container_width=True, hide_index=True)


# Tab 5: Cross-Platform Validation
with tab5:
    st.header("Cross-Platform Validation (TCGA → METABRIC)")

    st.subheader("External Validation")
    ext_img = load_image("external_validation_metabric.png")
    if ext_img:
        st.image(ext_img, use_container_width=True)

    ext_data = load_json("external_validation_metabric.json")
    if ext_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("METABRIC Samples", ext_data.get("test_samples", "—"))
        with col2:
            st.metric("Accuracy", f"{ext_data.get('accuracy', 0):.3f}")
        with col3:
            st.metric("Shared Genes", ext_data.get("shared_genes", "—"))

    st.subheader("Domain Adaptation (CORAL)")
    da_img = load_image("domain_adaptation_comparison.png")
    if da_img:
        st.image(da_img, use_container_width=True)

    da_data = load_json("domain_adaptation_results.json")
    if da_data:
        da_rows = []
        for strategy, metrics in da_data.items():
            da_rows.append({
                "Strategy": strategy,
                "Accuracy": metrics.get("accuracy", 0),
                "Macro F1": metrics.get("macro_f1", 0),
            })
        da_df = pd.DataFrame(da_rows)
        st.dataframe(da_df.round(4), use_container_width=True, hide_index=True)

        st.success(
            "CORAL improved cross-platform accuracy from 64.5% to 73.9% (+9.3pp). "
            "HER2 Recall: 25% → 55%, LumB: 35% → 58%."
        )


# Tab 6: Knowledge Graph
with tab6:
    st.header("Knowledge Graph (SHAP Top Genes)")

    kg_html_path = os.path.join(config.RESULTS_FIGURES, "knowledge_graph.html")
    if os.path.exists(kg_html_path):
        st.caption("Gene (blue) — Disease (red) — Pathway (green). Node size = SHAP importance. Green border = prognostic.")
        with open(kg_html_path, "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)
    else:
        st.warning("Knowledge graph not yet generated. Run build_kg first.")

    nodes_path = os.path.join(config.RESULTS_TABLES, "kg_nodes.csv")
    edges_path = os.path.join(config.RESULTS_TABLES, "kg_edges.csv")
    if os.path.exists(nodes_path):
        st.subheader("Graph Data")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Nodes**")
            st.dataframe(pd.read_csv(nodes_path), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Edges**")
            if os.path.exists(edges_path):
                st.dataframe(pd.read_csv(edges_path), use_container_width=True, hide_index=True)


# Sidebar
with st.sidebar:
    st.header("Pipeline Summary")
    st.markdown("""
    **Data**: TCGAbiolinks/GDC → DESeq2  
    **DEG**: Consensus (GEO ∩ DESeq2) → 516 genes  
    **Models**: XGBoost / Stacking / TabNet / Hierarchical / Multi-task  
    **Interpretation**: SHAP + TabNet Attention + Multi-task Gradient  
    **Validation**: KM Survival + Cox Regression + METABRIC + CORAL  
    **KG**: DisGeNET gene-disease-pathway mapping  
    """)
    st.divider()
    st.markdown("**Dataset**: TCGA-BRCA (1,099 samples)")
    st.markdown("**Best Single Model**: XGBoost (class weights)")
    st.markdown("**Best Ensemble**: Stacking (Acc 0.932)")
    st.markdown("**Best DL**: TabNet pretrained (Acc 0.909)")
    st.markdown("**Cross-Platform**: CORAL (Acc 0.739)")
    st.divider()
    st.markdown("**Gold Standard Biomarkers**")
    st.markdown("SFRP1 — independently prognostic (Cox multivariate)")
    st.markdown("TSLP — strongest survival association (p=0.0007)")
    st.markdown("MLPH, NPY1R — stable across all bootstraps")
    st.divider()
    
