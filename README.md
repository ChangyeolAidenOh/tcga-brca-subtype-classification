# TCGA Breast Cancer Subtype Classification & Biomarker Clinical Validation Pipeline

An end-to-end bioinformatics ML pipeline that classifies PAM50 breast cancer subtypes from TCGA RNA-seq data, identifies biomarker candidates through multi-method interpretation (SHAP, TabNet attention, multi-task gradient), and validates their clinical significance via Kaplan-Meier survival analysis, Cox proportional hazards regression, and Knowledge Graph mapping.

Data was downloaded from GDC using TCGAbiolinks (Colaprico et al., 2016) and differential expression analysis was performed with DESeq2 (Love et al., 2014) — the standard bioinformatics workflow.

**Independent Project**

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Motivation

This project addresses four questions:

1. **Can ML methods from medical image classification transfer to omics data?** — The same hierarchical classifier design (CXR: Normal vs Pneumonia → Bacterial vs Viral) is applied to PAM50 subtypes (ER+ vs ER- → LumA vs LumB / Basal vs HER2). Class weights proved optimal for imbalanced classes in both domains.

2. **Are ML-identified biomarkers clinically meaningful?** — Of 20 SHAP top genes, only 6 (30%) show statistically significant survival associations. In multivariate Cox regression controlling for subtype, only SFRP1 remains independently prognostic. ML classification importance ≠ clinical prognostic value.

3. **How robust are the identified biomarkers?** — Bootstrap stability analysis (100 iterations) reveals that only 9/516 genes are consistently selected. Cross-validation between cBioPortal and GDC/DESeq2 pipelines independently produced the same 3 gold standard genes (SFRP1, MLPH, NPY1R).

4. **Can domain adaptation improve cross-platform generalization?** — CORAL alignment improves TCGA→METABRIC accuracy from 64.5% to 73.9% (+9.3pp), an ML-specific approach rarely applied in bioinformatics.

---

## Dataset

| Source | Description | Samples | Features | Role |
|---|---|---|---|---|
| TCGA-BRCA | RNA-seq STAR-Counts via TCGAbiolinks/GDC | 1,099 | 43,160 genes | Training and internal validation |
| GEO GSE42568 | Microarray, Normal vs Tumor (NCBI GEO) | 104 | 54,675 probes → 21,655 genes | DEG-based feature candidate extraction |
| METABRIC | Microarray, independent cohort (cBioPortal) | 1,756 | 19,850 genes | External cross-platform validation |
| DisGeNET | Gene-disease association database | — | 17,000+ genes | Knowledge Graph construction |

### PAM50 Subtype Distribution (TCGA-BRCA, GDC)

| Subtype | Count | Proportion | Clinical Characteristic |
|---|---|---|---|
| Luminal A | 571 | 52.0% | ER+, low proliferation, best prognosis |
| Luminal B | 209 | 19.0% | ER+, high proliferation, intermediate |
| Basal-like | 197 | 17.9% | Triple-negative overlap, worst prognosis |
| HER2-enriched | 82 | 7.5% | HER2 overexpression, targeted therapy available |
| Normal-like | 40 | 3.6% | Resembles normal breast tissue, few specific markers |

### Feature Selection Pipeline

The feature space was reduced through a multi-stage process. First, low-expression genes were removed (43,160 → 28,932). Then, two independent DEG analyses were performed in parallel: DESeq2 on TCGA RNA-seq (1,111 tumor vs 113 normal, 6,646 DEGs) and Welch's t-test on GEO GSE42568 microarray (87 tumor vs 17 normal, 1,103 DEGs). The intersection of these two DEG lists produced **526 Consensus DEGs** — genes differentially expressed in both independent datasets using different platforms and statistical methods. Finally, Kruskal-Wallis testing across PAM50 subtypes (adj. p < 0.01) yielded the final **516 features**.

This consensus filter is a custom preprocessing step designed for this pipeline, not an existing tool.

---

## Pipeline

```
R: TCGAbiolinks + DESeq2                 Python: GEOparse
(GDC download, 1,231 files)              (GSE42568, 104 samples)
        │                                      │
  GDCprepare → DESeq2                    t-test + BH FDR
  (Tumor vs Normal, n=113)               (Normal vs Tumor)
  6,646 DEGs                             1,103 DEGs
        │                                      │
        └──── Consensus (GEO ∩ DESeq2) ────────┘
                     526 genes
                        │
                        ▼
              Kruskal-Wallis → 516 genes
                        │
     ┌──────────────────┼──────────────────────┐
     ▼                  ▼                      ▼
  XGBoost          Stacking            Hierarchical
  (class weights)  (LGB+XGB+CB OOF)   (L1: ER+/ER-/Normal
  Acc 0.905        Acc 0.900            L2a: LumA vs LumB 92.3%
  F1 0.852         F1 0.826             L2b: Basal vs Her2 98.2%)
     │                  │
     ├─ TabNet (pretrained, Acc 0.909, F1 0.864)
     │
     ├─ Ablation: DEG filter ON(516) vs OFF(43K), F1 diff 0.025
     │
     ├─ Multi-task: CE + α·Cox loss (α=0.1)
     │   Normal-like Recall 0.86, F1 0.803
     │
     └─ METABRIC External Validation
         Independent: Acc 0.645 → CORAL: Acc 0.739 (+9.3pp)
                        │
                        ▼
              SHAP (XGBoost native TreeSHAP)
              + TabNet Attention comparison
              + Multi-task gradient importance
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   Bootstrap        KM Survival     Cox Regression
   Stability        (6/20 sig)      Univariate: 6/29 sig
   (100x, 9         3 Gold Std:     Multivariate: SFRP1
   robust genes)    MLPH, SFRP1,    only independent
         │          NPY1R                  │
         └──────────────┬──────────────────┘
                        ▼
              Knowledge Graph (pyvis)
              + Streamlit Dashboard (6 tabs)
```

---

## Methodology

### Data Acquisition — TCGAbiolinks + GDC (Standard Approach)

TCGA-BRCA RNA-seq data (1,231 STAR-Counts files, 5.2 GB) was downloaded from GDC using TCGAbiolinks, the standard R package for TCGA data access. GDCprepare assembled a SummarizedExperiment with 60,660 genes x 1,231 samples, including 1,111 primary tumors, 113 solid tissue normal, and 7 metastatic samples. PAM50 subtype annotations were obtained from the TCGA Pan-Cancer Atlas molecular subtype data (paper_BRCA_Subtype_PAM50). Expression was DESeq2-normalized (size factor estimation) and log2-transformed for the Python classification pipeline.

### DEG Analysis — DESeq2 (R) + t-test (Python)

Two independent DEG analyses were performed in parallel on different data sources using different statistical methods. DESeq2 (negative binomial GLM with Wald test) on TCGA RNA-seq raw counts (1,111 tumor vs 113 normal) identified 6,646 DEGs. Welch's t-test with Benjamini-Hochberg FDR on GEO GSE42568 microarray (87 tumor vs 17 normal) identified 1,103 DEGs. The consensus set (intersection) of 526 genes represents genes confirmed as differentially expressed across both datasets, both platforms, and both statistical frameworks.

The choice of t-test for GEO (microarray data is pre-normalized, continuous) vs DESeq2 for TCGA (RNA-seq raw counts follow negative binomial distribution) reflects the appropriate statistical method for each data type.

### Classification Models

All models were trained on the same 516-feature consensus DEG dataset (1,099 samples) to ensure comparable results.

**XGBoost Baseline with Imbalance Comparison**: Three strategies compared — no handling (F1 0.794), class weights (F1 0.852), SMOTE (F1 0.851). Class weights achieved the best Macro F1, consistent with the CXR pneumonia project where WeightedRandomSampler was also optimal.

**Stacking Ensemble (OOF)**: LightGBM + XGBoost + CatBoost with Logistic Regression meta-learner, using Out-of-Fold predictions to prevent data leakage. Architecture directly transferred from the Stat Consulting Internship project. Achieved Acc 0.900, F1 0.826. The ensemble underperformed XGBoost baseline — with only 516 features, the diversity benefit of multiple tree models diminishes and the meta-learner lacks class-weight protection for minority classes.

**TabNet with Self-Supervised Pretraining**: Two-phase training — unsupervised pretraining learns feature structure from unlabeled data, then supervised fine-tuning with pretrained weights. This mirrors the CXR project's ImageNet-pretrained EfficientNet fine-tuning strategy. Pretraining improved TabNet from 83.6% to 90.9% accuracy (+7.3pp), achieving the best single-model performance. Attention-based feature importance showed only 15% overlap with SHAP top 20 (consensus: CEP55, CDC20, NAT1 — all cell proliferation markers), demonstrating that different interpretation methods capture different aspects of the same data.

**Hierarchical Classifier**: Clinical decision structure mirroring the CXR 2-stage design. Level 1 separates ER+ / ER- / Normal-like (Acc 0.955), Level 2a classifies LumA vs LumB within ER+ (Acc 0.923), Level 2b classifies Basal vs HER2 within ER- (Acc 0.982). This decomposition reveals that classification difficulty is concentrated in a single boundary — Luminal A vs Luminal B — while Basal and HER2 are near-perfectly separable once isolated.

**Multi-task Learning (Classification + Cox Survival)**: PyTorch network with shared encoder, classification head (CE loss), and survival head (Cox partial likelihood loss). Loss: CE + alpha * Cox, with alpha search over [0.01, 0.1, 0.5]. This mirrors the PINN project's multi-objective loss design (PDE residual + boundary condition). Best alpha=0.1 achieved F1 0.803 with Normal-like Recall 0.86. Multi-task top genes (TSLP, SFRP1) overlap with Cox regression's most significant survival genes, confirming the Cox loss guides learning toward clinically relevant features.

### Ablation Study — DEG Filter Effect

Comparing consensus DEG-filtered (516 genes) vs unfiltered (43,160 genes): 98.8% feature reduction with only 0.025 Macro F1 difference. However, HER2 Recall dropped notably (0.77 vs 0.88 unfiltered), suggesting some subtype-specific markers are excluded by the consensus filter. Practical implication: aggressive filtering trades recall on specific subtypes for computational efficiency and interpretability.

### External Validation — METABRIC + Domain Adaptation

TCGA-trained XGBoost was validated on the independent METABRIC cohort (1,756 samples, microarray). Independent standardization achieved 64.5% accuracy. CORAL (CORrelation ALignment) domain adaptation improved this to 73.9% (+9.3pp) by aligning the covariance structure of TCGA features to match METABRIC. HER2 Recall improved from 25.0% to 54.5%, LumB from 34.5% to 58.1%. This ML technique is standard in computer vision but rarely applied in genomics cross-platform studies.

### SHAP Interpretation

SHAP values were computed using XGBoost's native predict(pred_contribs=True) — the standard TreeSHAP algorithm. The shap.TreeExplainer library was not used due to a compatibility bug with XGBoost 2.x (multiclass base_score stored as vector, SHAP expects scalar). Top 20 genes were validated against known breast cancer markers: FOXA1 (rank 3) and GATA3 (rank 11) matched as established Luminal markers. SFRP1 (rank 2, Wnt pathway suppressor), MLPH (rank 1, PAM50 intrinsic gene), and NPY1R (rank 4) are well-characterized in breast cancer literature despite being classified as "novel" in our conservative reference list.

### Feature Stability Analysis

100 bootstrap iterations, each resampling the dataset and extracting SHAP top 20. Of 516 genes, 9 achieved robust stability (>=90%): MLPH (100%), SFRP1 (100%), FOXA1 (100%), NPY1R (100%), NAT1 (99%), SLC39A6 (99%), RRM2 (96%), CEP55 (96%), AGR3 (94%). Cross-validation: the same gold standard genes emerged from both the cBioPortal-based pipeline (925 features) and the GDC/DESeq2 pipeline (516 features), confirming data-source-independent robustness.

### Prediction Confidence

Prediction entropy identifies borderline classifications: 94.2% of samples classified with high confidence (entropy < 0.3, accuracy 99.6%). 64 patients flagged as borderline/low confidence. Among 21 misclassified samples, 81% fell in borderline/low categories. Primary confusion pair: LumA to LumB (10 of 21 errors), with LumB to LumA showing higher entropy (0.996) than LumA to LumB (0.655).

### Kaplan-Meier Survival Analysis

6 of 20 SHAP top genes (30%) showed significant survival associations: TSLP (p=0.010), MLPH (p=0.011), NPY1R (p=0.019), SFRP1 (p=0.022), SDC1 (p=0.024), CEP55 (p=0.048). FOXA1, despite being SHAP rank 3, showed no survival association (p=0.985) — a classification-only marker.

### Cox Proportional Hazards Regression

Univariate Cox identified 6/29 candidate genes as significant, with hazard ratios quantifying effect magnitude. TSLP (HR=0.861, p=0.0007) showed the strongest protective effect. In multivariate Cox controlling for PAM50 subtype, only SFRP1 (HR=0.927, p=0.038) remained independently prognostic. All other genes' survival effects were confounded by subtype — they predict survival because they predict subtype, not independently.

SHAP top 20 contained 5 Cox-significant genes vs multi-task top 20 with 2, indicating classification-focused features partially but not fully capture prognostic information.

---

## Results

### Classification Performance (all on 516 consensus DEG features)

| Model | Accuracy | Macro F1 | AUROC | Note |
|---|---|---|---|---|
| TabNet (pretrained) | **0.909** | **0.864** | — | Self-supervised + fine-tuning, best single model |
| XGBoost (class weights) | 0.905 | 0.852 | 0.988 | Best tree model |
| Stacking Ensemble (OOF) | 0.900 | 0.826 | 0.988 | LGB+XGB+CB, diversity limited at 516 features |
| Hierarchical Classifier | 0.900 | 0.845 | — | Level 2b: Basal vs HER2 = 98.2% |
| Multi-task (alpha=0.1) | 0.874 | 0.803 | — | Normal-like Recall 0.86 |
| Unfiltered (43,160 genes) | 0.904 | 0.828 | — | Ablation: full feature set |
| METABRIC (independent) | 0.645 | 0.550 | — | Cross-platform baseline |
| METABRIC (CORAL) | **0.739** | **0.693** | — | +9.3pp domain adaptation |

### Per-Class Performance (XGBoost, class weights)

| Subtype | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Basal | 1.00 | 0.97 | 0.99 | 40 |
| HER2 | 0.74 | 0.88 | 0.80 | 16 |
| Luminal A | 0.95 | 0.92 | 0.93 | 114 |
| Luminal B | 0.83 | 0.83 | 0.83 | 42 |
| Normal-like | 0.67 | 0.75 | 0.71 | 8 |

### Survival Validation Summary

| Gene | SHAP Rank | Stability | KM p-value | Cox Univariate | Cox Multivariate | Direction |
|---|---|---|---|---|---|---|
| SFRP1 | 2 | 100% | 0.022 | HR=0.939, p=0.031 | **HR=0.927, p=0.038** | Protective |
| TSLP | 20 | — | 0.010 | **HR=0.861, p=0.0007** | — | Protective |
| MLPH | 1 | 100% | 0.011 | ns | Confounded | — |
| NPY1R | 4 | 100% | 0.019 | HR=0.934, p=0.006 | Confounded | Protective |
| SDC1 | 19 | — | 0.024 | HR=1.191, p=0.015 | — | Risk |
| CEP55 | 8 | 96% | 0.048 | ns | — | — |

### Interpretation Method Comparison

| | SHAP Top 20 | TabNet Top 20 | Multi-task Top 20 |
|---|---|---|---|
| SHAP overlap | — | 3 (15%) | 3 (15%) |
| TabNet overlap | 3 | — | — |
| Multi-task overlap | 3 | — | — |
| Consensus genes | — | CEP55, CDC20, NAT1 | MAMDC2, SFRP1, TSLP |

### Prediction Confidence

| Category | Count | Proportion | Accuracy |
|---|---|---|---|
| High (entropy < 0.3) | 1,035 | 94.2% | 99.6% |
| Borderline (0.3-1.0) | 44 | 4.0% | 75.0% |
| Low (> 1.0) | 20 | 1.8% | 70.0% |

---

## Key Findings

### 1. Consensus DEG filter: 98.8% feature reduction, 2.5% performance cost
Two independent DEG analyses (GEO t-test on microarray + DESeq2 on RNA-seq) intersected to produce 526 consensus genes. This reduced 43,160 features to 516 (after Kruskal-Wallis) with only 0.025 Macro F1 loss, while improving computational efficiency by approximately 80x.

### 2. Self-supervised pretraining rescues TabNet on small genomic data
Without pretraining, TabNet achieved 83.6% accuracy — far below XGBoost (90.5%). With pretraining, TabNet reached 90.9%, surpassing XGBoost as the best single model. This mirrors the CXR project's finding that pretrained models outperform scratch training on limited data.

### 3. Hierarchical classification localizes difficulty to a single boundary
Level 2b (Basal vs HER2) = 98.2% accuracy. Level 2a (LumA vs LumB) = 92.3%. The flat 5-class problem is dominated by a single hard 2-class problem (Luminal A vs B), while Basal and HER2 are near-perfectly separable once isolated from Luminal subtypes.

### 4. SFRP1 is the only independently prognostic biomarker
In multivariate Cox controlling for subtype, only SFRP1 (Wnt signaling suppressor, HR=0.927, p=0.038) retains significance. MLPH and NPY1R — gold standard in univariate analysis — are confounded by subtype. TSLP shows the strongest univariate association (HR=0.861, p=0.0007) and is the only gene in both SHAP and multi-task top 20.

### 5. Different interpretation methods see different genes
SHAP (tree-based), TabNet attention, and multi-task gradients produce largely non-overlapping top 20 lists (10-15% overlap). Consensus genes across methods — CEP55, CDC20 (proliferation markers) and SFRP1, TSLP (survival markers) — are the most trustworthy candidates.

### 6. CORAL domain adaptation: +9.3pp cross-platform improvement
TCGA to METABRIC accuracy improved from 64.5% to 73.9% with CORAL covariance alignment. HER2 Recall doubled (25% to 55%), LumB improved from 35% to 58%. This ML technique is standard in computer vision but rarely applied in genomics cross-platform studies.

### 7. Prediction entropy reliably flags errors
94.2% of predictions are high-confidence with 99.6% accuracy. 81% of misclassified samples fall in borderline/low confidence, enabling targeted clinical review of uncertain cases.

### 8. Multi-task survival loss improves minority class performance
Cox partial likelihood as auxiliary loss (alpha=0.1) improved Normal-like Recall to 0.86, the highest across all models. The survival signal acts as a regularizer that prevents the model from ignoring minority classes — analogous to PINN's physics-informed loss constraining the solution space.

---

## Dashboard

Live demo: [Streamlit Cloud](https://tcga-brca-subtype-classification-m7rzyynaix.streamlit.app/)

```bash
streamlit run app/streamlit_app.py
```

| Tab | Content |
|---|---|
| Data Overview | PAM50 distribution, PCA scatter, GEO + DESeq2 volcano plots, ablation |
| Model Comparison | All models, imbalance strategy, hierarchical levels, TabNet, multi-task |
| Biomarker Discovery | SHAP summary/beeswarm, TabNet attention comparison, stability, confidence |
| Clinical Validation | KM survival curves (gene selector), Cox forest plot, multivariate results |
| Cross-Platform | METABRIC external validation, CORAL domain adaptation comparison |
| Knowledge Graph | Interactive gene-disease-pathway network (pyvis) |

---

## Project Structure

```
tcga-brca-subtype-classification/
├── config.py
├── requirements.txt
├── .gitignore
│
├── scripts/
│   └── tcga_download_and_deg.R              # TCGAbiolinks + DESeq2 (standard approach)
│
├── tcga_brca/
│   ├── data/
│   │   ├── geo_deg_analysis.py              # Stage 1: GEO DEG (t-test + BH FDR)
│   │   ├── tcga_data_loader.py              # Stage 2: Load R export + preprocessing
│   │   ├── tcga_data_loader_cbio.py         # Alternative: cBioPortal API loader
│   │   └── gdc_data_loader.py               # Alternative: GDC REST API loader
│   ├── features/
│   │   └── feature_selection.py             # Kruskal-Wallis + PCA
│   ├── models/
│   │   ├── baseline_xgboost.py              # XGBoost + imbalance comparison
│   │   ├── stacking_ensemble.py             # LGB+XGB+CB OOF stacking
│   │   ├── tabnet_classifier.py             # TabNet pretrain + fine-tune + SHAP comparison
│   │   ├── hierarchical_classifier.py       # Clinical decision structure
│   │   ├── multitask_survival.py            # CE + Cox multi-task (PyTorch)
│   │   ├── ablation_geo_filter.py           # DEG filter ON vs OFF
│   │   ├── external_validation.py           # METABRIC cross-platform
│   │   ├── domain_adaptation.py             # CORAL alignment
│   │   └── evaluate.py                      # Model comparison
│   ├── interpretation/
│   │   ├── shap_analysis.py                 # SHAP + known gene validation
│   │   ├── feature_stability.py             # 100x bootstrap stability
│   │   └── prediction_confidence.py         # Entropy-based uncertainty
│   ├── clinical_validation/
│   │   ├── survival_analysis.py             # KM + log-rank
│   │   └── cox_regression.py                # Uni/multivariate Cox + forest plot
│   └── knowledge_graph/
│       └── build_kg.py                      # DisGeNET KG (NetworkX + pyvis)
│
├── app/
│   ├── streamlit_app.py                     # 6-tab dashboard
│   └── data/                                # Small data files for Streamlit Cloud
│       └── tcga_brca_labels.csv
│
├── data/
│   ├── raw/                                 # GDC downloads + caches (gitignored)
│   ├── processed/                           # Processed CSVs (gitignored)
│   └── external/                            # DisGeNET
│
├── results/
│   ├── figures/                             # All visualizations
│   └── tables/                              # All result CSVs/JSONs
│
└── docs/
    └── methodology_notes.md                 # Decision log
```

---

## How to Run

### Prerequisites

R with BiocManager, TCGAbiolinks, DESeq2, SummarizedExperiment. Python 3.10+.

### Full Pipeline

```bash
git clone https://github.com/ChangyeolAidenOh/tcga-brca-subtype-classification.git
cd tcga-brca-subtype-classification
pip install -r requirements.txt

# Phase 1: R data acquisition (20-30 min first run, cached after)
Rscript -e 'install.packages("BiocManager"); BiocManager::install(c("TCGAbiolinks","DESeq2","SummarizedExperiment"))'
Rscript scripts/tcga_download_and_deg.R

# Phase 2: Python preprocessing
python -m tcga_brca.data.geo_deg_analysis
python -m tcga_brca.data.tcga_data_loader
python -m tcga_brca.features.feature_selection

# Phase 3: Modeling
python -m tcga_brca.models.baseline_xgboost
python -m tcga_brca.models.stacking_ensemble
python -m tcga_brca.models.hierarchical_classifier
python -m tcga_brca.models.tabnet_classifier
python -m tcga_brca.models.multitask_survival
python -m tcga_brca.models.ablation_geo_filter
python -m tcga_brca.models.external_validation
python -m tcga_brca.models.domain_adaptation
python -m tcga_brca.models.evaluate

# Phase 4: Interpretation + Validation
python -m tcga_brca.interpretation.shap_analysis
python -m tcga_brca.interpretation.feature_stability
python -m tcga_brca.interpretation.prediction_confidence
python -m tcga_brca.clinical_validation.survival_analysis
python -m tcga_brca.clinical_validation.cox_regression
python -m tcga_brca.knowledge_graph.build_kg

# Dashboard
streamlit run app/streamlit_app.py
```

R script downloads approximately 5.2 GB from GDC (cached after first run). METABRIC expression (approximately 300 MB) is cached after first API call.

---

## Dependencies

| Category | Tools |
|---|---|
| Data Acquisition (R) | TCGAbiolinks, DESeq2, SummarizedExperiment |
| Data Acquisition (Python) | GEOparse, requests |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| DL | PyTorch, pytorch-tabnet (self-supervised pretraining) |
| Interpretation | SHAP (via XGBoost native TreeSHAP) |
| Survival Analysis | lifelines (KM, log-rank, CoxPH) |
| Domain Adaptation | scipy.linalg (CORAL matrix operations) |
| Knowledge Graph | NetworkX, pyvis |
| Dashboard | Streamlit |
| Statistics | scipy, statsmodels |

---

## Methodology Transfer from Prior Projects

| Prior Project | Transferred Method | Application |
|---|---|---|
| CXR Pneumonia Detection | Hierarchical classifier (2-stage) | ER+/ER- → LumA/LumB, Basal/HER2 |
| CXR Pneumonia Detection | WeightedRandomSampler → class weights | Optimal imbalance strategy in both domains |
| CXR Pneumonia Detection | Pretrained fine-tuning (ImageNet → X-ray) | TabNet self-supervised pretrain → fine-tune |
| Stat Consulting Ensemble | LGB+XGB+CB OOF stacking | Same architecture on genomic data |
| GAM Parkinson's | Non-linearity diagnosis before model selection | Kruskal-Wallis (non-parametric) feature selection |
| PINN Lookback Options | Multi-objective loss (PDE + boundary) | CE + alpha * Cox loss, alpha search |
| NBA Salary Prediction | PCA dimensionality reduction | PCA subtype separation analysis |
| CNP VoC Pipeline | Streamlit dashboard, README conventions | 6-tab dashboard, ASCII pipeline |
| Consumer Signal Agent | Database schema design | KG graph schema |

---

## Technical Notes

### XGBoost-SHAP Compatibility
XGBoost 2.x stores multiclass base_score as a vector. SHAP's XGBTreeModelLoader expects a scalar, causing ValueError. Workaround: XGBoost native booster.predict(dmatrix, pred_contribs=True), which implements the same TreeSHAP algorithm internally.

### Cross-Platform Normalization
TCGA (RNA-seq, DESeq2-normalized) and METABRIC (microarray, Z-scores) require independent standardization. Fitting a scaler on TCGA and transforming METABRIC produces severe scale mismatch (accuracy drops to 12%). CORAL further aligns covariance structures after independent scaling.

### Consensus DEG Filter
Custom preprocessing: GEO microarray DEGs (t-test, 1,103 genes) intersected with TCGA RNA-seq DEGs (DESeq2, 6,646 genes) = 526 consensus genes. These two analyses used different datasets, different platforms, and different statistical methods — genes surviving this intersection are the most robust differential expression signals. Ablation confirms 98.8% feature reduction with 2.5% F1 cost.

### TabNet Pretraining
Self-supervised pretraining on unlabeled expression data (pretraining_ratio=0.5) learns feature correlations before classification. This is analogous to masked language modeling in NLP — the model learns gene co-expression patterns, then fine-tunes for subtype classification. Improved accuracy from 83.6% to 90.9%.

### Multi-task Loss Design
Total loss = CE(classification) + alpha * Cox(survival). alpha=0.01 underweights survival (F1=0.883), alpha=0.5 overweights it (F1=0.762), alpha=0.1 balances both (F1=0.803). This trade-off mirrors PINN's PDE residual weight tuning, where excessive physics constraints degrade boundary condition accuracy.
