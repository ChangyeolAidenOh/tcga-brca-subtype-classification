import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_EXTERNAL = os.path.join(PROJECT_ROOT, "data", "external")
RESULTS_FIGURES = os.path.join(PROJECT_ROOT, "results", "figures")
RESULTS_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
RESULTS_MODELS = os.path.join(PROJECT_ROOT, "results", "models")

# --- Data Sources ---
GEO_DATASET_ID = "GSE42568"
TCGA_PROJECT = "TCGA-BRCA"
GDC_API_ENDPOINT = "https://api.gdc.cancer.gov"
CBIO_API_ENDPOINT = "https://www.cbioportal.org/api"
DISGENET_TSV_PATH = os.path.join(DATA_EXTERNAL, "curated_gene_disease_associations.tsv")

# --- Preprocessing ---
LOG2_PSEUDOCOUNT = 1
LOW_EXPRESSION_THRESHOLD = 1
DEG_PVALUE_THRESHOLD = 0.01
DEG_LOG2FC_THRESHOLD = 1.5

# --- DEG Fallback ---
DEG_LOG2FC_FALLBACK = 1.0
DEG_MIN_GENE_COUNT = 100

# --- Modeling ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Interpretation ---
SHAP_TOP_N = 20
KM_SPLIT_METHOD = "median"

# --- KG ---
KG_MAX_GENES = 20

# --- Visualization ---
FIGURE_SIZE = (7, 5.6)
FIGURE_DPI = 100

# --- Known Cancer Genes (PAM50 subtype markers) ---
KNOWN_BREAST_CANCER_GENES = {
    "Luminal": ["ESR1", "PGR", "FOXA1", "GATA3"],
    "HER2": ["ERBB2", "GRB7"],
    "Basal": ["KRT5", "KRT14", "EGFR", "KRT17"],
    "Normal-like": ["ADIPOQ", "ADH1B"],  # few specific markers; SHAP results are exploratory
    "General": ["TP53", "MKI67", "BRCA1", "BRCA2"],
}
