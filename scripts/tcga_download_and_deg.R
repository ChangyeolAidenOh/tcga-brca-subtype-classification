#!/usr/bin/env Rscript
# ==============================================================
# TCGA-BRCA Data Acquisition & DESeq2 DEG Analysis
# ==============================================================
# Standard bioinformatics pipeline using TCGAbiolinks for data
# download from GDC and DESeq2 for differential expression.
#
# This is the canonical approach cited in bioinformatics papers:
#   "Data was downloaded from GDC using TCGAbiolinks (Colaprico et al., 2016)"
#
# Outputs CSV files consumed by the Python classification pipeline.
#
# Usage:
#   Rscript scripts/tcga_download_and_deg.R
#
# Installation (one-time):
#   install.packages("BiocManager")
#   BiocManager::install(c("TCGAbiolinks", "DESeq2", "SummarizedExperiment"))
#   install.packages(c("dplyr", "ggplot2"))
# ==============================================================

suppressPackageStartupMessages({
    library(TCGAbiolinks)
    library(SummarizedExperiment)
    library(DESeq2)
    library(dplyr)
    library(ggplot2)
})

cat("==============================================================\n")
cat("TCGA-BRCA Data Acquisition (TCGAbiolinks) + DESeq2\n")
cat("==============================================================\n\n")

# ------ Configuration ------
PROJECT <- "TCGA-BRCA"
DATA_DIR <- file.path(getwd(), "data")
RAW_DIR <- file.path(DATA_DIR, "raw")
PROCESSED_DIR <- file.path(DATA_DIR, "processed")
RESULTS_DIR <- file.path(getwd(), "results")
FIGURES_DIR <- file.path(RESULTS_DIR, "figures")
TABLES_DIR <- file.path(RESULTS_DIR, "tables")

for (d in c(RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

PVALUE_THRESHOLD <- 0.01
LOG2FC_THRESHOLD <- 1.5


# ==============================================================
# PART 1: Download TCGA-BRCA RNA-seq from GDC
# ==============================================================
cat("--------------------------------------------------------------\n")
cat("PART 1: Downloading TCGA-BRCA RNA-seq from GDC\n")
cat("--------------------------------------------------------------\n\n")

# Query GDC for STAR-Counts RNA-seq data
query_rnaseq <- GDCquery(
    project = PROJECT,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "STAR - Counts"
)

cat(sprintf("Query returned %d files\n", nrow(getResults(query_rnaseq))))

# Download (cached after first run)
GDCdownload(query_rnaseq, directory = RAW_DIR)

# Prepare SummarizedExperiment object
se <- GDCprepare(query_rnaseq, directory = RAW_DIR)

cat(sprintf("SummarizedExperiment: %d genes x %d samples\n", nrow(se), ncol(se)))

# Extract raw counts
counts_raw <- assay(se, "unstranded")
cat(sprintf("Raw counts matrix: %d genes x %d samples\n", nrow(counts_raw), ncol(counts_raw)))

# Extract gene metadata
gene_info <- as.data.frame(rowData(se))
cat(sprintf("Gene annotations: %d genes, columns: %s\n",
            nrow(gene_info), paste(colnames(gene_info), collapse=", ")))

# Extract sample metadata
sample_info <- as.data.frame(colData(se))
cat(sprintf("Sample metadata: %d samples, %d attributes\n",
            nrow(sample_info), ncol(sample_info)))


# ==============================================================
# PART 2: Extract and prepare clinical data
# ==============================================================
cat("\n--------------------------------------------------------------\n")
cat("PART 2: Clinical Data Extraction\n")
cat("--------------------------------------------------------------\n\n")

# Get PAM50 subtype (molecular subtype from GDC)
# TCGAbiolinks provides this in paper_BRCA_Subtype_PAM50 or similar
pam50_col <- NULL
for (col in colnames(sample_info)) {
    if (grepl("pam50|PAM50|subtype", col, ignore.case = TRUE)) {
        vals <- unique(sample_info[[col]])
        if (any(grepl("Lum|Basal|Her2|Normal", vals, ignore.case = TRUE))) {
            pam50_col <- col
            cat(sprintf("PAM50 column found: %s\n", col))
            break
        }
    }
}

# If not in colData, try TCGAquery_subtype
if (is.null(pam50_col)) {
    cat("PAM50 not in colData. Fetching via TCGAquery_subtype...\n")
    subtype_data <- TCGAquery_subtype("BRCA")
    cat(sprintf("Subtype data: %d rows, columns: %s\n",
                nrow(subtype_data),
                paste(head(colnames(subtype_data), 20), collapse=", ")))

    # Find PAM50 column
    for (col in colnames(subtype_data)) {
        if (grepl("pam50", col, ignore.case = TRUE)) {
            pam50_col <- col
            cat(sprintf("PAM50 column: %s\n", col))
            break
        }
    }

    if (!is.null(pam50_col)) {
        pam50_labels <- subtype_data[, c("patient", pam50_col)]
        colnames(pam50_labels) <- c("patient", "PAM50")
    }
} else {
    pam50_labels <- data.frame(
        patient = sample_info$patient,
        PAM50 = sample_info[[pam50_col]],
        stringsAsFactors = FALSE
    )
}

# Standardize PAM50 labels
if (exists("pam50_labels")) {
    pam50_labels$PAM50_standard <- case_when(
        grepl("LumA|Luminal A", pam50_labels$PAM50, ignore.case = TRUE) ~ "BRCA_LumA",
        grepl("LumB|Luminal B", pam50_labels$PAM50, ignore.case = TRUE) ~ "BRCA_LumB",
        grepl("Basal", pam50_labels$PAM50, ignore.case = TRUE) ~ "BRCA_Basal",
        grepl("Her2", pam50_labels$PAM50, ignore.case = TRUE) ~ "BRCA_Her2",
        grepl("Normal", pam50_labels$PAM50, ignore.case = TRUE) ~ "BRCA_Normal",
        TRUE ~ NA_character_
    )
    pam50_labels <- pam50_labels[!is.na(pam50_labels$PAM50_standard), ]

    cat(sprintf("\nPAM50 distribution:\n"))
    print(table(pam50_labels$PAM50_standard))
}

# Sample type (Primary Tumor vs Solid Tissue Normal)
sample_type <- data.frame(
    barcode = colnames(se),
    sample_type = sample_info$sample_type,
    patient = sample_info$patient,
    stringsAsFactors = FALSE
)

cat(sprintf("\nSample type distribution:\n"))
print(table(sample_type$sample_type))

# Survival data
surv_cols <- c("patient", "vital_status", "days_to_death", "days_to_last_follow_up")
available_surv <- intersect(surv_cols, colnames(sample_info))
if (length(available_surv) >= 2) {
    surv_data <- sample_info[, available_surv, drop = FALSE]
    surv_data <- as.data.frame(surv_data)
    surv_data$sampleId <- colnames(se)

    surv_data$os_months <- ifelse(
        surv_data$vital_status == "Dead",
        as.numeric(surv_data$days_to_death) / 30.44,
        as.numeric(surv_data$days_to_last_follow_up) / 30.44
    )
    surv_data$os_event <- ifelse(surv_data$vital_status == "Dead", 1, 0)

    surv_clean <- surv_data[!is.na(surv_data$os_months), c("sampleId", "os_months", "os_event")]
    cat(sprintf("\nSurvival data: %d patients, %d events\n",
                nrow(surv_clean), sum(surv_clean$os_event)))
}


# ==============================================================
# PART 3: DESeq2 Differential Expression (Tumor vs Normal)
# ==============================================================
cat("\n--------------------------------------------------------------\n")
cat("PART 3: DESeq2 DEG Analysis (Tumor vs Normal)\n")
cat("--------------------------------------------------------------\n\n")

# Separate tumor and normal samples
is_tumor <- grepl("Primary|Tumor", sample_type$sample_type, ignore.case = TRUE)
is_normal <- grepl("Normal|Solid Tissue Normal", sample_type$sample_type, ignore.case = TRUE)

n_tumor <- sum(is_tumor)
n_normal <- sum(is_normal)
cat(sprintf("Tumor samples: %d, Normal samples: %d\n", n_tumor, n_normal))

if (n_normal >= 3) {
    # Build DESeq2 dataset with tumor vs normal
    deg_samples <- sample_type[is_tumor | is_normal, ]
    deg_counts <- counts_raw[, deg_samples$barcode]

    condition <- ifelse(
        grepl("Normal", deg_samples$sample_type, ignore.case = TRUE),
        "Normal", "Tumor"
    )
    col_data_deg <- data.frame(
        condition = factor(condition, levels = c("Normal", "Tumor")),
        row.names = deg_samples$barcode
    )

    # Filter low-expression genes
    keep <- rowSums(deg_counts >= 10) >= 3
    deg_counts_filtered <- deg_counts[keep, ]
    cat(sprintf("Genes after filtering: %d\n", nrow(deg_counts_filtered)))

    # Use gene symbols as row names
    gene_symbols <- gene_info$gene_name[match(rownames(deg_counts_filtered), rownames(gene_info))]
    valid_symbols <- !is.na(gene_symbols) & gene_symbols != "" & !duplicated(gene_symbols)
    deg_counts_filtered <- deg_counts_filtered[valid_symbols, ]
    rownames(deg_counts_filtered) <- gene_symbols[valid_symbols]

    # Create DESeqDataSet
    dds <- DESeqDataSetFromMatrix(
        countData = deg_counts_filtered,
        colData = col_data_deg,
        design = ~ condition
    )

    # Run DESeq2
    cat("Running DESeq2...\n")
    dds <- DESeq(dds)

    # Results with shrinkage
    res <- results(dds, contrast = c("condition", "Tumor", "Normal"), alpha = PVALUE_THRESHOLD)

    tryCatch({
        library(apeglm)
        res <- lfcShrink(dds, coef = "condition_Tumor_vs_Normal", type = "apeglm")
        cat("Applied apeglm shrinkage\n")
    }, error = function(e) {
        cat("apeglm not available, using default results\n")
    })

    res_df <- as.data.frame(res)
    res_df$gene <- rownames(res_df)
    res_df$significant <- !is.na(res_df$padj) &
                          res_df$padj < PVALUE_THRESHOLD &
                          abs(res_df$log2FoldChange) > LOG2FC_THRESHOLD

    n_up <- sum(res_df$significant & res_df$log2FoldChange > 0, na.rm = TRUE)
    n_down <- sum(res_df$significant & res_df$log2FoldChange < 0, na.rm = TRUE)

    cat(sprintf("\nDESeq2 DEG Results:\n"))
    cat(sprintf("  Total DEGs: %d (Up: %d, Down: %d)\n", n_up + n_down, n_up, n_down))

    # Save DESeq2 results
    write.csv(res_df, file.path(TABLES_DIR, "deseq2_full_results.csv"), row.names = FALSE)
    sig_df <- res_df[res_df$significant & !is.na(res_df$significant), ]
    write.csv(sig_df, file.path(TABLES_DIR, "deseq2_significant_genes.csv"), row.names = FALSE)
    writeLines(sig_df$gene, file.path(PROCESSED_DIR, "deseq2_feature_candidates.txt"))

    # Volcano plot
    png(file.path(FIGURES_DIR, "deseq2_volcano_plot.png"), width = 700, height = 560, res = 100)
    par(mar = c(5, 5, 4, 2))
    col_vec <- ifelse(
        res_df$significant & res_df$log2FoldChange > 0, "#E24B4A",
        ifelse(res_df$significant & res_df$log2FoldChange < 0, "#378ADD", "#B4B2A9")
    )
    col_vec[is.na(col_vec)] <- "#B4B2A9"
    plot(res_df$log2FoldChange, -log10(res_df$padj),
         col = col_vec, pch = 16, cex = 0.5,
         xlab = "log2 Fold Change", ylab = "-log10(adjusted p-value)",
         main = sprintf("DESeq2 — TCGA-BRCA Tumor vs Normal\nUp: %d | Down: %d", n_up, n_down))
    abline(h = -log10(PVALUE_THRESHOLD), lty = 2, col = "gray60")
    abline(v = c(-LOG2FC_THRESHOLD, LOG2FC_THRESHOLD), lty = 2, col = "gray60")
    dev.off()
    cat("DESeq2 volcano plot saved\n")

    # Compare with GEO t-test DEGs
    geo_path <- file.path(PROCESSED_DIR, "geo_deg_significant_genes.csv")
    if (file.exists(geo_path)) {
        geo_df <- read.csv(geo_path)
        geo_genes <- geo_df$gene
        deseq_genes <- sig_df$gene
        overlap <- intersect(geo_genes, deseq_genes)

        cat(sprintf("\n--- GEO t-test vs TCGA DESeq2 Comparison ---\n"))
        cat(sprintf("GEO t-test DEGs:    %d\n", length(geo_genes)))
        cat(sprintf("TCGA DESeq2 DEGs:   %d\n", length(deseq_genes)))
        cat(sprintf("Overlap:            %d (%.1f%% of GEO)\n",
                    length(overlap), length(overlap) / max(length(geo_genes), 1) * 100))

        writeLines(overlap, file.path(PROCESSED_DIR, "consensus_deg_genes.txt"))
        cat(sprintf("Consensus genes saved: %d\n", length(overlap)))
    }

} else {
    cat("Insufficient normal samples for DESeq2. Skipping DEG analysis.\n")
}


# ==============================================================
# PART 4: Export for Python Pipeline
# ==============================================================
cat("\n--------------------------------------------------------------\n")
cat("PART 4: Exporting Data for Python Pipeline\n")
cat("--------------------------------------------------------------\n\n")

# Expression matrix — normalized counts (for ML classification)
# Use variance stabilizing transformation (vst) or log2(normalized counts + 1)
if (exists("dds")) {
    # Use all tumor samples for classification
    dds_all <- DESeqDataSetFromMatrix(
        countData = counts_raw[keep, is_tumor][valid_symbols, ],
        colData = data.frame(
            row.names = colnames(counts_raw)[is_tumor],
            dummy = rep("tumor", n_tumor)
        ),
        design = ~ 1
    )
    dds_all <- estimateSizeFactors(dds_all)
    norm_counts <- counts(dds_all, normalized = TRUE)
    rownames(norm_counts) <- gene_symbols[valid_symbols]

    # Log2 transform
    expr_log2 <- log2(norm_counts + 1)
    expr_df <- as.data.frame(t(expr_log2))

    cat(sprintf("Normalized expression matrix: %d samples x %d genes\n",
                nrow(expr_df), ncol(expr_df)))
} else {
    # Fallback: use all tumor samples with simple normalization
    tumor_counts <- counts_raw[, is_tumor]
    lib_sizes <- colSums(tumor_counts)
    norm_counts <- sweep(tumor_counts, 2, lib_sizes, "/") * 1e6  # CPM
    expr_df <- as.data.frame(t(log2(norm_counts + 1)))
    cat(sprintf("CPM expression matrix: %d samples x %d genes\n",
                nrow(expr_df), ncol(expr_df)))
}

# Map sample barcodes to patient IDs for PAM50 matching
barcode_to_patient <- sample_type$patient[match(rownames(expr_df), sample_type$barcode)]

# Save expression
expr_path <- file.path(PROCESSED_DIR, "tcga_brca_expression.csv")
write.csv(expr_df, expr_path)
cat(sprintf("Expression saved: %s\n", expr_path))

# Save labels (PAM50)
if (exists("pam50_labels")) {
    expr_labels <- pam50_labels$PAM50_standard[match(barcode_to_patient, pam50_labels$patient)]
    names(expr_labels) <- rownames(expr_df)
    expr_labels <- expr_labels[!is.na(expr_labels)]

    labels_df <- data.frame(sampleId = names(expr_labels), PAM50 = expr_labels)
    labels_path <- file.path(PROCESSED_DIR, "tcga_brca_labels.csv")
    write.csv(labels_df, labels_path, row.names = FALSE)
    cat(sprintf("Labels saved: %s (%d samples with PAM50)\n", labels_path, nrow(labels_df)))
}

# Save survival data
if (exists("surv_clean")) {
    surv_path <- file.path(PROCESSED_DIR, "tcga_brca_survival.csv")
    write.csv(surv_clean, surv_path, row.names = FALSE)
    cat(sprintf("Survival data saved: %s\n", surv_path))
}

# Save raw counts (for reference)
raw_path <- file.path(RAW_DIR, "tcga_brca_raw_counts.csv")
if (exists("deg_counts_filtered")) {
    write.csv(deg_counts_filtered, raw_path)
    cat(sprintf("Raw counts saved: %s\n", raw_path))
}

cat("\n==============================================================\n")
cat("EXPORT COMPLETE\n")
cat("==============================================================\n")
cat("Files ready for Python pipeline:\n")
cat(sprintf("  Expression: %s\n", expr_path))
if (exists("labels_df")) cat(sprintf("  Labels:     %s\n", labels_path))
if (exists("surv_clean")) cat(sprintf("  Survival:   %s\n", surv_path))
cat("\nNext: python -m tcga_brca.features.feature_selection\n")
