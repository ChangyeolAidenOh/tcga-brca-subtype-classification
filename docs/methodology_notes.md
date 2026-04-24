# Methodology Notes

This document records all methodology pivots, design decisions, and their rationale.

---

## Pre-project decisions

### DEG threshold fallback plan
- Primary: adj. p-value < 0.01, |log2FC| > 1.5
- If DEG count < 100: relax to |log2FC| > 1.0
- Rationale: 1.5 is conservative; too few DEGs would limit feature selection downstream

### Hierarchical classifier design
- Decision point: Day 3-4 (during TCGA data download and EDA)
- Key question: how many Solid Tissue Normal samples exist in TCGA-BRCA?
- Option 1: Solid Tissue Normal vs Tumor (binary) → Tumor: PAM50 5-class
- Option 2: Exclude Solid Tissue Normal, PAM50 5-class direct classification
- Note: PAM50 Normal-like is a SUBTYPE, not normal tissue

### Normal-like subtype handling
- Normal-like has ~60 samples (~5%), smallest class
- Few subtype-specific markers (ADIPOQ, ADH1B)
- SHAP results for Normal-like will be exploratory
- Fallback: if Normal-like causes severe instability, consider 4-class (document rationale)

---

## During-project notes

(to be filled during execution)
