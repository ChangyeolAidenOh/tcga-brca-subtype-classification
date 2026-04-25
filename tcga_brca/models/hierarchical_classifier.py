"""
Hierarchical Classifier — Clinical Decision Structure
=======================================================
Instead of flat 5-class, classify in levels mirroring clinical workflow:
  Level 1: ER+ (LumA, LumB) vs ER- (Basal, Her2) vs Normal-like
  Level 2a: LumA vs LumB (within ER+)
  Level 2b: Basal vs Her2 (within ER-)

CXR transfer: Normal vs Pneumonia → Bacterial vs Viral

Usage:
    python -m tcga_brca.models.hierarchical_classifier
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_data():
    """Load expression and labels."""
    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")

    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    common = expr.index.intersection(labels.index)
    return expr.loc[common], labels.loc[common]


def create_hierarchy_labels(labels):
    """Map PAM50 subtypes to hierarchical levels.

    Level 1: ER_pos (LumA, LumB) / ER_neg (Basal, Her2) / Normal
    Level 2a: LumA vs LumB (within ER_pos)
    Level 2b: Basal vs Her2 (within ER_neg)
    """
    level1_map = {
        "BRCA_LumA": "ER_pos",
        "BRCA_LumB": "ER_pos",
        "BRCA_Basal": "ER_neg",
        "BRCA_Her2": "ER_neg",
        "BRCA_Normal": "Normal_like",
    }
    level1 = labels.map(level1_map)

    level2a_mask = labels.isin(["BRCA_LumA", "BRCA_LumB"])
    level2b_mask = labels.isin(["BRCA_Basal", "BRCA_Her2"])

    return level1, level2a_mask, level2b_mask


def train_level(X_train, y_train, X_test, y_test, level_name):
    """Train XGBoost for one level of the hierarchy."""
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    y_te_enc = le.transform(y_test)
    class_names = le.classes_

    sw = compute_sample_weight("balanced", y_tr_enc)

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X_train, y_tr_enc, sample_weight=sw)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_te_enc, y_pred)
    f1 = f1_score(y_te_enc, y_pred, average="macro")

    print(f"\n  {level_name}:")
    print(f"    Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(f"    Classes: {list(class_names)}")
    print(classification_report(y_te_enc, y_pred, target_names=class_names, zero_division=0))

    return model, le, y_pred, y_proba, acc, f1


def run_hierarchical_classification(expr, labels):
    """Run full hierarchical classification pipeline."""
    level1, l2a_mask, l2b_mask = create_hierarchy_labels(labels)

    X_train, X_test, y_train_full, y_test_full = train_test_split(
        expr.values, labels.values,
        test_size=config.TEST_SIZE, stratify=labels.values,
        random_state=config.RANDOM_STATE,
    )

    # Corresponding level1 labels
    train_idx = expr.index[expr.index.isin(
        expr.iloc[:len(X_train)].index
    )]

    # Simpler approach: split with index tracking
    indices = np.arange(len(expr))
    train_idx, test_idx = train_test_split(
        indices, test_size=config.TEST_SIZE, stratify=labels.values,
        random_state=config.RANDOM_STATE,
    )

    X_train_all = expr.values[train_idx]
    X_test_all = expr.values[test_idx]
    y_train_full = labels.values[train_idx]
    y_test_full = labels.values[test_idx]

    level1_train = pd.Series(y_train_full).map({
        "BRCA_LumA": "ER_pos", "BRCA_LumB": "ER_pos",
        "BRCA_Basal": "ER_neg", "BRCA_Her2": "ER_neg",
        "BRCA_Normal": "Normal_like",
    }).values
    level1_test = pd.Series(y_test_full).map({
        "BRCA_LumA": "ER_pos", "BRCA_LumB": "ER_pos",
        "BRCA_Basal": "ER_neg", "BRCA_Her2": "ER_neg",
        "BRCA_Normal": "Normal_like",
    }).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_all)
    X_test_s = scaler.transform(X_test_all)

    # Level 1: ER+ vs ER- vs Normal-like
    print("=" * 50)
    print("LEVEL 1: ER+ vs ER- vs Normal-like")
    print("=" * 50)
    model_l1, le_l1, pred_l1, proba_l1, acc_l1, f1_l1 = train_level(
        X_train_s, level1_train, X_test_s, level1_test, "Level 1"
    )

    # Level 2a: LumA vs LumB (ER+ only)
    print("=" * 50)
    print("LEVEL 2a: Luminal A vs Luminal B (within ER+)")
    print("=" * 50)
    tr_erpos = y_train_full == "BRCA_LumA"
    tr_erpos |= y_train_full == "BRCA_LumB"
    te_erpos = y_test_full == "BRCA_LumA"
    te_erpos |= y_test_full == "BRCA_LumB"

    if te_erpos.sum() > 0:
        model_l2a, le_l2a, pred_l2a, proba_l2a, acc_l2a, f1_l2a = train_level(
            X_train_s[tr_erpos], y_train_full[tr_erpos],
            X_test_s[te_erpos], y_test_full[te_erpos], "Level 2a (Luminal)"
        )
    else:
        acc_l2a, f1_l2a = 0, 0

    # Level 2b: Basal vs Her2 (ER- only)
    print("=" * 50)
    print("LEVEL 2b: Basal vs HER2 (within ER-)")
    print("=" * 50)
    tr_erneg = y_train_full == "BRCA_Basal"
    tr_erneg |= y_train_full == "BRCA_Her2"
    te_erneg = y_test_full == "BRCA_Basal"
    te_erneg |= y_test_full == "BRCA_Her2"

    if te_erneg.sum() > 0:
        model_l2b, le_l2b, pred_l2b, proba_l2b, acc_l2b, f1_l2b = train_level(
            X_train_s[tr_erneg], y_train_full[tr_erneg],
            X_test_s[te_erneg], y_test_full[te_erneg], "Level 2b (ER-negative)"
        )
    else:
        acc_l2b, f1_l2b = 0, 0

    # Combine hierarchical predictions into final 5-class
    print("=" * 50)
    print("COMBINED HIERARCHICAL RESULT")
    print("=" * 50)
    pred_l1_decoded = le_l1.inverse_transform(pred_l1)
    final_pred = np.empty(len(y_test_full), dtype=object)

    for i in range(len(y_test_full)):
        if pred_l1_decoded[i] == "Normal_like":
            final_pred[i] = "BRCA_Normal"
        elif pred_l1_decoded[i] == "ER_pos":
            # Find position in ER+ subset
            erpos_indices = np.where(
                (pred_l1_decoded == "ER_pos") &
                (np.arange(len(pred_l1_decoded)) <= i)
            )[0]
            pos_in_erpos = len(erpos_indices) - 1
            if te_erpos.sum() > 0 and pos_in_erpos < len(pred_l2a):
                final_pred[i] = le_l2a.inverse_transform([pred_l2a[pos_in_erpos]])[0]
            else:
                final_pred[i] = "BRCA_LumA"  # fallback
        elif pred_l1_decoded[i] == "ER_neg":
            erneg_indices = np.where(
                (pred_l1_decoded == "ER_neg") &
                (np.arange(len(pred_l1_decoded)) <= i)
            )[0]
            pos_in_erneg = len(erneg_indices) - 1
            if te_erneg.sum() > 0 and pos_in_erneg < len(pred_l2b):
                final_pred[i] = le_l2b.inverse_transform([pred_l2b[pos_in_erneg]])[0]
            else:
                final_pred[i] = "BRCA_Basal"  # fallback

    # Simpler approach: use ground truth grouping for Level 2 evaluation
    # (hierarchical = Level 1 routes, Level 2 classifies within group)
    final_pred_simple = np.empty(len(y_test_full), dtype=object)
    for i in range(len(y_test_full)):
        if pred_l1_decoded[i] == "Normal_like":
            final_pred_simple[i] = "BRCA_Normal"
        elif pred_l1_decoded[i] == "ER_pos":
            final_pred_simple[i] = "BRCA_LumA"  # placeholder
        elif pred_l1_decoded[i] == "ER_neg":
            final_pred_simple[i] = "BRCA_Basal"  # placeholder

    # For true hierarchical: route by L1, then apply L2 within each group
    # Use boolean mask approach
    final_hierarchical = pred_l1_decoded.copy()
    erpos_test_mask = pred_l1_decoded == "ER_pos"
    erneg_test_mask = pred_l1_decoded == "ER_neg"

    if erpos_test_mask.sum() > 0 and te_erpos.sum() > 0:
        l2a_model_pred = model_l2a.predict(X_test_s[erpos_test_mask])
        l2a_decoded = le_l2a.inverse_transform(l2a_model_pred)
        final_hierarchical[erpos_test_mask] = l2a_decoded

    if erneg_test_mask.sum() > 0 and te_erneg.sum() > 0:
        l2b_model_pred = model_l2b.predict(X_test_s[erneg_test_mask])
        l2b_decoded = le_l2b.inverse_transform(l2b_model_pred)
        final_hierarchical[erneg_test_mask] = l2b_decoded

    final_hierarchical[pred_l1_decoded == "Normal_like"] = "BRCA_Normal"

    acc_h = accuracy_score(y_test_full, final_hierarchical)
    f1_h = f1_score(y_test_full, final_hierarchical, average="macro")

    print(f"Hierarchical Accuracy: {acc_h:.4f} | Macro F1: {f1_h:.4f}")
    print(classification_report(y_test_full, final_hierarchical, zero_division=0))

    # Compare with flat classifier
    print("=" * 50)
    print("FLAT vs HIERARCHICAL COMPARISON")
    print("=" * 50)

    # Load flat result
    flat_path = os.path.join(config.RESULTS_TABLES, "xgboost_summary.json")
    if os.path.exists(flat_path):
        with open(flat_path) as f:
            flat = json.load(f)
        flat_acc = flat["test_metrics"]["accuracy"]
        flat_f1 = flat["test_metrics"]["macro_f1"]
        print(f"Flat:         Accuracy {flat_acc:.4f} | Macro F1 {flat_f1:.4f}")
    print(f"Hierarchical: Accuracy {acc_h:.4f} | Macro F1 {f1_h:.4f}")

    return {
        "level1": {"accuracy": float(acc_l1), "macro_f1": float(f1_l1)},
        "level2a_luminal": {"accuracy": float(acc_l2a), "macro_f1": float(f1_l2a)},
        "level2b_erneg": {"accuracy": float(acc_l2b), "macro_f1": float(f1_l2b)},
        "combined": {"accuracy": float(acc_h), "macro_f1": float(f1_h)},
    }, y_test_full, final_hierarchical


def plot_hierarchical_results(y_test, y_pred, results, save_dir):
    """Plot hierarchical classification results."""
    os.makedirs(save_dir, exist_ok=True)

    # Confusion matrix
    class_order = ["BRCA_Basal", "BRCA_Her2", "BRCA_LumA", "BRCA_LumB", "BRCA_Normal"]
    present = [c for c in class_order if c in set(y_test) | set(y_pred)]
    cm = confusion_matrix(y_test, y_pred, labels=present)

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    short = [c.replace("BRCA_", "") for c in present]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=short, yticklabels=short, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = results["combined"]["accuracy"]
    f1 = results["combined"]["macro_f1"]
    ax.set_title(f"Hierarchical Classifier\nAccuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "hierarchical_confusion_matrix.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Hierarchical Classifier — Clinical Decision Structure")
    print("=" * 60)
    print("Level 1: ER+ vs ER- vs Normal-like")
    print("Level 2a: LumA vs LumB | Level 2b: Basal vs Her2")
    print("Transfer: CXR Normal vs Pneumonia → Bacterial vs Viral")

    expr, labels = load_data()
    print(f"Data: {expr.shape[0]} samples x {expr.shape[1]} genes")

    results, y_test, y_pred = run_hierarchical_classification(expr, labels)

    plot_hierarchical_results(y_test, y_pred, results, config.RESULTS_FIGURES)

    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    with open(os.path.join(config.RESULTS_TABLES, "hierarchical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nHierarchical classifier complete.")


if __name__ == "__main__":
    main()
