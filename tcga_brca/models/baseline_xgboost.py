"""
Stage 3-2: Baseline XGBoost + Class Imbalance Handling
=======================================================
Train XGBoost baseline with 3 imbalance strategies:
  1. No handling
  2. Class weights (compute_sample_weight)
  3. SMOTE oversampling

CXR transfer: WeightedRandomSampler → compute_sample_weight
Normal-like (~3.7%) parallels CXR Normal class Recall 0.63

Usage:
    python -m tcga_brca.models.baseline_xgboost
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
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
    """Load statistically selected features from Stage 3-1."""
    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")
        print("Using full expression (run feature_selection first for filtered set)")

    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    common = expr.index.intersection(labels.index)
    expr = expr.loc[common]
    labels = labels.loc[common]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    print(f"Data: {expr.shape[0]} samples x {expr.shape[1]} features")
    print(f"Classes: {dict(zip(class_names, np.bincount(y)))}")
    return expr.values, y, class_names, le, expr.columns.tolist()


def train_xgboost(X_train, y_train, X_test, y_test, sample_weights=None, label=""):
    """Train and evaluate XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    try:
        auroc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | AUROC: {auroc:.4f}")

    return model, y_pred, y_proba, {"accuracy": acc, "macro_f1": f1_macro, "auroc": auroc}


def run_imbalance_comparison(X, y, class_names):
    """Compare 3 imbalance handling strategies."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        stratify=y, random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Strategy 1: No handling
    model1, pred1, proba1, metrics1 = train_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test,
        label="No imbalance handling"
    )
    results["no_handling"] = metrics1

    # Strategy 2: Class weights
    sw = compute_sample_weight("balanced", y_train)
    model2, pred2, proba2, metrics2 = train_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test,
        sample_weights=sw, label="Class weights (balanced)"
    )
    results["class_weights"] = metrics2

    # Strategy 3: SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
        model3, pred3, proba3, metrics3 = train_xgboost(
            X_train_sm, y_train_sm, X_test_scaled, y_test,
            label="SMOTE oversampling"
        )
        results["smote"] = metrics3
    except ImportError:
        print("\nimbalanced-learn not installed. Skipping SMOTE.")
        print("  Install with: pip install imbalanced-learn")
        model3, pred3, proba3 = None, None, None

    # Select best strategy
    best_strategy = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\nBest strategy: {best_strategy} (Macro F1: {results[best_strategy]['macro_f1']:.4f})")

    # Use best model for detailed report
    if best_strategy == "no_handling":
        best_model, best_pred, best_proba = model1, pred1, proba1
    elif best_strategy == "class_weights":
        best_model, best_pred, best_proba = model2, pred2, proba2
    else:
        best_model, best_pred, best_proba = model3, pred3, proba3

    # Classification report
    print(f"\nClassification Report ({best_strategy}):")
    print(classification_report(y_test, best_pred, target_names=class_names))

    return best_model, best_pred, best_proba, y_test, results, best_strategy, scaler


def run_cv(X, y, class_names, strategy="class_weights"):
    """5-fold stratified cross-validation with best strategy."""
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        sw = compute_sample_weight("balanced", y_tr) if strategy == "class_weights" else None

        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "macro_f1": f1})

    fold_df = pd.DataFrame(fold_metrics)
    print(f"\n{config.CV_FOLDS}-Fold CV Results ({strategy}):")
    print(f"  Accuracy: {fold_df['accuracy'].mean():.4f} +/- {fold_df['accuracy'].std():.4f}")
    print(f"  Macro F1: {fold_df['macro_f1'].mean():.4f} +/- {fold_df['macro_f1'].std():.4f}")
    return fold_df


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)

    short_names = [c.replace("BRCA_", "") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("XGBoost Baseline — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_results(results, cv_df, best_strategy, class_names, y_test, y_pred, save_dir):
    """Save all baseline results."""
    os.makedirs(save_dir, exist_ok=True)

    # Comparison table
    comp_df = pd.DataFrame(results).T
    comp_df.to_csv(os.path.join(save_dir, "xgboost_imbalance_comparison.csv"))

    # CV results
    cv_df.to_csv(os.path.join(save_dir, "xgboost_cv_results.csv"), index=False)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(save_dir, "xgboost_classification_report.csv"))

    # Summary
    summary = {
        "best_strategy": best_strategy,
        "test_metrics": results[best_strategy],
        "cv_accuracy_mean": float(cv_df["accuracy"].mean()),
        "cv_accuracy_std": float(cv_df["accuracy"].std()),
        "cv_f1_mean": float(cv_df["macro_f1"].mean()),
        "cv_f1_std": float(cv_df["macro_f1"].std()),
    }
    with open(os.path.join(save_dir, "xgboost_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {save_dir}")


def main():
    print("=" * 60)
    print("Stage 3-2: XGBoost Baseline + Imbalance Comparison")
    print("=" * 60)

    X, y, class_names, le, feature_names = load_data()

    # Imbalance strategy comparison
    best_model, best_pred, best_proba, y_test, results, best_strategy, scaler = (
        run_imbalance_comparison(X, y, class_names)
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_test, best_pred, class_names,
        os.path.join(config.RESULTS_FIGURES, "xgboost_confusion_matrix.png")
    )

    # 5-fold CV with best strategy
    cv_df = run_cv(X, y, class_names, strategy=best_strategy)

    # Save
    save_results(results, cv_df, best_strategy, class_names, y_test, best_pred, config.RESULTS_TABLES)

    # Save model for SHAP (Stage 4)
    import pickle
    os.makedirs(config.RESULTS_MODELS, exist_ok=True)
    with open(os.path.join(config.RESULTS_MODELS, "xgboost_baseline.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "label_encoder": le, "feature_names": feature_names}, f)

    print("\nStage 3-2 complete.")


if __name__ == "__main__":
    main()
