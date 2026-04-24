"""
Stage 3-3: Stacking Ensemble (LightGBM + XGBoost + CatBoost)
=============================================================
OOF-based stacking ensemble with Logistic Regression meta-learner.
Direct transfer from Stat Consulting Internship project.

Usage:
    python -m tcga_brca.models.stacking_ensemble
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_data():
    """Load statistically selected features."""
    expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(config.DATA_PROCESSED, "tcga_brca_expression.csv")

    expr = pd.read_csv(expr_path, index_col=0)
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()

    common = expr.index.intersection(labels.index)
    expr, labels = expr.loc[common], labels.loc[common]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    print(f"Data: {expr.shape[0]} samples x {expr.shape[1]} features")
    print(f"Classes: {dict(zip(class_names, np.bincount(y)))}")
    return expr.values, y, class_names, le, expr.columns.tolist()


def get_base_models(n_classes):
    """Define base models — same trio as Stat Consulting project."""
    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, num_leaves=31,
            is_unbalance=True, random_state=config.RANDOM_STATE, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            auto_class_weights="Balanced",
            random_state=config.RANDOM_STATE, verbose=0,
        ),
    }
    return models


def generate_oof_predictions(X_train, y_train, X_test, n_classes):
    """Generate Out-of-Fold predictions for stacking.

    OOF prevents data leakage — direct transfer from Stat Consulting.
    """
    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    models = get_base_models(n_classes)

    oof_train = np.zeros((X_train.shape[0], n_classes * len(models)))
    oof_test = np.zeros((X_test.shape[0], n_classes * len(models)))

    base_scores = {}

    for m_idx, (name, model) in enumerate(models.items()):
        print(f"\n  {name}:")
        col_start = m_idx * n_classes
        col_end = col_start + n_classes

        test_preds_folds = np.zeros((config.CV_FOLDS, X_test.shape[0], n_classes))
        fold_accs = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            if name == "XGBoost":
                sw = compute_sample_weight("balanced", y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)

            oof_train[val_idx, col_start:col_end] = model.predict_proba(X_val)
            test_preds_folds[fold] = model.predict_proba(X_test)

            fold_acc = accuracy_score(y_val, model.predict(X_val))
            fold_accs.append(fold_acc)
            print(f"    Fold {fold + 1}: accuracy {fold_acc:.4f}")

        oof_test[:, col_start:col_end] = test_preds_folds.mean(axis=0)
        base_scores[name] = {"mean_acc": np.mean(fold_accs), "std_acc": np.std(fold_accs)}
        print(f"    Mean: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")

    return oof_train, oof_test, base_scores


def train_meta_learner(oof_train, y_train, oof_test):
    """Train Logistic Regression meta-learner on OOF predictions."""
    meta = LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_STATE, multi_class="multinomial"
    )
    meta.fit(oof_train, y_train)
    y_pred = meta.predict(oof_test)
    y_proba = meta.predict_proba(oof_test)
    return meta, y_pred, y_proba


def main():
    print("=" * 60)
    print("Stage 3-3: Stacking Ensemble (LightGBM + XGBoost + CatBoost)")
    print("=" * 60)

    X, y, class_names, le, feature_names = load_data()
    n_classes = len(class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Generate OOF predictions
    print("\nGenerating OOF predictions...")
    oof_train, oof_test, base_scores = generate_oof_predictions(
        X_train_s, y_train, X_test_s, n_classes
    )

    # Meta-learner
    print("\nTraining meta-learner (Logistic Regression)...")
    meta, y_pred, y_proba = train_meta_learner(oof_train, y_train, oof_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    try:
        auroc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    print(f"\nStacking Ensemble Results:")
    print(f"  Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | AUROC: {auroc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    short_names = [c.replace("BRCA_", "") for c in class_names]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stacking Ensemble — Confusion Matrix")
    plt.tight_layout()
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)
    fig.savefig(
        os.path.join(config.RESULTS_FIGURES, "stacking_confusion_matrix.png"),
        dpi=config.FIGURE_DPI, bbox_inches="tight"
    )
    plt.close(fig)

    # Save results
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    summary = {
        "stacking_accuracy": float(acc),
        "stacking_macro_f1": float(f1),
        "stacking_auroc": float(auroc),
        "base_model_scores": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in base_scores.items()},
    }
    with open(os.path.join(config.RESULTS_TABLES, "stacking_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).T.to_csv(
        os.path.join(config.RESULTS_TABLES, "stacking_classification_report.csv")
    )

    # Save model for SHAP
    import pickle
    os.makedirs(config.RESULTS_MODELS, exist_ok=True)
    with open(os.path.join(config.RESULTS_MODELS, "stacking_ensemble.pkl"), "wb") as f:
        pickle.dump({
            "meta": meta, "scaler": scaler, "label_encoder": le,
            "feature_names": feature_names, "base_scores": base_scores,
        }, f)

    print("\nStage 3-3 complete.")


if __name__ == "__main__":
    main()
