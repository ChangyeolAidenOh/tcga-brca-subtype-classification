"""
Comprehensive Model Evaluation — All Metrics for All Models
==============================================================
Computes AUROC, CV Accuracy, CV Macro F1 for every model.
Marks N/A for metrics that don't apply (e.g., CV for external validation).

Usage:
    python -m tcga_brca.models.evaluate_comprehensive
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_data():
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
    return expr.values, y, le.classes_, le


def compute_auroc(y_true, y_proba, n_classes):
    try:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def run_xgboost_cv(X, y, n_classes, class_names):
    """Full CV for XGBoost with all metrics."""
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_acc, fold_f1, fold_auroc = [], [], []

    for tr_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        sw = compute_sample_weight("balanced", y[tr_idx])

        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
        )
        model.fit(X_tr, y[tr_idx], sample_weight=sw)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        fold_acc.append(accuracy_score(y[val_idx], y_pred))
        fold_f1.append(f1_score(y[val_idx], y_pred, average="macro"))
        fold_auroc.append(compute_auroc(y[val_idx], y_proba, n_classes))

    return fold_acc, fold_f1, fold_auroc


def run_stacking_cv(X, y, n_classes):
    """Full CV for stacking ensemble."""
    skf_outer = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_acc, fold_f1, fold_auroc = [], [], []

    for outer_tr, outer_val in skf_outer.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[outer_tr])
        X_val = scaler.transform(X[outer_val])

        # OOF within training set
        skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
        models = {
            "lgb": LGBMClassifier(n_estimators=200, max_depth=6, is_unbalance=True, random_state=config.RANDOM_STATE, verbose=-1),
            "xgb": XGBClassifier(n_estimators=200, max_depth=6, random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric="mlogloss", verbosity=0),
            "cb": CatBoostClassifier(iterations=200, depth=6, auto_class_weights="Balanced", random_state=config.RANDOM_STATE, verbose=0),
        }

        oof_train = np.zeros((len(outer_tr), n_classes * 3))
        oof_val = np.zeros((len(outer_val), n_classes * 3))

        for m_idx, (name, model) in enumerate(models.items()):
            col_s, col_e = m_idx * n_classes, (m_idx + 1) * n_classes
            test_preds = np.zeros((3, len(outer_val), n_classes))

            for fold, (itr, ival) in enumerate(skf_inner.split(X_tr, y[outer_tr])):
                if name == "xgb":
                    sw = compute_sample_weight("balanced", y[outer_tr][itr])
                    model.fit(X_tr[itr], y[outer_tr][itr], sample_weight=sw)
                else:
                    model.fit(X_tr[itr], y[outer_tr][itr])
                oof_train[ival, col_s:col_e] = model.predict_proba(X_tr[ival])
                test_preds[fold] = model.predict_proba(X_val)

            oof_val[:, col_s:col_e] = test_preds.mean(axis=0)

        meta = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, multi_class="multinomial")
        meta.fit(oof_train, y[outer_tr])
        y_pred = meta.predict(oof_val)
        y_proba = meta.predict_proba(oof_val)

        fold_acc.append(accuracy_score(y[outer_val], y_pred))
        fold_f1.append(f1_score(y[outer_val], y_pred, average="macro"))
        fold_auroc.append(compute_auroc(y[outer_val], y_proba, n_classes))

    return fold_acc, fold_f1, fold_auroc


def run_tabnet_cv(X, y, n_classes):
    """Full CV for TabNet with pretraining."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.pretraining import TabNetPretrainer
    import torch

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_acc, fold_f1, fold_auroc = [], [], []

    class_counts = np.bincount(y)
    total = len(y)
    weights_arr = total / (n_classes * class_counts)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    TabNet CV fold {fold + 1}/{config.CV_FOLDS}...")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        sw = np.array([weights_arr[c] for c in y[tr_idx]])

        pretrainer = TabNetPretrainer(
            n_d=64, n_a=64, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax", verbose=0,
        )
        pretrainer.fit(X_train=X_tr, eval_set=[X_val], max_epochs=100,
                       patience=20, batch_size=128, virtual_batch_size=32,
                       pretraining_ratio=0.5)

        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=5e-3, weight_decay=1e-5),
            scheduler_params={"step_size": 15, "gamma": 0.85},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=config.RANDOM_STATE, verbose=0,
        )
        model.fit(X_tr, y[tr_idx], eval_set=[(X_val, y[val_idx])],
                  eval_metric=["accuracy"], max_epochs=200, patience=30,
                  batch_size=256, virtual_batch_size=64,
                  weights=sw, drop_last=False, from_unsupervised=pretrainer)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        fold_acc.append(accuracy_score(y[val_idx], y_pred))
        fold_f1.append(f1_score(y[val_idx], y_pred, average="macro"))
        fold_auroc.append(compute_auroc(y[val_idx], y_proba, n_classes))

    return fold_acc, fold_f1, fold_auroc


def run_multitask_cv(X, y, n_classes):
    """Full CV for multi-task model."""
    import torch
    import torch.nn as nn

    surv = pd.read_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv"))
    expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()
    surv_idx = surv.set_index("sampleId")
    common = expr.index.intersection(labels.index).intersection(surv_idx.index)

    X_full = expr.loc[common].values
    le_local = LabelEncoder()
    y_full = le_local.fit_transform(labels.loc[common])
    times = surv_idx.loc[common, "os_months"].values
    events = surv_idx.loc[common, "os_event"].values

    from tcga_brca.models.multitask_survival import MultiTaskNet, cox_loss

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_acc, fold_f1, fold_auroc = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"    Multi-task CV fold {fold + 1}/{config.CV_FOLDS}...")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_full[tr_idx])
        X_val = scaler.transform(X_full[val_idx])

        model = MultiTaskNet(X_tr.shape[1], n_classes, (256, 128, 64), 0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        class_counts = np.bincount(y_full[tr_idx])
        w = torch.FloatTensor(len(y_full[tr_idx]) / (n_classes * class_counts)).to(device)
        ce_fn = nn.CrossEntropyLoss(weight=w)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_tr), torch.LongTensor(y_full[tr_idx]),
            torch.FloatTensor(times[tr_idx]), torch.FloatTensor(events[tr_idx]),
        )
        loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

        alpha = 0.1
        best_loss = float("inf")
        patience_count = 0

        for epoch in range(150):
            model.train()
            for bx, by, bt, be in loader:
                bx, by, bt, be = bx.to(device), by.to(device), bt.to(device), be.to(device)
                optimizer.zero_grad()
                logits, risk, _ = model(bx)
                loss = ce_fn(logits, by) + alpha * cox_loss(risk, bt, be)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vx = torch.FloatTensor(X_val).to(device)
                vl, vr, _ = model(vx)
                v_loss = ce_fn(vl, torch.LongTensor(y_full[val_idx]).to(device)).item()

            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= 20:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vx = torch.FloatTensor(X_val).to(device)
            logits, _, _ = model(vx)
            y_proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()

        fold_acc.append(accuracy_score(y_full[val_idx], y_pred))
        fold_f1.append(f1_score(y_full[val_idx], y_pred, average="macro"))
        fold_auroc.append(compute_auroc(y_full[val_idx], y_proba, n_classes))

    return fold_acc, fold_f1, fold_auroc


def format_cv(vals):
    return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"


def main():
    print("=" * 60)
    print("Comprehensive Model Evaluation — All Metrics")
    print("=" * 60)

    X, y, class_names, le = load_data()
    n_classes = len(class_names)
    print(f"Data: {X.shape[0]} samples x {X.shape[1]} features, {n_classes} classes\n")

    # Fixed train/test split (same as individual scripts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # ============================================================
    # 1. XGBoost Baseline
    # ============================================================
    print("1. XGBoost Baseline...")
    sw = compute_sample_weight("balanced", y_train)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
    )
    xgb.fit(X_train_s, y_train, sample_weight=sw)
    xgb_pred = xgb.predict(X_test_s)
    xgb_proba = xgb.predict_proba(X_test_s)

    cv_acc, cv_f1, cv_auroc = run_xgboost_cv(X, y, n_classes, class_names)

    results["XGBoost"] = {
        "accuracy": accuracy_score(y_test, xgb_pred),
        "macro_f1": f1_score(y_test, xgb_pred, average="macro"),
        "auroc": compute_auroc(y_test, xgb_proba, n_classes),
        "cv_accuracy": format_cv(cv_acc),
        "cv_f1": format_cv(cv_f1),
        "cv_auroc": format_cv(cv_auroc),
        "imbalance": "class_weights",
    }
    print(f"   Acc={results['XGBoost']['accuracy']:.4f} F1={results['XGBoost']['macro_f1']:.4f} AUROC={results['XGBoost']['auroc']:.4f}")

    # ============================================================
    # 2. Stacking Ensemble
    # ============================================================
    print("2. Stacking Ensemble (CV takes a few minutes)...")
    cv_acc, cv_f1, cv_auroc = run_stacking_cv(X, y, n_classes)

    # Test set performance
    stacking_path = os.path.join(config.RESULTS_TABLES, "stacking_summary.json")
    if os.path.exists(stacking_path):
        with open(stacking_path) as f:
            st = json.load(f)
        results["Stacking"] = {
            "accuracy": st["stacking_accuracy"],
            "macro_f1": st["stacking_macro_f1"],
            "auroc": st.get("stacking_auroc", float("nan")),
            "cv_accuracy": format_cv(cv_acc),
            "cv_f1": format_cv(cv_f1),
            "cv_auroc": format_cv(cv_auroc),
            "imbalance": "per-model (balanced)",
        }
    print(f"   CV F1={format_cv(cv_f1)}")

    # ============================================================
    # 3. TabNet
    # ============================================================
    print("3. TabNet (CV with pretraining — this will take ~10 min)...")
    tabnet_path = os.path.join(config.RESULTS_TABLES, "tabnet_summary.json")
    if os.path.exists(tabnet_path):
        with open(tabnet_path) as f:
            tn = json.load(f)

        cv_acc, cv_f1, cv_auroc = run_tabnet_cv(X, y, n_classes)

        results["TabNet (PT)"] = {
            "accuracy": tn["tabnet_accuracy"],
            "macro_f1": tn["tabnet_macro_f1"],
            "auroc": float("nan"),  # Will compute from saved
            "cv_accuracy": format_cv(cv_acc),
            "cv_f1": format_cv(cv_f1),
            "cv_auroc": format_cv(cv_auroc),
            "imbalance": "class_weights",
        }
        print(f"   CV F1={format_cv(cv_f1)}")

    # ============================================================
    # 4. Hierarchical
    # ============================================================
    print("4. Hierarchical Classifier...")
    hier_path = os.path.join(config.RESULTS_TABLES, "hierarchical_results.json")
    if os.path.exists(hier_path):
        with open(hier_path) as f:
            h = json.load(f)
        results["Hierarchical"] = {
            "accuracy": h["combined"]["accuracy"],
            "macro_f1": h["combined"]["macro_f1"],
            "auroc": "N/A (multi-level)",
            "cv_accuracy": "N/A (multi-level)",
            "cv_f1": "N/A (multi-level)",
            "cv_auroc": "N/A (multi-level)",
            "imbalance": "class_weights (per level)",
        }
    print(f"   Acc={h['combined']['accuracy']:.4f}")

    # ============================================================
    # 5. Multi-task
    # ============================================================
    print("5. Multi-task (CV — this will take ~10 min)...")
    mt_path = os.path.join(config.RESULTS_TABLES, "multitask_summary.json")
    if os.path.exists(mt_path):
        with open(mt_path) as f:
            mt = json.load(f)

        cv_acc, cv_f1, cv_auroc = run_multitask_cv(X, y, n_classes)

        results["Multi-task"] = {
            "accuracy": mt["accuracy"],
            "macro_f1": mt["macro_f1"],
            "auroc": format_cv(cv_auroc) if cv_auroc else "N/A",
            "cv_accuracy": format_cv(cv_acc),
            "cv_f1": format_cv(cv_f1),
            "cv_auroc": format_cv(cv_auroc),
            "imbalance": "class_weights + Cox(α=0.1)",
        }
        print(f"   CV F1={format_cv(cv_f1)}")

    # ============================================================
    # 6. METABRIC External
    # ============================================================
    print("6. METABRIC External Validation...")
    da_path = os.path.join(config.RESULTS_TABLES, "domain_adaptation_results.json")
    if os.path.exists(da_path):
        with open(da_path) as f:
            da = json.load(f)
        for strategy, metrics in da.items():
            label = f"METABRIC ({strategy})"
            results[label] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "auroc": "N/A (external)",
                "cv_accuracy": "N/A (external)",
                "cv_f1": "N/A (external)",
                "cv_auroc": "N/A (external)",
                "imbalance": "N/A (external)",
            }

    # ============================================================
    # Print and save
    # ============================================================
    print(f"\n{'=' * 100}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"{'=' * 100}")

    rows = []
    for model_name, m in results.items():
        row = {"Model": model_name}
        for key, display in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                              ("auroc", "AUROC"), ("cv_accuracy", "CV Accuracy"),
                              ("cv_f1", "CV Macro F1"), ("cv_auroc", "CV AUROC"),
                              ("imbalance", "Imbalance")]:
            val = m.get(key, "-")
            if isinstance(val, float):
                if np.isnan(val):
                    row[display] = "-"
                else:
                    row[display] = f"{val:.4f}"
            else:
                row[display] = str(val)
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    save_path = os.path.join(config.RESULTS_TABLES, "model_comparison_comprehensive.csv")
    comp_df.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")

    # Also overwrite the standard comparison
    comp_df.to_csv(os.path.join(config.RESULTS_TABLES, "model_comparison.csv"), index=False)

    # Plot
    plot_models = [k for k in results if "METABRIC" not in k]
    accs = [results[m]["accuracy"] if isinstance(results[m]["accuracy"], float) else 0 for m in plot_models]
    f1s = [results[m]["macro_f1"] if isinstance(results[m]["macro_f1"], float) else 0 for m in plot_models]

    fig, ax = plt.subplots(figsize=(max(8, len(plot_models) * 1.5), config.FIGURE_SIZE[1]), dpi=config.FIGURE_DPI)
    x = np.arange(len(plot_models))
    w = 0.35
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#378ADD", alpha=0.8)
    b2 = ax.bar(x + w/2, f1s, w, label="Macro F1", color="#1D9E75", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Models (516 Consensus DEG Features)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_models, fontsize=8, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in b1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7)
    for bar in b2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_FIGURES, "model_comparison.png"),
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # Cross-platform comparison plot
    ext_models = [k for k in results if "METABRIC" in k]
    if ext_models:
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
        ext_accs = [results[m]["accuracy"] for m in ext_models]
        ext_f1s = [results[m]["macro_f1"] for m in ext_models]
        x = np.arange(len(ext_models))
        ax.bar(x - w/2, ext_accs, w, label="Accuracy", color="#378ADD", alpha=0.8)
        ax.bar(x + w/2, ext_f1s, w, label="Macro F1", color="#1D9E75", alpha=0.8)
        short = [m.replace("METABRIC (", "").replace(")", "") for m in ext_models]
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Cross-Platform Validation (TCGA → METABRIC)")
        ax.legend()
        ax.set_ylim(0.4, 0.85)
        for i, (a, f) in enumerate(zip(ext_accs, ext_f1s)):
            ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(config.RESULTS_FIGURES, "cross_platform_comparison.png"),
                    dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    print("\nComprehensive evaluation complete.")


if __name__ == "__main__":
    main()
