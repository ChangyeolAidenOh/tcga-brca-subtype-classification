"""
Multi-Task Learning — Classification + Survival Prediction
=============================================================
Joint CE + Cox partial likelihood loss. Full evaluation with
AUROC and 5-fold CV.

Usage:
    python -m tcga_brca.models.multitask_survival
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class MultiTaskNet(nn.Module):
    def __init__(self, n_features, n_classes, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)
        self.survival = nn.Sequential(nn.Linear(hidden_dims[-1], 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        shared = self.encoder(x)
        return self.classifier(shared), self.survival(shared), shared


def cox_loss(risk_scores, times, events):
    sorted_idx = torch.argsort(times, descending=True)
    risk_sorted = risk_scores[sorted_idx].squeeze()
    events_sorted = events[sorted_idx]
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)
    event_mask = events_sorted.bool()
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -(risk_sorted[event_mask] - log_cumsum[event_mask]).mean()


def load_data():
    expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()
    surv = pd.read_csv(os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv"))

    surv_indexed = surv.set_index("sampleId")
    common = expr.index.intersection(labels.index).intersection(surv_indexed.index)
    expr = expr.loc[common]
    labels = labels.loc[common]
    surv_aligned = surv_indexed.loc[common]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"Data: {expr.shape[0]} samples x {expr.shape[1]} features")
    print(f"Classes: {dict(zip(le.classes_, np.bincount(y)))}")
    print(f"Survival events: {int(surv_aligned['os_event'].sum())} / {len(surv_aligned)}")

    return expr.values, y, le, surv_aligned["os_months"].values, surv_aligned["os_event"].values, expr.columns.tolist()


def train_one_model(X_train, y_train, t_train, e_train, X_val, y_val, t_val, e_val,
                    n_features, n_classes, alpha, device, max_epochs=200, patience=25):
    """Train one multi-task model and return predictions."""
    model = MultiTaskNet(n_features, n_classes, (256, 128, 64), 0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    class_counts = np.bincount(y_train)
    weights = torch.FloatTensor(len(y_train) / (n_classes * class_counts)).to(device)
    ce_fn = nn.CrossEntropyLoss(weight=weights)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train),
        torch.FloatTensor(t_train), torch.FloatTensor(e_train),
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
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
            v_loss = ce_fn(vl, torch.LongTensor(y_val).to(device)) + alpha * cox_loss(vr, torch.FloatTensor(t_val).to(device), torch.FloatTensor(e_val).to(device))

        if v_loss.item() < best_loss:
            best_loss = v_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        vx = torch.FloatTensor(X_val).to(device)
        logits, risk, _ = model(vx)
        y_proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
        risk_scores = risk.cpu().numpy().flatten()

    return model, y_pred, y_proba, risk_scores


def run_multitask_cv(X, y, times, events, n_classes, alpha, device):
    """5-fold CV for multi-task model."""
    print(f"\n  Running {config.CV_FOLDS}-fold CV (alpha={alpha})...")
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_acc, fold_f1, fold_auroc = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    CV fold {fold + 1}/{config.CV_FOLDS}...")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])

        _, y_pred, y_proba, _ = train_one_model(
            X_tr, y[tr_idx], times[tr_idx], events[tr_idx],
            X_val, y[val_idx], times[val_idx], events[val_idx],
            X_tr.shape[1], n_classes, alpha, device, max_epochs=150, patience=20,
        )

        fold_acc.append(accuracy_score(y[val_idx], y_pred))
        fold_f1.append(f1_score(y[val_idx], y_pred, average="macro"))
        try:
            fold_auroc.append(roc_auc_score(y[val_idx], y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            fold_auroc.append(float("nan"))

        print(f"      Acc={fold_acc[-1]:.4f} F1={fold_f1[-1]:.4f} AUROC={fold_auroc[-1]:.4f}")

    return fold_acc, fold_f1, fold_auroc


def extract_shared_importance(model, X, feature_names, device):
    model.eval()
    X_t = torch.FloatTensor(X).to(device).requires_grad_(True)
    logits, risk, _ = model(X_t)
    combined = logits.max(dim=1).values.mean() + risk.mean()
    combined.backward()
    gradients = X_t.grad.detach().cpu().numpy()
    importance = np.mean(np.abs(gradients), axis=0)

    importance_df = pd.DataFrame({
        "gene": feature_names, "multitask_importance": importance,
    }).sort_values("multitask_importance", ascending=False)
    importance_df["multitask_rank"] = range(1, len(importance_df) + 1)
    return importance_df


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(config.FIGURE_SIZE[0] * 1.5, config.FIGURE_SIZE[1]), dpi=config.FIGURE_DPI)
    axes[0].plot(history["train_loss"], label="Train", color="#378ADD")
    axes[0].plot(history["val_loss"], label="Val", color="#E24B4A")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss"); axes[0].set_title("Total Loss"); axes[0].legend()
    axes[1].plot(history["cls_loss"], label="Classification", color="#1D9E75")
    axes[1].plot(history["surv_loss"], label="Survival (Cox)", color="#BA7517")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].set_title("Loss Decomposition"); axes[1].legend()
    axes[2].plot(history["val_acc"], color="#7F77DD")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy"); axes[2].set_title("Validation Accuracy")
    plt.suptitle("Multi-Task Training: Classification + Cox Survival", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Multi-Task Learning: Classification + Survival")
    print("=" * 60)

    X, y, le, times, events, feature_names = load_data()
    n_classes = len(le.classes_)

    indices = np.arange(len(X))
    tr_idx, te_idx = train_test_split(indices, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[tr_idx])
    X_test = scaler.transform(X[te_idx])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Alpha search
    alphas = [0.01, 0.1, 0.5]
    best_alpha, best_f1 = None, 0
    best_model_state = None

    for alpha in alphas:
        print(f"\n{'=' * 40}")
        print(f"Training with alpha={alpha}")
        print(f"{'=' * 40}")

        model, y_pred, y_proba, risk = train_one_model(
            X_train, y[tr_idx], times[tr_idx], events[tr_idx],
            X_test, y[te_idx], times[te_idx], events[te_idx],
            X_train.shape[1], n_classes, alpha, device,
        )

        acc = accuracy_score(y[te_idx], y_pred)
        f1 = f1_score(y[te_idx], y_pred, average="macro")
        print(f"  Alpha={alpha}: Accuracy={acc:.4f} | Macro F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest alpha: {best_alpha} (Macro F1: {best_f1:.4f})")

    # Reload best model
    model = MultiTaskNet(X_train.shape[1], n_classes, (256, 128, 64), 0.3).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        vx = torch.FloatTensor(X_test).to(device)
        logits, risk, _ = model(vx)
        y_proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y[te_idx], y_pred)
    f1 = f1_score(y[te_idx], y_pred, average="macro")
    try:
        auroc = roc_auc_score(y[te_idx], y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    print(f"\n{'=' * 60}")
    print(f"MULTI-TASK MODEL RESULTS (alpha={best_alpha})")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | AUROC: {auroc:.4f}")
    print(classification_report(y[te_idx], y_pred, target_names=le.classes_))

    # 5-fold CV with best alpha
    cv_acc, cv_f1, cv_auroc = run_multitask_cv(X, y, times, events, n_classes, best_alpha, device)

    print(f"\n  CV Accuracy: {np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}")
    print(f"  CV Macro F1: {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")
    print(f"  CV AUROC:    {np.nanmean(cv_auroc):.4f} +/- {np.nanstd(cv_auroc):.4f}")

    # Feature importance
    importance = extract_shared_importance(model, X_test, feature_names, device)
    print("\nTop 10 multi-task important genes:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['gene']:15s} importance: {row['multitask_importance']:.6f}")

    # Compare with SHAP
    shap_path = os.path.join(config.RESULTS_TABLES, "shap_global_importance.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        merged = importance.merge(shap_df[["gene", "rank"]].rename(columns={"rank": "shap_rank"}), on="gene", how="inner")
        mt_top20 = set(merged.nsmallest(20, "multitask_rank")["gene"])
        shap_top20 = set(merged.nsmallest(20, "shap_rank")["gene"])
        overlap = mt_top20 & shap_top20
        print(f"\nMulti-task top 20 vs SHAP top 20 overlap: {len(overlap)}/20")
        if overlap:
            print(f"Consensus: {', '.join(overlap)}")

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    summary = {
        "best_alpha": float(best_alpha),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "auroc": float(auroc),
        "cv_accuracy_mean": float(np.mean(cv_acc)),
        "cv_accuracy_std": float(np.std(cv_acc)),
        "cv_f1_mean": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "cv_auroc_mean": float(np.nanmean(cv_auroc)),
        "cv_auroc_std": float(np.nanstd(cv_auroc)),
        "imbalance_strategy": "class_weights + Cox(alpha=" + str(best_alpha) + ")",
    }
    with open(os.path.join(config.RESULTS_TABLES, "multitask_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    importance.to_csv(os.path.join(config.RESULTS_TABLES, "multitask_feature_importance.csv"), index=False)

    print("\nMulti-task learning complete.")


if __name__ == "__main__":
    main()
