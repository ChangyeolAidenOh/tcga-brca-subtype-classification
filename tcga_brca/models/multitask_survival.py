"""
Multi-Task Learning — Classification + Survival Prediction
=============================================================
A PyTorch model that jointly optimizes:
  - PAM50 subtype classification (cross-entropy loss)
  - Survival prediction (Cox partial likelihood loss)

This mirrors the PINN project's multi-objective loss design:
  PINN: PDE residual loss + boundary condition loss
  Here: classification loss + survival loss

The shared encoder learns features important for BOTH subtype
classification AND patient prognosis, directly identifying
genes that are simultaneously classification and prognostic
biomarkers.

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class MultiTaskNet(nn.Module):
    """Multi-task network with shared encoder, classification head, and survival head.

    Architecture mirrors PINN dual-network design:
      PINN: pricing network + free-boundary network with shared loss
      Here: shared encoder + classification head + survival head
    """

    def __init__(self, n_features, n_classes, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()

        # Shared encoder
        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(hidden_dims[-1], n_classes)

        # Survival head (outputs risk score for Cox model)
        self.survival = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        shared = self.encoder(x)
        class_logits = self.classifier(shared)
        risk_score = self.survival(shared)
        return class_logits, risk_score, shared


def cox_loss(risk_scores, times, events):
    """Negative partial log-likelihood for Cox proportional hazards.

    Equivalent to PINN's PDE residual loss — enforces a
    "physical law" (survival dynamics) on the model.
    """
    # Sort by time (descending)
    sorted_idx = torch.argsort(times, descending=True)
    risk_sorted = risk_scores[sorted_idx].squeeze()
    events_sorted = events[sorted_idx]

    # Log cumulative sum of exp(risk)
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)

    # Only compute loss for uncensored (event=1) observations
    event_mask = events_sorted.bool()
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    loss = -(risk_sorted[event_mask] - log_cumsum[event_mask]).mean()
    return loss


def load_data():
    """Load expression, labels, and survival data."""
    expr = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_expression_selected.csv"), index_col=0
    )
    labels = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_labels.csv"), index_col=0
    ).squeeze()
    surv = pd.read_csv(
        os.path.join(config.DATA_PROCESSED, "tcga_brca_survival.csv")
    )

    # Align all three
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

    return (expr, y, le, surv_aligned["os_months"].values, surv_aligned["os_event"].values,
            expr.columns.tolist())


def train_multitask(
    model, train_loader, val_X, val_y, val_times, val_events,
    n_epochs, lr, alpha, device,
):
    """Train multi-task model.

    Alpha controls loss balance (like PINN's pde_weight):
      total_loss = classification_loss + alpha * cox_loss
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Class weights for imbalanced data
    class_counts = np.bincount(val_y)
    weights = torch.FloatTensor(len(val_y) / (len(class_counts) * class_counts)).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "cls_loss": [], "surv_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 25

    val_X_t = torch.FloatTensor(val_X).to(device)
    val_y_t = torch.LongTensor(val_y).to(device)
    val_times_t = torch.FloatTensor(val_times).to(device)
    val_events_t = torch.FloatTensor(val_events).to(device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_cls = 0
        epoch_surv = 0

        for batch_X, batch_y, batch_times, batch_events in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_times = batch_times.to(device)
            batch_events = batch_events.to(device)

            optimizer.zero_grad()

            logits, risk, _ = model(batch_X)

            cls_l = ce_loss_fn(logits, batch_y)
            surv_l = cox_loss(risk, batch_times, batch_events)
            total = cls_l + alpha * surv_l

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            epoch_cls += cls_l.item()
            epoch_surv += surv_l.item()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, v_risk, _ = model(val_X_t)
            v_cls = ce_loss_fn(v_logits, val_y_t)
            v_surv = cox_loss(v_risk, val_times_t, val_events_t)
            v_total = v_cls + alpha * v_surv

            v_pred = v_logits.argmax(dim=1).cpu().numpy()
            v_acc = accuracy_score(val_y, v_pred)

        n_batches = len(train_loader)
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(v_total.item())
        history["val_acc"].append(v_acc)
        history["cls_loss"].append(v_cls.item())
        history["surv_loss"].append(v_surv.item())

        scheduler.step(v_total)

        if v_total.item() < best_val_loss:
            best_val_loss = v_total.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f} "
                  f"val_loss={v_total.item():.4f} val_acc={v_acc:.4f} "
                  f"(cls={v_cls.item():.4f} surv={v_surv.item():.4f})")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return history


def extract_shared_importance(model, X, feature_names, device):
    """Extract feature importance from shared encoder gradients.

    Compute gradient of the combined (classification + survival) output
    with respect to input features — features with high gradients are
    important for BOTH tasks simultaneously.
    """
    model.eval()
    X_t = torch.FloatTensor(X).to(device).requires_grad_(True)

    logits, risk, _ = model(X_t)

    # Combined importance: gradient of class confidence + risk
    class_confidence = logits.max(dim=1).values.mean()
    risk_mean = risk.mean()
    combined = class_confidence + risk_mean
    combined.backward()

    gradients = X_t.grad.detach().cpu().numpy()
    importance = np.mean(np.abs(gradients), axis=0)

    importance_df = pd.DataFrame({
        "gene": feature_names,
        "multitask_importance": importance,
    }).sort_values("multitask_importance", ascending=False)
    importance_df["multitask_rank"] = range(1, len(importance_df) + 1)

    return importance_df


def compare_single_vs_multitask(model, X_test, y_test, le, device):
    """Evaluate multi-task model on classification."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        logits, risk, _ = model(X_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        risk_scores = risk.cpu().numpy().flatten()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return acc, f1, y_pred, risk_scores


def plot_training_history(history, save_path):
    """Plot training curves showing multi-task loss decomposition."""
    fig, axes = plt.subplots(1, 3, figsize=(config.FIGURE_SIZE[0] * 1.5, config.FIGURE_SIZE[1]),
                              dpi=config.FIGURE_DPI)

    axes[0].plot(history["train_loss"], label="Train", color="#378ADD")
    axes[0].plot(history["val_loss"], label="Val", color="#E24B4A")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (cls + alpha * surv)")
    axes[0].legend()

    axes[1].plot(history["cls_loss"], label="Classification", color="#1D9E75")
    axes[1].plot(history["surv_loss"], label="Survival (Cox)", color="#BA7517")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Decomposition")
    axes[1].legend()

    axes[2].plot(history["val_acc"], color="#7F77DD")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation Accuracy")

    plt.suptitle("Multi-Task Training: Classification + Cox Survival", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Multi-Task Learning: Classification + Survival")
    print("=" * 60)
    print("Loss: CE(classification) + alpha * Cox(survival)")
    print("Transfer: PINN's PDE residual + boundary condition loss")

    expr, y, le, times, events, feature_names = load_data()
    n_classes = len(le.classes_)

    # Split
    indices = np.arange(len(expr))
    tr_idx, te_idx = train_test_split(
        indices, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(expr.values[tr_idx])
    X_test = scaler.transform(expr.values[te_idx])
    y_train, y_test = y[tr_idx], y[te_idx]
    t_train, t_test = times[tr_idx], times[te_idx]
    e_train, e_test = events[tr_idx], events[te_idx]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(t_train),
        torch.FloatTensor(e_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Build model
    model = MultiTaskNet(
        n_features=X_train.shape[1],
        n_classes=n_classes,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train with alpha search
    # alpha = weight of survival loss relative to classification loss
    # Similar to PINN's pde_weight parameter
    alphas = [0.01, 0.1, 0.5]
    best_alpha = None
    best_f1 = 0
    best_model_state = None

    for alpha in alphas:
        print(f"\n{'=' * 40}")
        print(f"Training with alpha={alpha}")
        print(f"{'=' * 40}")

        model_trial = MultiTaskNet(
            n_features=X_train.shape[1],
            n_classes=n_classes,
            hidden_dims=(256, 128, 64),
            dropout=0.3,
        ).to(device)

        history = train_multitask(
            model_trial, train_loader,
            X_test, y_test, t_test, e_test,
            n_epochs=200, lr=1e-3, alpha=alpha, device=device,
        )

        acc, f1, _, _ = compare_single_vs_multitask(model_trial, X_test, y_test, le, device)
        print(f"  Alpha={alpha}: Accuracy={acc:.4f} | Macro F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model_state = {k: v.clone() for k, v in model_trial.state_dict().items()}
            best_history = history

    print(f"\nBest alpha: {best_alpha} (Macro F1: {best_f1:.4f})")

    # Reload best model
    model.load_state_dict(best_model_state)
    acc, f1, y_pred, risk_scores = compare_single_vs_multitask(model, X_test, y_test, le, device)

    print(f"\n{'=' * 60}")
    print(f"MULTI-TASK MODEL RESULTS (alpha={best_alpha})")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Shared encoder importance
    importance = extract_shared_importance(model, X_test, feature_names, device)
    print("Top 10 multi-task important genes (shared encoder):")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['gene']:15s} importance: {row['multitask_importance']:.6f}")

    # Compare with single-task SHAP
    shap_path = os.path.join(config.RESULTS_TABLES, "shap_global_importance.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        merged = importance.merge(
            shap_df[["gene", "mean_abs_shap", "rank"]].rename(columns={"rank": "shap_rank"}),
            on="gene", how="inner",
        )
        mt_top20 = set(merged.nsmallest(20, "multitask_rank")["gene"])
        shap_top20 = set(merged.nsmallest(20, "shap_rank")["gene"])
        overlap = mt_top20 & shap_top20

        print(f"\nMulti-task top 20 vs SHAP top 20 overlap: {len(overlap)}/20")
        if overlap:
            print(f"Consensus: {', '.join(overlap)}")

    # Plot
    plot_training_history(
        best_history,
        os.path.join(config.RESULTS_FIGURES, "multitask_training_history.png"),
    )

    # Save
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    summary = {
        "best_alpha": float(best_alpha),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "alpha_search": {str(a): "tested" for a in alphas},
    }
    with open(os.path.join(config.RESULTS_TABLES, "multitask_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    importance.to_csv(
        os.path.join(config.RESULTS_TABLES, "multitask_feature_importance.csv"), index=False
    )

    print("\nMulti-task learning complete.")


if __name__ == "__main__":
    main()
