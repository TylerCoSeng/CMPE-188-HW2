import sys
import os
import json
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metadata

def get_task_metadata():
    return {
        "task_id": "mlp_lvl2_autograd_modules",
        "algorithm": "MLP classifier, nn.Module + autograd",
        "dataset": "MNIST (torchvision) or synthetic fallback",
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
    }


# Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Device

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset / dataloaders

def _make_synthetic_dataloaders(batch_size: int, device):
    print("  [dataset] Using synthetic fallback dataset (10-class, 128-dim).")
    N, D, C = 2000, 128, 10
    torch.manual_seed(0)
    centroids = torch.randn(C, D)
    X_list, y_list = [], []
    per_class = N // C
    for c in range(C):
        X_list.append(centroids[c] + 0.5 * torch.randn(per_class, D))
        y_list.append(torch.full((per_class,), c, dtype=torch.long))
    X = torch.cat(X_list)
    y = torch.cat(y_list)

    mu, std = X.mean(0), X.std(0).clamp(min=1e-6)
    X = (X - mu) / std

    dataset = TensorDataset(X, y)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, D, C


def make_dataloaders(batch_size: int = 256, data_root: str = "./data"):
    try:
        import torchvision
        import torchvision.transforms as transforms
        print("  [dataset] Loading MNIST via torchvision …")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_full = torchvision.datasets.MNIST(
            root=data_root, train=True,  download=True, transform=transform)
        val_ds     = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform)

        n_val   = 10_000
        n_train = len(train_full) - n_val
        train_ds, _ = random_split(train_full, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)
        input_dim, num_classes = 28 * 28, 10
        print(f"  [dataset] MNIST ready. train={len(train_ds)}  val={len(val_ds)}")
        return train_loader, val_loader, input_dim, num_classes

    except Exception as exc:
        print(f"  [dataset] torchvision/MNIST unavailable ({exc}).")
        device = get_device()
        return _make_synthetic_dataloaders(batch_size, device)


# Model

class MLP(nn.Module):

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list,
        num_classes: int,
        dropout_p:   float = 0.3,
        use_bn:      bool  = True,
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


def build_model(input_dim: int, num_classes: int, device,
                hidden_dims=None, dropout_p: float = 0.3, use_bn: bool = True):
    if hidden_dims is None:
        hidden_dims = [256, 128]
    model = MLP(input_dim, hidden_dims, num_classes,
                dropout_p=dropout_p, use_bn=use_bn)
    return model.to(device)


# Metrics helpers

def macro_f1(all_preds: torch.Tensor, all_labels: torch.Tensor,
             num_classes: int) -> float:

    f1s = []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().float()
        fp = ((all_preds == c) & (all_labels != c)).sum().float()
        fn = ((all_preds != c) & (all_labels == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1.item())
    return sum(f1s) / len(f1s)


# Train

def train(model, train_loader, val_loader, device,
          epochs: int = 10, lr: float = 1e-3):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y_batch.size(0)
            preds    = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        val_metrics = evaluate(model, val_loader, device, verbose=False)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  epoch {epoch:3d}/{epochs}"
              f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
              f"  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}")

    return history


# Evaluate

def evaluate(model, loader, device, verbose: bool = True,
             split_name: str = ""):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            running_loss += loss.item() * y_batch.size(0)
            preds    = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    num_classes = int(all_labels.max().item()) + 1

    metrics = {
        "loss":     running_loss / total,
        "accuracy": correct / total,
        "macro_f1": macro_f1(all_preds, all_labels, num_classes),
    }

    if verbose:
        tag = f"[{split_name}] " if split_name else ""
        print(f"  {tag}loss={metrics['loss']:.4f}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}")

    return metrics


# Predict

def predict(model, X: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
    return logits.argmax(dim=1).cpu()


# Save artifacts

def save_artifacts(model, metrics: dict, history: dict,
                   task_id: str = "mlp_lvl2_autograd_modules"):
    out_dir = os.path.join("output", task_id)
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"  Saved weights  -> {weights_path}")

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics  -> {json_path}")

    plot_path = os.path.join(out_dir, "training_curves.png")
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"  Saved plot     -> {plot_path}")


# Main

if __name__ == "__main__":
    print("=== Task:", get_task_metadata()["task_id"], "===")

    EPOCHS        = 10
    BATCH_SIZE    = 256
    LR            = 1e-3
    HIDDEN_DIMS   = [256, 128]
    DROPOUT_P     = 0.3
    USE_BN        = True
    ACC_THRESHOLD = 0.92
    F1_THRESHOLD  = 0.90

    set_seed(42)
    device = get_device()
    print(f"  Device: {device}")

    train_loader, val_loader, input_dim, num_classes = make_dataloaders(
        batch_size=BATCH_SIZE)

    model = build_model(input_dim, num_classes, device,
                        hidden_dims=HIDDEN_DIMS,
                        dropout_p=DROPOUT_P,
                        use_bn=USE_BN)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    print("\n--- Training ---")
    history = train(model, train_loader, val_loader, device,
                    epochs=EPOCHS, lr=LR)

    print("\n--- Final Evaluation ---")
    train_metrics = evaluate(model, train_loader, device,
                             verbose=True, split_name="train")
    val_metrics   = evaluate(model, val_loader,   device,
                             verbose=True, split_name="val")

    print("\n--- Saving Artifacts ---")
    all_metrics = {"train": train_metrics, "val": val_metrics,
                   "config": {"epochs": EPOCHS, "lr": LR,
                               "hidden_dims": HIDDEN_DIMS,
                               "dropout_p": DROPOUT_P, "use_bn": USE_BN}}
    save_artifacts(model, all_metrics, history)

    assert val_metrics["accuracy"] >= ACC_THRESHOLD, (
        f"Val accuracy {val_metrics['accuracy']:.4f} < threshold {ACC_THRESHOLD}")
    assert val_metrics["macro_f1"] >= F1_THRESHOLD, (
        f"Val macro-F1 {val_metrics['macro_f1']:.4f} < threshold {F1_THRESHOLD}")

    print(f"\n✓ All assertions passed. "
          f"val_acc={val_metrics['accuracy']:.4f}  "
          f"val_f1={val_metrics['macro_f1']:.4f}")
    sys.exit(0)