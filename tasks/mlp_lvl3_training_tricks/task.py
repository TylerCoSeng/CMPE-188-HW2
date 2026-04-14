import sys
import os
import json
import random
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metadata

def get_task_metadata():
    return {
        "task_id": "mlp_lvl3_training_tricks",
        "algorithm": "MLP + LR scheduler + AMP + grad clipping + checkpointing",
        "dataset": "MNIST (torchvision) or synthetic fallback",
        "optimizer": "AdamW",
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

def _synthetic_dataloaders(batch_size: int, device):
    print("  [dataset] Using synthetic fallback (10-class, 128-dim, 4000 samples).")
    N, D, C = 4000, 128, 10
    torch.manual_seed(0)
    centroids = torch.randn(C, D)
    Xs, ys = [], []
    for c in range(C):
        Xs.append(centroids[c] + 0.4 * torch.randn(N // C, D))
        ys.append(torch.full((N // C,), c, dtype=torch.long))
    X = torch.cat(Xs)
    y = torch.cat(ys)
    mu, std = X.mean(0), X.std(0).clamp(min=1e-6)
    X = (X - mu) / std

    ds = TensorDataset(X, y)
    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, D, C


def make_dataloaders(batch_size: int = 256, data_root: str = "./data"):
    try:
        import torchvision
        import torchvision.transforms as T
        print("  [dataset] Loading MNIST …")
        tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_full = torchvision.datasets.MNIST(
            root=data_root, train=True,  download=True, transform=tfm)
        val_ds     = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=tfm)
        n_val   = 10_000
        n_train = len(train_full) - n_val
        train_ds, _ = random_split(
            train_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        print(f"  [dataset] MNIST ready  train={len(train_ds)}  val={len(val_ds)}")
        return train_loader, val_loader, 28 * 28, 10
    except Exception as exc:
        print(f"  [dataset] MNIST unavailable ({exc}).")
        return _synthetic_dataloaders(batch_size, get_device())


# Model

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list,
                 num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout_p)]
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
        return self.net(self.flatten(x))


def build_model(input_dim: int, num_classes: int, device,
                hidden_dims=None, dropout_p: float = 0.3):
    if hidden_dims is None:
        hidden_dims = [512, 256, 128]
    return MLP(input_dim, hidden_dims, num_classes, dropout_p).to(device)


# AMP helper

def make_scaler_and_autocast(device):

    if device.type == "cuda":
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("  [AMP] Mixed precision ENABLED (CUDA).")
            return scaler, autocast
        except Exception:
            pass
    from contextlib import contextmanager

    @contextmanager
    def _noop_autocast():
        yield

    class _NoopScaler:
        def scale(self, loss): return loss
        def step(self, opt):   opt.step()
        def update(self):      pass
        def unscale_(self, o): pass

    print("  [AMP] Mixed precision DISABLED (CPU or AMP unavailable).")
    return _NoopScaler(), _noop_autocast


# Metrics

def _macro_f1(preds: torch.Tensor, labels: torch.Tensor, C: int) -> float:
    f1s = []
    for c in range(C):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1s.append((2 * p * r / (p + r + 1e-8)).item())
    return sum(f1s) / len(f1s)


# Evaluate

def evaluate(model, loader, device, split_name: str = "", num_classes: int = 10):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    C = max(num_classes, int(all_labels.max().item()) + 1)

    metrics = {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "macro_f1": _macro_f1(all_preds, all_labels, C),
    }
    if split_name:
        print(f"  [{split_name:5s}]  loss={metrics['loss']:.4f}  "
              f"acc={metrics['accuracy']:.4f}  f1={metrics['macro_f1']:.4f}")
    return metrics


# Predict

def predict(model, X: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).argmax(1).cpu()


# Train

def train(model, train_loader, val_loader, device,
          epochs: int = 12, lr: float = 1e-3,
          grad_clip: float = 1.0, checkpoint_dir: str = "output/mlp_lvl3_training_tricks",
          num_classes: int = 10):

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    scaler, autocast = make_scaler_and_autocast(device)

    history = {k: [] for k in
               ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    best_val_acc = -1.0
    best_epoch   = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                logits = model(X)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            preds    = logits.detach().argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_loss = running_loss / total
        train_acc  = correct / total
        val_m      = evaluate(model, val_loader, device, num_classes=num_classes)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["lr"].append(current_lr)

        print(f"  epoch {epoch:3d}/{epochs}"
              f"  lr={current_lr:.2e}"
              f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
              f"  val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch   = epoch
            torch.save({
                "epoch":      epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "val_metrics": val_m,
            }, best_ckpt_path)
            print(f"    ✓ New best checkpoint saved  (val_acc={best_val_acc:.4f})")

    print(f"\n  Best checkpoint: epoch {best_epoch}  val_acc={best_val_acc:.4f}")
    return history, best_ckpt_path


# Checkpoint resume helpers

def load_best_checkpoint(model, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def verify_checkpoint_reproduces(model_factory_fn, checkpoint_path: str,
                                  val_loader, device, reference_metrics: dict,
                                  num_classes: int, tol: float = 1e-3):

    fresh_model = model_factory_fn()
    fresh_model, ckpt = load_best_checkpoint(fresh_model, checkpoint_path, device)
    reproduced = evaluate(fresh_model, val_loader, device,
                          split_name="ckpt_reload", num_classes=num_classes)

    for key in ("accuracy", "macro_f1"):
        stored = ckpt["val_metrics"][key]
        live   = reproduced[key]
        diff   = abs(stored - live)
        assert diff <= tol, (
            f"Checkpoint reproduce mismatch on {key}: "
            f"stored={stored:.6f}  reloaded={live:.6f}  diff={diff:.6e} > tol={tol}")
        print(f"  ✓ {key}: stored={stored:.6f}  reloaded={live:.6f}  diff={diff:.2e}")

    return reproduced


# Save artifacts

def save_artifacts(model, all_metrics: dict, history: dict,
                   task_id: str = "mlp_lvl3_training_tricks"):
    out_dir = os.path.join("output", task_id)
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "final_model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"  Saved weights    -> {weights_path}")

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved metrics    -> {json_path}")

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, history["lr"], color="purple")
    axes[2].set_title("Learning Rate (Cosine)"); axes[2].set_xlabel("Epoch")

    fig.suptitle("mlp_lvl3_training_tricks – Training Curves", fontsize=12)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"  Saved plot       -> {plot_path}")


# Main

if __name__ == "__main__":
    print("=== Task:", get_task_metadata()["task_id"], "===")

    EPOCHS        = 12
    BATCH_SIZE    = 256
    LR            = 1e-3
    GRAD_CLIP     = 1.0
    HIDDEN_DIMS   = [512, 256, 128]
    DROPOUT_P     = 0.3
    ACC_THRESHOLD = 0.92
    F1_THRESHOLD  = 0.90
    CKPT_TOL      = 1e-3
    TASK_ID       = "mlp_lvl3_training_tricks"
    OUT_DIR       = os.path.join("output", TASK_ID)

    set_seed(42)
    device = get_device()
    print(f"  Device: {device}")

    train_loader, val_loader, input_dim, num_classes = make_dataloaders(
        batch_size=BATCH_SIZE)

    def model_factory():
        return build_model(input_dim, num_classes, device,
                           hidden_dims=HIDDEN_DIMS, dropout_p=DROPOUT_P)

    model = model_factory()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    print("\n--- Training ---")
    history, best_ckpt_path = train(
        model, train_loader, val_loader, device,
        epochs=EPOCHS, lr=LR, grad_clip=GRAD_CLIP,
        checkpoint_dir=OUT_DIR, num_classes=num_classes)

    print("\n--- Loading Best Checkpoint for Final Evaluation ---")
    model, ckpt = load_best_checkpoint(model, best_ckpt_path, device)
    print(f"  Restored epoch {ckpt['epoch']}  "
          f"(stored val_acc={ckpt['val_metrics']['accuracy']:.4f})")

    print("\n--- Final Evaluation (best checkpoint) ---")
    train_metrics = evaluate(model, train_loader, device,
                             split_name="train", num_classes=num_classes)
    val_metrics   = evaluate(model, val_loader,   device,
                             split_name="val",   num_classes=num_classes)

    print("\n--- Checkpoint Reproducibility Check ---")
    reproduced = verify_checkpoint_reproduces(
        model_factory, best_ckpt_path, val_loader, device,
        reference_metrics=val_metrics, num_classes=num_classes, tol=CKPT_TOL)

    print("\n--- Saving Artifacts ---")
    all_metrics = {
        "train":       train_metrics,
        "val":         val_metrics,
        "reproduced":  reproduced,
        "best_epoch":  ckpt["epoch"],
        "config": {
            "epochs": EPOCHS, "lr": LR, "grad_clip": GRAD_CLIP,
            "hidden_dims": HIDDEN_DIMS, "dropout_p": DROPOUT_P,
        },
    }
    save_artifacts(model, all_metrics, history, task_id=TASK_ID)

    assert val_metrics["accuracy"] >= ACC_THRESHOLD, (
        f"Val accuracy {val_metrics['accuracy']:.4f} < threshold {ACC_THRESHOLD}")
    assert val_metrics["macro_f1"] >= F1_THRESHOLD, (
        f"Val macro-F1 {val_metrics['macro_f1']:.4f} < threshold {F1_THRESHOLD}")

    print(f"\n✓ All assertions passed. "
          f"val_acc={val_metrics['accuracy']:.4f}  "
          f"val_f1={val_metrics['macro_f1']:.4f}")
    sys.exit(0)