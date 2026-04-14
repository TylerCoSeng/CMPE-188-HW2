import sys
import os
import json
import random
import itertools
import time
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metadata

def get_task_metadata():
    return {
        "task_id":   "mlp_lvl4_hparam_sweep",
        "algorithm": "MLP grid-search over depth, width, lr, weight_decay",
        "dataset":   "MNIST (torchvision) or synthetic fallback",
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

def _synthetic_dataloaders(batch_size: int):
    print("  [dataset] Using synthetic fallback (10-class, 128-dim, 4000 samples).")
    N, D, C = 4000, 128, 10
    torch.manual_seed(0)
    centroids = torch.randn(C, D)
    Xs, ys = [], []
    for c in range(C):
        Xs.append(centroids[c] + 0.45 * torch.randn(N // C, D))
        ys.append(torch.full((N // C,), c, dtype=torch.long))
    X = torch.cat(Xs)
    y = torch.cat(ys)
    mu, std = X.mean(0), X.std(0).clamp(min=1e-6)
    X = (X - mu) / std

    ds    = TensorDataset(X, y)
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
        full_train = torchvision.datasets.MNIST(
            root=data_root, train=True,  download=True, transform=tfm)
        full_val   = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=tfm)

        TRAIN_N, VAL_N = 12_000, 2_000
        train_ds, _ = random_split(
            full_train, [TRAIN_N, len(full_train) - TRAIN_N],
            generator=torch.Generator().manual_seed(42))
        val_ds, _ = random_split(
            full_val, [VAL_N, len(full_val) - VAL_N],
            generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        print(f"  [dataset] MNIST ready  train={TRAIN_N}  val={VAL_N}")
        return train_loader, val_loader, 28 * 28, 10
    except Exception as exc:
        print(f"  [dataset] MNIST unavailable ({exc}).")
        return _synthetic_dataloaders(batch_size)


# Model

class MLP(nn.Module):

    def __init__(self, input_dim: int, num_classes: int,
                 depth: int, width: int, dropout_p: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        layers, prev = [], input_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, width),
                       nn.BatchNorm1d(width),
                       nn.ReLU(inplace=True),
                       nn.Dropout(dropout_p)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(self.flatten(x))


def build_model(input_dim: int, num_classes: int, depth: int, width: int,
                device, dropout_p: float = 0.2) -> MLP:
    return MLP(input_dim, num_classes, depth, width, dropout_p).to(device)


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

def evaluate(model, loader, device, num_classes: int, split_name: str = ""):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * y.size(0)
            preds    = logits.argmax(1)
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
        print(f"    [{split_name:5s}]  loss={metrics['loss']:.4f}  "
              f"acc={metrics['accuracy']:.4f}  f1={metrics['macro_f1']:.4f}")
    return metrics


# Train one configuration

def train_one(cfg: dict, train_loader, val_loader,
              input_dim: int, num_classes: int, device) -> dict:

    set_seed(cfg.get("seed", 42))
    model = build_model(input_dim, num_classes,
                        depth=cfg["depth"], width=cfg["width"],
                        device=device, dropout_p=cfg.get("dropout_p", 0.2))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)

    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    elapsed = time.time() - t0
    val_m = evaluate(model, val_loader, device, num_classes=num_classes)
    return {
        "cfg":       cfg,
        "val_acc":   val_m["accuracy"],
        "val_loss":  val_m["loss"],
        "val_f1":    val_m["macro_f1"],
        "elapsed_s": round(elapsed, 2),
        "model":     model,
    }


# Grid search

def run_sweep(train_loader, val_loader, input_dim, num_classes, device,
              epochs_per_cfg: int = 5) -> list:

    grid = {
        "depth": [1, 2, 3],
        "width": [128, 256],
        "lr":    [1e-3, 3e-3],
        "wd":    [1e-4, 1e-3],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n  [sweep] {len(combos)} configurations × {epochs_per_cfg} epochs each")

    results = []
    for i, values in enumerate(combos, 1):
        cfg = dict(zip(keys, values))
        cfg["epochs"]    = epochs_per_cfg
        cfg["dropout_p"] = 0.2
        cfg["seed"]      = 42
        tag = (f"depth={cfg['depth']} width={cfg['width']} "
               f"lr={cfg['lr']:.0e} wd={cfg['wd']:.0e}")
        print(f"  [{i:2d}/{len(combos)}] {tag}", end="  ")
        result = train_one(cfg, train_loader, val_loader,
                           input_dim, num_classes, device)
        print(f"val_acc={result['val_acc']:.4f}  "
              f"f1={result['val_f1']:.4f}  "
              f"({result['elapsed_s']:.1f}s)")
        results.append({k: v for k, v in result.items() if k != "model"})
        results[-1]["_model_ref"] = result["model"]

    return results


# Predict

def predict(model, X: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).argmax(1).cpu()


# Save artifacts

def save_artifacts(best_model, all_metrics: dict, sweep_results: list,
                   task_id: str = "mlp_lvl4_hparam_sweep"):
    out_dir = os.path.join("output", task_id)
    os.makedirs(out_dir, exist_ok=True)

    weights_path = os.path.join(out_dir, "best_model_weights.pt")
    torch.save(best_model.state_dict(), weights_path)
    print(f"  Saved weights    -> {weights_path}")

    clean_sweep = [{k: v for k, v in r.items() if k != "_model_ref"}
                   for r in sweep_results]
    all_metrics["sweep"] = clean_sweep
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved metrics    -> {json_path}")

    csv_path = os.path.join(out_dir, "leaderboard.csv")
    fieldnames = ["rank", "depth", "width", "lr", "wd",
                  "val_acc", "val_loss", "val_f1", "elapsed_s"]
    sorted_results = sorted(clean_sweep,
                             key=lambda r: r["val_acc"], reverse=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(sorted_results, 1):
            writer.writerow({
                "rank":      rank,
                "depth":     r["cfg"]["depth"],
                "width":     r["cfg"]["width"],
                "lr":        r["cfg"]["lr"],
                "wd":        r["cfg"]["wd"],
                "val_acc":   round(r["val_acc"],  4),
                "val_loss":  round(r["val_loss"],  4),
                "val_f1":    round(r["val_f1"],    4),
                "elapsed_s": r["elapsed_s"],
            })
    print(f"  Saved leaderboard-> {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    accs = [r["val_acc"] for r in clean_sweep]
    ax.bar(range(len(accs)), accs, color="steelblue", alpha=0.8)
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    ax.bar(best_idx, accs[best_idx], color="crimson", label="best")
    ax.axhline(all_metrics.get("baseline_val_acc", 0),
               color="orange", linestyle="--", label="baseline")
    ax.set_xlabel("Config index")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Sweep – Val Accuracy per Configuration")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "sweep_results.png")
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"  Saved sweep plot -> {plot_path}")


# Main

if __name__ == "__main__":
    print("=== Task:", get_task_metadata()["task_id"], "===")

    BATCH_SIZE        = 256
    EPOCHS_PER_CFG    = 5
    EPOCHS_BEST_FINAL = 12
    TASK_ID           = "mlp_lvl4_hparam_sweep"

    ACC_THRESHOLD     = 0.90
    BEAT_BASELINE_BY  = 0.01

    BASELINE_CFG = {
        "depth": 1, "width": 128, "lr": 1e-3, "wd": 1e-4,
        "epochs": EPOCHS_PER_CFG, "dropout_p": 0.2, "seed": 42,
    }

    set_seed(42)
    device = get_device()
    print(f"  Device: {device}")

    train_loader, val_loader, input_dim, num_classes = make_dataloaders(
        batch_size=BATCH_SIZE)

    print("\n--- Baseline Training ---")
    baseline_result = train_one(BASELINE_CFG, train_loader, val_loader,
                                input_dim, num_classes, device)
    baseline_val_acc = baseline_result["val_acc"]
    print(f"  Baseline val_acc={baseline_val_acc:.4f}  "
          f"f1={baseline_result['val_f1']:.4f}  "
          f"({baseline_result['elapsed_s']:.1f}s)")

    print("\n--- Hyperparameter Sweep ---")
    sweep_results = run_sweep(train_loader, val_loader,
                              input_dim, num_classes, device,
                              epochs_per_cfg=EPOCHS_PER_CFG)

    sorted_results = sorted(sweep_results, key=lambda r: r["val_acc"], reverse=True)
    print("\n--- Leaderboard (top 5) ---")
    print(f"  {'Rank':>4}  {'Depth':>5}  {'Width':>5}  {'LR':>7}  "
          f"{'WD':>7}  {'Val Acc':>8}  {'Val F1':>7}")
    print("  " + "-" * 58)
    for rank, r in enumerate(sorted_results[:5], 1):
        c = r["cfg"]
        print(f"  {rank:>4}  {c['depth']:>5}  {c['width']:>5}  "
              f"{c['lr']:>7.0e}  {c['wd']:>7.0e}  "
              f"{r['val_acc']:>8.4f}  {r['val_f1']:>7.4f}")

    best_cfg_sweep = sorted_results[0]["cfg"]
    print(f"\n  Best sweep config: {best_cfg_sweep}")
    print(f"  Sweep best val_acc={sorted_results[0]['val_acc']:.4f}")

    print(f"\n--- Re-training Best Config for {EPOCHS_BEST_FINAL} epochs ---")
    best_cfg_full = {**best_cfg_sweep,
                     "epochs": EPOCHS_BEST_FINAL,
                     "dropout_p": 0.2,
                     "seed": 42}
    best_result = train_one(best_cfg_full, train_loader, val_loader,
                            input_dim, num_classes, device)
    best_model = best_result["model"]

    print("\n--- Final Evaluation (best config, full training) ---")
    train_metrics = evaluate(best_model, train_loader, device,
                             num_classes=num_classes, split_name="train")
    val_metrics   = evaluate(best_model, val_loader,   device,
                             num_classes=num_classes, split_name="val")

    print("\n--- Saving Artifacts ---")
    all_metrics = {
        "baseline":         {"cfg": BASELINE_CFG,
                             "val_acc": round(baseline_val_acc, 4),
                             "val_f1":  round(baseline_result["val_f1"], 4)},
        "best_cfg":         best_cfg_full,
        "best_val_acc":     round(val_metrics["accuracy"], 4),
        "best_val_f1":      round(val_metrics["macro_f1"], 4),
        "train":            train_metrics,
        "val":              val_metrics,
        "baseline_val_acc": round(baseline_val_acc, 4),
    }
    save_artifacts(best_model, all_metrics, sweep_results, task_id=TASK_ID)

    # Assertions
    assert val_metrics["accuracy"] >= ACC_THRESHOLD, (
        f"Best val_acc {val_metrics['accuracy']:.4f} < threshold {ACC_THRESHOLD}")

    margin = val_metrics["accuracy"] - baseline_val_acc
    if baseline_val_acc >= 0.999:
        best_loss = val_metrics["loss"]
        baseline_loss = all_metrics["baseline"].get("val_loss",
                                                    baseline_result.get("val_loss", float("inf")))
        assert margin >= 0 or best_loss <= baseline_loss, (
            f"Saturated baseline ({baseline_val_acc:.4f}): best config val_acc "
            f"({val_metrics['accuracy']:.4f}) is lower AND val_loss "
            f"({best_loss:.4f}) > baseline val_loss ({baseline_loss:.4f})")
        print(f"  ✓ Baseline saturated ({baseline_val_acc:.4f}); "
              f"tie-break by val_loss: best={best_loss:.4f}  "
              f"baseline={baseline_loss:.4f}")
    else:
        assert margin >= BEAT_BASELINE_BY, (
            f"Best config val_acc ({val_metrics['accuracy']:.4f}) does not beat "
            f"baseline ({baseline_val_acc:.4f}) by required margin {BEAT_BASELINE_BY} "
            f"(actual margin={margin:.4f})")

    print(f"\n✓ All assertions passed.")
    print(f"  val_acc={val_metrics['accuracy']:.4f}  "
          f"val_f1={val_metrics['macro_f1']:.4f}  "
          f"margin_over_baseline={margin:.4f}")
    sys.exit(0)