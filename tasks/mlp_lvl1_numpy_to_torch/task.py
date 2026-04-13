import sys
import os
import json
import random

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metadata

def get_task_metadata():
    return {
        "task_id": "mlp_lvl1_numpy_to_torch",
        "algorithm": "2-layer MLP, manual backprop",
        "dataset": "XOR",
        "framework": "PyTorch (no autograd)",
    }


# Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


# Device

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset / dataloaders

def make_dataloaders(device):
    """
    XOR dataset: 4 canonical points, replicated for a stable split.
    Returns plain tensors (no DataLoader needed for 4-point XOR).
    """
    X_all = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=torch.float32, device=device)

    y_all = torch.tensor([[0.0], [1.0], [1.0], [0.0]],
                         dtype=torch.float32, device=device)

    X_train, y_train = X_all, y_all
    X_val,   y_val   = X_all, y_all

    return (X_train, y_train), (X_val, y_val)


# Model: raw parameter tensors (no nn.Module)

def build_model(input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1,
                device=None):
    """
    Returns a dict of weight/bias tensors initialised with Xavier uniform.
    requires_grad=False because we update them manually.
    """
    if device is None:
        device = get_device()

    def xavier(fan_in, fan_out):
        limit = (6.0 / (fan_in + fan_out)) ** 0.5
        return torch.empty(fan_in, fan_out, device=device).uniform_(-limit, limit)

    params = {
        "W1": xavier(input_dim, hidden_dim),
        "b1": torch.zeros(1, hidden_dim, device=device),
        "W2": xavier(hidden_dim, output_dim),
        "b2": torch.zeros(1, output_dim, device=device),
    }
    return params


# Activation

def sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


# Forward pass

def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# BCE Loss

def bce_loss(y_hat, y):
    eps = 1e-8
    return -torch.mean(y * torch.log(y_hat + eps) +
                       (1 - y) * torch.log(1 - y_hat + eps))


# Manual backward pass

def backward(y, params, cache):
    """
    Computes gradients by hand using the chain rule (see module docstring).
    Returns a dict of gradients with the same keys as params.
    """
    A2, A1, X = cache["A2"], cache["A1"], cache["X"]
    N = float(y.shape[0])

    # Output layer
    dA2 = -(y / (A2 + 1e-8)) + (1 - y) / (1 - A2 + 1e-8)
    dZ2 = dA2 * A2 * (1.0 - A2)
    dW2 = A1.t() @ dZ2 / N
    db2 = torch.mean(dZ2, dim=0, keepdim=True)

    # Hidden layer
    dA1 = dZ2 @ params["W2"].t()
    dZ1 = dA1 * A1 * (1.0 - A1)
    dW1 = X.t() @ dZ1 / N
    db1 = torch.mean(dZ1, dim=0, keepdim=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# Train

def train(params, train_data, lr: float = 1.0, epochs: int = 5000):
    """
    Gradient-descent loop.  No optimizer, no loss.backward().
    Returns list of per-epoch training losses.
    """
    X_train, y_train = train_data
    loss_history = []

    for epoch in range(1, epochs + 1):
        y_hat, cache = forward(X_train, params)
        loss = bce_loss(y_hat, y_train)

        grads = backward(y_train, params, cache)

        for key in params:
            params[key] = params[key] - lr * grads[key]

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d}  loss={loss.item():.6f}")

    return loss_history


# Evaluate

def evaluate(params, data, split_name: str = ""):
    X, y = data
    with torch.no_grad():
        y_hat, _ = forward(X, params)
    loss = bce_loss(y_hat, y).item()
    preds = (y_hat >= 0.5).float()
    acc = (preds == y).float().mean().item()
    if split_name:
        print(f"  [{split_name}]  loss={loss:.6f}  acc={acc:.4f}")
    return {"loss": loss, "accuracy": acc}


# Predict

def predict(params, X):
    with torch.no_grad():
        y_hat, _ = forward(X, params)
    return (y_hat >= 0.5).float()


# Save artifacts

def save_artifacts(metrics: dict, loss_history: list,
                   task_id: str = "mlp_lvl1_numpy_to_torch"):
    out_dir = os.path.join("output", task_id)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics  -> {json_path}")

    plot_path = os.path.join(out_dir, "loss_curve.png")
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training Loss (manual backprop, XOR)")
    fig.savefig(plot_path, dpi=80)
    plt.close(fig)
    print(f"  Saved loss plot -> {plot_path}")


# Main

if __name__ == "__main__":
    print("=== Task:", get_task_metadata()["task_id"], "===")

    set_seed(42)
    device = get_device()
    print(f"  Device: {device}")

    train_data, val_data = make_dataloaders(device)
    params = build_model(input_dim=2, hidden_dim=4, output_dim=1, device=device)

    print("\n--- Training ---")
    loss_history = train(params, train_data, lr=1.0, epochs=5000)

    print("\n--- Evaluation ---")
    train_metrics = evaluate(params, train_data, split_name="train")
    val_metrics   = evaluate(params, val_data,   split_name="val")

    print("\n--- Predictions ---")
    X_all, y_all = train_data
    preds = predict(params, X_all)
    for i in range(len(X_all)):
        xi = X_all[i].tolist()
        print(f"  XOR{xi} -> pred={int(preds[i].item())}  true={int(y_all[i].item())}")

    print("\n--- Saving Artifacts ---")
    all_metrics = {
        "train": train_metrics,
        "val":   val_metrics,
    }
    save_artifacts(all_metrics, loss_history)

    ACC_THRESHOLD = 0.95
    assert val_metrics["accuracy"] >= ACC_THRESHOLD, (
        f"Val accuracy {val_metrics['accuracy']:.4f} < threshold {ACC_THRESHOLD}"
    )
    assert train_metrics["accuracy"] >= ACC_THRESHOLD, (
        f"Train accuracy {train_metrics['accuracy']:.4f} < threshold {ACC_THRESHOLD}"
    )

    print(f"\n✓ All assertions passed. "
          f"Val acc={val_metrics['accuracy']:.4f} >= {ACC_THRESHOLD}")
    sys.exit(0)