"""Train the SketchCNN model and export to OpenVINO IR format."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path so we can import the model definition
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from airsketch.sketch_cnn import SketchCNN


class GaussianNoise:
    """Add random Gaussian noise to a tensor."""

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma


def load_data(data_path: Path, batch_size: int = 128):
    """Load the .npz split and return DataLoaders."""
    data = np.load(data_path)

    def make_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        X_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
        y_t = torch.from_numpy(y).long()
        return TensorDataset(X_t, y_t)

    train_ds = make_dataset(data["X_train"], data["y_train"])
    val_ds = make_dataset(data["X_val"], data["y_val"])
    test_ds = make_dataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to a WHOLE batch at once (CPU-friendly).

    A single rotation + translation is sampled per batch and applied with one
    vectorized affine_grid / grid_sample, instead of looping per image. This is
    100x+ faster on CPU while still providing useful regularization, and
    Gaussian noise is added per element.
    """
    import torch.nn.functional as F

    n = x.shape[0]
    # Random rotation in radians (±15°) and small translation, sampled per batch
    angle = (torch.rand(1).item() * 2 - 1) * (15 * math.pi / 180)
    tx = (torch.rand(1).item() * 2 - 1) * 0.10  # fraction of width
    ty = (torch.rand(1).item() * 2 - 1) * 0.10
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    theta = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=x.dtype
    ).unsqueeze(0).repeat(n, 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    x = x + torch.randn_like(x) * 0.05
    return x.clamp(0, 1)


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
) -> None:
    """Train the model, export ONNX, convert to OpenVINO IR."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training device: {device}")

    # Load data
    train_loader, val_loader, test_loader = load_data(data_path, batch_size)

    # Load class names to get num_classes
    labels_path = output_dir / "class_names.json"
    with open(labels_path) as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Model
    model = SketchCNN(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = augment_batch(X_batch)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:2d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"lr={current_lr:.1e}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Test accuracy
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            test_correct += (logits.argmax(1) == y_batch).sum().item()
            test_total += len(y_batch)

    test_acc = test_correct / test_total
    print(f"\nTest accuracy: {test_acc:.3f} ({test_correct}/{test_total})")

    if test_acc < 0.80:
        print("WARNING: Test accuracy below 80%. Model may need more training or data.")

    # Convert directly from PyTorch to OpenVINO IR (skip ONNX file issues)
    model = model.to("cpu")
    model.eval()
    import openvino as ov
    dummy_input = torch.randn(1, 1, 28, 28)
    ov_model = ov.convert_model(model, example_input=dummy_input)
    ir_path = output_dir / "sketch_classifier.xml"
    ov.save_model(ov_model, str(ir_path))
    print(f"\nOpenVINO IR saved: {ir_path}")
    print(f"Done! Model ready at: {ir_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train(
        data_path=project_root / "data" / "quickdraw" / "quickdraw_split.npz",
        output_dir=project_root / "models",
        epochs=15,
        batch_size=256,
        patience=4,
    )
