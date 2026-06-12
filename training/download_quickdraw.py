"""Download Quick Draw .npy files for selected categories and create train/val/test split."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

CATEGORIES = [
    "triangle", "square", "circle",
    "house", "car", "tree", "star",
    "cat", "flower", "sun", "airplane", "fish",
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


def download_category(category: str, output_dir: Path) -> np.ndarray:
    """Download a single category .npy file and return the array."""
    url = f"{BASE_URL}/{category}.npy"
    filepath = output_dir / f"{category}.npy"

    if filepath.exists():
        print(f"  {category}: already downloaded, loading...")
        return np.load(filepath)

    print(f"  {category}: downloading from {url}...")
    urllib.request.urlretrieve(url, filepath)
    return np.load(filepath)


def create_split(
    output_dir: Path,
    models_dir: Path,
    max_per_class: int = 10000,
    seed: int = 42,
) -> None:
    """Download all categories, subsample, split, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    print(f"Downloading {len(CATEGORIES)} categories (max {max_per_class} per class)...\n")

    for label_idx, category in enumerate(CATEGORIES):
        data = download_category(category, output_dir)
        # data shape: (N, 784) uint8
        n_samples = min(len(data), max_per_class)
        indices = rng.choice(len(data), n_samples, replace=False)
        subset = data[indices].reshape(n_samples, 28, 28)

        all_X.append(subset)
        all_y.append(np.full(n_samples, label_idx, dtype=np.int64))
        print(f"    -> {category}: {n_samples} samples")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split: 80% train, 10% val, 10% test
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    split_path = output_dir / "quickdraw_split.npz"
    np.savez_compressed(
        split_path,
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_val=X[n_train:n_train + n_val],
        y_val=y[n_train:n_train + n_val],
        X_test=X[n_train + n_val:],
        y_test=y[n_train + n_val:],
    )

    # Save class names for runtime
    labels_path = models_dir / "class_names.json"
    with open(labels_path, "w") as f:
        json.dump(CATEGORIES, f)

    print(f"\nSaved split to: {split_path}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n - n_train - n_val}")
    print(f"Saved class names to: {labels_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    create_split(
        output_dir=project_root / "data" / "quickdraw",
        models_dir=project_root / "models",
        max_per_class=10000,
    )
