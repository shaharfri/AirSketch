"""CNN model definition for sketch classification (28x28 grayscale input).

Training-only (the app runs the exported OpenVINO IR via ``sketch_classifier``,
not PyTorch). The ``torch`` import is guarded so that merely *importing* this
module never fails when torch is missing or can't load (e.g. PyInstaller's
submodule scan in a low-memory isolated subprocess, which otherwise dies with
WinError 1455). When torch is present the real ``SketchCNN`` class is defined
exactly as before, so training is unaffected.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:   # ImportError, or OSError(1455) under memory pressure
    _HAS_TORCH = False


if _HAS_TORCH:
    class SketchCNN(nn.Module):
        """Simple CNN for 28x28 grayscale sketch classification.

        Architecture: 3 conv blocks (32→64→128) + 2 FC layers.
        ~330K parameters, ~2MB as ONNX.
        """

        def __init__(self, num_classes: int = 12):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 28x28 → 14x14
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 14x14 → 7x7
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 7x7 → 3x3
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128 * 3 * 3, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return self.classifier(x)

else:
    class SketchCNN:  # type: ignore[no-redef]
        """Stub used only when PyTorch is unavailable (e.g. build-time import
        scans). Constructing it is an error — training needs real PyTorch."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required to build/train SketchCNN, but it could not "
                "be imported. Install torch (the [train] extra)."
            )
