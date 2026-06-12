"""Image generation via OpenVINO GenAI — sketch-to-image using LCM img2img.

Usage:
    gen = ImageGenerator()
    gen.generate_async(canvas_image, "a beautiful house")
    # ... poll gen.is_generating / gen.result ...
    output_image = gen.result
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

try:
    import openvino_genai as ov_genai
    HAS_IMAGE_GEN = True
except ImportError:
    HAS_IMAGE_GEN = False

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_LCM_MODEL = _MODELS_DIR / "lcm-dreamshaper-ov"


class ImageGenerator:
    """Generate images from sketches using LCM via OpenVINO GenAI."""

    def __init__(self, model_path: str | None = None, device: str = "CPU"):
        if not HAS_IMAGE_GEN:
            raise RuntimeError("openvino-genai is not installed")
        path = model_path or str(_LCM_MODEL)
        if not Path(path).exists():
            raise FileNotFoundError(f"Image model not found: {path}")
        self._pipe = ov_genai.Image2ImagePipeline(path, device)
        self._generating = False
        self._result: np.ndarray | None = None

    def generate(self, sketch: np.ndarray, prompt: str, strength: float = 0.75,
                 num_steps: int = 4, size: int = 384) -> np.ndarray:
        """Generate image from sketch (blocking). Returns BGR image."""
        input_img = self._prepare_input(sketch, size)
        result = self._pipe.generate(
            prompt,
            input_img,
            strength=strength,
            num_inference_steps=num_steps,
        )
        return self._to_cv2(result)

    def generate_async(self, sketch: np.ndarray, prompt: str, strength: float = 0.75,
                       num_steps: int = 4, size: int = 384) -> None:
        """Start generation in a background thread."""
        self._result = None
        self._generating = True
        thread = threading.Thread(
            target=self._generate_bg,
            args=(sketch, prompt, strength, num_steps, size),
            daemon=True,
        )
        thread.start()

    def _generate_bg(self, sketch: np.ndarray, prompt: str,
                     strength: float, num_steps: int, size: int) -> None:
        try:
            self._result = self.generate(sketch, prompt, strength, num_steps, size)
        except Exception:
            self._result = None
        finally:
            self._generating = False

    @property
    def is_generating(self) -> bool:
        return self._generating

    @property
    def result(self) -> np.ndarray | None:
        return self._result

    @staticmethod
    def _prepare_input(sketch: np.ndarray, size: int) -> ov_genai.Tensor:
        """Convert canvas BGR image to RGB tensor for the pipeline."""
        # Resize sketch to target size
        resized = cv2.resize(sketch, (size, size))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to the format expected by openvino-genai
        return ov_genai.Tensor(rgb)

    @staticmethod
    def _to_cv2(result) -> np.ndarray:
        """Convert pipeline output to BGR numpy array."""
        # openvino-genai returns an ov_genai.Tensor or numpy array
        if hasattr(result, 'data'):
            img = np.array(result.data, dtype=np.uint8)
        else:
            img = np.array(result, dtype=np.uint8)
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
