"""OpenVINO inference abstraction — device-agnostic model loading and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


class InferenceEngine:
    """Load an OpenVINO IR model and run inference on any available device.

    Usage:
        engine = InferenceEngine("models/shape_classifier.xml", device="AUTO")
        result = engine.infer(input_array)
    """

    def __init__(self, model_path: str, device: str = "AUTO"):
        if not HAS_OPENVINO:
            raise RuntimeError(
                "OpenVINO is not installed. Install with: pip install openvino"
            )
        self._core = ov.Core()
        model = self._core.read_model(model_path)
        self._compiled = self._core.compile_model(model, device)
        self._input_layer = self._compiled.input(0)
        self._output_layer = self._compiled.output(0)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on a single input tensor."""
        result = self._compiled([input_tensor])
        return result[self._output_layer]

    @property
    def input_shape(self) -> list[int] | None:
        shape = self._input_layer.partial_shape
        if shape.is_dynamic:
            return None
        return [d.get_length() for d in shape]

    @property
    def output_shape(self) -> list[int] | None:
        shape = self._output_layer.partial_shape
        if shape.is_dynamic:
            return None
        return [d.get_length() for d in shape]

    @staticmethod
    def available_devices() -> list[str]:
        """List all available inference devices."""
        if not HAS_OPENVINO:
            return []
        return ov.Core().available_devices

    @staticmethod
    def convert_onnx_to_ir(onnx_path: str, output_dir: str) -> str:
        """Convert an ONNX model to OpenVINO IR format.

        Returns path to the generated .xml file.
        """
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO is not installed.")
        model = ov.convert_model(onnx_path)
        output = Path(output_dir) / Path(onnx_path).stem
        ov.save_model(model, str(output) + ".xml")
        return str(output) + ".xml"
