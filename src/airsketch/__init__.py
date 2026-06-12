"""AirSketch — air-drawing with multi-stroke notebook, live snap-to-shape,
CNN sketch recognition, and optional Qwen-VL semantic enrichment.

A merger of AirDraw AR (multi-stroke diagrams + VLM + live snap) and
Skysketch (CNN classifier + InferenceEngine + tests + voice + image-gen).
"""
from airsketch.beautifier import beautify_diagram, points_for_primitive
from airsketch.camera import Camera
from airsketch.diagram_analyzer import (
    DiagramAnalyzer,
    LocalAnalyzer,
    OpenVINOQwenVLAnalyzer,
    create_analyzer,
)
from airsketch.exporter import export_html, export_json
from airsketch.gesture_detector import IndexPointingDetector, PinchDetector
from airsketch.hand_tracker import HandTracker, MediaPipeHandTracker
from airsketch.inference_engine import InferenceEngine
from airsketch.notebook import Notebook
from airsketch.primitive_classifier import Primitive, PrimitiveClassifier, PrimitiveKind
from airsketch.shape_recognizer import ShapeRecognizer
from airsketch.sketch_classifier import SketchClassifier
from airsketch.stroke import Diagram, DiagramAnalysis, DiagramStatus, Stroke
from airsketch.video_source import VideoSource

__version__ = "0.1.0"

__all__ = [
    "beautify_diagram", "points_for_primitive",
    "Camera",
    "DiagramAnalyzer", "LocalAnalyzer", "OpenVINOQwenVLAnalyzer", "create_analyzer",
    "export_html", "export_json",
    "IndexPointingDetector", "PinchDetector",
    "HandTracker", "MediaPipeHandTracker",
    "InferenceEngine",
    "Notebook",
    "Primitive", "PrimitiveClassifier", "PrimitiveKind",
    "ShapeRecognizer",
    "SketchClassifier",
    "Diagram", "DiagramAnalysis", "DiagramStatus", "Stroke",
    "VideoSource",
]
