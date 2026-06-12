"""Diagram analyzer interface + LocalAnalyzer fallback + OpenVINO Qwen2-VL implementation."""
import json
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

import cv2
import numpy as np

from airsketch.config import RecognitionResult
from airsketch.stroke import DiagramAnalysis


class DiagramAnalyzer(ABC):
    """Interface — produces a title/description/topic/tags for a finalized diagram."""

    name: str = "base"

    @abstractmethod
    def analyze(
        self,
        canvas: np.ndarray,
        shapes: List[RecognitionResult],
        primitives: list | None = None,
    ) -> DiagramAnalysis:
        ...

    def release(self) -> None:
        pass


# ----------------------------------------------------------------------------
# Local geometric fallback — no model required, runs in <1 ms
# ----------------------------------------------------------------------------

class LocalAnalyzer(DiagramAnalyzer):
    """Heuristic analyzer using only the geometric ShapeRecognizer output."""

    name = "local"

    TOPIC_BY_SHAPE = {
        "circle": "geometry",
        "ellipse": "geometry",
        "triangle": "geometry",
        "rectangle": "geometry",
        "polygon": "geometry",
        "line": "diagram",
        "arrow": "diagram",
        "curve": "sketch",
        "dot": "symbol",
        "star": "symbol",
        "heart": "symbol",
    }

    def analyze(
        self,
        canvas: np.ndarray,
        shapes: List[RecognitionResult],
        primitives: list | None = None,
    ) -> DiagramAnalysis:
        # Prefer primitive list if available — it's per-stroke and reliable
        if primitives:
            counts = Counter(p.kind for p in primitives if p.kind != "dot")
        else:
            recognized = [s for s in shapes if s.label != "unknown"]
            counts = Counter(s.label for s in recognized) if recognized else Counter()

        if not counts:
            return DiagramAnalysis(
                title="Free-form Sketch",
                description="An abstract hand-drawn sketch.",
                topic="sketch",
                tags=["sketch", "abstract"],
                confidence=0.30,
                analyzer_name=self.name,
            )

        parts = [
            f"{n} {lbl}{'s' if n > 1 else ''}"
            for lbl, n in counts.most_common()
        ]
        description = f"Diagram with {', '.join(parts)}."

        if len(counts) == 1:
            lbl = next(iter(counts))
            title = f"{lbl.capitalize()} Diagram"
        else:
            title = " + ".join(lbl.capitalize() for lbl, _ in counts.most_common(3))

        top_label = counts.most_common(1)[0][0]
        topic = self.TOPIC_BY_SHAPE.get(top_label, "diagram")
        tags = list(counts.keys())

        return DiagramAnalysis(
            title=title,
            description=description,
            topic=topic,
            tags=tags,
            confidence=0.60,
            analyzer_name=self.name,
        )


# ----------------------------------------------------------------------------
# OpenVINO + Qwen2-VL VLM analyzer
# ----------------------------------------------------------------------------

class OpenVINOQwenVLAnalyzer(DiagramAnalyzer):
    """VLM analyzer using OpenVINO GenAI's VLMPipeline.

    Loads `OpenVINO/Qwen2-VL-2B-Instruct-int4-ov` (or any compatible IR) and
    asks for a JSON-structured response with title/description/topic/tags.
    """

    name = "openvino_qwen2vl"

    # The prompt explicitly tells Qwen NOT to re-identify shapes — geometric
    # detection already did that. Qwen's job is semantic enrichment only.
    PROMPT_TEMPLATE = (
        "A user drew a diagram in the air. Our geometric detector already "
        "identified the strokes as: {shapes_str}.\n\n"
        "Look at the rendered image (clean shapes on a white background) and "
        "tell us what the diagram REPRESENTS semantically — not what shapes "
        "it contains.\n\n"
        "Respond with ONLY a single JSON object (no markdown, no commentary):\n"
        '  "title":        a short interpretive title (5 words max). '
        'For a single primitive, just use the shape name. For compositions, '
        'name what they likely represent (e.g. "Flowchart", "Graph", "Equation").\n'
        '  "description":  one sentence about what the diagram likely represents '
        'or could be used for.\n'
        '  "topic":        one of [math, geometry, physics, chemistry, biology, '
        "flowchart, graph, equation, sketch, symbol, other]\n"
        '  "tags":         list of 2-4 short keywords\n\n'
        "Do NOT contradict the geometric detection. If we said the strokes are "
        "1 triangle, your title should not say 'circle'.\n"
    )

    def __init__(self, model_dir: str, device: str = "CPU", max_tokens: int = 150):
        import openvino_genai as ov_genai  # noqa: F401

        self._device = device
        self._max_tokens = max_tokens
        # Always-available local fallback for the factual title
        self._local = LocalAnalyzer()
        print(f"  Loading Qwen2-VL pipeline on device={device}...", flush=True)
        self._pipe = ov_genai.VLMPipeline(model_dir, device)
        print(f"  Qwen2-VL pipeline ready.", flush=True)

    def analyze(
        self,
        canvas: np.ndarray,
        shapes: List[RecognitionResult],
        primitives: list | None = None,
    ) -> DiagramAnalysis:
        # 1) Always start from the local geometric analysis — this gives the
        #    factually correct title from primitive detection.
        local = self._local.analyze(canvas, shapes, primitives=primitives)

        # 2) For trivial cases (single shape, no composition), skip the VLM
        #    entirely — Qwen-2B can actually hurt accuracy here, and we save
        #    7-15 seconds of inference per diagram.
        non_dot_count = sum(1 for p in (primitives or []) if p.kind != "dot")
        if non_dot_count <= 1:
            local.confidence = 0.85
            local.analyzer_name = self.name + ":local-only"
            return local

        # 3) Multi-primitive composition — ask Qwen for semantic enrichment.
        try:
            vlm_result = self._call_vlm(canvas, shapes, primitives)
        except Exception as e:
            # VLM failed → return the local analysis with a note
            local.analyzer_name = self.name + f":vlm-failed({type(e).__name__})"
            return local

        # 4) Merge: prefer LOCAL for title (it's based on confident primitive
        #    detection), VLM for description/topic/tags (semantic context).
        merged_title = local.title
        # If VLM produced a clearly more meaningful title (e.g. "Flowchart"
        # instead of "Rectangle + Arrow + Rectangle"), prefer it — but only
        # when the VLM title contains a known composition keyword.
        composition_words = {
            "flowchart", "graph", "equation", "chart", "diagram", "tree",
            "network", "map", "process", "cycle", "timeline", "table", "plot",
        }
        vlm_title_lower = vlm_result.title.lower()
        if any(w in vlm_title_lower for w in composition_words):
            merged_title = vlm_result.title

        merged_tags = list(dict.fromkeys(local.tags + vlm_result.tags))[:5]

        return DiagramAnalysis(
            title=merged_title,
            description=vlm_result.description or local.description,
            topic=vlm_result.topic if vlm_result.topic != "sketch" else local.topic,
            tags=merged_tags,
            confidence=0.85,
            raw_response=vlm_result.raw_response,
            analyzer_name=self.name,
        )

    def _call_vlm(
        self,
        canvas: np.ndarray,
        shapes: List[RecognitionResult],
        primitives: list | None,
    ) -> DiagramAnalysis:
        import openvino as ov

        if primitives:
            counts = Counter(p.kind for p in primitives if p.kind != "dot")
            shapes_str = (
                ", ".join(f"{n} {lbl}" for lbl, n in counts.most_common())
                or "free-form strokes"
            )
        else:
            shapes_str = self._shapes_to_text(shapes)
        prompt = self.PROMPT_TEMPLATE.format(shapes_str=shapes_str)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image_tensor = ov.Tensor(np.ascontiguousarray(rgb))

        try:
            raw = self._pipe.generate(
                prompt,
                images=[image_tensor],
                max_new_tokens=self._max_tokens,
                do_sample=False,
            )
        except TypeError:
            raw = self._pipe.generate(
                prompt,
                images=[image_tensor],
                max_new_tokens=self._max_tokens,
            )

        text = str(raw)
        return self._parse_response(text, shapes)

    @staticmethod
    def _shapes_to_text(shapes: List[RecognitionResult]) -> str:
        if not shapes:
            return "no clear geometric shapes"
        counts = Counter(s.label for s in shapes if s.label != "unknown")
        if not counts:
            return "free-form strokes"
        return ", ".join(f"{n} {lbl}" for lbl, n in counts.most_common())

    def _parse_response(
        self, text: str, shapes: List[RecognitionResult]
    ) -> DiagramAnalysis:
        # Try to extract a JSON object from the response.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                data = json.loads(json_text)
                tags_raw = data.get("tags", [])
                if not isinstance(tags_raw, list):
                    tags_raw = [str(tags_raw)]
                return DiagramAnalysis(
                    title=str(data.get("title", "Untitled"))[:60],
                    description=str(data.get("description", ""))[:240],
                    topic=str(data.get("topic", "sketch")).lower()[:24],
                    tags=[str(t)[:24] for t in tags_raw][:4],
                    confidence=0.85,
                    raw_response=text,
                    analyzer_name=self.name,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: use the raw response as the description, generate a title heuristically.
        first_line = text.strip().split("\n", 1)[0].strip().strip("\"' .")
        title = first_line[:50] if first_line else "Diagram"
        return DiagramAnalysis(
            title=title,
            description=text.strip()[:240],
            topic="sketch",
            tags=[s.label for s in shapes[:3] if s.label != "unknown"] or ["sketch"],
            confidence=0.50,
            raw_response=text,
            analyzer_name=self.name + ":fallback",
        )

    def release(self) -> None:
        try:
            del self._pipe
        except Exception:
            pass


# ----------------------------------------------------------------------------
# CNN Analyzer (Quick, Draw! sketch classifier)
# Adopted from Skysketch — recognizes complex objects like house, car, cat,
# tree, star, flower, sun, airplane, fish in addition to geometric shapes.
# ----------------------------------------------------------------------------

class CNNAnalyzer(DiagramAnalyzer):
    """Sketch classifier built on Skysketch's Quick-Draw CNN via OpenVINO."""

    name = "cnn"

    TOPIC_BY_LABEL = {
        # geometric → routed to LocalAnalyzer-style result
        "triangle": "geometry", "square": "geometry", "circle": "geometry",
        # Quick, Draw! categories
        "house": "object", "car": "vehicle", "tree": "nature",
        "star": "symbol", "cat": "animal", "flower": "nature",
        "sun": "nature", "airplane": "vehicle", "fish": "animal",
    }

    def __init__(
        self,
        sketch_classifier,
        local_fallback: "LocalAnalyzer | None" = None,
        confidence_threshold: float = 0.55,
    ):
        self._cls = sketch_classifier
        self._local = local_fallback or LocalAnalyzer()
        self._threshold = confidence_threshold

    def analyze(
        self,
        canvas: np.ndarray,
        shapes: List[RecognitionResult],
        primitives: list | None = None,
    ) -> DiagramAnalysis:
        # Local primitive analysis is always cheap and accurate for simple cases
        local = self._local.analyze(canvas, shapes, primitives)

        # Single-primitive cases: trust the geometric detector
        non_dot = sum(1 for p in (primitives or []) if p.kind != "dot")
        if non_dot <= 1:
            local.analyzer_name = self.name + ":single-prim"
            return local

        # Multi-primitive composition — try the CNN
        try:
            label, conf = self._cls.classify(canvas)
        except Exception as e:
            local.analyzer_name = self.name + f":cnn-failed({type(e).__name__})"
            return local

        if label == "unknown" or conf < self._threshold:
            return local

        topic = self.TOPIC_BY_LABEL.get(label, "sketch")
        return DiagramAnalysis(
            title=label.capitalize(),
            description=f"Recognized as a {label} ({conf:.0%} confidence).",
            topic=topic,
            tags=list(dict.fromkeys([label] + local.tags))[:4],
            confidence=float(conf),
            analyzer_name=self.name,
        )


# ----------------------------------------------------------------------------
# Chained analyzer — CNN → VLM enrichment
# ----------------------------------------------------------------------------

class ChainedAnalyzer(DiagramAnalyzer):
    """Three-tier analyzer chain:
       1. Geometric (LocalAnalyzer) for single primitives
       2. CNN sketch classifier for compositions (house, car, cat, ...)
       3. VLM (Qwen2-VL) for semantic enrichment when present

    Each tier can be enabled/disabled. The chain returns the most-confident
    interpretation, with VLM contributing description/topic/tags when active.
    """

    name = "chain"

    def __init__(
        self,
        cnn_analyzer: "CNNAnalyzer | None" = None,
        vlm_analyzer: "OpenVINOQwenVLAnalyzer | None" = None,
        local_analyzer: "LocalAnalyzer | None" = None,
    ):
        self._cnn = cnn_analyzer
        self._vlm = vlm_analyzer
        self._local = local_analyzer or LocalAnalyzer()

    def analyze(
        self,
        canvas: np.ndarray,
        shapes: List[RecognitionResult],
        primitives: list | None = None,
    ) -> DiagramAnalysis:
        # Stage 1: CNN if available, else local
        base = (self._cnn or self._local).analyze(canvas, shapes, primitives)

        # Stage 2: VLM enrichment for compositions
        non_dot = sum(1 for p in (primitives or []) if p.kind != "dot")
        if self._vlm is None or non_dot <= 1:
            return base

        try:
            vlm = self._vlm._call_vlm(canvas, shapes, primitives)
        except Exception:
            return base

        # Merge: CNN/local owns the title (factual), VLM enriches description/topic/tags
        composition_words = {
            "flowchart", "graph", "equation", "chart", "diagram", "tree",
            "network", "map", "process", "cycle", "timeline", "table", "plot",
        }
        title = base.title
        if any(w in vlm.title.lower() for w in composition_words):
            title = vlm.title

        merged_tags = list(dict.fromkeys(base.tags + vlm.tags))[:5]
        return DiagramAnalysis(
            title=title,
            description=vlm.description or base.description,
            topic=vlm.topic if vlm.topic != "sketch" else base.topic,
            tags=merged_tags,
            confidence=max(base.confidence, vlm.confidence),
            raw_response=vlm.raw_response,
            analyzer_name=self.name,
        )

    def release(self) -> None:
        if self._vlm:
            self._vlm.release()


# ----------------------------------------------------------------------------
# Device discovery + model download helpers
# ----------------------------------------------------------------------------

def list_openvino_devices() -> List[str]:
    """Return available OpenVINO devices, or [] on import failure."""
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except Exception:
        return []


def pick_device(preferred: str = "AUTO") -> str:
    """Pick an OpenVINO device, falling back gracefully.

    `preferred` may be 'AUTO', 'CPU', 'GPU', 'NPU'.
    'AUTO' policy: GPU > NPU > CPU.
    """
    available = list_openvino_devices()
    if not available:
        return "CPU"

    if preferred != "AUTO":
        # Try exact match, then prefix match (e.g. 'GPU' matches 'GPU.0')
        if preferred in available:
            return preferred
        for d in available:
            if d.startswith(preferred):
                return d
        print(
            f"  WARNING: requested device {preferred} not available "
            f"(found {available}); falling back to AUTO selection."
        )

    for priority in ("GPU", "NPU", "CPU"):
        for d in available:
            if d == priority or d.startswith(priority):
                return d
    return available[0]


def download_vlm_model(
    repo_id: str,
    cache_dir: str | None = None,
    offline_only: bool = False,
) -> str:
    """Download (or locate cached) HuggingFace model. Returns local directory.

    If `offline_only` is True, only succeeds when the model is already cached;
    never touches the network.
    """
    from huggingface_hub import snapshot_download

    if offline_only:
        print(f"  Looking up cached model {repo_id} (offline, no download)...", flush=True)
    else:
        print(f"  Resolving model {repo_id} (download if missing)...", flush=True)

    kwargs = {"repo_id": repo_id, "cache_dir": cache_dir}
    # local_dir_use_symlinks was removed in newer huggingface_hub; pass conditionally
    try:
        local_dir = snapshot_download(local_files_only=offline_only, **kwargs)
    except TypeError:
        local_dir = snapshot_download(**kwargs)
    print(f"  Model directory: {local_dir}", flush=True)
    return local_dir


def ensure_ov_tokenizer(model_dir: str) -> bool:
    """If the OpenVINO tokenizer IR is missing, convert it from the HF tokenizer
    files that came with the model.

    Returns True if the OV tokenizer is present (either was already, or was
    just converted). False on irrecoverable failure.
    """
    xml = os.path.join(model_dir, "openvino_tokenizer.xml")
    if os.path.exists(xml):
        return True

    # The OV tokenizer file may live in the resolved (non-symlink) target dir
    # but be missing from the snapshot dir. Either way we'll regenerate it.
    print(f"  openvino_tokenizer.xml missing — auto-converting from tokenizer.json")

    tok_json = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_json):
        print(f"  ERROR: no tokenizer.json found in {model_dir}; cannot convert")
        return False

    try:
        # Resolve symlink for tokenizers library (it doesn't follow symlinks on Windows)
        tok_path = os.path.realpath(tok_json)

        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(model_dir)
            tok_object = hf_tok
        except ImportError:
            # Fall back to raw tokenizers (smaller dep) — may still need
            # transformers internally for some tokenizer types
            from tokenizers import Tokenizer
            tok_object = Tokenizer.from_file(tok_path)

        from openvino_tokenizers import convert_tokenizer
        import openvino as ov

        ov_tok, ov_det = convert_tokenizer(tok_object, with_detokenizer=True)
        ov.save_model(ov_tok, xml)
        ov.save_model(ov_det, os.path.join(model_dir, "openvino_detokenizer.xml"))
        print(f"  Tokenizer converted: {xml}")
        return True

    except ImportError as e:
        print(f"  ERROR: tokenizer conversion needs `transformers` "
              f"(pip install transformers[sentencepiece] tiktoken): {e}")
        return False
    except Exception as e:
        print(f"  ERROR: tokenizer conversion failed: {type(e).__name__}: {e}")
        return False


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------

def _try_cnn_analyzer(config) -> "CNNAnalyzer | None":
    """Attempt to load Skysketch's Quick-Draw CNN. Returns None if model is missing."""
    try:
        from airsketch.sketch_classifier import SketchClassifier
        cnn_device = getattr(config, "cnn_device", "AUTO")
        classifier = SketchClassifier(device=cnn_device)
        print(f"[analyzer] CNN sketch classifier loaded on device={cnn_device}")
        print(f"[analyzer]   labels: {classifier.labels}")
        return CNNAnalyzer(classifier)
    except FileNotFoundError as e:
        print(f"[analyzer] CNN not loaded: {e}")
        print(f"[analyzer]   To enable: train the CNN via `python -m training.train_sketch_cnn`")
    except Exception as e:
        print(f"[analyzer] CNN load failed: {type(e).__name__}: {e}")
    return None


def _try_vlm_analyzer(config) -> "OpenVINOQwenVLAnalyzer | None":
    """Attempt to load Qwen-VL via OpenVINO GenAI. Returns None if disabled/failed."""
    if not getattr(config, "vlm_enabled", False):
        return None
    try:
        device = pick_device(config.vlm_device)
        print(f"[analyzer] OpenVINO devices: {list_openvino_devices()}")
        print(f"[analyzer] VLM device: {device}")
        os.makedirs(config.vlm_model_cache_dir, exist_ok=True)
        # Prefer a bundled, ready-to-use local VLM dir (offline, no download).
        local = os.path.join(config.vlm_model_cache_dir, "qwen2-vl-2b-ov")
        if os.path.isdir(local) and any(f.endswith(".xml") for f in os.listdir(local)):
            model_dir = local
            print(f"[analyzer] Using bundled VLM: {local}")
        else:
            model_dir = download_vlm_model(
                config.vlm_model_id,
                cache_dir=config.vlm_model_cache_dir,
                offline_only=config.vlm_offline_only,
            )
        if not ensure_ov_tokenizer(model_dir):
            raise RuntimeError(
                "OpenVINO tokenizer IR missing; install transformers or try a different repo"
            )
        return OpenVINOQwenVLAnalyzer(
            model_dir, device=device, max_tokens=config.vlm_max_tokens
        )
    except Exception as e:
        print(f"[analyzer] VLM load failed: {type(e).__name__}: {e}")
        return None


def create_analyzer(config) -> DiagramAnalyzer:
    """Construct the three-tier analyzer chain:
        local geometric  →  CNN sketch classifier  →  Qwen-VL enrichment

    Tiers that can't be loaded are skipped — the chain degrades gracefully
    to whichever tiers ARE available, all the way down to LocalAnalyzer.
    """
    local = LocalAnalyzer()
    cnn = _try_cnn_analyzer(config) if getattr(config, "cnn_enabled", True) else None
    vlm = _try_vlm_analyzer(config)

    if cnn is None and vlm is None:
        print("[analyzer] Using LocalAnalyzer (no CNN, no VLM).")
        return local

    print(f"[analyzer] Chain: local + "
          f"{'CNN' if cnn else '-'} + "
          f"{'VLM' if vlm else '-'}")
    return ChainedAnalyzer(cnn_analyzer=cnn, vlm_analyzer=vlm, local_analyzer=local)
