# Model Download Instructions

All models go in the `models/` directory (gitignored).

## Quick Setup (all models)

```bash
micromamba activate skysketch
pip install optimum[openvino]

# Qwen3 LLM (required for shape descriptions)
optimum-cli export openvino --model Qwen/Qwen3-1.7B --weight-format int4 --trust-remote-code models/qwen3-1.7b-ov

# Whisper STT (required for voice input)
optimum-cli export openvino --model openai/whisper-base --trust-remote-code models/whisper-base-ov

# LCM DreamShaper (required for image generation)
huggingface-cli download OpenVINO/LCM_Dreamshaper_v7-int8-ov --local-dir models/lcm-dreamshaper-ov
```

## MediaPipe Hand Model

Downloaded automatically by `setup_env.sh`, or manually:

```bash
curl -L -o models/hand_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

## CNN Shape Classifier

Already included as `models/sketch_classifier.xml` + `models/sketch_classifier.bin`.
To retrain:

```bash
pip install torch torchvision
python -m training.download_quickdraw
python -m training.train_sketch_cnn
```

## Model Sizes

| Model | Size | Purpose |
|-------|------|---------|
| hand_landmarker.task | 7.6 MB | Hand tracking |
| sketch_classifier.xml+bin | 800 KB | Shape CNN |
| qwen3-1.7b-ov/ | ~1 GB | LLM fun facts |
| whisper-base-ov/ | ~150 MB | Speech-to-text |
| lcm-dreamshaper-ov/ | ~1.5 GB | Image generation |

## Fallback: Qwen2.5-1.5B

If Qwen3 fails to export, Qwen2.5 works as fallback:

```bash
optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct --weight-format int4 models/qwen2.5-1.5b-ov
```

The app auto-detects which model is available (prefers Qwen3).
