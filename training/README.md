# Training the Sketch Classifier

## Prerequisites

```bash
micromamba activate skysketch
pip install torch torchvision  # or: pip install -e ".[train]"
```

## Steps

### 1. Download data

Downloads 12 categories from Google Quick Draw (28x28 grayscale sketches):

```bash
python -m training.download_quickdraw
```

This creates `data/quickdraw/quickdraw_split.npz` (~90MB) with 120K samples.

### 2. Train

```bash
python -m training.train_sketch_cnn
```

Trains a 3-layer CNN (~330K params) for ~5-10 minutes on M1 Mac.
Produces:
- `models/sketch_classifier.xml` — OpenVINO IR model
- `models/sketch_classifier.bin` — Model weights
- `models/class_names.json` — Label mapping

Expected test accuracy: 85-90%.

### 3. Use

After training, the main app automatically loads the CNN:

```bash
python -m skysketch.main
```

Draw complex objects (house, cat, car) — the CNN classifies them when contour detection can't.

## Categories

triangle, square, circle, house, car, tree, star, cat, flower, sun, airplane, fish

## Retraining

To add new categories:
1. Edit `CATEGORIES` list in `training/download_quickdraw.py`
2. Re-run download: `python -m training.download_quickdraw`
3. Re-run training: `python -m training.train_sketch_cnn`

The runtime classifier picks up the new `class_names.json` automatically.
