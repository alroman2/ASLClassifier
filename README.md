# ASL Classifier 🖐️🤟

**ASL Classifier** is an end-to-end, Docker-packaged pipeline for detecting three common American Sign Language gestures—**“I Love You”, “Thank You”, and “Yes”**—in images or video frames.  
It is built on **PyTorch + TorchVision Faster R-CNN** and comes with ready-to-run Docker Compose targets for **GPU** and **CPU-only** training or inference.

> *If you are looking for a quick demo, jump to **[🔧 Quick Start](#quick-start)**.*

---

## Table of Contents
1. [Project Motivation](#project-motivation)
2. [Features](#features)
3. [Model Architecture](#model-architecture)
4. [Dataset & Directory Layout](#dataset--directory-layout)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
   - Training
   - Inference / Evaluation
7. [Project Structure](#project-structure)
8. [Extending to New Classes](#extending-to-new-classes)
9. [Results](#results)
10. [Roadmap](#roadmap)
11. [Contributing](#contributing)
12. [License & Citation](#license--citation)

---

## Project Motivation
Accurate sign-language recognition lowers communication barriers for the Deaf and Hard-of-Hearing community.  
This repository demonstrates a **clean, reproducible MLOps workflow** (Docker, Compose, GPU/CPU parity) that enables you to:

- Retrain the model on a custom set of signs
- Benchmark Faster R-CNN on small, hand-annotated datasets
- Package the trained detector for downstream apps (mobile, embedded, web)

A short technical report with background research and experiments is included at [`ASL_Classifier_Report.pdf`](ASL_Classifier_Report.pdf).

---

## Features

| ✨ Feature | Description |
|-----------|-------------|
| **3-class object detector** | Detects *I Love You*, *Thank You*, *Yes* signs in real time |
| **Dockerised workflow** | One-line commands for training & inference on **Windows/Linux/NVIDIA GPUs**, **macOS (CPU/Metal)**, or **CPU-only** |
| **Albumentations data pipeline** | Geometric & blur augmentations for better generalisation |
| **Auto-logging** | Saves annotated loss curves and checkpoints every *n* epochs |
| **Modular code** | Separate modules for config, dataset, transforms, engine, metrics |

---

## Model Architecture
```
Faster-R-CNN (resnet50 + FPN, pretrained on ImageNet)
      │
      ├── Backbone (freezes first two blocks)
      ├── Region Proposal Network
      └── Detection Head
            └── cls_score: 4 neurons (background + 3 ASL classes)
```

---

## Dataset & Directory Layout
The repo ships with a **sample Pascal-VOC style** dataset in `train/` and `test/`:

```
train/
 ├── ily_***.jpg
 ├── ily_***.xml         # bounding-box
 ├── thankyou_***.jpg
 ├── thankyou_***.xml
 ├── yes_***.jpg
 └── yes_***.xml
```

- **Images:** RGB, variable resolution
- **Annotations:** XML (`<object><bndbox>…</bndbox></object>`)
- **Classes:** `["background", "I love you", "Thank You", "Yes"]`

> **Tip:** Replace these folders with your own JPEG+XML pairs to retrain.

---

## Quick Start

### 1. Clone & enter
```bash
git clone https://github.com/alroman2/ASLClassifier.git
cd ASLClassifier
```

### 2. Ensure Docker & (optionally) NVIDIA Container Toolkit are installed
- GPU recommended, but **CPU-only** mode is supported.

### 3. Run!
| Goal | GPU / Windows-Linux | CPU-only | macOS |
|------|--------------------|----------|-------|
| **Train** | `docker-compose up train` | `docker-compose up train-no-gpu` | `docker-compose up train-mac` |
| **Infer** | `docker-compose up inference` | `docker-compose up inference-no-gpu` | `docker-compose up inference-mac` |

*Checkpoints land in `outputs/`, predictions in `test_predictions/`.*

---

## Detailed Usage

### Training Parameters
Edit `src/config.py` to tweak:
- `BATCH_SIZE`, `NUM_EPOCHS`, `RESIZE_TO`
- `CLASSES` list and `NUM_CLASSES`
- Plot/checkpoint frequency (`SAVE_PLOTS_EPOCH`, `SAVE_MODEL_EPOCH`)

To visualise augmentations before training, set `VISUALIZE_TRANSFORMED_IMAGES = True`.

### Inference
`src/inference.py` loads the latest checkpoint from `outputs/`, runs on images in `test_data/`, and writes annotated images + JSON predictions to `test_predictions/`.

### Metrics
`src/metrics.py` computes mAP@0.5 and per-class recall/precision (WIP—see Roadmap).

---

## Project Structure
```
├── Docker/                # CUDA-11.6 (Linux) and macOS base images
├── docker-compose.yml     # One-liners for every scenario
├── src/
│   ├── config.py          # Hyper-parameters & paths
│   ├── datasets.py        # Pascal-VOC Dataset + DataLoaders
│   ├── model.py           # Faster R-CNN factory
│   ├── engine.py          # Train / validation loops
│   ├── inference.py       # Batch inference script
│   └── utils.py           # Augs, Averager, helpers
├── train/, test/          # Images + XML labels
└── outputs/               # Checkpoints & loss plots (auto-created)
```

---

## Extending to New Classes
1. Add new class names to `CLASSES` and increment `NUM_CLASSES`.
2. Collect JPEG images, label each sign with a single bounding box in VOC XML format.
3. Place images + XML annotations in `train/` & `test/`.
4. Re-run **Training**—that’s it! The model head auto-resizes.

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| FPS (RTX 3060) | ~22 | 512 × 512 inference |

Loss curves are saved every 2 epochs in `outputs/` (editable via `SAVE_PLOTS_EPOCH`).

We note that extensible models will need much larger amounts of data, but this serves as proof of concept.

---

## Roadmap
- [ ] Expand dataset to full ASL alphabet (A-Z)

---

## Contributing
Pull requests are welcome! Please open an issue first to discuss major changes.  
Ensure your code passes **`flake8`** and **pre-commit hooks**.

---

## License & Citation
This project is released under the **MIT License** (see `LICENSE`).  
If you use this work in academic research, please cite the repository and the original datasets you train on.

---

> _Made with ❤️ and PyTorch by [@alroman2](https://github.com/alroman2)_
