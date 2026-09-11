---
title: VisIG Explainer
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: ResNet18 + Integrated Gradients image attribution demo
---

# VisIG Explainer

VisIG Explainer is an interactive web application that classifies images with a **pretrained ResNet18** model and explains each prediction using **Captum Integrated Gradients (IG)**.

Upload an image, and the app shows the predicted ImageNet class, confidence, top-5 predictions, and a pixel-level attribution heatmap that highlights regions influencing the model’s decision.

## What it does

- **Classification** — ResNet18 pretrained on ImageNet-1K (1000 classes, inference only)
- **Attribution** — Integrated Gradients from a selectable baseline (black, gray, or white)
- **Integration control** — adjustable `n_steps` with Captum convergence **delta**
- **Visualization** — attribution heatmap and overlay on the model input
- **Baseline analysis** — compare how explanations change across baselines
- **Test images** — bundled good and challenging examples for experimentation

## How it works

```
Image upload
    → optional input blur
    → ImageNet preprocessing
    → ResNet18 prediction
    → RGB baseline (same preprocessing)
    → Integrated Gradients (n_steps, target = predicted class)
    → heatmap + overlay
```

Integrated Gradients estimates how much each input pixel contributed to the selected class by integrating gradients along a path from the baseline to the input. The **`n_steps`** parameter controls how many points are used in that numerical integration; **delta** reports Captum’s completeness/convergence error for the estimate.

## Project structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── ResNet18_Test_Images/
    ├── Good_Cases/
    └── Challenging_Cases/
```

## Requirements

- Python 3.10+
- CPU or GPU (CUDA used automatically when available)

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

## Usage

1. **Explain Image** — upload an image, choose a baseline and `n_steps`, then click **Explain / Classify**
2. **Compare Baselines** — view black, gray, and white baseline overlays for the same image
3. **Course Notes & Results** — in-app notes on ResNet18, IG, and baselines; refresh the results table when needed

## Tech stack

- PyTorch & torchvision — ResNet18
- Captum — Integrated Gradients
- Gradio — web interface
- PIL, NumPy, Matplotlib — image handling and plots
