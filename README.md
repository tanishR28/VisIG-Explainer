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

**Interactive image classification and pixel-level attribution for ImageNet using ResNet18 and Captum Integrated Gradients.**

VisIG Explainer is a Gradio web app that classifies uploaded images with a pretrained ResNet18 model and visualizes *why* the model made its prediction using Integrated Gradients (IG). Compare black, gray, and white baselines, tune integration steps (`n_steps`), inspect convergence delta, and explore bundled good and challenging test cases.


## Features

- Upload any compatible image (JPG / PNG / WEBP)
- **ResNet18 (ImageNet-1K)** inference with **top-5** predictions and confidence
- **Captum Integrated Gradients** with configurable **`n_steps`** (10–200)
- **Convergence delta** reporting (`return_convergence_delta=True`)
- **Baselines:** black, gray, and white RGB reference inputs
- Optional **input image blur** before preprocessing (live preview)
- **Heatmap** and **overlay** visualizations
- **Baseline comparison** tab with correlation analysis
- Bundled **Good Cases** and **Challenging Cases** test images
- In-app course notes and experiment results table

---

## Project structure

```
visig-explainer/
├── app.py                      # Gradio UI + ML / XAI pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore
└── ResNet18_Test_Images/
    ├── Good_Cases/             # Clear classification examples
    └── Challenging_Cases/      # Occlusion, context, sketch cases
```

---

## Requirements

- Python 3.10+
- CPU supported (app runs on CPU by default)
- ~2 GB disk space for PyTorch and torchvision weights

---

## Setup

From the repository root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run locally

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

### Temporary public demo link

In `app.py`, change the launch line to:

```python
demo.launch(share=True)
```

Gradio will print a temporary public URL (useful for viva / remote demo).

---

## How to use

1. Open the **Explain Image** tab.
2. Upload an image or select a bundled test case.
3. Choose a **Baseline** (Black / Gray / White).
4. Set **Integration Steps (`n_steps`)** — default `50`.
5. Optionally enable **Blur input image**.
6. Click **Explain / Classify**.
7. Review prediction, top-5 classes, convergence **delta**, heatmap, and overlay.

Use **Compare Baselines** to see all three baselines side by side.

---

## Pipeline

```
Upload image
    ↓
(Optional) blur input
    ↓
torchvision preprocess (ImageNet)
    ↓
Pretrained ResNet18 → prediction + confidence
    ↓
RGB baseline → same preprocess
    ↓
Captum Integrated Gradients (n_steps, convergence delta)
    ↓
Attribution map → heatmap + overlay
    ↓
Gradio UI
```


## Deploy on Hugging Face Spaces

1. Create a Space with SDK **Gradio**.
2. Connect this repo or upload:
   - `app.py`
   - `requirements.txt`
   - `ResNet18_Test_Images/`
3. Select **CPU Basic** (free tier).
4. Use moderate `n_steps` (25–50) on CPU for faster responses.

---

## Push to GitHub (quick reference)

```bash
cd path/to/this/repo
git remote add origin https://github.com/YOUR_USERNAME/visig-explainer.git
git push -u origin main
```

Create the GitHub repo **empty** (no README / .gitignore) before pushing.

---

## Bundled test images

| Set | File | Notes |
|-----|------|-------|
| Good | `good_01_dog.jpg` | Clear dog |
| Good | `good_02_elephant.jpg` | Clear elephant |
| Good | `good_03_car.jpg` | Clear car |
| Good | `good_04_eagle.jpg` | Clear bird |
| Good | `good_05_coffee.jpg` | Clear coffee cup |
| Challenging | `challenge_01_dog_sunglasses.jpg` | Occlusion |
| Challenging | `challenge_02_laptop_forest.jpg` | Unusual context |
| Challenging | `challenge_03_bird_camouflage.jpg` | Camouflage |
| Challenging | `challenge_04_toy_car.jpg` | Toy vs real |
| Challenging | `challenge_05_sketch_elephant.jpg` | Sketch |

---

## Tech stack

| Library | Role |
|---------|------|
| [PyTorch](https://pytorch.org/) + [torchvision](https://pytorch.org/vision/) | ResNet18 inference |
| [Captum](https://captum.ai/) | Integrated Gradients |
| [Gradio](https://gradio.app/) | Web UI |
| PIL, NumPy, Matplotlib | Images and plots |

---

## License

Educational / coursework use. Add an institutional license if required.

---

## Author

Explainable AI coursework — **VisIG Explainer**
