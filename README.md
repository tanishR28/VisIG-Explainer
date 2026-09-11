# Experiment 5: ResNet18 + Integrated Gradients

Interactive Explainable AI demo for image classification and attribution using **pretrained ResNet18 (ImageNet-1K)** and **Captum Integrated Gradients**.

Built for an Explainable AI course project with a Gradio UI, baseline comparison, convergence delta reporting, and bundled test images.

## Features

- Upload any compatible image (JPG/PNG)
- **ResNet18** inference with top-5 predictions and confidence
- **Integrated Gradients** explanations with Captum
- **Baselines:** black, gray, and white RGB references
- **`n_steps` control** (10–200) with **convergence delta** display
- Optional **input image blur** before model preprocessing
- Side-by-side **heatmap** and **overlay** visualizations
- **Baseline comparison** tab (black vs gray vs white)
- Bundled **Good Cases** and **Challenging Cases** test images
- Course notes and experiment results table in the UI

## Project structure

```
.
├── app.py                      # Gradio app + ML/XAI pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore
└── ResNet18_Test_Images/
    ├── Good_Cases/             # 5 clear classification examples
    └── Challenging_Cases/      # 5 harder/occlusion/context examples
```

## Requirements

- Python 3.10+ recommended
- CPU is supported (GPU optional; app defaults to CPU)
- ~2 GB free disk space for PyTorch + torchvision weights

## Setup

From this repository root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run locally

```bash
python app.py
```

Open the URL shown in the terminal (default: http://127.0.0.1:7860).

### Temporary public link (demo / viva)

Change the launch line in `app.py` to:

```python
demo.launch(share=True)
```

Gradio will print a temporary public `gradio.live` URL.

## How to use the app

1. Open the **Explain Image** tab.
2. Upload an image or pick a sample from **Good Cases** / **Challenging Cases**.
3. Choose a **Baseline** (Black / Gray / White).
4. Set **Integration Steps (`n_steps`)** — default `50`.
5. Optionally enable **Blur input image** to blur before ResNet18/IG.
6. Click **Explain / Classify**.
7. Inspect:
   - Predicted class and confidence
   - Top 5 predictions
   - IG convergence **delta**
   - Model input, heatmap, and overlay

Use the **Compare Baselines** tab to view all three baselines on one image.

## Technical pipeline

```
Upload image
    ↓
(Optional) blur input image
    ↓
torchvision preprocess (ImageNet)
    ↓
Pretrained ResNet18 → prediction + confidence
    ↓
Solid-color RGB baseline → same preprocess
    ↓
Captum Integrated Gradients (n_steps, return_convergence_delta=True)
    ↓
Attribution map → heatmap + overlay
    ↓
Gradio UI
```

## Key XAI concepts (viva notes)

| Term | Meaning |
|------|---------|
| **Baseline** | Reference input IG integrates from (black / gray / white) |
| **`n_steps`** | Number of interpolation points in the IG Riemann sum |
| **Delta** | Captum completeness/convergence error for the IG estimate |
| **Input features** | Image pixels/channels — not `n_steps` or delta |
| **Attribution map** | Per-pixel contribution toward the selected class logit |

Try the same image with `n_steps = 10, 50, 100, 200` and compare **delta** and heatmaps.

## Deploy on Hugging Face Spaces

1. Create a new Space with SDK **Gradio**.
2. Upload `app.py`, `requirements.txt`, and `ResNet18_Test_Images/`.
3. Use **CPU Basic** hardware (free tier).
4. Prefer moderate `n_steps` (25–50) on CPU for faster responses.

Space file layout:

```
app.py
requirements.txt
README.md
ResNet18_Test_Images/
```

## Test images (bundled)

| Case | File | Notes |
|------|------|-------|
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

## Libraries

- [PyTorch](https://pytorch.org/) + [torchvision](https://pytorch.org/vision/) — ResNet18
- [Captum](https://captum.ai/) — Integrated Gradients
- [Gradio](https://gradio.app/) — UI
- PIL, NumPy, Matplotlib — image handling and plots

## License

Course / educational use. Add a license file if required by your institution.

## Author

Explainable AI — Experiment 5
