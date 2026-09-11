import io
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image, ImageFilter
from torchvision.models import ResNet18_Weights, resnet18

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
TEST_IMAGE_ROOT = APP_DIR / "ResNet18_Test_Images"
IMAGE_SIZE = 224

# Project identity (UI, docs, deployment)
PROJECT_NAME = "VisIG Explainer"
PROJECT_TAGLINE = (
    "Interactive image classification and pixel-level attribution using "
    "pretrained ResNet18 and Captum Integrated Gradients."
)

BASELINE_COLORS = {
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
}

BASELINE_TYPES = list(BASELINE_COLORS.keys())

# Default integration steps for Captum IG Riemann sum approximation.
DEFAULT_N_STEPS = 50
INTEGRATION_STEP_CHOICES = [10, 25, 50, 100, 150, 200]

TEST_IMAGES = {
    "Good Cases": [
        "good_01_dog.jpg",
        "good_02_elephant.jpg",
        "good_03_car.jpg",
        "good_04_eagle.jpg",
        "good_05_coffee.jpg",
    ],
    "Challenging Cases": [
        "challenge_01_dog_sunglasses.jpg",
        "challenge_02_laptop_forest.jpg",
        "challenge_03_bird_camouflage.jpg",
        "challenge_04_toy_car.jpg",
        "challenge_05_sketch_elephant.jpg",
    ],
}

CASE_FOLDER_MAP = {
    "Good Cases": "Good_Cases",
    "Challenging Cases": "Challenging_Cases",
}

# Loaded once at startup and reused for all requests.
MODEL = None
PREPROCESS = None
CATEGORIES = None
IG = None

# Hugging Face Spaces set SPACE_ID at runtime. Skip heavy startup jobs there.
IS_HF_SPACE = os.getenv("SPACE_ID") is not None


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model():
    """Load pretrained ResNet18 and Captum Integrated Gradients explainer."""
    global MODEL, PREPROCESS, CATEGORIES, IG

    if MODEL is not None:
        return MODEL, PREPROCESS, CATEGORIES, IG

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()

    MODEL = model
    PREPROCESS = weights.transforms()
    CATEGORIES = weights.meta["categories"]
    IG = IntegratedGradients(model)

    return MODEL, PREPROCESS, CATEGORIES, IG


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------


def predict_image(image, top_k=5):
    """
    Run inference on a PIL image.

    Returns label, confidence percentage, class id, preprocessed input tensor,
    and top-k predictions as (label, confidence_pct) tuples.
    """
    model, preprocess, categories, _ = load_model()

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)

    class_id = predicted_class.item()
    label = categories[class_id]
    confidence_pct = confidence.item() * 100

    top_probs, top_indices = torch.topk(probabilities.squeeze(0), top_k)
    top_predictions = [
        (categories[idx.item()], top_probs[i].item() * 100)
        for i, idx in enumerate(top_indices)
    ]

    return label, confidence_pct, class_id, input_tensor, top_predictions


def format_top_predictions(top_predictions):
    """Format top-k predictions as a markdown table for the UI."""
    lines = [
        "| Rank | Class | Confidence |",
        "| --- | --- | --- |",
    ]
    for rank, (pred_label, pred_conf) in enumerate(top_predictions, start=1):
        lines.append(f"| {rank} | {pred_label} | {pred_conf:.2f}% |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def create_baseline(image, baseline_type):
    """
    Create a baseline tensor for Integrated Gradients.

    IG compares the input to a reference (baseline) input. The baseline is built
    as a solid RGB image and passed through the same preprocessing pipeline as
    the original image so both tensors live in the same feature space.
    """
    _, preprocess, _, _ = load_model()

    baseline_key = baseline_type.lower()
    if baseline_key not in BASELINE_COLORS:
        raise ValueError(
            f"Unknown baseline type '{baseline_type}'. "
            f"Choose from: {', '.join(BASELINE_TYPES)}"
        )

    color = BASELINE_COLORS[baseline_key]
    baseline_image = Image.new("RGB", image.size, color)
    baseline = preprocess(baseline_image).unsqueeze(0)
    return baseline


# -----------------------------------------------------------------------------
# Integrated Gradients
# -----------------------------------------------------------------------------


def generate_integrated_gradients(input_tensor, baseline, target, n_steps=50):
    """
    Compute raw Integrated Gradients attributions for the selected class.

    n_steps controls how many interpolation points are used along the straight
    path from baseline to input when approximating the integral numerically.
    More steps usually improve approximation quality but increase compute time.

    Captum can also return delta, the convergence/completeness error:
    how well the attributions satisfy the IG completeness axiom for the
    selected target class. Pixels/channels are the input features; n_steps and
    delta describe the integration procedure, not additional input features.
    """
    _, _, _, ig = load_model()

    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return attributions, delta


def convergence_delta_value(delta):
    """Convert Captum's convergence delta tensor to a plain float."""
    if isinstance(delta, torch.Tensor):
        return float(delta.detach().cpu().reshape(-1)[0].item())
    return float(delta)


def format_convergence_delta(delta, n_steps, baseline_type, class_label):
    """Format IG convergence delta for display in the Gradio UI."""
    delta_value = convergence_delta_value(delta)
    return (
        f"**Integration steps (n_steps):** {int(n_steps)}\n\n"
        f"**Convergence delta:** {delta_value:.6f}\n\n"
        f"Computed for prediction **{class_label}** with **{baseline_type.title()}** baseline.\n\n"
        "Delta is Captum's completeness/convergence error for the IG estimate. "
        "Smaller absolute values generally mean the numerical integration is "
        "more consistent with the model output change from baseline to input."
    )


def create_heatmap(attributions, mode="magnitude"):
    """
    Convert raw attributions into a 2D importance map.

    mode='magnitude': absolute channel sum (default, good for overlay)
    mode='signed': signed channel sum (positive vs negative contribution)
    """
    attr = attributions.squeeze(0)

    if mode == "signed":
        heatmap = attr.sum(dim=0).detach().cpu().numpy()
        max_abs = np.abs(heatmap).max()
        if max_abs > 0:
            heatmap = heatmap / max_abs
        return heatmap

    heatmap = attr.abs().sum(dim=0).detach().cpu().numpy()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap


def blur_input_image(image, radius=5):
    """
    Apply Gaussian blur to the uploaded image before model preprocessing.

    The blurred RGB image is what ResNet18 and Integrated Gradients receive.
    """
    if radius <= 0:
        return image.convert("RGB")

    return image.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=int(radius))
    )


def prepare_model_image(image, blur_input=False, blur_radius=5):
    """Return the PIL image that will be preprocessed and sent to the model."""
    if blur_input:
        return blur_input_image(image, blur_radius)
    return image.convert("RGB")


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def pil_resize_224(image):
    """Resize image to model input size for consistent visualization."""
    return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def figure_to_pil(fig):
    """Convert a matplotlib figure to a PIL image for Gradio display."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def create_heatmap_image(heatmap, mode="magnitude", title="Integrated Gradients Heatmap"):
    """Render a standalone heatmap image."""
    fig, ax = plt.subplots(figsize=(4, 4))

    if mode == "signed":
        im = ax.imshow(heatmap, cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        im = ax.imshow(heatmap, cmap="hot", vmin=0, vmax=1)

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return figure_to_pil(fig)


def create_overlay(image, heatmap, title="IG Overlay"):
    """Overlay the attribution heatmap on the original image."""
    original = pil_resize_224(image)
    original_np = np.array(original)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original_np)
    ax.imshow(heatmap, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return figure_to_pil(fig)


def create_comparison_panel(image, overlays, titles, suptitle=""):
    """Build a multi-panel comparison figure for baseline analysis."""
    fig, axes = plt.subplots(1, len(overlays), figsize=(5 * len(overlays), 5))

    if len(overlays) == 1:
        axes = [axes]

    for ax, overlay, title in zip(axes, overlays, titles):
        ax.imshow(overlay)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout()
    return figure_to_pil(fig)


def attribution_correlation(heatmap_a, heatmap_b):
    """Pearson correlation between two flattened attribution maps."""
    flat_a = heatmap_a.flatten()
    flat_b = heatmap_b.flatten()

    if flat_a.std() == 0 or flat_b.std() == 0:
        return float("nan")

    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def describe_baseline_observation(heatmap, baseline_type, confidence_pct):
    """
    Generate a neutral observation based on heatmap statistics.

    These notes describe spatial patterns only, not ground-truth correctness.
    """
    total = heatmap.sum()
    max_val = heatmap.max()
    high_threshold = max_val * 0.6
    high_fraction = (heatmap >= high_threshold).mean()

    if confidence_pct < 50:
        confidence_note = "Low prediction confidence; inspect heatmap carefully."
    elif confidence_pct < 70:
        confidence_note = "Moderate prediction confidence."
    else:
        confidence_note = "High prediction confidence."

    if high_fraction > 0.25:
        spread_note = "Attribution is spread across a relatively large area."
    elif high_fraction > 0.12:
        spread_note = "Attribution is concentrated on a moderate-sized region."
    else:
        spread_note = "Attribution is concentrated on a compact region."

    peak_note = (
        f"Peak normalized attribution under {baseline_type} baseline: {max_val:.2f}."
    )

    return f"{confidence_note} {spread_note} {peak_note}"


def compare_baselines_text(black_map, gray_map, white_map):
    """Summarize factual differences between baseline attributions."""
    corr_bg = attribution_correlation(black_map, gray_map)
    corr_bw = attribution_correlation(black_map, white_map)
    corr_gw = attribution_correlation(gray_map, white_map)

    def fmt(name, value):
        if np.isnan(value):
            return f"{name}: not comparable (flat attribution map)"
        return f"{name}: {value:.2f}"

    lines = [
        "### Baseline Comparison (factual)",
        "",
        "Pearson correlation of flattened attribution maps:",
        f"- {fmt('Black vs Gray', corr_bg)}",
        f"- {fmt('Black vs White', corr_bw)}",
        f"- {fmt('Gray vs White', corr_gw)}",
        "",
        "Interpretation guide:",
        "- Values close to 1.0 mean broadly similar spatial patterns.",
        "- Lower values mean the baseline choice changes where attribution appears.",
        "- Correlation measures pattern similarity, not whether the model is correct.",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def explain_image(
    image,
    baseline_type,
    blur_input=False,
    blur_radius=5,
    n_steps=DEFAULT_N_STEPS,
):
    """Run the full explanation pipeline for one image and baseline."""
    model_image = prepare_model_image(image, blur_input, blur_radius)

    label, confidence_pct, class_id, input_tensor, top_predictions = predict_image(
        model_image
    )
    baseline = create_baseline(model_image, baseline_type)

    attributions, delta = generate_integrated_gradients(
        input_tensor,
        baseline,
        class_id,
        n_steps=n_steps,
    )
    heatmap = create_heatmap(attributions, mode="magnitude")

    blur_suffix = f", input blur={blur_radius}" if blur_input else ""
    model_input_view = pil_resize_224(model_image)
    heatmap_img = create_heatmap_image(
        heatmap,
        title=f"IG Heatmap ({baseline_type.title()} baseline, n_steps={n_steps}{blur_suffix})",
    )
    overlay = create_overlay(
        model_image,
        heatmap,
        title=f"IG Overlay ({baseline_type.title()} baseline, n_steps={n_steps}{blur_suffix})",
    )

    return {
        "label": label,
        "confidence_pct": confidence_pct,
        "class_id": class_id,
        "top_predictions": top_predictions,
        "original": model_input_view,
        "heatmap": heatmap_img,
        "overlay": overlay,
        "heatmap_array": heatmap,
        "model_image": model_image,
        "delta": delta,
        "n_steps": n_steps,
        "baseline_type": baseline_type,
    }


def explain_all_baselines(image, n_steps=DEFAULT_N_STEPS):
    """Generate explanations for black, gray, and white baselines."""
    label, confidence_pct, class_id, input_tensor, _ = predict_image(image)

    results = {}
    heatmaps = {}

    for baseline_type in BASELINE_TYPES:
        baseline = create_baseline(image, baseline_type)
        attributions, _ = generate_integrated_gradients(
            input_tensor,
            baseline,
            class_id,
            n_steps=n_steps,
        )
        heatmap = create_heatmap(attributions, mode="magnitude")
        heatmaps[baseline_type] = heatmap
        results[baseline_type] = create_overlay(
            image,
            heatmap,
            title=f"{baseline_type.title()} Baseline",
        )

    comparison_text = compare_baselines_text(
        heatmaps["black"],
        heatmaps["gray"],
        heatmaps["white"],
    )

    original = pil_resize_224(image)
    panel = create_comparison_panel(
        image,
        [original, results["black"], results["gray"], results["white"]],
        ["Original", "Black Baseline", "Gray Baseline", "White Baseline"],
        suptitle=f"Prediction: {label} | Confidence: {confidence_pct:.2f}%",
    )

    return {
        "label": label,
        "confidence_pct": confidence_pct,
        "panel": panel,
        "comparison_text": comparison_text,
        "heatmaps": heatmaps,
    }


# -----------------------------------------------------------------------------
# Test image helpers
# -----------------------------------------------------------------------------


def list_test_images(case_set):
    """Return available filenames for a test case set."""
    if case_set not in TEST_IMAGES:
        return gr.update(choices=[], value=None)

    folder_name = CASE_FOLDER_MAP[case_set]
    folder_path = TEST_IMAGE_ROOT / folder_name

    if folder_path.is_dir():
        available = sorted(
            name
            for name in TEST_IMAGES[case_set]
            if (folder_path / name).is_file()
        )
    else:
        available = TEST_IMAGES[case_set]

    default = available[0] if available else None
    return gr.update(choices=available, value=default)


def load_test_image(case_set, filename):
    """Load a provided test image into the upload component."""
    if not case_set or not filename:
        return None

    folder_name = CASE_FOLDER_MAP.get(case_set)
    if folder_name is None:
        return None

    image_path = TEST_IMAGE_ROOT / folder_name / filename
    if not image_path.is_file():
        return None

    return Image.open(image_path).convert("RGB")


def test_images_available():
    """Check whether the bundled test image folders exist."""
    return TEST_IMAGE_ROOT.is_dir()


# -----------------------------------------------------------------------------
# Experiment results table
# -----------------------------------------------------------------------------


def build_results_table():
    """
    Build a results table for all 10 test images across three baselines.

    Observations are neutral descriptions of attribution patterns.
    """
    rows = []

    for case_set, filenames in TEST_IMAGES.items():
        folder_name = CASE_FOLDER_MAP[case_set]

        for filename in filenames:
            image_path = TEST_IMAGE_ROOT / folder_name / filename
            if not image_path.is_file():
                continue

            image = Image.open(image_path).convert("RGB")
            label, confidence_pct, class_id, input_tensor, _ = predict_image(image)

            baseline_maps = {}
            for baseline_type in BASELINE_TYPES:
                baseline = create_baseline(image, baseline_type)
                attributions, _ = generate_integrated_gradients(
                    input_tensor,
                    baseline,
                    class_id,
                    n_steps=DEFAULT_N_STEPS,
                )
                baseline_maps[baseline_type] = create_heatmap(attributions)

            for baseline_type in BASELINE_TYPES:
                observation = describe_baseline_observation(
                    baseline_maps[baseline_type],
                    baseline_type,
                    confidence_pct,
                )
                rows.append(
                    [
                        case_set,
                        filename,
                        label,
                        f"{confidence_pct:.2f}%",
                        baseline_type.title(),
                        observation,
                    ]
                )

    columns = [
        "Case",
        "Image",
        "Prediction",
        "Confidence",
        "Baseline",
        "Observation",
    ]
    return rows, columns


# -----------------------------------------------------------------------------
# Input blur preview helpers
# -----------------------------------------------------------------------------


def describe_input_blur_preview(original, blurred, radius):
    """Explain blur strength and what image is sent to the model."""
    original_224 = np.array(pil_resize_224(original)).astype(np.float32)
    blurred_224 = np.array(pil_resize_224(blurred)).astype(np.float32)

    mean_diff = np.abs(original_224 - blurred_224).mean()
    corr = float(np.corrcoef(original_224.flatten(), blurred_224.flatten())[0, 1])

    if corr >= 0.95:
        similarity = "very similar"
    elif corr >= 0.85:
        similarity = "similar"
    elif corr >= 0.65:
        similarity = "moderately changed"
    else:
        similarity = "strongly changed"

    return (
        f"**Blur radius: {radius}** (PIL `GaussianBlur`)\n\n"
        "Blur is applied to the **uploaded input image** before preprocessing. "
        "The blurred image is then sent to ResNet18 and Integrated Gradients.\n\n"
        "**Original upload (preview at 224×224)**\n"
        f"- Shape: `{original_224.shape[0]}×{original_224.shape[1]}×3`\n"
        f"- Pixel value range: `{original_224.min():.0f}` to `{original_224.max():.0f}`\n\n"
        "**Blurred input sent to model**\n"
        f"- Same shape `{blurred_224.shape[0]}×{blurred_224.shape[1]}×3`\n"
        f"- Mean absolute pixel change vs original: **{mean_diff:.1f}**\n"
        f"- Pixel correlation with original: **{corr:.2f}** ({similarity})\n\n"
        "Pipeline: blurred image → `preprocess()` → ResNet18 → IG attribution.\n"
        "Higher radius removes more fine detail before classification and explanation."
    )


def blur_preview_from_input(image, blur_input, blur_radius):
    """Build live preview of original vs blurred model input."""
    if not blur_input:
        return (
            False,
            None,
            None,
            "*Blur preview disabled. Enable **Blur input image** to inspect the "
            "model input.*",
        )

    if image is None:
        return (
            True,
            None,
            None,
            "*Upload an image to preview how blur changes the input sent to the model.*",
        )

    radius = int(blur_radius)
    blurred = blur_input_image(image, radius)
    original_preview = pil_resize_224(image)
    blurred_preview = pil_resize_224(blurred)
    info = describe_input_blur_preview(image, blurred, radius)

    return True, original_preview, blurred_preview, info


def on_blur_preview(image, blur_input, blur_radius):
    """Live preview while the blur radius slider moves."""
    if not blur_input:
        return (
            gr.update(visible=False),
            None,
            None,
            "*Blur preview disabled. Enable **Blur input image** to inspect the "
            "model input.*",
        )

    show, original_preview, blurred_preview, info = blur_preview_from_input(
        image,
        blur_input,
        blur_radius,
    )

    return (
        gr.update(visible=show),
        original_preview,
        blurred_preview,
        info,
    )


def on_blur_controls(image, blur_input, blur_radius):
    """Toggle blur controls and refresh the live input preview."""
    preview = on_blur_preview(image, blur_input, blur_radius)
    return (gr.update(visible=blur_input), *preview)


# -----------------------------------------------------------------------------
# Gradio handlers
# -----------------------------------------------------------------------------


def on_explain(image, baseline, blur_input, blur_radius, n_steps):
    """Handle Explain / Classify button click."""
    empty_preview = (gr.update(visible=False), None, None, "")

    if image is None:
        return (
            "Please upload an image.",
            "",
            "",
            None,
            None,
            None,
            "",
            *empty_preview,
        )

    baseline_key = baseline.lower()
    n_steps = int(n_steps)
    result = explain_image(
        image,
        baseline_key,
        blur_input=blur_input,
        blur_radius=int(blur_radius),
        n_steps=n_steps,
    )

    show, original_preview, blurred_preview, info = blur_preview_from_input(
        image,
        blur_input,
        blur_radius,
    )

    delta_text = format_convergence_delta(
        result["delta"],
        result["n_steps"],
        result["baseline_type"],
        result["label"],
    )

    return (
        result["label"],
        f"{result['confidence_pct']:.2f}%",
        delta_text,
        result["original"],
        result["heatmap"],
        result["overlay"],
        format_top_predictions(result["top_predictions"]),
        gr.update(visible=show),
        original_preview,
        blurred_preview,
        info,
    )


def on_compare_baselines(image):
    """Handle baseline comparison request."""
    if image is None:
        return None, "Please upload an image to compare baselines."

    result = explain_all_baselines(image)
    return result["panel"], result["comparison_text"]


def on_refresh_results():
    """Rebuild the experiment results table."""
    if not test_images_available():
        return (
            [],
            "Test images folder not found. Add ResNet18_Test_Images/ to generate results.",
        )

    rows, _ = build_results_table()
    return rows, f"Generated {len(rows)} result rows."


# -----------------------------------------------------------------------------
# Course notes markdown
# -----------------------------------------------------------------------------

ANALYSIS_MARKDOWN = f"""
# {PROJECT_NAME}

{PROJECT_TAGLINE}

## About ResNet18

ResNet18 is a convolutional neural network with 18 layers and residual (skip)
connections. Residual connections help deeper networks train and represent
features more effectively.

In this project we use **pretrained ImageNet-1K weights** from torchvision.
ImageNet-1K contains **1000 object categories**. The model has already learned
general visual features from a large dataset.

We perform **inference only** — the model is not trained or fine-tuned in this
project. Given a preprocessed image, ResNet18 outputs class scores for all
1000 categories.

## About Integrated Gradients

Integrated Gradients (IG) is an attribution method from Captum. It estimates
how much each input feature (here, each pixel channel) contributed to the
model's prediction for a selected class.

IG integrates gradients along a straight path from a **baseline input** to the
actual input. Intuitively, it answers: "How much did changing each pixel from
the baseline toward the input affect the predicted class score?"

The integral is approximated numerically using **`n_steps` interpolation
points** (a Riemann sum). Captum also returns **`delta`**, the convergence /
completeness error that checks how well the attributions sum matches the
change in model output for the selected class. **Pixels are the input features;
`n_steps` and `delta` describe the integration procedure, not extra input
features.**

The attribution map shows **contribution toward the selected prediction**, not
absolute importance in isolation. Higher values indicate regions that pushed
the model toward that class logit.

## About Baselines

A **baseline** (reference input) represents a neutral starting point. Integrated
Gradients requires this reference because attribution is measured relative to
that starting point.

In this project we use three solid-color RGB baselines:

| Baseline | RGB Color | Role |
|----------|-----------|------|
| Black | (0, 0, 0) | Strong absence-of-signal reference |
| Gray | (128, 128, 128) | Mid-intensity neutral reference |
| White | (255, 255, 255) | Bright neutral reference |

Each baseline image is preprocessed with the **same pipeline** as the input
image. Changing the baseline can change the attribution map because IG measures
change **from that specific reference** to the input.

Similar patterns across baselines suggest stable explanations; large differences
suggest the reference choice materially affects the attribution.

## Model Focus Analysis

The heatmap and overlay should be **visually inspected** to study where the
model's prediction comes from.

Useful inspection questions:

- Do high-attribution regions overlap with the main object in the image?
- Does the model also highlight background or context regions?
- For challenging cases (sunglasses, camouflage, unusual context, sketches),
  is attribution spread, fragmented, or concentrated?

**Important:** This application does not automatically decide whether the model
is "correct" or "wrong" in what it focuses on. Attribution shows what influenced
the prediction, and relevance must be judged by inspecting the image and heatmap
together.
"""


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------


def build_ui():
    """Build the Gradio application."""
    load_model()

    with gr.Blocks(
        title=f"{PROJECT_NAME} | ResNet18 + Integrated Gradients",
    ) as demo:
        gr.Markdown(
            f"# {PROJECT_NAME}\n"
            f"{PROJECT_TAGLINE}\n\n"
            "Upload an image, choose a baseline, and inspect model predictions "
            "with Captum Integrated Gradients explanations."
        )

        with gr.Tabs():
            # -----------------------------------------------------------------
            # Tab 1: Explain Image
            # -----------------------------------------------------------------
            with gr.Tab("Explain Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                        )

                        baseline_input = gr.Radio(
                            choices=["Black", "Gray", "White"],
                            value="Black",
                            label="Baseline",
                        )

                        n_steps_input = gr.Dropdown(
                            choices=INTEGRATION_STEP_CHOICES,
                            value=DEFAULT_N_STEPS,
                            label="Integration Steps (n_steps)",
                            info="Number of interpolation points for IG integration.",
                        )

                        blur_input_checkbox = gr.Checkbox(
                            label="Blur input image",
                            value=False,
                            info="Blur the uploaded image before ResNet18 and IG.",
                        )
                        blur_radius_input = gr.Slider(
                            minimum=1,
                            maximum=15,
                            step=1,
                            value=5,
                            label="Blur radius",
                            visible=False,
                        )

                        with gr.Group(visible=False) as blur_preview_group:
                            gr.Markdown("### Live Input Blur Preview")
                            blur_info_output = gr.Markdown()
                            with gr.Row():
                                blur_raw_preview = gr.Image(
                                    label="Original upload (224×224 preview)",
                                    type="pil",
                                )
                                blur_out_preview = gr.Image(
                                    label="Blurred input sent to model",
                                    type="pil",
                                )

                        explain_btn = gr.Button(
                            "Explain / Classify",
                            variant="primary",
                        )

                        gr.Markdown("### Optional: Provided Test Images")

                        case_dropdown = gr.Dropdown(
                            choices=list(TEST_IMAGES.keys()),
                            value="Good Cases",
                            label="Test Case Set",
                        )
                        sample_dropdown = gr.Dropdown(
                            choices=TEST_IMAGES["Good Cases"],
                            value=TEST_IMAGES["Good Cases"][0],
                            label="Sample Image",
                        )

                        if not test_images_available():
                            gr.Markdown(
                                "*Warning: `ResNet18_Test_Images/` was not found. "
                                "Upload any compatible image, or add the test folders.*"
                            )

                    with gr.Column(scale=2):
                        prediction_output = gr.Textbox(
                            label="Predicted Class",
                        )
                        confidence_output = gr.Textbox(
                            label="Confidence",
                        )
                        convergence_delta_output = gr.Markdown(
                            label="IG Convergence Delta",
                        )
                        gr.Markdown("### Top 5 Predictions")
                        top5_output = gr.Markdown()

                        with gr.Row():
                            original_output = gr.Image(
                                label="Model Input (224×224)",
                                type="pil",
                            )
                            heatmap_output = gr.Image(
                                label="Integrated Gradients Heatmap",
                                type="pil",
                            )
                            overlay_output = gr.Image(
                                label="IG Overlay",
                                type="pil",
                            )

                case_dropdown.change(
                    list_test_images,
                    inputs=[case_dropdown],
                    outputs=[sample_dropdown],
                )

                sample_dropdown.change(
                    load_test_image,
                    inputs=[case_dropdown, sample_dropdown],
                    outputs=[image_input],
                )

                blur_preview_inputs = [
                    image_input,
                    blur_input_checkbox,
                    blur_radius_input,
                ]
                blur_preview_outputs = [
                    blur_preview_group,
                    blur_raw_preview,
                    blur_out_preview,
                    blur_info_output,
                ]

                blur_input_checkbox.change(
                    on_blur_controls,
                    inputs=blur_preview_inputs,
                    outputs=[
                        blur_radius_input,
                        *blur_preview_outputs,
                    ],
                )

                blur_radius_input.change(
                    on_blur_preview,
                    inputs=blur_preview_inputs,
                    outputs=blur_preview_outputs,
                )

                image_input.change(
                    on_blur_preview,
                    inputs=blur_preview_inputs,
                    outputs=blur_preview_outputs,
                )

                explain_btn.click(
                    on_explain,
                    inputs=[
                        image_input,
                        baseline_input,
                        blur_input_checkbox,
                        blur_radius_input,
                        n_steps_input,
                    ],
                    outputs=[
                        prediction_output,
                        confidence_output,
                        convergence_delta_output,
                        original_output,
                        heatmap_output,
                        overlay_output,
                        top5_output,
                        blur_preview_group,
                        blur_raw_preview,
                        blur_out_preview,
                        blur_info_output,
                    ],
                )

            # -----------------------------------------------------------------
            # Tab 2: Compare Baselines
            # -----------------------------------------------------------------
            with gr.Tab("Compare Baselines"):
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_image_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                        )
                        compare_case_dropdown = gr.Dropdown(
                            choices=list(TEST_IMAGES.keys()),
                            value="Good Cases",
                            label="Test Case Set",
                        )
                        compare_sample_dropdown = gr.Dropdown(
                            choices=TEST_IMAGES["Good Cases"],
                            value=TEST_IMAGES["Good Cases"][0],
                            label="Sample Image",
                        )
                        compare_btn = gr.Button(
                            "Compare Baselines",
                            variant="primary",
                        )

                    with gr.Column(scale=2):
                        compare_panel_output = gr.Image(
                            label="Baseline Comparison",
                            type="pil",
                        )
                        compare_text_output = gr.Markdown()

                compare_case_dropdown.change(
                    list_test_images,
                    inputs=[compare_case_dropdown],
                    outputs=[compare_sample_dropdown],
                )

                compare_sample_dropdown.change(
                    load_test_image,
                    inputs=[compare_case_dropdown, compare_sample_dropdown],
                    outputs=[compare_image_input],
                )

                compare_btn.click(
                    on_compare_baselines,
                    inputs=[compare_image_input],
                    outputs=[compare_panel_output, compare_text_output],
                )

            # -----------------------------------------------------------------
            # Tab 3: Course Notes & Results
            # -----------------------------------------------------------------
            with gr.Tab("Course Notes & Results"):
                gr.Markdown(ANALYSIS_MARKDOWN)

                gr.Markdown("## Experiment Results")

                refresh_btn = gr.Button("Refresh Results Table", variant="secondary")
                results_status = gr.Markdown()

                results_table = gr.Dataframe(
                    headers=[
                        "Case",
                        "Image",
                        "Prediction",
                        "Confidence",
                        "Baseline",
                        "Observation",
                    ],
                    datatype=["str"] * 6,
                    interactive=False,
                    wrap=True,
                )

                refresh_btn.click(
                    on_refresh_results,
                    inputs=[],
                    outputs=[results_table, results_status],
                )

        # Avoid 30 IG runs on Space startup; user clicks Refresh instead.
        if not IS_HF_SPACE:
            demo.load(
                on_refresh_results,
                inputs=[],
                outputs=[results_table, results_status],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
