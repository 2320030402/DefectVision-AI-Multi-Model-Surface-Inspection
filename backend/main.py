"""
Capstone-1 Backend — Industrial Surface Defect Detection API

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health           -> service status + which model is loaded
    GET  /models            -> list checkpoint files found in ./checkpoints
    POST /predict            -> multipart image upload -> prediction + Grad-CAM + evidence consistency test
"""
import io
import os
import base64

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from model_defs import MODEL_REGISTRY, GRADCAM_TARGET_LAYER, CLASSES

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL", "resnet18")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = FastAPI(title="Surface Defect Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your actual frontend origin before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_loaded_models = {}  # cache: model_name -> torch.nn.Module


def _checkpoint_path(model_name: str) -> str:
    # Accepts either "resnet18" (-> resnet18_best.pth) or an exact filename
    candidate = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(CHECKPOINT_DIR, f"{model_name}_month2_final.pth")
    if os.path.exists(candidate2):
        return candidate2
    raise FileNotFoundError(
        f"No checkpoint found for '{model_name}' in {CHECKPOINT_DIR}. "
        f"Expected {model_name}_best.pth or {model_name}_month2_final.pth"
    )


def load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. "
                                                      f"Choose from {list(MODEL_REGISTRY.keys())}")
    try:
        ckpt_path = _checkpoint_path(model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model = MODEL_REGISTRY[model_name]()
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    _loaded_models[model_name] = model
    return model


def compute_gradcam_map(model, model_name, input_tensor, pred_idx):
    """Compute a normalized Grad-CAM map for the predicted class."""
    activations = {}
    gradients = {}

    target_layer = GRADCAM_TARGET_LAYER[model_name](model)

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        output = model(input_tensor)
        score = output[0, pred_idx]
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    acts = activations["value"][0]
    grads = gradients["value"][0]
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def make_gradcam_overlay(model, model_name, input_tensor, orig_image_np, pred_idx):
    """Return a Grad-CAM overlay and the normalized CAM map."""
    cam = compute_gradcam_map(model, model_name, input_tensor, pred_idx)
    cam_resized = cv2.resize(cam, (orig_image_np.shape[1], orig_image_np.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.55 * orig_image_np + 0.45 * heatmap).astype(np.uint8)
    return overlay, cam_resized


def make_evidence_views(orig_image_np, cam_resized, threshold=0.50):
    """
    Create controlled evidence-preserved and evidence-removed views.

    Pixels in the strongest Grad-CAM region are retained in the preserved view
    and suppressed in the removed view. A blurred version is used for the
    suppressed area to reduce hard-edged masking artifacts.
    """
    cam = np.clip(cam_resized, 0.0, 1.0)
    binary_mask = (cam >= threshold).astype(np.float32)

    # Smooth the binary mask so the intervention is not a hard rectangular cut.
    mask = cv2.GaussianBlur(binary_mask, (0, 0), sigmaX=3)
    mask = np.expand_dims(mask, axis=2)

    # Blur provides a natural-looking baseline while removing fine evidence.
    blurred = cv2.GaussianBlur(orig_image_np, (0, 0), sigmaX=12)

    evidence_preserved = (
        mask * orig_image_np + (1.0 - mask) * blurred
    ).astype(np.uint8)

    evidence_removed = (
        (1.0 - mask) * orig_image_np + mask * blurred
    ).astype(np.uint8)

    return evidence_preserved, evidence_removed


def predict_class_probability(model, image_np, target_idx):
    """Run inference on an RGB numpy image and return the target-class probability."""
    pil = Image.fromarray(image_np.astype(np.uint8), mode="RGB")
    tensor = preprocess(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    return float(probs[target_idx].item()), probs.cpu().numpy()


def robust_scale_map(x: np.ndarray):
    """Robustly normalize a response map to [0, 1] using percentile bounds."""
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def compute_visual_anomaly_map(image_np: np.ndarray):
    """
    Training-free visual anomaly screening.

    The detector estimates how unusual each local texture/gradient response is
    relative to the image itself. Unlike simple percentile normalization, the
    screening score is NOT forced high on every image: it is based on robust
    z-scores and the amount of genuinely extreme response.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Remove slow illumination/background variation before measuring texture.
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0)
    residual = np.abs(gray - local_mean)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    def robust_z(x):
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = max(1.4826 * mad, 1e-4)
        return np.maximum((x - med) / scale, 0.0)

    residual_z = robust_z(residual)
    gradient_z = robust_z(gradient)

    # Texture + edge evidence. Cap extreme outliers so a single pixel cannot
    # dominate the visualization.
    response = 0.65 * np.clip(residual_z / 4.0, 0.0, 1.0) + \
               0.35 * np.clip(gradient_z / 4.0, 0.0, 1.0)
    response = cv2.GaussianBlur(response.astype(np.float32), (0, 0), sigmaX=1.5)

    # A visual-only normalization makes the heatmap readable without affecting
    # the actual screening score.
    lo = float(np.percentile(response, 1))
    hi = float(np.percentile(response, 99))
    anomaly_map = np.clip((response - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # Detect only genuinely extreme responses. The threshold is tied to the
    # robust response distribution rather than always selecting the top 10%.
    raw_threshold = 2.5
    raw_combined = 0.65 * residual_z + 0.35 * gradient_z
    mask = (raw_combined >= raw_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    extreme = raw_combined[raw_combined >= raw_threshold]
    if extreme.size:
        excess_strength = float(np.mean(np.clip((extreme - raw_threshold) / 2.5, 0.0, 1.0)))
    else:
        excess_strength = 0.0
    area_fraction = float(np.mean(mask > 0))

    # The score is a screening index, not a probability. It stays low when
    # there are few extreme pixels and rises when both strength and area grow.
    area_component = min(area_fraction / 0.08, 1.0)
    score = float(np.clip(0.70 * excess_strength + 0.30 * area_component, 0.0, 1.0))

    return anomaly_map, mask, score, raw_threshold

def make_anomaly_heatmap(anomaly_map: np.ndarray):
    """Create a standalone false-color anomaly heatmap.

    Red/yellow regions represent stronger locally unusual texture or gradient
    responses; blue regions represent weaker responses. This is a screening
    visualization and is not a defect segmentation mask.
    """
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(anomaly_map, 0.0, 1.0)), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def make_anomaly_overlay(image_np: np.ndarray, anomaly_map: np.ndarray, mask: np.ndarray):
    heat = cv2.applyColorMap(np.uint8(255 * anomaly_map), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.68 * image_np + 0.32 * heat).astype(np.uint8)

    # Add a thin contour around the detected anomalous region.
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 1)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)



def compute_evidence_profile(cam_resized: np.ndarray, anomaly_map: np.ndarray, anomaly_mask: np.ndarray):
    """Build a spatial evidence profile from model attention + anomaly response.

    The profile is intentionally separate from the Grad-CAM consistency metric.
    It combines the model's attention with the independent anomaly response so
    the displayed spatial outline corresponds to regions that are actually
    highlighted by the inspection pipeline rather than a generic shape.
    """
    cam = np.clip(cam_resized.astype(np.float32), 0.0, 1.0)
    anomaly = np.clip(anomaly_map.astype(np.float32), 0.0, 1.0)

    # Combine both evidence sources. The anomaly detector contributes slightly
    # more so the spatial profile follows visible unusual surface regions.
    evidence_map = 0.45 * cam + 0.55 * anomaly
    evidence_map = cv2.GaussianBlur(evidence_map, (0, 0), sigmaX=2.0)
    evidence_map = robust_scale_map(evidence_map)

    threshold = float(np.percentile(evidence_map, 82))
    mask = (evidence_map >= threshold).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove very small connected components while retaining meaningful
    # defect/anomaly regions.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    min_area = max(12, int(mask.size * 0.00025))
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    mask = cleaned

    area = float(np.mean(mask > 0) * 100.0)
    ys, xs = np.where(mask > 0)
    h, w = mask.shape

    if len(xs) == 0:
        cx, cy = w / 2.0, h / 2.0
        orientation = 0.0
        pattern = "DIFFUSE"
        location = "center"
    else:
        cx, cy = float(xs.mean()), float(ys.mean())
        if len(xs) >= 2:
            coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            cov = np.cov(coords, rowvar=False)
            if np.ndim(cov) == 2 and cov.shape == (2, 2):
                vals, vecs = np.linalg.eigh(cov)
                vec = vecs[:, int(np.argmax(vals))]
                orientation = float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)
            else:
                orientation = 0.0
        else:
            orientation = 0.0

        contours, _ = cv2.findContours((mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        component_areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
        total_area = float(mask.sum())
        largest = component_areas[0] if component_areas else 0.0
        compactness = largest / max(total_area, 1.0)
        count = len(component_areas)

        if compactness >= 0.55 and count <= 3:
            pattern = "CONCENTRATED"
        elif count <= 8:
            pattern = "CLUSTERED"
        else:
            pattern = "DISTRIBUTED"

        horizontal = cx / max(w, 1)
        vertical = cy / max(h, 1)
        hpos = "left" if horizontal < 0.33 else "right" if horizontal > 0.66 else "middle"
        vpos = "top" if vertical < 0.33 else "bottom" if vertical > 0.66 else "center"
        location = f"{vpos}-{hpos}"

    # Render a clean spatial evidence map: dark background, highlighted
    # evidence regions, and contours matching the actual combined mask.
    base = np.full((h, w, 3), 20, dtype=np.uint8)
    heat = cv2.applyColorMap(np.uint8(255 * evidence_map), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    strength = np.expand_dims(evidence_map, axis=2)
    profile = (0.30 * base.astype(np.float32) + 0.70 * heat.astype(np.float32) * (0.35 + 0.65 * strength)).clip(0, 255).astype(np.uint8)

    # Dim low-evidence background so the detected regions remain visually clear.
    low = evidence_map < threshold
    profile[low] = (0.35 * profile[low] + 0.65 * base[low]).astype(np.uint8)

    profile_bgr = cv2.cvtColor(profile, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours((mask * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(profile_bgr, contours, -1, (255, 175, 45), 2)
    cv2.circle(profile_bgr, (int(round(cx)), int(round(cy))), 6, (255, 175, 45), -1)
    profile_overlay = cv2.cvtColor(profile_bgr, cv2.COLOR_BGR2RGB)

    return {
        "pattern": pattern,
        "location": location,
        "area_percent": round(area, 2),
        "orientation_degrees": round(orientation, 1),
        "interpretation": (
            f"{pattern.capitalize()} evidence is concentrated in the {location} area; "
            f"the highlighted evidence occupies {area:.1f}% of the image."
        ),
        "profile_overlay_base64": image_to_base64(profile_overlay),
        "profile_basis": "combined Grad-CAM attention + independent anomaly response",
    }

def anomaly_status(score: float):
    if score >= 0.65:
        return "HIGH ANOMALY"
    if score >= 0.40:
        return "MODERATE ANOMALY"
    return "LOW ANOMALY"

def image_to_base64(img_np: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode overlay image")
    return base64.b64encode(buf).decode("utf-8")



def calculate_evidence_consistency(original_confidence, preserved_confidence, removed_confidence):
    """
    Evidence Consistency Score (0-1).

    A high score is obtained when:
      - the prediction remains strong after preserving the highlighted evidence
      - the prediction weakens substantially after removing the highlighted evidence

    This is a project-specific intervention score, not a calibrated probability
    of correctness.
    """
    eps = 1e-8
    retention = min(1.0, preserved_confidence / max(original_confidence, eps))
    removal_effect = 1.0 - min(1.0, removed_confidence / max(original_confidence, eps))
    score = 0.5 * (retention + removal_effect)
    return float(np.clip(score, 0.0, 1.0))


def evidence_status(score):
    """Conservative UI labels for the experimental evidence-consistency metric."""
    if score >= 0.75:
        return "CONSISTENT"
    if score >= 0.50:
        return "PARTIALLY CONSISTENT"
    return "INCONSISTENT"


def compute_severity_assessment(anomaly_score: float, anomalous_area_percent: float,
                                classification_confidence: float,
                                evidence_consistency: float | None = None,
                                pattern: str | None = None):
    """Compute an explainable inspection-priority index.

    This is a project-specific screening index, not a measurement of physical
    damage depth or production loss. It combines signals already produced by
    the inspection pipeline so the operator can prioritize manual review.
    """
    anomaly_component = float(np.clip(anomaly_score, 0.0, 1.0))
    area_component = float(np.clip(anomalous_area_percent / 10.0, 0.0, 1.0))
    confidence_component = float(np.clip(classification_confidence, 0.0, 1.0))
    evidence_component = (
        float(np.clip(evidence_consistency, 0.0, 1.0))
        if evidence_consistency is not None else 0.5
    )

    # The index prioritizes unusual visual evidence and affected extent while
    # using classifier confidence/evidence support as secondary context.
    score = 0.40 * anomaly_component + 0.25 * area_component + \
            0.20 * evidence_component + 0.15 * confidence_component
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.70:
        level, priority = "HIGH", "P1"
        action = "Immediate manual inspection recommended"
    elif score >= 0.45:
        level, priority = "MODERATE", "P2"
        action = "Review during the next inspection cycle"
    else:
        level, priority = "LOW", "P3"
        action = "Routine monitoring recommended"

    factors = []
    if anomaly_component >= 0.65:
        factors.append("high anomaly strength")
    elif anomaly_component >= 0.40:
        factors.append("moderate anomaly strength")
    else:
        factors.append("low anomaly strength")

    if anomalous_area_percent >= 10:
        factors.append("large affected area")
    elif anomalous_area_percent >= 3:
        factors.append("moderate affected area")
    else:
        factors.append("small affected area")

    if evidence_consistency is not None:
        if evidence_consistency >= 0.75:
            factors.append("strong evidence consistency")
        elif evidence_consistency >= 0.50:
            factors.append("mixed evidence consistency")
        else:
            factors.append("weak evidence consistency")

    if pattern:
        factors.append(f"{pattern.lower()} spatial pattern")

    return {
        "score": round(score, 4),
        "score_percent": round(score * 100.0, 1),
        "level": level,
        "priority": priority,
        "action": action,
        "factors": factors,
        "basis": "anomaly strength + affected area + evidence consistency + classifier confidence",
        "interpretation": (
            f"{level.title()} inspection priority based on the combined visual evidence. "
            "This index supports manual triage; it is not a physical damage measurement."
        ),
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "default_model": DEFAULT_MODEL_NAME,
        "classes": CLASSES,
    }


@app.get("/models")
def list_models():
    available = []
    if os.path.isdir(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith(".pth"):
                available.append(f)
    return {"checkpoint_dir": CHECKPOINT_DIR, "checkpoints_found": available,
            "registered_architectures": list(MODEL_REGISTRY.keys())}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(default="auto", description="auto | resnet18 | mobilenet_v2 | custom_cnn"),
    explain: bool = Query(default=True, description="Include Grad-CAM overlay in the response"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    # Automatic model selection: evaluate the same image with all three
    # registered CNNs, compare their predictions, and select the strongest
    # candidate. The detailed explainability pipeline is then run only for
    # the selected model.
    model_comparison = []
    if model_name == "auto":
        candidates = []
        for candidate_name in MODEL_REGISTRY.keys():
            candidate_model = load_model(candidate_name)
            candidate_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                candidate_logits = candidate_model(candidate_tensor)
                candidate_probs = F.softmax(candidate_logits, dim=1)[0]
            candidate_idx = int(torch.argmax(candidate_probs).item())
            candidate_class = CLASSES[candidate_idx]
            candidate_conf = float(candidate_probs[candidate_idx].item())
            candidates.append({
                "model": candidate_name,
                "display_name": candidate_name.replace("_", " ").upper(),
                "predicted_class": candidate_class,
                "confidence": candidate_conf,
                "class_probabilities": {CLASSES[i]: float(candidate_probs[i].item()) for i in range(len(CLASSES))},
            })

        class_counts = {}
        for item in candidates:
            class_counts[item["predicted_class"]] = class_counts.get(item["predicted_class"], 0) + 1
        majority_class = max(class_counts, key=class_counts.get)
        for item in candidates:
            agreement = 1.0 if item["predicted_class"] == majority_class else 0.0
            # Confidence is the primary signal; agreement provides a modest
            # tie-breaking/consensus bonus rather than pretending to be a
            # calibrated probability of correctness.
            item["selection_score"] = 0.90 * item["confidence"] + 0.10 * agreement

        candidates.sort(key=lambda x: x["selection_score"], reverse=True)
        best = candidates[0]
        selected_model_name = best["model"]
        model_comparison = candidates
        model_name = selected_model_name
    elif model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Choose auto or {list(MODEL_REGISTRY.keys())}")

    model = load_model(model_name)

    input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx].item())

    class_probabilities = {CLASSES[i]: float(probs[i].item()) for i in range(len(CLASSES))}

    response = {
        "model_used": model_name,
        "model_display_name": model_name.replace("_", " ").upper(),
        "model_selection_mode": "automatic" if model_comparison else "manual",
        "predicted_class": pred_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
    }
    if model_comparison:
        response["model_comparison"] = model_comparison
        response["model_selection"] = {
            "selected_model": model_name,
            "selected_display_name": model_name.replace("_", " ").upper(),
            "selection_method": "confidence + 10% consensus bonus",
            "reason": "All three CNN models were evaluated on the same image; the highest project-defined selection score was chosen for the detailed inspection.",
        }

    # Training-free visual anomaly screening. This is independent of the
    # classifier, so it can flag unusual surface texture even when the model
    # confidently maps the image to one of its known classes.
    anomaly_image = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
    anomaly_map, anomaly_mask, anomaly_score, anomaly_threshold = compute_visual_anomaly_map(anomaly_image)
    anomaly_heatmap = make_anomaly_heatmap(anomaly_map)
    anomaly_overlay = make_anomaly_overlay(anomaly_image, anomaly_map, anomaly_mask)
    anomaly_area_percent = float(np.mean(anomaly_mask > 0) * 100.0)

    response["anomaly_detection"] = {
        "score": round(anomaly_score, 4),
        "score_percent": round(anomaly_score * 100.0, 1),
        "status": anomaly_status(anomaly_score),
        "anomalous_area_percent": round(anomaly_area_percent, 2),
        "threshold": round(anomaly_threshold, 4),
        "method": "local texture residual + gradient screening",
        "heatmap_base64": image_to_base64(anomaly_heatmap),
        "overlay_base64": image_to_base64(anomaly_overlay),
        "interpretation": (
            "Strong unusual surface patterns detected; inspect the highlighted region."
            if anomaly_score >= 0.65
            else "Some unusual surface patterns detected; manual inspection is recommended."
            if anomaly_score >= 0.40
            else "No strong unusual texture pattern was detected by the screening layer."
        ),
    }
    response["anomaly_heatmap_base64"] = image_to_base64(anomaly_heatmap)
    response["anomaly_overlay_base64"] = image_to_base64(anomaly_overlay)


    # Initial severity/priority assessment. Evidence consistency is added later
    # when explain=True, once the intervention test has been completed.
    response["severity_assessment"] = compute_severity_assessment(
        anomaly_score,
        anomaly_area_percent,
        confidence,
        evidence_consistency=None,
    )
    response["verdict"] = "PASS" if anomaly_score < 0.40 else "DEFECT DETECTED"

    if explain:
        orig_resized = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE)))
        overlay, cam_resized = make_gradcam_overlay(
            model, model_name, input_tensor, orig_resized, pred_idx
        )
        response["gradcam_overlay_base64"] = image_to_base64(overlay)

        # Experimental feature: challenge the explanation by intervening on the
        # region identified by Grad-CAM and measuring how the same prediction reacts.
        evidence_preserved, evidence_removed = make_evidence_views(
            orig_resized, cam_resized, threshold=0.50
        )

        preserved_confidence, _ = predict_class_probability(
            model, evidence_preserved, pred_idx
        )
        removed_confidence, _ = predict_class_probability(
            model, evidence_removed, pred_idx
        )

        consistency = calculate_evidence_consistency(
            confidence, preserved_confidence, removed_confidence
        )

        response["evidence_consistency"] = {
            "score": consistency,
            "score_percent": round(consistency * 100.0, 1),
            "status": evidence_status(consistency),
            "original_confidence": confidence,
            "evidence_preserved_confidence": preserved_confidence,
            "evidence_removed_confidence": removed_confidence,
            "cam_threshold": 0.50,
            "interpretation": (
                "Prediction is supported by the highlighted evidence region."
                if consistency >= 0.75
                else "Prediction shows mixed dependence on the highlighted evidence region."
                if consistency >= 0.50
                else "Prediction is not strongly dependent on the highlighted evidence region; review recommended."
            ),
        }

        response["evidence_preserved_base64"] = image_to_base64(evidence_preserved)
        response["evidence_removed_base64"] = image_to_base64(evidence_removed)
        response["evidence_profile"] = compute_evidence_profile(cam_resized, anomaly_map, anomaly_mask)

        profile_pattern = response["evidence_profile"].get("pattern")
        response["severity_assessment"] = compute_severity_assessment(
            anomaly_score,
            anomaly_area_percent,
            confidence,
            evidence_consistency=consistency,
            pattern=profile_pattern,
        )

    return response


@app.get("/api")
def api_info():
    return {"message": "Surface Defect Detection API is running. See /health, /models, /predict (POST)."}


# --------------------------------------------------------------------------
# Serve the frontend from this same server/port — avoids needing a second
# process on a second port, which is a common source of connectivity issues
# (Windows Firewall, browser cross-origin quirks, etc.) between two separate
# local servers. With this mount, http://localhost:8000/ serves the UI
# directly, and the frontend's own fetch() calls to /health, /predict, etc.
# are same-origin, so no separate frontend server is needed at all.
# Must be added LAST — routes defined above are matched first, and this
# StaticFiles mount only catches whatever wasn't matched by an explicit route.
# --------------------------------------------------------------------------
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
