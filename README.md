# Industrial Surface Defect Intelligence — Complete Full-Stack Project

Full-stack local inspection project. The application has **three UI pages**: Home, Detection, and About. There is **no separate Results page**; all inspection results are rendered directly inside Detection.

## Implemented features

The Detection page presents the complete inspection pipeline in one scrollable result panel. The independent anomaly screen is intentionally placed directly after the classifier probability distribution so it is visible as a first-class inspection stage.

1. **Six-class defect classification** using the included ResNet18, MobileNetV2 and Custom CNN checkpoints.
2. **Independent visual anomaly screening** using a training-free robust local texture-residual + gradient method. The score is based on extreme-response strength and anomalous area rather than a forced top-percentile score. It reports a screening score, unusual-area percentage, status, standalone anomaly heatmap, and heatmap overlay.
3. **Grad-CAM explainability** for the selected model and predicted class.
4. **Evidence Consistency Test**: preserves the Grad-CAM evidence region, suppresses it in a second image, reruns the classifier, and reports original/preserved/removed confidence plus an experimental consistency score.
5. **Spatial Evidence Profile**: combines Grad-CAM attention with the independent anomaly response to report evidence pattern, approximate location, area and orientation, with a spatial profile overlay. The outline is an inspection visualization, not a pixel-accurate defect mask.
6. **Single-page detection workflow**: upload → classify → anomaly screen → explain → validate → spatial profile.
7. **Three model choices**: `resnet18`, `mobilenet_v2`, `custom_cnn`.
8. **Backend health indicator** showing CPU/GPU availability.
9. **FastAPI serves the frontend**, so only one local server is required.

## Run on Windows

Open PowerShell in the `backend` folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open:

`http://localhost:8000/`

Do **not** open the HTML files directly from File Explorer.

## Checkpoints

The included files are:

- `resnet18_month2_final.pth`
- `mobilenet_v2_best.pth`
- `custom_cnn_best.pth`

## Important interpretation note

The anomaly detector is an **independent visual screening layer**, not a trained/calibrated anomaly probability model. The evidence-consistency score is also an experimental intervention metric. These should be presented as inspection aids unless they are further validated experimentally.


## Team members
- T. Nishkha — 2320030139 — Frontend & UI Developer
- PLV. Abhiram — 2320030294 — Backend & API Developer
- G. SaiAbhiRam Reddy — 2320030402 — Machine Learning & Computer Vision Developer

## Anomaly heatmap
The `/predict` response now exposes the anomaly heatmap both inside `anomaly_detection` and as backward-compatible top-level fields. The frontend renders both the standalone anomaly heatmap and the heatmap overlay separately from Grad-CAM.


## Added feature: Severity & Priority Assessment

The detection page now includes an explainable inspection-priority index derived from the existing anomaly strength, anomalous area, evidence consistency, and classifier confidence signals. It reports LOW/MODERATE/HIGH severity, P1/P2/P3 priority, action guidance, and contributing factors. This is a triage aid, not a calibrated probability or physical damage measurement.

## New Feature — Inspection Report Export

After a completed inspection, use **EXPORT INSPECTION REPORT** in the results panel to generate a self-contained HTML report containing:
- classifier result and class probabilities
- anomaly score, status, heatmap, and overlay
- severity index, severity level, priority, action, and factors
- evidence consistency metrics and preserved/removed evidence views
- spatial evidence profile
- Grad-CAM attention overlay

The generated HTML can be printed/saved as PDF directly from the browser. No external reporting service is required.

## Automatic 3-Model Inspection

The Detection station now runs each uploaded image through all three registered CNN checkpoints automatically: ResNet18, MobileNetV2, and Custom CNN. Their predicted class, confidence, and project-defined selection score are displayed in a three-model benchmark. The highest-scoring model becomes the selected model for the detailed inspection pipeline (Grad-CAM, evidence consistency, spatial evidence, severity/priority) and for the exported inspection report.

The selection score is a project-specific ranking aid based primarily on classification confidence with a modest consensus bonus. It is not a calibrated probability of model correctness.
