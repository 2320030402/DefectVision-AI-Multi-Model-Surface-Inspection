# DefectVision AI — Multi-Model Surface Inspection

DefectVision AI is an AI-based industrial surface defect inspection system.

It checks a surface image using three different deep learning models and automatically selects the best model. It also provides visual explanations and additional inspection information.

## Features

- Automatic inspection using 3 AI models:
  - ResNet18
  - MobileNetV2
  - Custom CNN
- Six-class surface defect classification
- Automatic best-model selection
- Anomaly heatmap
- Severity and priority assessment
- Grad-CAM visualization
- Evidence consistency check
- Spatial evidence profile
- Complete class probability results
- Inspection report export
- Backend health status

## Inspection Workflow

```text
Upload Image
     ↓
3 Model Classification
     ↓
Select Best Model
     ↓
Anomaly Heatmap
     ↓
Severity / Priority
     ↓
Evidence Consistency
     ↓
Spatial Profile
     ↓
Grad-CAM
     ↓
Export Report
```

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

## Deployment

The application is deployed using Render.

Live application:

`https://defectvision-ai-multi-model-surface-94kz.onrender.com`

## Checkpoints

The included files are:

- `resnet18_month2_final.pth`
- `mobilenet_v2_best.pth`
- `custom_cnn_best.pth`

## Important interpretation note

The anomaly detector is an **independent visual screening layer**, not a trained/calibrated anomaly probability model. The evidence-consistency score is also an experimental intervention metric. These should be presented as inspection aids unless they are further validated experimentally.

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

## Team members
- T. Nishkha — 2320030139 — Frontend & UI Developer
- PLV. Abhiram — 2320030294 — Backend & API Developer
- G. SaiAbhiRam Reddy — 2320030402 — Machine Learning & Computer Vision Developer

## Future Improvements
- Improve model accuracy
- Add more defect classes
- Add larger datasets
- Improve anomaly detection
- Add batch image inspection
- Improve report generation
- Add inspection history
- Add database support
- Improve cloud performance
