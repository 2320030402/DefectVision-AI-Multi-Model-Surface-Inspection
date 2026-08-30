const API_BASE = "";
const $ = (id) => document.getElementById(id);

async function checkHealth() {
  const dot = $("statusDot"), text = $("statusText");
  if (!dot || !text) return;
  try {
    const r = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!r.ok) throw new Error("Backend unavailable");
    const d = await r.json();
    dot.className = "dot live";
    text.textContent = `backend online · ${String(d.device || "").includes("cuda") ? "GPU" : "CPU"}`;
  } catch (e) {
    dot.className = "dot down";
    text.textContent = "backend unreachable";
  }
}
checkHealth();

const upload = $("upload");
const input = $("fileInput");
const preview = $("previewImg");
const copy = $("uploadCopy");
const run = $("runBtn");
const name = $("fileName");
const output = $("liveResult");
const scan = $("scanline");
const state = $("resultState");
let selectedFile = null;
let latestInspection = null;

if (upload && input) {
  upload.addEventListener("click", () => input.click());
  upload.addEventListener("dragover", (e) => {
    e.preventDefault();
    upload.classList.add("dragging");
  });
  upload.addEventListener("dragleave", () => upload.classList.remove("dragging"));
  upload.addEventListener("drop", (e) => {
    e.preventDefault();
    upload.classList.remove("dragging");
    if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files[0]) selectFile(e.target.files[0]);
  });
}

function selectFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select a JPG or PNG image.");
    return;
  }
  selectedFile = file;
  if (name) name.textContent = file.name.toUpperCase();
  const reader = new FileReader();
  reader.onload = (e) => {
    if (preview) {
      preview.src = e.target.result;
      preview.style.display = "block";
    }
    if (copy) copy.style.display = "none";
  };
  reader.readAsDataURL(file);
  if (run) run.disabled = false;
}

if (run) run.addEventListener("click", runInspection);

async function runInspection() {
  if (!selectedFile) return;
  run.disabled = true;
  run.textContent = "SCANNING…";
  if (scan) scan.classList.add("active");
  if (state) state.textContent = "PROCESSING";
  output.innerHTML = `<div class="processing-state"><div class="processing-spinner"></div><strong>Running inspection pipeline</strong><span>3-model comparison · classification · anomaly screening · Grad-CAM · evidence validation · spatial profile</span></div>`;

  const fd = new FormData();
  fd.append("file", selectedFile);
  try {
    const r = await fetch(`${API_BASE}/predict?model_name=auto&explain=true`, {
      method: "POST",
      body: fd,
      cache: "no-store"
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(e.detail || "Request failed");
    }
    renderResults(await r.json());
    if (state) state.textContent = "COMPLETE";
  } catch (e) {
    output.innerHTML = `<div class="empty error-state"><strong>Inspection failed</strong><br><br>${escapeHtml(e.message)}</div>`;
    if (state) state.textContent = "ERROR";
  } finally {
    if (scan) scan.classList.remove("active");
    run.disabled = false;
    run.textContent = "RUN FULL INSPECTION";
  }
}

function renderResults(d) {
  latestInspection = d;
  let html = "";
  const classes = Object.entries(d.class_probabilities || {}).sort((a, b) => b[1] - a[1]);
  const pass = d.verdict === "PASS";

  const selectedModelLabel = d.model_selection?.selected_display_name || d.model_display_name || d.model_used || "UNKNOWN";
  html += `<div class="verdict ${pass ? "pass" : ""}">
    <div><span class="result-kicker">CLASSIFICATION RESULT · BEST MODEL: ${escapeHtml(selectedModelLabel)}</span><span class="verdict-label">${escapeHtml(formatClass(d.predicted_class))}</span></div>
    <span class="confidence-badge">${pct(d.confidence)}</span>
  </div>`;


  html += `<div class="probability-heading"><span>CLASS PROBABILITIES</span><span>${classes.length} CLASSES · SELECTED MODEL ${escapeHtml(d.model_display_name || d.model_used || "UNKNOWN")}</span></div>`;
  classes.forEach(([n, p], i) => {
    html += `<div class="prob-row ${i === 0 ? "top" : ""}">
      <div class="prob-row-head"><span>${escapeHtml(formatClass(n))}</span><span class="val">${pct(p)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, Number(p) * 100)}%"></div></div>
    </div>`;
  });

  // IMPORTANT: anomaly screening is rendered immediately after classification,
  // making the independent anomaly feature visible without needing to scroll.
  const a = d.anomaly_detection;
  if (a) {
    const sc = String(a.status || "").toUpperCase();
    const cls = sc.includes("HIGH") ? "bad" : sc.includes("MODERATE") ? "warn" : "good";
    html += `<section class="result-section anomaly-section">
      <div class="result-section-title"><span>VISUAL ANOMALY SCREEN</span><span>INDEPENDENT CHECK</span></div>
      <div class="anomaly-summary">
        <div class="anomaly-score-row">
          <div><div class="anomaly-kicker">SCREENING SCORE</div><div class="anomaly-score">${Number(a.score_percent || 0).toFixed(1)}%</div></div>
          <span class="anomaly-status ${cls}">${escapeHtml(a.status || "UNKNOWN")}</span>
        </div>
        <div class="anomaly-description">${escapeHtml(a.interpretation || "")}</div>
        <div class="anomaly-metrics">
          ${card("Anomalous area", `${Number(a.anomalous_area_percent || 0).toFixed(2)}%`)}
          ${card("Method", "Texture + gradient")}
          ${card("Role", "Independent screen")}
        </div>
        <div class="anomaly-note">Independent of the classifier. This screen identifies locally unusual texture or gradient patterns and should be treated as an inspection aid, not a calibrated probability of defectiveness. Red/yellow areas indicate stronger anomaly responses; blue areas indicate weaker responses.</div>
        <div class="anomaly-views">
          <div class="anomaly-view">
            <div class="anomaly-view-title">ANOMALY HEATMAP</div>
            ${img(a.anomaly_heatmap_base64 || d.anomaly_heatmap_base64, "Standalone anomaly heatmap", "anomaly-image")}
          </div>
          <div class="anomaly-view">
            <div class="anomaly-view-title">HEATMAP OVERLAY</div>
            ${img(a.anomaly_overlay_base64 || d.anomaly_overlay_base64, "Anomaly heatmap over surface image", "anomaly-image")}
          </div>
        </div>
      </div>
    </section>`;
  }

  const sev = d.severity_assessment;
  if (sev) {
    const level = String(sev.level || "LOW").toUpperCase();
    const severityClass = level === "HIGH" ? "bad" : level === "MODERATE" ? "warn" : "good";
    const factors = Array.isArray(sev.factors) ? sev.factors : [];
    html += `<section class="result-section severity-section">
      <div class="result-section-title"><span>SEVERITY &amp; PRIORITY ASSESSMENT</span><span>INSPECTION TRIAGE</span></div>
      <div class="severity-summary">
        <div class="severity-top">
          <div>
            <div class="severity-kicker">INSPECTION SEVERITY INDEX</div>
            <div class="severity-score">${Number(sev.score_percent || 0).toFixed(1)}%</div>
          </div>
          <div class="severity-badges">
            <span class="severity-level ${severityClass}">${escapeHtml(level)}</span>
            <span class="priority-badge">${escapeHtml(sev.priority || "P3")}</span>
          </div>
        </div>
        <div class="severity-bar"><div class="severity-bar-fill ${severityClass}" style="width:${Math.min(100, Math.max(0, Number(sev.score_percent || 0)))}%"></div></div>
        <div class="severity-action"><strong>${escapeHtml(sev.action || "Routine monitoring recommended")}</strong></div>
        <div class="severity-description">${escapeHtml(sev.interpretation || "")}</div>
        <div class="severity-factors">${factors.map(f => `<span>• ${escapeHtml(f)}</span>`).join("")}</div>
        <div class="severity-note">This is an explainable inspection-priority index derived from the existing anomaly, area, evidence-consistency, and classifier signals. It is not a physical damage measurement or calibrated probability of failure.</div>
      </div>
    </section>`;
  }

  const ec = d.evidence_consistency;
  if (ec) {
    const cls = ec.status === "CONSISTENT" ? "good" : ec.status === "INCONSISTENT" ? "bad" : "warn";
    html += `<section class="result-section">
      <div class="result-section-title"><span>EVIDENCE CONSISTENCY</span><span>${escapeHtml(ec.status || "EXPERIMENTAL")}</span></div>
      <div class="evidence-summary">
        <div class="evidence-score-row"><span class="evidence-score">${Number(ec.score_percent || 0).toFixed(1)}%</span><span class="evidence-status ${cls}">${escapeHtml(ec.status || "EXPERIMENTAL")}</span></div>
        <div class="evidence-description">${escapeHtml(ec.interpretation || "")}</div>
        <div class="evidence-metrics">${metric("Original", ec.original_confidence)}${metric("Preserved", ec.evidence_preserved_confidence)}${metric("Removed", ec.evidence_removed_confidence)}</div>
        <div class="evidence-note">The highlighted Grad-CAM region is preserved in one image and suppressed in another. The model is rerun to measure how strongly the prediction depends on that region.</div>
        <div class="evidence-views">${view("Evidence preserved", d.evidence_preserved_base64)}${view("Evidence removed", d.evidence_removed_base64)}</div>
      </div>
    </section>`;
  }

  const ep = d.evidence_profile;
  if (ep) {
    html += `<section class="result-section">
      <div class="result-section-title"><span>INSPECTION EVIDENCE PROFILE</span><span>SPATIAL ANALYSIS</span></div>
      <div class="profile-grid">${card("Pattern", ep.pattern)}${card("Location", ep.location)}${card("Evidence area", `${Number(ep.area_percent || 0).toFixed(1)}%`)}${card("Orientation", `${Number(ep.orientation_degrees || 0).toFixed(1)}°`)}</div>
      <div class="profile-interpretation">${escapeHtml(ep.interpretation || "")}</div>
      <div class="profile-note">Spatial profile combines Grad-CAM model attention with the independent anomaly response. The outline marks the resulting inspection evidence regions; it is not a pixel-accurate defect mask.</div>
      ${img(ep.profile_overlay_base64, "Combined spatial evidence profile", "profile-overlay")}
    </section>`;
  }

  if (d.gradcam_overlay_base64) {
    html += `<section class="result-section">
      <div class="result-section-title"><span>GRAD-CAM ATTENTION</span><span>MODEL FOCUS</span></div>
      <div class="image-caption">Heatmap shows regions that most influenced the selected classifier output; it is not a pixel-accurate defect mask.</div>
      ${img(d.gradcam_overlay_base64, "Grad-CAM attention overlay", "gradcam-image")}
    </section>`;
  }

  html += `<div class="report-actions">
    <button class="report-btn" type="button" onclick="exportInspectionReport()">EXPORT INSPECTION REPORT</button>
    <span class="report-hint">Self-contained HTML report · printable to PDF</span>
  </div>`;

  output.innerHTML = html;
}

function exportInspectionReport() {
  const d = latestInspection;
  if (!d) return;
  const a = d.anomaly_detection || {};
  const ec = d.evidence_consistency || {};
  const sev = d.severity_assessment || {};
  const ep = d.evidence_profile || {};
  const generated = new Date().toLocaleString();
  const esc = escapeHtml;
  const image = (b, label) => b ? `<figure><figcaption>${esc(label)}</figcaption><img src="data:image/png;base64,${b}" alt="${esc(label)}"></figure>` : "";
  const rows = Object.entries(d.class_probabilities || {}).sort((x,y)=>y[1]-x[1]).map(([n,v]) => `<tr><td>${esc(formatClass(n))}</td><td>${pct(v)}</td></tr>`).join("");
  const factors = (sev.factors || []).map(x => `<li>${esc(x)}</li>`).join("");
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Surface Defect Inspection Report</title><style>
  body{font-family:Arial,sans-serif;background:#f4f6f8;color:#18222d;margin:0;padding:32px} .page{max-width:1000px;margin:auto;background:#fff;padding:34px;box-shadow:0 2px 14px #ccd2d8} h1{margin:0 0 6px} h2{margin:28px 0 12px;border-bottom:1px solid #dce2e8;padding-bottom:8px} .muted{color:#657487;font-size:13px} .hero{display:flex;justify-content:space-between;gap:20px;align-items:center;border:1px solid #dce2e8;padding:18px;border-radius:8px} .big{font-size:30px;font-weight:700} .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px} .metric{border:1px solid #dce2e8;padding:12px;border-radius:6px}.label{font-size:10px;color:#68788a;text-transform:uppercase;letter-spacing:.08em}.value{font-size:18px;font-weight:700;margin-top:5px} table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #e4e8ec} figure{margin:0;border:1px solid #dce2e8;padding:8px;border-radius:6px}figure img{width:100%;max-height:420px;object-fit:contain}figcaption{font-size:11px;color:#657487;margin-bottom:7px}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px} li{margin:5px 0}.note{background:#f5f7f9;border-left:3px solid #e0a02f;padding:12px;font-size:12px;color:#59697a}@media print{body{background:#fff;padding:0}.page{box-shadow:none}}
  </style></head><body><div class="page">
  <div class="muted">INDUSTRIAL SURFACE DEFECT INTELLIGENCE · INSPECTION REPORT</div><h1>${esc(formatClass(d.predicted_class || "Unknown"))}</h1><div class="muted">Model: ${esc(d.model_display_name || d.model_used || "Unknown")} · Generated: ${esc(generated)}</div>
  <div class="hero" style="margin-top:18px"><div><div class="label">Classification confidence</div><div class="big">${pct(d.confidence)}</div></div><div><div class="label">Verdict</div><div class="big">${esc(d.verdict || "—")}</div></div></div>
  ${Array.isArray(d.model_comparison) ? `<h2>3-Model Benchmark</h2><p class="muted">The uploaded image was evaluated by all three CNN models. The selected model was ${esc((d.model_selection && d.model_selection.selected_display_name) || d.model_display_name || "Unknown")} using the project-defined confidence + consensus selection score.</p><table><thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th><th>Selection Score</th></tr></thead><tbody>${d.model_comparison.map(m => `<tr><td>${esc(m.display_name)}</td><td>${esc(formatClass(m.predicted_class))}</td><td>${pct(m.confidence)}</td><td>${pct(m.selection_score)}</td></tr>`).join("")}</tbody></table>` : ""}
  <h2>Classification</h2><table><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>${rows}</tbody></table>
  <h2>Visual Anomaly Screening</h2><div class="grid"><div class="metric"><div class="label">Score</div><div class="value">${Number(a.score_percent||0).toFixed(1)}%</div></div><div class="metric"><div class="label">Status</div><div class="value">${esc(a.status||"—")}</div></div><div class="metric"><div class="label">Anomalous area</div><div class="value">${Number(a.anomalous_area_percent||0).toFixed(2)}%</div></div><div class="metric"><div class="label">Method</div><div class="value">Texture + gradient</div></div></div><p class="muted">${esc(a.interpretation||"")}</p><div class="images">${image(a.anomaly_heatmap_base64 || d.anomaly_heatmap_base64,"Anomaly heatmap")}${image(a.anomaly_overlay_base64 || d.anomaly_overlay_base64,"Anomaly overlay")}</div>
  <h2>Severity & Priority</h2><div class="grid"><div class="metric"><div class="label">Severity index</div><div class="value">${Number(sev.score_percent||0).toFixed(1)}%</div></div><div class="metric"><div class="label">Severity</div><div class="value">${esc(sev.level||"—")}</div></div><div class="metric"><div class="label">Priority</div><div class="value">${esc(sev.priority||"—")}</div></div><div class="metric"><div class="label">Action</div><div class="value">${esc(sev.action||"—")}</div></div></div><ul>${factors}</ul>
  <h2>Evidence Consistency</h2><div class="grid"><div class="metric"><div class="label">Score</div><div class="value">${Number(ec.score_percent||0).toFixed(1)}%</div></div><div class="metric"><div class="label">Status</div><div class="value">${esc(ec.status||"—")}</div></div><div class="metric"><div class="label">Original</div><div class="value">${pct(ec.original_confidence)}</div></div><div class="metric"><div class="label">Removed</div><div class="value">${pct(ec.evidence_removed_confidence)}</div></div></div><p class="muted">${esc(ec.interpretation||"")}</p><div class="images">${image(d.evidence_preserved_base64,"Evidence preserved")}${image(d.evidence_removed_base64,"Evidence removed")}</div>
  <h2>Spatial Evidence Profile</h2><div class="grid"><div class="metric"><div class="label">Pattern</div><div class="value">${esc(ep.pattern||"—")}</div></div><div class="metric"><div class="label">Location</div><div class="value">${esc(ep.location||"—")}</div></div><div class="metric"><div class="label">Evidence area</div><div class="value">${Number(ep.area_percent||0).toFixed(1)}%</div></div><div class="metric"><div class="label">Orientation</div><div class="value">${Number(ep.orientation_degrees||0).toFixed(1)}°</div></div></div><p class="muted">${esc(ep.interpretation||"")}</p>${image(ep.profile_overlay_base64,"Combined spatial evidence profile")}
  <h2>Grad-CAM Attention</h2>${image(d.gradcam_overlay_base64,"Grad-CAM attention overlay")}
  <p class="note">This report combines classifier output, independent anomaly screening, severity/priority triage, evidence consistency, spatial evidence, and Grad-CAM. These explainability and triage metrics are inspection aids and are not calibrated probabilities of physical failure.</p>
  </div></body></html>`;
  const blob = new Blob([html], {type:"text/html;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g,"-");
  link.href = url; link.download = `surface_defect_inspection_${stamp}.html`;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function pct(v) { return `${(Number(v || 0) * 100).toFixed(1)}%`; }
function card(l, v) { return `<div class="profile-card"><div class="profile-label">${escapeHtml(l)}</div><div class="profile-value">${escapeHtml(String(v ?? "—"))}</div></div>`; }
function metric(l, v) { return `<div class="evidence-metric"><div class="metric-label">${escapeHtml(l)}</div><div class="metric-value">${pct(v)}</div></div>`; }
function view(t, b) { return b ? `<div class="evidence-view"><div class="evidence-view-title">${escapeHtml(t)}</div><img src="data:image/png;base64,${b}" alt="${escapeHtml(t)}"></div>` : ""; }
function img(b, a, c) { return b ? `<div class="image-frame"><img class="${c || ""}" src="data:image/png;base64,${b}" alt="${escapeHtml(a)}"></div>` : ""; }
function formatClass(v) { return String(v || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()); }
function escapeHtml(v) { return String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;"); }
