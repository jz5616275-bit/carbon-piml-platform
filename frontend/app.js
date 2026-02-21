// frontend/app.js
const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || "http://127.0.0.1:5000/api";

const state = {
  uploadId: null,
  schema: null,
  modeDetected: null,
  scaleUsed: null,
  featureCols: [],
  lastResponse: null,
};

function el(id) {
  return document.getElementById(id);
}

function show(node) {
  node.classList.remove("hidden");
}

function hide(node) {
  node.classList.add("hidden");
}

function setText(id, text) {
  const node = el(id);
  if (!node) return;
  node.textContent = text;
}

function jsonPretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function fmtNum(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function setStatus(boxId, text, isError = false) {
  const box = el(boxId);
  if (!box) return;
  box.textContent = text;
  box.style.borderColor = isError ? "rgba(255,91,122,0.55)" : "rgba(255,255,255,0.12)";
}

function getJsonHeaders() {
  return { "Content-Type": "application/json" };
}

async function uploadCsv(file) {
  const form = new FormData();
  form.append("file", file);

  const resp = await fetch(`${API_BASE}/uploads`, {
    method: "POST",
    body: form,
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || "Upload failed.");
  return data;
}

function inferFrontMode(uploadResp) {
  const schema = uploadResp.schema || uploadResp.detected_schema || {};
  const featureCols = (schema.feature_cols || schema.featureCols || []) || [];
  const mode = uploadResp.mode_detected || uploadResp.modeDetected || (featureCols.length ? "advanced" : "basic");
  const scale = uploadResp.scale_used || uploadResp.scaleUsed || "monthly";
  return { schema, featureCols, mode, scale };
}

function rebuildDisturbUI() {
  const enabled = el("disturbEnabled").checked;
  el("disturbPanel").classList.toggle("disabled", !enabled);

  hide(el("basicDisturb"));
  hide(el("advancedDisturb"));

  if (!enabled) return;
  if (!state.modeDetected) return;

  if (state.modeDetected === "basic") {
    show(el("basicDisturb"));
  } else {
    show(el("advancedDisturb"));
    buildFeatureSliders(state.featureCols);
  }
}

function buildFeatureSliders(cols) {
  const host = el("featureSliders");
  host.innerHTML = "";

  if (!cols || !cols.length) {
    const div = document.createElement("div");
    div.className = "disturb__hint";
    div.textContent = "No feature columns detected.";
    host.appendChild(div);
    return;
  }

  cols.forEach((c) => {
    const wrap = document.createElement("div");
    wrap.className = "sliderRow";

    const label = document.createElement("div");
    label.className = "sliderRow__label";
    label.textContent = `${c} change (%)`;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-20";
    slider.max = "20";
    slider.value = "0";
    slider.dataset.feature = c;

    const valueBox = document.createElement("div");
    valueBox.className = "sliderRow__value";
    valueBox.textContent = "0%";

    slider.addEventListener("input", () => {
      valueBox.textContent = `${slider.value}%`;
    });

    wrap.appendChild(label);
    wrap.appendChild(slider);
    wrap.appendChild(valueBox);

    const inputLine = document.createElement("div");
    inputLine.className = "row";
    inputLine.style.marginTop = "6px";

    const input = document.createElement("input");
    input.className = "input";
    input.type = "number";
    input.step = "0.1";
    input.min = "-95";
    input.max = "200";
    input.value = "0";
    input.dataset.featureInput = c;

    input.addEventListener("input", () => {
      const v = clamp(Number(input.value || 0), -95, 200);
      slider.value = String(clamp(v, -20, 20));
      valueBox.textContent = `${slider.value}%`;
    });

    const inputLabel = document.createElement("div");
    inputLabel.className = "card__hint";
    inputLabel.style.margin = "0";
    inputLabel.style.width = "220px";
    inputLabel.textContent = "Or type value (%)";
    inputLine.appendChild(inputLabel);
    inputLine.appendChild(input);

    host.appendChild(wrap);
    host.appendChild(inputLine);

    const hr = document.createElement("div");
    hr.style.height = "1px";
    hr.style.background = "rgba(255,255,255,0.10)";
    hr.style.margin = "10px 0";
    host.appendChild(hr);
  });
}

function readDisturbance() {
  const enabled = el("disturbEnabled").checked;
  if (!enabled) return { enabled: false };

  if (!state.modeDetected) return { enabled: false };

  if (state.modeDetected === "basic") {
    const v = Number(el("globalPctInput").value || 0);
    return { enabled: true, global_pct: v / 100.0 };
  }

  const featurePct = {};
  const inputs = document.querySelectorAll("[data-feature-input]");
  inputs.forEach((inp) => {
    const key = inp.dataset.featureInput;
    const v = Number(inp.value || 0);
    featurePct[key] = v / 100.0;
  });
  return { enabled: true, feature_pct: featurePct };
}

function syncBasicDisturbControls() {
  const slider = el("globalPct");
  const num = el("globalPctInput");
  const out = el("globalPctVal");

  slider.addEventListener("input", () => {
    out.textContent = slider.value;
    num.value = slider.value;
  });

  num.addEventListener("input", () => {
    const v = clamp(Number(num.value || 0), -95, 200);
    num.value = String(v);
    slider.value = String(clamp(v, -20, 20));
    out.textContent = slider.value;
  });
}

function readEvaluation() {
  const enabled = el("evalEnabled").checked;
  if (!enabled) return { enabled: false };

  const mode = el("evalMode").value;
  if (mode === "last12") {
    const k = clamp(Number(el("evalK").value || 12), 2, 2000);
    return { enabled: true, split: { mode: "last12", test_points: k } };
  }
  return { enabled: true, split: { mode: "ratio", test_ratio: 0.2 } };
}

function readPhysics() {
  const physicsMode = el("physicsMode").value;
  const nonNegative = el("nonNegativeToggle").checked;
  const maxChangeRate = Number(el("maxChangeRate").value || 0.25);
  const capRaw = el("capValue").value;
  const capValue = capRaw === "" ? null : Number(capRaw);

  return {
    physics_mode: physicsMode,
    physics: {
      non_negative: nonNegative,
      max_change_rate: maxChangeRate,
      cap_value: capValue,
    },
  };
}

async function runPrediction() {
  if (!state.uploadId) throw new Error("Please upload a CSV first.");

  const horizon = Number(el("horizonSelect").value);
  const disturbance = readDisturbance();
  const evaluation = readEvaluation();
  const physics = readPhysics();

  const payload = {
    upload_id: state.uploadId,
    horizon_months: horizon,
    physics_mode: physics.physics_mode,
    physics: physics.physics,
    disturbance,
    evaluation,
    scenario_name: el("scenarioName").value || "",
  };

  const resp = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: getJsonHeaders(),
    body: JSON.stringify(payload),
  });

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || "Prediction failed.");
  return data;
}

function getPidFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get("pid");
}

function setPidInUrl(pid) {
  const url = new URL(window.location.href);
  url.searchParams.set("pid", pid);
  window.history.replaceState({}, "", url.toString());
}

function cacheLastResponse(resp) {
  try {
    if (!resp || !resp.prediction_id) return;
    sessionStorage.setItem("last_prediction_id", resp.prediction_id);
    sessionStorage.setItem(`prediction_cache__${resp.prediction_id}`, JSON.stringify(resp));
  } catch {
    // ignore
  }
}

function readCachedResponse(pid) {
  try {
    const raw = sessionStorage.getItem(`prediction_cache__${pid}`);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

async function fetchPrediction(pid) {
  const resp = await fetch(`${API_BASE}/predictions/${encodeURIComponent(pid)}`, { method: "GET" });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || "Failed to fetch prediction.");
  return data;
}

function pickSeriesFromDoc(doc, view) {
  const outputs = doc.outputs || {};
  const observed = outputs.observed || [];
  const block = outputs[view] || outputs.original || {};
  const baseline = block.baseline || [];
  const piml = block.piml || [];
  return { observed, baseline, piml, block };
}

function computeForecastPhysicsFromBlock(block) {
  // Your backend currently guarantees physics metrics for disturbed only.
  // For original view, we compute quick checks on the front-end so KPIs always show something.
  const baseline = block.baseline || [];
  const piml = block.piml || [];

  function count(series) {
    let neg = 0;
    for (const p of series) {
      const v = Number(p.value);
      if (Number.isFinite(v) && v < 0) neg += 1;
    }
    return { negatives: neg };
  }

  return { baseline: count(baseline), piml: count(piml) };
}

function renderResults(doc) {
  state.lastResponse = doc;

  setText("pidBox", doc.prediction_id || "—");
  setText("modeBox", doc.mode_used || doc.mode_detected || "—");
  setText("methodBox", doc.method || "—");
  setText("rawBox", jsonPretty(doc));

  const hasEval = !!doc.evaluation && !!doc.evaluation.metrics;
  if (hasEval) {
    const b = doc.evaluation.metrics.baseline?.accuracy || {};
    const p = doc.evaluation.metrics.piml?.accuracy || {};

    setText("kpiBaseErr", `RMSE ${fmtNum(b.rmse)} | MAE ${fmtNum(b.mae)} | MAPE ${fmtNum(b.mape)}`);
    setText("kpiPimlErr", `RMSE ${fmtNum(p.rmse)} | MAE ${fmtNum(p.mae)} | MAPE ${fmtNum(p.mape)}`);

    const improve = (b.rmse && p.rmse) ? ((b.rmse - p.rmse) / Math.max(b.rmse, 1e-9)) * 100 : null;
    setText("kpiImprove", improve === null ? "—" : `${fmtNum(improve, 1)}%`);
  } else {
    setText("kpiBaseErr", "—");
    setText("kpiPimlErr", "—");
    setText("kpiImprove", "—");
  }

  const viewSel = el("viewSelect");
  const disturbedExists = !!doc.outputs?.disturbed;
  viewSel.querySelector('option[value="disturbed"]').disabled = !disturbedExists;
  if (!disturbedExists) viewSel.value = "original";

  drawChartForView(viewSel.value);

  setStatus("resultsStatus", "Loaded.", false);
}

function drawChartForView(view) {
  if (!state.lastResponse) return;

  const doc = state.lastResponse;
  const { observed, baseline, piml, block } = pickSeriesFromDoc(doc, view);

  // physics KPI
  const disturbedPhys = block.physics;
  if (disturbedPhys && disturbedPhys.baseline && disturbedPhys.piml) {
    const pb = disturbedPhys.baseline;
    const pp = disturbedPhys.piml;
    setText(
      "kpiPhys",
      `B: neg ${pb.negatives}, cap ${pb.cap_violations}, jump ${pb.jump_violations} | P: neg ${pp.negatives}, cap ${pp.cap_violations}, jump ${pp.jump_violations}`
    );
  } else {
    const quick = computeForecastPhysicsFromBlock(block);
    setText("kpiPhys", `B: neg ${quick.baseline.negatives} | P: neg ${quick.piml.negatives}`);
  }

  const canvas = el("chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  const pad = { l: 50, r: 16, t: 18, b: 40 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;

  const dates = [];
  const map = new Map();

  function add(series, key) {
    series.forEach((p) => {
      const d = p.date;
      if (!map.has(d)) map.set(d, { date: d });
      map.get(d)[key] = Number(p.value);
    });
  }

  add(observed, "obs");
  add(baseline, "base");
  add(piml, "piml");

  Array.from(map.keys()).sort().forEach((d) => dates.push(d));
  const rows = dates.map((d) => map.get(d));

  const values = [];
  rows.forEach((r) => {
    ["obs", "base", "piml"].forEach((k) => {
      if (r[k] !== undefined && Number.isFinite(r[k])) values.push(r[k]);
    });
  });

  if (!values.length) {
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("No data to plot.", 20, 30);
    return;
  }

  const yMin = Math.min(...values);
  const yMax = Math.max(...values);
  const yPad = (yMax - yMin) * 0.08 || 1.0;
  const y0 = yMin - yPad;
  const y1 = yMax + yPad;

  function xAt(i) {
    if (rows.length <= 1) return pad.l;
    return pad.l + (i / (rows.length - 1)) * innerW;
  }
  function yAt(v) {
    const t = (v - y0) / (y1 - y0);
    return pad.t + innerH - t * innerH;
  }

  ctx.strokeStyle = "rgba(255,255,255,0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + innerH);
  ctx.lineTo(pad.l + innerW, pad.t + innerH);
  ctx.stroke();

  const ticks = 4;
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "12px ui-sans-serif";
  for (let i = 0; i <= ticks; i++) {
    const v = y0 + (i / ticks) * (y1 - y0);
    const y = yAt(v);
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + innerW, y);
    ctx.stroke();
    ctx.fillText(fmtNum(v, 2), 6, y + 4);
  }

  function plotLine(key, stroke) {
    let started = false;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((r, i) => {
      const v = r[key];
      if (v === undefined || !Number.isFinite(v)) return;
      const x = xAt(i);
      const y = yAt(v);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (started) ctx.stroke();
  }

  plotLine("obs", "rgba(255,255,255,0.9)");
  plotLine("base", "rgba(58,123,253,0.95)");
  plotLine("piml", "rgba(40,215,201,0.95)");

  const step = Math.max(1, Math.floor(rows.length / 6));
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.font = "11px ui-sans-serif";
  for (let i = 0; i < rows.length; i += step) {
    const x = xAt(i);
    const label = rows[i].date.slice(0, 7);
    ctx.save();
    ctx.translate(x, pad.t + innerH + 18);
    ctx.rotate(-0.35);
    ctx.fillText(label, -14, 0);
    ctx.restore();
  }
}

function downloadText(filename, text, mime = "application/json") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();

  URL.revokeObjectURL(url);
}

function exportJson() {
  if (!state.lastResponse) return;
  downloadText("prediction.json", jsonPretty(state.lastResponse));
}

function exportCsv() {
  if (!state.lastResponse) return;

  const view = el("viewSelect").value;
  const { observed, baseline, piml } = pickSeriesFromDoc(state.lastResponse, view);

  const map = new Map();
  function add(series, key) {
    series.forEach((p) => {
      const d = p.date;
      if (!map.has(d)) map.set(d, { date: d });
      map.get(d)[key] = Number(p.value);
    });
  }
  add(observed, "observed");
  add(baseline, "baseline");
  add(piml, "piml");

  const dates = Array.from(map.keys()).sort();
  const lines = ["date,observed,baseline,piml"];
  dates.forEach((d) => {
    const r = map.get(d);
    const o = r.observed ?? "";
    const b = r.baseline ?? "";
    const p = r.piml ?? "";
    lines.push(`${d},${o},${b},${p}`);
  });

  downloadText(`series_${view}.csv`, lines.join("\n"), "text/csv");
}

/* ---------- Setup page ---------- */

function initSetupPage() {
  el("uploadBtn").addEventListener("click", async () => {
    const file = el("fileInput").files?.[0];
    if (!file) {
      setStatus("runStatus", "Choose a CSV file first.", true);
      return;
    }

    setStatus("runStatus", "Uploading...", false);
    setText("uploadStatus", "Uploading...");

    try {
      const uploadResp = await uploadCsv(file);

      const uploadId = uploadResp.upload_id || uploadResp.id || uploadResp._id;
      if (!uploadId) throw new Error("Upload response missing upload_id.");

      const info = inferFrontMode(uploadResp);

      state.uploadId = uploadId;
      state.schema = info.schema;
      state.featureCols = info.featureCols;
      state.modeDetected = info.mode;
      state.scaleUsed = info.scale;

      setText("uploadStatus", `Uploaded. ID: ${uploadId}`);
      setText("modeStatus", state.modeDetected);
      setText("scaleStatus", state.scaleUsed);
      el("schemaBox").textContent = jsonPretty(state.schema || {});

      rebuildDisturbUI();
      setStatus("runStatus", "Upload done. Configure settings and run.", false);
    } catch (err) {
      setText("uploadStatus", "Upload failed.");
      setStatus("runStatus", err.message || String(err), true);
    }
  });

  el("disturbEnabled").addEventListener("change", rebuildDisturbUI);
  syncBasicDisturbControls();

  el("resetBtn").addEventListener("click", () => {
    el("scenarioName").value = "";
    el("horizonSelect").value = "12";
    el("physicsMode").value = "full";
    el("nonNegativeToggle").checked = true;
    el("maxChangeRate").value = "0.25";
    el("capValue").value = "";

    el("disturbEnabled").checked = false;
    el("globalPct").value = "0";
    el("globalPctInput").value = "0";
    setText("globalPctVal", "0");

    el("evalEnabled").checked = true;
    el("evalMode").value = "ratio";
    el("evalK").value = "12";

    rebuildDisturbUI();
    setStatus("runStatus", "Reset done.", false);
  });

  el("runBtn").addEventListener("click", async () => {
    setStatus("runStatus", "Running...", false);
    try {
      const resp = await runPrediction();
      cacheLastResponse(resp);

      const pid = resp.prediction_id;
      if (!pid) throw new Error("Missing prediction_id in response.");

      // Go to results page with pid
      window.location.href = `./results.html?pid=${encodeURIComponent(pid)}`;
    } catch (err) {
      setStatus("runStatus", err.message || String(err), true);
    }
  });

  rebuildDisturbUI();
  setStatus("runStatus", "Ready.", false);
}

/* ---------- Results page ---------- */

async function initResultsPage() {
  const viewSel = el("viewSelect");
  const refreshBtn = el("refreshBtn");

  async function load(pid) {
    setStatus("resultsStatus", "Loading...", false);

    const cached = readCachedResponse(pid);
    if (cached) {
      renderResults(cached);
      return;
    }

    const doc = await fetchPrediction(pid);
    // GET returns a doc-shaped object; keep cache anyway
    try {
      sessionStorage.setItem(`prediction_cache__${pid}`, JSON.stringify(doc));
    } catch {
      // ignore
    }
    renderResults(doc);
  }

  viewSel.addEventListener("change", () => {
    drawChartForView(viewSel.value);
  });

  el("exportJsonBtn").addEventListener("click", exportJson);
  el("exportCsvBtn").addEventListener("click", exportCsv);

  refreshBtn.addEventListener("click", async () => {
    const pid = getPidFromUrl();
    if (!pid) return;
    try {
      const doc = await fetchPrediction(pid);
      try {
        sessionStorage.setItem(`prediction_cache__${pid}`, JSON.stringify(doc));
      } catch {
        // ignore
      }
      renderResults(doc);
    } catch (err) {
      setStatus("resultsStatus", err.message || String(err), true);
    }
  });

  // Pick pid from URL, otherwise fall back to last prediction id
  let pid = getPidFromUrl();
  if (!pid) {
    pid = sessionStorage.getItem("last_prediction_id");
    if (pid) setPidInUrl(pid);
  }

  if (!pid) {
    setStatus("resultsStatus", "No prediction id. Go to Setup and run first.", true);
    return;
  }

  try {
    await load(pid);
  } catch (err) {
    setStatus("resultsStatus", err.message || String(err), true);
  }
}

/* ---------- Boot ---------- */

function init() {
  const page = document.body.dataset.page;
  if (page === "setup") initSetupPage();
  if (page === "results") initResultsPage();
}

window.addEventListener("DOMContentLoaded", init);