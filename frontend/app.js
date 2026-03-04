const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || "http://127.0.0.1:5000/api";
const AUTH0_CFG = (window.APP_CONFIG && window.APP_CONFIG.AUTH0) || null;
const state = {
  uploadId: null,
  schema: null,
  modeDetected: null,
  scaleUsed: null,
  featureCols: [],
  lastResponse: null,
  activeTab: "compare",
  activeSegment: "test_forecast",
  auth0: null,
  authed: false,
  user: null,
  token: null,
};

function el(id) { return document.getElementById(id); }
function show(node) { if (node) node.classList.remove("hidden"); }
function hide(node) { if (node) node.classList.add("hidden"); }
function setText(id, text) {
  const node = el(id);
  if (!node) return;
  node.textContent = text;
}

function setHtml(id, html) {
  const node = el(id);
  if (!node) return;
  node.innerHTML = html;
}

function jsonPretty(obj) { return JSON.stringify(obj, null, 2); }
function fmtNum(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
function setStatus(boxId, text, isError = false) {
  const box = el(boxId);
  if (!box) return;
  box.textContent = text;
  box.style.borderColor = isError ? "rgba(255,91,122,0.55)" : "rgba(255,255,255,0.12)";
}

function getJsonHeaders() { return { "Content-Type": "application/json" }; }

/* Horizon */
function buildHorizonOptions(scale) {
  const s = String(scale || "monthly").toLowerCase();
  if (s === "daily") {
    return [
      { value: 30, label: "30 days" },
      { value: 60, label: "60 days" },
      { value: 90, label: "90 days" },
    ];
  }
  if (s === "yearly") {
    return [
      { value: 5, label: "5 years" },
      { value: 10, label: "10 years" },
      { value: 20, label: "20 years" },
      { value: 30, label: "30 years" },
    ];
  }
  return [
    { value: 3, label: "3 months" },
    { value: 6, label: "6 months" },
    { value: 12, label: "12 months" },
    { value: 24, label: "24 months" },
  ];
}

function refreshHorizonSelect({ keepValue = true } = {}) {
  const sel = el("horizonSelect");
  if (!sel) return;

  const scale = state.scaleUsed || "monthly";
  const opts = buildHorizonOptions(scale);
  const prev = keepValue ? String(sel.value || "") : "";
  sel.innerHTML = "";

  for (const o of opts) {
    const opt = document.createElement("option");
    opt.value = String(o.value);
    opt.textContent = o.label;
    sel.appendChild(opt);
  }

  const stillOk = opts.some((o) => String(o.value) === prev);
  if (stillOk && keepValue) {
    sel.value = prev;
  } else {
    sel.value = String(opts[0].value);
  }

  const hint = el("horizonHint");
  if (hint) {
    const s = String(scale).toLowerCase();
    hint.textContent =
      s === "daily" ? "Daily data: horizon is measured in days."
      : (s === "yearly" ? "Yearly data: horizon is measured in years."
      : "Monthly data: horizon is measured in months.");
  }
}

function resetHorizonToDefault() {
  const sel = el("horizonSelect");
  if (!sel) return;
  const scale = String(state.scaleUsed || "monthly").toLowerCase();
  sel.value = scale === "daily" ? "60" : (scale === "yearly" ? "10" : "12");
}

/* Auth0 */
function auth0ErrCode(err) {
  if (!err) return "";
  if (typeof err === "string") return err;
  return err.error || err.error_description || err.code || err.message || "";
}

function getRedirectUri() {
  const cfg = AUTH0_CFG && AUTH0_CFG.redirectUri;
  if (cfg) return cfg;
  return window.location.origin + window.location.pathname;
}

function authConfigured() {
  return !!AUTH0_CFG && !!AUTH0_CFG.domain && !!AUTH0_CFG.clientId && !!AUTH0_CFG.audience;
}

async function loginInteractive({ signup = false, forceConsent = false } = {}) {
  if (!state.auth0) throw new Error("Auth0 client not ready.");

  const authorizationParams = {
    redirect_uri: getRedirectUri(),
    audience: AUTH0_CFG.audience,
  };

  if (signup) authorizationParams.screen_hint = "signup";
  if (forceConsent) authorizationParams.prompt = "consent";

  await state.auth0.loginWithRedirect({ authorizationParams });
}

async function ensureTokenInteractive() {
  if (!state.auth0) throw new Error("Auth0 client not ready.");
  if (state.token) return state.token;

  const authed = await state.auth0.isAuthenticated();
  state.authed = !!authed;
  if (!state.authed) {
    state.user = null;
    state.token = null;
    throw new Error("Please login first.");
  }

  try {
    state.user = await state.auth0.getUser();
    state.token = await state.auth0.getTokenSilently();
    return state.token;
  } catch (err) {
    const code = auth0ErrCode(err);

    if (String(code).includes("consent_required") || String(code).includes("Consent required")) {
      setStatus("runStatus", "Auth0: consent required. Click Login again to grant consent.", true);
      setStatus("resultsStatus", "Auth0: consent required. Click Login again to grant consent.", true);
      throw err;
    }

    if (String(code).includes("login_required") || String(code).includes("interaction_required")) {
      setStatus("runStatus", "Auth0 session expired. Click Login to continue.", true);
      setStatus("resultsStatus", "Auth0 session expired. Click Login to continue.", true);
      throw err;
    }

    setStatus("runStatus", `Auth0 token error: ${String(code)}`, true);
    setStatus("resultsStatus", `Auth0 token error: ${String(code)}`, true);
    throw err;
  }
}

async function initAuth() {
  bindAuthButtons();

  if (!authConfigured()) {
    setStatus("runStatus", "Auth0 config missing in config.js", true);
    setStatus("resultsStatus", "Auth0 config missing in config.js", true);
    refreshAuthUI();
    return;
  }

  if (!window.auth0 || !window.auth0.createAuth0Client) {
    setStatus("runStatus", "Auth0 SDK not loaded.", true);
    setStatus("resultsStatus", "Auth0 SDK not loaded.", true);
    refreshAuthUI();
    return;
  }

  state.auth0 = await window.auth0.createAuth0Client({
    domain: AUTH0_CFG.domain,
    clientId: AUTH0_CFG.clientId,
    cacheLocation: "localstorage",
    authorizationParams: {
      redirect_uri: getRedirectUri(),
      audience: AUTH0_CFG.audience,
    },
  });

  const qs = new URLSearchParams(window.location.search);
  const hasCallback = qs.has("code") && qs.has("state");
  if (hasCallback) {
    await state.auth0.handleRedirectCallback();
    const url = new URL(window.location.href);
    url.searchParams.delete("code");
    url.searchParams.delete("state");
    window.history.replaceState({}, document.title, url.toString());
  }

  try {
    state.authed = await state.auth0.isAuthenticated();
    if (state.authed) {
      state.user = await state.auth0.getUser();
      try {
        state.token = await state.auth0.getTokenSilently();
      } catch (errTok) {
        state.token = null;
        const code = auth0ErrCode(errTok);
        if (String(code).includes("consent_required") || String(code).includes("Consent required")) {
          setStatus("runStatus", "Auth0: consent required. Click Login again to grant consent.", true);
          setStatus("resultsStatus", "Auth0: consent required. Click Login again to grant consent.", true);
        } else {
          setStatus("runStatus", `Auth0 token error: ${String(code)}`, true);
          setStatus("resultsStatus", `Auth0 token error: ${String(code)}`, true);
        }
      }
    } else {
      state.user = null;
      state.token = null;
    }
  } finally {
    refreshAuthUI();
  }
}

function bindAuthButtons() {
  const loginBtn = el("loginBtn");
  const signupBtn = el("signupBtn");
  const logoutBtn = el("logoutBtn");

  if (loginBtn && !loginBtn.dataset.bound) {
    loginBtn.dataset.bound = "1";
    loginBtn.addEventListener("click", async () => {
      try {
        const forceConsent = state.authed && !state.token;
        await loginInteractive({ signup: false, forceConsent });
      } catch (e) {
        setStatus("runStatus", (e && e.message) || String(e), true);
        setStatus("resultsStatus", (e && e.message) || String(e), true);
      }
    });
  }

  if (signupBtn && !signupBtn.dataset.bound) {
    signupBtn.dataset.bound = "1";
    signupBtn.addEventListener("click", async () => {
      try {
        await loginInteractive({ signup: true, forceConsent: false });
      } catch (e) {
        setStatus("runStatus", (e && e.message) || String(e), true);
        setStatus("resultsStatus", (e && e.message) || String(e), true);
      }
    });
  }

  if (logoutBtn && !logoutBtn.dataset.bound) {
    logoutBtn.dataset.bound = "1";
    logoutBtn.addEventListener("click", async () => {
      try {
        if (!state.auth0) return;
        state.auth0.logout({ logoutParams: { returnTo: getRedirectUri() } });
      } catch (e) {
        setStatus("runStatus", (e && e.message) || String(e), true);
        setStatus("resultsStatus", (e && e.message) || String(e), true);
      }
    });
  }
}

function refreshAuthUI() {
  const label = el("authUserLabel");
  const loginBtn = el("loginBtn");
  const signupBtn = el("signupBtn");
  const logoutBtn = el("logoutBtn");

  if (state.authed && state.user) {
    const who = state.user.email || state.user.name || state.user.nickname || state.user.sub || "User";
    if (label) label.textContent = who;

    hide(loginBtn);
    hide(signupBtn);
    show(logoutBtn);
  } else {
    if (label) label.textContent = "Not logged in";

    show(loginBtn);
    show(signupBtn);
    hide(logoutBtn);
  }

  const uploadBtn = el("uploadBtn");
  const runBtn = el("runBtn");
  if (uploadBtn) uploadBtn.disabled = !state.authed;
  if (runBtn) runBtn.disabled = !state.authed;

  const page = document.body.dataset.page;
  if (!state.authed) {
    if (page === "setup") setStatus("runStatus", "Please login to upload and run.", true);
    if (page === "results") setStatus("resultsStatus", "Please login to view results.", true);
  }
}

/* API */
async function apiFetch(path, options = {}) {
  if (!state.auth0) throw new Error("Auth0 not ready. Check config.js and SDK loading.");
  await ensureTokenInteractive();
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${state.token}`);
  const isForm = options.body && options.body instanceof FormData;
  if (!isForm) {
    if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  }

  const resp = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await resp.json().catch(() => ({}));

  if (!resp.ok) {
    const msg = data.error || data.message || `Request failed (${resp.status}).`;
    const hint = data.hint ? `\nHint: ${data.hint}` : "";
    const details = data.details ? `\nDetails: ${data.details}` : "";
    throw new Error(msg + hint + details);
  }
  return data;
}

/* Upload */
async function uploadCsv(file) {
  const form = new FormData();
  form.append("file", file);

  return await apiFetch("/uploads", {
    method: "POST",
    body: form,
  });
}

function inferFrontMode(uploadResp) {
  const schema = uploadResp.schema || uploadResp.detected_schema || {};
  const featureCols = (schema.feature_cols || schema.featureCols || []) || [];
  const mode = uploadResp.mode_detected || uploadResp.modeDetected || (featureCols.length ? "advanced" : "basic");
  const scale = uploadResp.scale_used || uploadResp.scaleUsed || "monthly";
  return { schema, featureCols, mode, scale };
}

/* Recent runs */
async function fetchRecentPredictions(limit = 5) {
  const n = clamp(Number(limit || 5), 1, 50);
  return await apiFetch(`/predictions?limit=${encodeURIComponent(String(n))}`, { method: "GET" });
}

function fmtShortDate(iso) {
  if (!iso) return "-";
  try {
    const d = new Date(iso);
    return d.toISOString().split("T")[0];
  } catch {
    return String(iso);
  }
}

function renderRecentRuns(items) {
  const host = el("recentRunsList");
  const status = el("recentRunsStatus");
  if (!host) return;

  host.innerHTML = "";
  const list = Array.isArray(items) ? items : [];
  if (!list.length) {
    host.innerHTML = `
      <div class="recentRuns__empty">
        No runs yet. Run a prediction to see history here.
      </div>
    `;
    if (status) status.textContent = "—";
    return;
  }

  const frag = document.createDocumentFragment();
  list.forEach((it, idx) => {
    const pid = it.prediction_id || "";
    const when = fmtShortDate(it.created_at);
    const scenario = (it.scenario_name || "").trim();
    const title = scenario ? scenario : `Run #${idx + 1}`;
    const a = document.createElement("a");
    a.className = "recentRun";
    a.href = `./results.html?pid=${encodeURIComponent(pid)}`;
    const left = document.createElement("div");
    left.className = "recentRun__left";
    const t = document.createElement("div");
    t.className = "recentRun__title";
    t.textContent = title;
    const meta = document.createElement("div");
    meta.className = "recentRun__meta";
    meta.textContent = when;
    const chips = document.createElement("div");
    chips.className = "recentRun__chips";
    const mkChip = (text, kind = "soft") => {
      const s = document.createElement("span");
      s.className = `chip chip--${kind}`;
      s.textContent = text;
      return s;
    };

    chips.appendChild(mkChip(it.scale_used || "—"));
    chips.appendChild(mkChip(it.mode_used || "—"));
    chips.appendChild(mkChip(it.physics_mode || "—"));
    const adj = Number(it.num_adjusted ?? 0);
    chips.appendChild(mkChip(`adjusted ${adj}`, adj > 0 ? "ok" : "muted"));
    left.appendChild(t);
    left.appendChild(meta);
    left.appendChild(chips);
    const right = document.createElement("div");
    right.className = "recentRun__right";
    const pill = document.createElement("div");
    pill.className = "pidPill";
    pill.textContent = pid ? `${String(pid).slice(0, 8)}…` : "—";
    right.appendChild(pill);
    a.appendChild(left);
    a.appendChild(right);
    frag.appendChild(a);
  });

  host.appendChild(frag);
  if (status) {
    status.textContent = `Showing last ${list.length} runs.`;
  }
}

async function refreshRecentRuns() {
  if (!state.authed) {
    const host = el("recentRunsList");
    const status = el("recentRunsStatus");
    if (host) host.innerHTML = "";
    if (status) status.textContent = "Login to view recent runs.";
    return;
  }
  const status = el("recentRunsStatus");
  if (status) status.textContent = "Loading...";
  try {
    const resp = await fetchRecentPredictions(5);
    renderRecentRuns(resp.items || []);
  } catch (err) {
    const host = el("recentRunsList");
    if (host) {
      host.innerHTML = "";
      const li = document.createElement("li");
      li.style.opacity = "0.85";
      li.textContent = `Failed to load recent runs: ${(err && err.message) || String(err)}`;
      host.appendChild(li);
    }
    if (status) status.textContent = "—";
  }
}

/* Disturb UI */
function rebuildDisturbUI() {
  const enabled = el("disturbEnabled")?.checked;
  if (!el("disturbPanel")) return;
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
  if (!host) return;
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
      const input = host.querySelector(`[data-feature-input="${c}"]`);
      if (input) input.value = String(slider.value);
    });

    wrap.appendChild(label);
    wrap.appendChild(slider);
    wrap.appendChild(valueBox);
    const inputLine = document.createElement("div");
    inputLine.className = "row";
    inputLine.style.marginTop = "6px";
    const inputLabel = document.createElement("div");
    inputLabel.className = "card__hint";
    inputLabel.style.margin = "0";
    inputLabel.style.width = "220px";
    inputLabel.textContent = "Or type value (%)";
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
      input.value = String(v);
      slider.value = String(clamp(v, -20, 20));
      valueBox.textContent = `${slider.value}%`;
    });

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
  const enabled = el("disturbEnabled")?.checked;
  if (!enabled) return { enabled: false };
  if (!state.modeDetected) return { enabled: false };
  if (state.modeDetected === "basic") {
    const v = Number(el("globalPctInput")?.value || 0);
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
  if (!slider || !num || !out) return;

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

/* Evaluation */
function readEvaluation() {
  const enabled = el("evalEnabled")?.checked;
  if (!enabled) return { enabled: false };

  const mode = el("evalMode")?.value;
  if (mode === "last12") {
    const k = clamp(Number(el("evalK")?.value || 12), 2, 2000);
    return { enabled: true, split: { mode: "last12", test_points: k } };
  }
  return { enabled: true, split: { mode: "ratio", test_ratio: 0.2 } };
}

/* Physics UI */
function updatePhysicsUI() {
  const smooth = !!el("phySmooth")?.checked;
  const cap = !!el("phyCap")?.checked;

  if (smooth) show(el("smoothnessGroup"));
  else hide(el("smoothnessGroup"));

  if (cap) show(el("capGroup"));
  else hide(el("capGroup"));

  syncPhysicsUi();
}

function syncPhysicsUi() {
  const strength = el("smoothStrength");
  const mcr = el("maxChangeRate");
  const smooth = !!el("phySmooth")?.checked;
  if (!strength || !mcr) return;

  const preset = String(strength.value || "auto_normal").toLowerCase();
  const allowCustom = smooth && preset === "custom";

  mcr.disabled = !allowCustom;
  if (!allowCustom) mcr.value = "";
}

function smoothnessPresetRate(preset, scaleUsed) {
  const s = String(scaleUsed || "monthly").toLowerCase();
  const p = String(preset || "auto_normal").toLowerCase();

  if (p === "auto_strict") return s === "daily" ? 0.05 : (s === "yearly" ? 0.20 : 0.10);
  if (p === "auto_loose") return s === "daily" ? 0.12 : (s === "yearly" ? 0.55 : 0.35);
  return null;
}

function readPhysics() {
  const nonNeg = !!el("phyNonNeg")?.checked;
  const smooth = !!el("phySmooth")?.checked;
  const cap = !!el("phyCap")?.checked;
  let physicsMode = "none";
  if (cap && smooth) physicsMode = "full";
  else if (cap) physicsMode = "cap";
  else if (smooth) physicsMode = "smoothness";
  else if (nonNeg) physicsMode = "non_negative";

  const applyTo = "test_forecast";
  const capRaw = el("capValue")?.value;
  const capValue = capRaw === "" ? null : Number(capRaw);
  const physics = {
    non_negative: nonNeg,
    cap_value: capValue,
    apply_to: applyTo,
  };

  if (smooth) {
    const preset = String(el("smoothStrength")?.value || "auto_normal").toLowerCase();

    if (preset === "custom") {
      const raw = String(el("maxChangeRate")?.value || "").trim();
      if (raw !== "") {
        const v = Number(raw);
        if (Number.isFinite(v)) physics.max_change_rate = v;
      }
    } else {
      const v = smoothnessPresetRate(preset, state.scaleUsed || "monthly");
      if (v !== null) physics.max_change_rate = v;
    }
  }

  return { physics_mode: physicsMode, physics };
}

/* Predict */
async function runPrediction() {
  if (!state.uploadId) throw new Error("Please upload a CSV first.");
  const horizon = Number(el("horizonSelect")?.value);
  const disturbance = readDisturbance();
  const evaluation = readEvaluation();
  const phy = readPhysics();
  const payload = {
    upload_id: state.uploadId,
    horizon_months: horizon,
    physics_mode: phy.physics_mode,
    physics: phy.physics,
    disturbance,
    evaluation,
    scenario_name: el("scenarioName")?.value || "",
  };

  return await apiFetch("/predict", {
    method: "POST",
    headers: getJsonHeaders(),
    body: JSON.stringify(payload),
  });
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
  } catch {}
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
  return await apiFetch(`/predictions/${encodeURIComponent(pid)}`, { method: "GET" });
}

function pickBlock(doc, view) {
  const outputs = doc.outputs || {};
  const observed = outputs.observed || [];
  const block = outputs[view] || outputs.original || {};
  const observedDisturbed = (view === "disturbed" && block && block.observed_disturbed) ? block.observed_disturbed : null;
  return { observed, observedDisturbed, block };
}

function buildSegmentDateSets(observed, meta) {
  const dates = observed.map((p) => p.date);
  const nHist = Number(meta?.n_history ?? dates.length);
  const nTest = Number(meta?.n_test ?? 0);
  const fitEnd = Math.max(0, nHist - nTest);
  const fitDates = new Set(dates.slice(0, fitEnd));
  const testDates = new Set(dates.slice(fitEnd, nHist));

  return { fitDates, testDates };
}

function filterSeriesBySegment(series, segment, segSets) {
  const out = [];
  for (const p of series || []) {
    const kind = p.kind || "";
    const d = p.date;

    if (segment === "all") { out.push(p); continue; }
    if (segment === "fit") { if (segSets.fitDates.has(d)) out.push(p); continue; }
    if (segment === "test") { if (segSets.testDates.has(d)) out.push(p); continue; }
    if (segment === "forecast") { if (kind === "forecast") out.push(p); continue; }
    if (segment === "test_forecast") {
      if (segSets.testDates.has(d) || kind === "forecast") out.push(p);
      continue;
    }
  }
  return out;
}

function renderHints(doc, view) {
  const viewBox = el("viewHintBox");
  const whyBox = el("whyNoBox");
  if (!viewBox || !whyBox) return;
  const outputs = doc?.outputs || {};
  const block = outputs[view] || outputs.original || {};

  if (view === "disturbed") {
    const note = block.disturbance_note || "Disturbed is a what-if scenario. Real observed history is unchanged; the disturbance affects the scenario inputs used to generate baseline and corrected series. Accuracy is not scored because there is no ground truth for the disturbed future.";
    setHtml("viewHintBox", note);
    show(viewBox);
  } else {
    hide(viewBox);
    setHtml("viewHintBox", "");
  }

  const why = block.why_no_correction;
  if (why) {
    setText("whyNoBox", why);
    show(whyBox);
  } else {
    hide(whyBox);
    setText("whyNoBox", "");
  }
}

/* plotting */
function drawLineChart(canvas, seriesList, opts = {}) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 50, r: 16, t: 18, b: 40 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const map = new Map();
  function addSeries(series, key) {
    for (const p of series || []) {
      const d = p.date;
      if (!map.has(d)) map.set(d, { date: d });
      map.get(d)[key] = Number(p.value);
    }
  }
  seriesList.forEach((s) => addSeries(s.data, s.key));

  const dates = Array.from(map.keys()).sort();
  const rows = dates.map((d) => map.get(d));
  const values = [];
  for (const r of rows) {
    for (const s of seriesList) {
      const v = r[s.key];
      if (v !== undefined && Number.isFinite(v)) values.push(v);
    }
  }

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

  function plot(key, stroke) {
    let started = false;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((r, i) => {
      const v = r[key];
      if (v === undefined || !Number.isFinite(v)) return;
      const x = xAt(i);
      const y = yAt(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else { ctx.lineTo(x, y); }
    });
    if (started) ctx.stroke();
  }

  for (const s of seriesList) plot(s.key, s.stroke);

  const step = Math.max(1, Math.floor(rows.length / 6));
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.font = "11px ui-sans-serif";
  for (let i = 0; i < rows.length; i += step) {
    const x = xAt(i);
    const label = rows[i].date.slice(0, 10);
    ctx.save();
    ctx.translate(x, pad.t + innerH + 18);
    ctx.rotate(-0.35);
    ctx.fillText(label, -18, 0);
    ctx.restore();
  }
}

function drawDelta(canvas, deltaSeries) {
  drawLineChart(canvas, [{ key: "delta", data: deltaSeries, stroke: "rgba(40,215,201,0.95)" }]);
}

function drawViolations(canvas, violSeries) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 110, r: 16, t: 18, b: 26 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const rules = ["non_negative", "cap", "rate_limit"];
  const rows = rules.length;
  const dates = (violSeries || []).map((p) => p.date).sort();
  if (!dates.length) {
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("No violations in selected segment.", 20, 30);
    return;
  }

  function xAt(i) {
    if (dates.length <= 1) return pad.l;
    return pad.l + (i / (dates.length - 1)) * innerW;
  }

  function yAtRow(r) {
    return pad.t + (r / Math.max(1, rows - 1)) * innerH;
  }

  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.font = "12px ui-sans-serif";
  rules.forEach((r, i) => {
    ctx.fillText(r, 12, yAtRow(i) + 4);
  });

  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + innerH);
  ctx.lineTo(pad.l + innerW, pad.t + innerH);
  ctx.stroke();

  for (let i = 0; i < dates.length; i++) {
    const d = dates[i];
    const p = violSeries.find((x) => x.date === d);
    if (!p) continue;

    const hits = p.rules_hit || [];
    const x = xAt(i);

    rules.forEach((r, ridx) => {
      if (!hits.includes(r)) return;
      const y = yAtRow(ridx);

      ctx.fillStyle = "rgba(58,123,253,0.95)";
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

/* Export */
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

  const view = el("viewSelect")?.value || "original";
  const { observed, observedDisturbed, block } = pickBlock(state.lastResponse, view);
  const meta = block.meta || {};
  const segSets = buildSegmentDateSets(observed, meta);
  const baseline = filterSeriesBySegment(block.baseline || [], state.activeSegment, segSets);
  const piml = filterSeriesBySegment(block.piml || [], state.activeSegment, segSets);
  const obs = filterSeriesBySegment(observed || [], state.activeSegment, segSets);
  const obsD = observedDisturbed ? filterSeriesBySegment(observedDisturbed || [], state.activeSegment, segSets) : null;
  const map = new Map();
  function add(series, key) {
    (series || []).forEach((p) => {
      const d = p.date;
      if (!map.has(d)) map.set(d, { date: d });
      map.get(d)[key] = Number(p.value);
    });
  }

  add(obs, "observed");
  if (obsD) add(obsD, "observed_disturbed");
  add(baseline, "baseline");
  add(piml, "piml");

  const dates = Array.from(map.keys()).sort();
  const header = obsD ? "date,observed,observed_disturbed,baseline,piml" : "date,observed,baseline,piml";
  const lines = [header];

  dates.forEach((d) => {
    const r = map.get(d);
    const o = r.observed ?? "";
    const od = obsD ? (r.observed_disturbed ?? "") : null;
    const b = r.baseline ?? "";
    const p = r.piml ?? "";
    lines.push(obsD ? `${d},${o},${od},${b},${p}` : `${d},${o},${b},${p}`);
  });

  downloadText(`series_${view}_${state.activeSegment}.csv`, lines.join("\n"), "text/csv");
}

/* Results rendering */
function setActiveTab(tab) {
  state.activeTab = tab;
  const tabCompare = el("tabCompare");
  const tabDelta = el("tabDelta");
  const tabViol = el("tabViol");

  [tabCompare, tabDelta, tabViol].forEach((b) => b && b.classList.remove("tabbtn--active"));
  if (tab === "compare") tabCompare?.classList.add("tabbtn--active");
  if (tab === "delta") tabDelta?.classList.add("tabbtn--active");
  if (tab === "viol") tabViol?.classList.add("tabbtn--active");

  if (tab === "compare") {
    show(el("panelCompare"));
    hide(el("panelDelta"));
    hide(el("panelViol"));
  } else if (tab === "delta") {
    hide(el("panelCompare"));
    show(el("panelDelta"));
    hide(el("panelViol"));
  } else {
    hide(el("panelCompare"));
    hide(el("panelDelta"));
    show(el("panelViol"));
  }

  drawActiveView();
}

function setActiveSegment(seg) {
  state.activeSegment = seg;
  const btns = document.querySelectorAll(".segbtn");
  btns.forEach((b) => b.classList.toggle("segbtn--active", b.dataset.seg === seg));
  drawActiveView();
}

/* Test KPI */
function _toValueMap(series) {
  const m = new Map();
  (series || []).forEach((p) => {
    if (!p || !p.date) return;
    const v = Number(p.value);
    if (!Number.isFinite(v)) return;
    m.set(p.date, v);
  });
  return m;
}

function _computeKpiTest(observed, predSeries, segSets) {
  const obsTest = filterSeriesBySegment(observed || [], "test", segSets);
  const predTest = filterSeriesBySegment(predSeries || [], "test", segSets);
  const om = _toValueMap(obsTest);
  const pm = _toValueMap(predTest);

  let n = 0;
  let sumAbs = 0.0;
  let sumSq = 0.0;
  let sumApe = 0.0;

  for (const [d, y] of om.entries()) {
    if (!pm.has(d)) continue;
    const yhat = pm.get(d);
    if (!Number.isFinite(y) || !Number.isFinite(yhat)) continue;

    const e = yhat - y;
    n += 1;
    sumAbs += Math.abs(e);
    sumSq += e * e;

    const denom = Math.max(Math.abs(y), 1e-9);
    sumApe += Math.abs(e) / denom;
  }

  if (n <= 0) return null;

  const rmse = Math.sqrt(sumSq / n);
  const mae = sumAbs / n;
  const mape = (sumApe / n) * 100.0;

  return { n, rmse, mae, mape };
}

function renderResults(doc) {
  state.lastResponse = doc;

  setText("pidBox", doc.prediction_id || "—");
  setText("modeBox", doc.mode_used || doc.mode_detected || "—");
  setText("methodBox", doc.method || "—");
  setText("scaleBox", doc.scale_used || "—");

  const applyTo = doc.params?.physics_effective?.apply_to
    || doc.params?.physics?.apply_to
    || doc.params?.physics?.applyTo
    || "test_forecast";
  setText("applyToBox", applyTo);
  setText("rawBox", jsonPretty(doc));
  setText("kpiBaseErr", "—");
  setText("kpiPimlErr", "—");
  setText("kpiImprove", "—");

  const viewSel = el("viewSelect");
  const disturbedExists = !!doc.outputs?.disturbed;
  if (viewSel) {
    const opt = viewSel.querySelector('option[value="disturbed"]');
    if (opt) opt.disabled = !disturbedExists;
    if (!disturbedExists) viewSel.value = "original";
  }

  setActiveTab("compare");
  setActiveSegment("test_forecast");
  drawActiveView();
  setStatus("resultsStatus", "Loaded.", false);
}

const PLOT_EPS = 1e-6;
const HIDE_PIML_IN_FIT = true;
const SHOW_PIML_ONLY_WHEN_DIFF = true;

function _toMap(series) {
  const m = new Map();
  (series || []).forEach((p) => {
    if (!p || !p.date) return;
    m.set(p.date, Number(p.value));
  });
  return m;
}

function _maskedPimlSeries(pimlSeries, baseSeries, eps = PLOT_EPS) {
  const bm = _toMap(baseSeries);
  const out = [];
  for (const p of pimlSeries || []) {
    const pv = Number(p.value);
    const bv = bm.has(p.date) ? bm.get(p.date) : NaN;
    const diff = Number.isFinite(pv) && Number.isFinite(bv) ? Math.abs(pv - bv) : NaN;
    if (Number.isFinite(diff) && diff <= eps) out.push({ ...p, value: NaN });
    else out.push({ ...p, value: pv });
  }
  return out;
}

function _hasAnyVisiblePoint(series) {
  for (const p of series || []) {
    const v = Number(p?.value);
    if (Number.isFinite(v)) return true;
  }
  return false;
}

function _calcAdjustStatsForSegment(baseSeries, pimlSeries, eps = PLOT_EPS) {
  const bm = _toMap(baseSeries);
  let n = 0;
  let sumAbs = 0.0;
  let nBoth = 0;

  for (const p of pimlSeries || []) {
    const pv = Number(p.value);
    const bv = bm.has(p.date) ? bm.get(p.date) : NaN;
    if (!Number.isFinite(pv) || !Number.isFinite(bv)) continue;
    nBoth += 1;
    const d = Math.abs(pv - bv);
    if (d > eps) {
      n += 1;
      sumAbs += d;
    }
  }

  const meanAbs = n > 0 ? (sumAbs / n) : 0.0;
  const ratio = nBoth > 0 ? (n / nBoth) : 0.0;
  return { n_adjusted: n, n_total: nBoth, adjusted_ratio: ratio, mean_abs_adjustment: meanAbs };
}

function _getPhysMode(doc) {
  const physMode =
    doc?.params?.physics_effective?.physics_mode ||
    doc?.params?.physics_effective?.physicsMode ||
    doc?.params?.physics_mode ||
    doc?.params?.physicsMode ||
    doc?.physics_mode ||
    doc?.physicsMode ||
    "unknown";
  return String(physMode || "unknown").toLowerCase();
}

function drawActiveView() {
  if (!state.lastResponse) return;

  const view = el("viewSelect")?.value || "original";
  renderHints(state.lastResponse, view);

  const { observed, observedDisturbed, block } = pickBlock(state.lastResponse, view);
  const meta = block.meta || {};
  const segSets = buildSegmentDateSets(observed, meta);
  const obsSeg = filterSeriesBySegment(observed, state.activeSegment, segSets);
  const obsDSeg = observedDisturbed ? filterSeriesBySegment(observedDisturbed, state.activeSegment, segSets) : null;
  const baseSeg = filterSeriesBySegment(block.baseline || [], state.activeSegment, segSets);
  const pimlSegRaw = filterSeriesBySegment(block.piml || [], state.activeSegment, segSets);
  const violSeg = filterSeriesBySegment(block.violations_series || [], state.activeSegment, segSets);
  const kpi = block.kpi_test || {};
  const kb = kpi.baseline || {};
  const kp = kpi.piml || {};
  const hasBase = (kb.rmse !== null && kb.rmse !== undefined);
  const hasPiml = (kp.rmse !== null && kp.rmse !== undefined);

  if (hasBase) {
    setText("kpiBaseErr", `RMSE ${fmtNum(kb.rmse)} | MAE ${fmtNum(kb.mae)} | MAPE ${fmtNum(kb.mape, 2)}%`);
  } else {
    setText("kpiBaseErr", "—");
  }

  if (hasPiml) {
    setText("kpiPimlErr", `RMSE ${fmtNum(kp.rmse)} | MAE ${fmtNum(kp.mae)} | MAPE ${fmtNum(kp.mape, 2)}%`);
  } else {
    setText("kpiPimlErr", "—");
  }

  if (hasBase && hasPiml) {
    const improve = ((Number(kb.rmse) - Number(kp.rmse)) / Math.max(Number(kb.rmse), 1e-9)) * 100;
    setText("kpiImprove", `${fmtNum(improve, 1)}%`);
  } else {
    setText("kpiImprove", "—");
  }
  const byRule = {};
  for (const p of violSeg) {
    const rules = p.rules_hit || [];
    for (const r of rules) byRule[r] = (byRule[r] || 0) + 1;
  }
  const parts = ["non_negative", "cap", "rate_limit"].map((r) => `${r} ${byRule[r] || 0}`);
  setText("kpiPhys", `B vs P shown by rule hits: ${parts.join(", ")}`);
  const cs = block.correction_summary || {};
  const numAdj = (cs.num_adjusted ?? cs.n_adjusted_points);
  const ratio = (cs.adjusted_ratio ?? cs.adjustedRatio);
  const meanAbs = (cs.mean_abs_adjustment ?? cs.meanAbsAdjustment);
  const ratioText = (ratio !== null && ratio !== undefined) ? `${fmtNum(Number(ratio) * 100, 1)}%` : "—";
  setText("kpiAdjPts", (numAdj === null || numAdj === undefined) ? "—" : `${numAdj} (${ratioText})`);
  setText("kpiAdjMean", (meanAbs === null || meanAbs === undefined) ? "—" : fmtNum(meanAbs));
  const segStats = _calcAdjustStatsForSegment(baseSeg, pimlSegRaw, PLOT_EPS);
  const segRatioText = `${fmtNum(segStats.adjusted_ratio * 100, 1)}%`;
  setText("kpiAdjPtsSeg", `In current view (segment): ${segStats.n_adjusted} (${segRatioText})`);
  setText("kpiAdjMeanSeg", `In current view (segment): ${fmtNum(segStats.mean_abs_adjustment)}`);
  const legendObsDist = el("legendObsDisturbed");
  if (legendObsDist) {
    if (view === "disturbed" && obsDSeg && obsDSeg.length) show(legendObsDist);
    else hide(legendObsDist);
  }

  if (state.activeTab === "compare") {
    const series = [
      { key: "obs", data: obsSeg.map((p) => ({ ...p, value: p.value })), stroke: "rgba(255,255,255,0.9)" },
    ];
    if (view === "disturbed" && obsDSeg && obsDSeg.length) {
      series.push({ key: "obsD", data: obsDSeg.map((p) => ({ ...p, value: p.value })), stroke: "rgba(255,195,80,0.95)" });
    }
    series.push({ key: "base", data: baseSeg.map((p) => ({ ...p, value: p.value })), stroke: "rgba(58,123,253,0.95)" });
    const seg = String(state.activeSegment || "test_forecast").toLowerCase();
    const hideInFit = HIDE_PIML_IN_FIT && seg === "fit";
    const physMode = _getPhysMode(state.lastResponse);
    const backendSaysNoAdj = (numAdj !== null && numAdj !== undefined) ? (Number(numAdj) <= 0) : false;
    const backendModeNone = physMode === "none";
    let pimlToPlot = pimlSegRaw.map((p) => ({ ...p, value: p.value }));
    if (SHOW_PIML_ONLY_WHEN_DIFF) {
      pimlToPlot = _maskedPimlSeries(pimlToPlot, baseSeg, PLOT_EPS);
    }

    const hasVisible = _hasAnyVisiblePoint(pimlToPlot);
    const shouldPlotPiml = !hideInFit && !backendModeNone && !backendSaysNoAdj && hasVisible;

    if (shouldPlotPiml) {
      series.push({ key: "piml", data: pimlToPlot, stroke: "rgba(40,215,201,0.95)" });
    }

    drawLineChart(el("compareChart"), series);
    return;
  }

  if (state.activeTab === "delta") {
    const deltaSeg = filterSeriesBySegment(block.delta_series || [], state.activeSegment, segSets);
    drawDelta(el("deltaChart"), deltaSeg.map((p) => ({ date: p.date, value: p.delta, kind: p.kind })));
    setText("deltaNum", cs.num_adjusted ?? "—");
    setText("deltaMax", cs.max_abs_adjustment !== undefined ? fmtNum(cs.max_abs_adjustment) : "—");
    setText("deltaMean", cs.mean_abs_adjustment !== undefined ? fmtNum(cs.mean_abs_adjustment) : "—");
    setText("deltaRatio", cs.adjusted_ratio !== undefined ? `${fmtNum(cs.adjusted_ratio * 100, 1)}%` : "—");
    return;
  }

  const violFiltered = filterSeriesBySegment(block.violations_series || [], state.activeSegment, segSets);
  drawViolations(el("violChart"), violFiltered);

  const br = cs.by_rule_counts || {};
  const msg = Object.keys(br).length
    ? Object.entries(br).map(([k, v]) => `${k}:${v}`).join(" | ")
    : "—";
  setText("violSummary", msg);
}

/* Setup page */
function initSetupPage() {
  refreshHorizonSelect({ keepValue: false });
  resetHorizonToDefault();
  ["phyNonNeg", "phySmooth", "phyCap"].forEach((id) => {
    el(id)?.addEventListener("change", updatePhysicsUI);
  });
  el("smoothStrength")?.addEventListener("change", syncPhysicsUi);
  updatePhysicsUI();
  // Recent runs
  refreshRecentRuns();
  el("uploadBtn")?.addEventListener("click", async () => {
    if (!state.authed) {
      setStatus("runStatus", "Please login first.", true);
      return;
    }

    const file = el("fileInput")?.files?.[0];
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
      refreshHorizonSelect({ keepValue: false });
      resetHorizonToDefault();

      if (el("schemaBox")) {
        const schemaText = jsonPretty(state.schema || {});
        const warnings = uploadResp.warnings || [];
        const warnText = warnings.length ? "\n\nWarnings:\n- " + warnings.join("\n- ") : "\n\nWarnings:\n- none";
        el("schemaBox").textContent = schemaText + warnText;
      }

      rebuildDisturbUI();
      updatePhysicsUI();
      setStatus("runStatus", "Upload done. Configure settings and run.", false);
    } catch (err) {
      setText("uploadStatus", "Upload failed.");
      setStatus("runStatus", (err && err.message) || String(err), true);
    }
  });

  el("disturbEnabled")?.addEventListener("change", rebuildDisturbUI);
  syncBasicDisturbControls();

  el("resetBtn")?.addEventListener("click", () => {
    if (el("scenarioName")) el("scenarioName").value = "";
    refreshHorizonSelect({ keepValue: false });
    resetHorizonToDefault();
    if (el("phyNonNeg")) el("phyNonNeg").checked = true;
    if (el("phySmooth")) el("phySmooth").checked = true;
    if (el("phyCap")) el("phyCap").checked = false;
    if (el("smoothStrength")) el("smoothStrength").value = "auto_normal";
    if (el("maxChangeRate")) el("maxChangeRate").value = "";
    if (el("capValue")) el("capValue").value = "";
    if (el("disturbEnabled")) el("disturbEnabled").checked = false;
    if (el("globalPct")) el("globalPct").value = "0";
    if (el("globalPctInput")) el("globalPctInput").value = "0";
    setText("globalPctVal", "0");
    if (el("evalEnabled")) el("evalEnabled").checked = true;
    if (el("evalMode")) el("evalMode").value = "ratio";
    if (el("evalK")) el("evalK").value = "12";

    rebuildDisturbUI();
    updatePhysicsUI();
    setStatus("runStatus", "Reset done.", false);
  });

  el("runBtn")?.addEventListener("click", async () => {
    if (!state.authed) {
      setStatus("runStatus", "Please login first.", true);
      return;
    }

    setStatus("runStatus", "Running...", false);
    try {
      const resp = await runPrediction();
      cacheLastResponse(resp);
      refreshRecentRuns();
      const pid = resp.prediction_id;
      if (!pid) throw new Error("Missing prediction_id in response.");
      window.location.href = `./results.html?pid=${encodeURIComponent(pid)}`;
    } catch (err) {
      setStatus("runStatus", (err && err.message) || String(err), true);
    }
  });

  rebuildDisturbUI();
  refreshAuthUI();
}

/* Results page */
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
    try { sessionStorage.setItem(`prediction_cache__${pid}`, JSON.stringify(doc)); } catch {}
    renderResults(doc);
  }

  viewSel?.addEventListener("change", () => { drawActiveView(); });
  el("tabCompare")?.addEventListener("click", () => setActiveTab("compare"));
  el("tabDelta")?.addEventListener("click", () => setActiveTab("delta"));
  el("tabViol")?.addEventListener("click", () => setActiveTab("viol"));
  const segBtns = document.querySelectorAll(".segbtn");
  segBtns.forEach((b) => { b.addEventListener("click", () => setActiveSegment(b.dataset.seg)); });
  el("exportJsonBtn")?.addEventListener("click", exportJson);
  el("exportCsvBtn")?.addEventListener("click", exportCsv);
  refreshBtn?.addEventListener("click", async () => {
    const pid = getPidFromUrl();
    if (!pid) return;
    try {
      const doc = await fetchPrediction(pid);
      try { sessionStorage.setItem(`prediction_cache__${pid}`, JSON.stringify(doc)); } catch {}
      renderResults(doc);
    } catch (err) {
      setStatus("resultsStatus", (err && err.message) || String(err), true);
    }
  });

  let pid = getPidFromUrl();
  if (!pid) {
    pid = sessionStorage.getItem("last_prediction_id");
    if (pid) setPidInUrl(pid);
  }

  if (!pid) {
    setStatus("resultsStatus", "No prediction id. Go to Setup and run first.", true);
    return;
  }

  if (!state.authed) {
    setStatus("resultsStatus", "Please login to view results.", true);
    return;
  }

  try {
    await load(pid);
  } catch (err) {
    setStatus("resultsStatus", (err && err.message) || String(err), true);
  }
}

async function init() {
  await initAuth();
  const page = document.body.dataset.page;
  if (page === "setup") initSetupPage();
  if (page === "results") initResultsPage();
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((e) => {
    const msg = e && e.message ? e.message : String(e);
    if (el("runStatus")) setStatus("runStatus", msg, true);
    if (el("resultsStatus")) setStatus("resultsStatus", msg, true);
  });
});