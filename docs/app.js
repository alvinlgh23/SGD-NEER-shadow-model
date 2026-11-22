/* Interactive shadow S$NEER dashboard.
 *
 * data.json (written by export_web.py) carries the shadow index for each
 * weight mode plus the MAS policy calendar. The policy band is recomputed here
 * with the same rules as policy_band.py, so slope and band-width edits update
 * every number on the page instantly.
 */
"use strict";

const DAY_MS = 86400000;
const TAGS = { slope_up: "↑ slope", slope_down: "↓ slope", recentre: "re-centre" };
const MOVE_LABELS = { slope_up: "Slope raised", slope_down: "Slope reduced", recentre: "Re-centred" };
const MIN_REGIME_OBS = 120;

let DATA = null;
const state = { mode: null, band: 2, slopes: {} };

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtPct = (v, d = 2, sign = true) => (sign && v > 0 ? "+" : "") + v.toFixed(d) + "%";
const monthTag = (iso) => {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleString("en-GB", { month: "short", timeZone: "UTC" }) + " " + String(d.getUTCFullYear()).slice(2);
};
const longDate = (iso) => new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB",
  { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const mean = (arr, a, b) => { let s = 0; for (let i = a; i < b; i++) s += arr[i]; return s / (b - a); };

function bandEvents() {
  return DATA.events.filter((e) => e.action !== "hold");
}

function slopeAt(iso) {
  // Only band-changing events set the slope; "hold" keeps whatever is in force.
  let slope = DATA.meta.initial_slope_pct;
  for (const e of bandEvents()) if (e.date <= iso) slope = state.slopes[e.date];
  return slope;
}

// ---------- model: mirror of policy_band.build_piecewise_center ----------
function computeBand(index) {
  const dates = DATA.dates;
  const n = dates.length;
  const t = dates.map((d) => Date.parse(d + "T00:00:00Z"));
  const logI = index.map(Math.log);

  const growth = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const days = Math.max((t[i] - t[i - 1]) / DAY_MS, 0);
    growth[i] = growth[i - 1] + (slopeAt(dates[i]) / 100) * days / 365.25;
  }

  const recentres = bandEvents().filter((e) => e.action === "recentre" && e.date > dates[0] && e.date <= dates[n - 1]);
  const bounds = [0, ...recentres.map((e) => dates.findIndex((d) => d >= e.date)), n];

  const logC = new Float64Array(n);
  let offset = 0;
  for (let i = bounds[0]; i < bounds[1]; i++) offset += logI[i] - growth[i];
  offset /= bounds[1] - bounds[0];
  for (let i = bounds[0]; i < bounds[1]; i++) logC[i] = offset + growth[i];

  const lookback = DATA.meta.recentre_lookback;
  for (let k = 1; k < bounds.length - 1; k++) {
    const start = bounds[k], end = bounds[k + 1];
    const lo = Math.max(start - lookback, 0);
    const level = start > lo ? mean(logI, lo, start) : logI[start];
    for (let i = start; i < end; i++) logC[i] = level + growth[i] - growth[start];
  }

  const half = state.band / 100;
  const center = Array.from(logC, Math.exp);
  const upper = center.map((c) => c * (1 + half));
  const lower = center.map((c) => c * (1 - half));
  const deviation = index.map((v, i) => (v / center[i] - 1) * 100);
  const outsideCount = deviation.filter((d) => Math.abs(d) > state.band).length;
  return { center, upper, lower, deviation, share: outsideCount / n, gap: deviation[n - 1] };
}

function zoneText(gap) {
  if (gap > state.band) return "above the estimated band";
  if (gap >= 0) return "in the upper half of the band";
  if (gap >= -state.band) return "in the lower half of the band";
  return "below the estimated band";
}

// ---------- chart styling ----------
function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: css("--ink-2") },
    margin: { l: 52, r: 16, t: 12, b: 40 },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--axis"), font: { color: css("--ink"), size: 12 } },
    xaxis: { gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") } },
    yaxis: { gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted") } },
    legend: { orientation: "h", x: 0, y: -0.14, font: { color: css("--ink-2") } },
    showlegend: true,
  }, extra);
}
const PLOT_CONFIG = { responsive: true, displayModeBar: false };
const narrow = () => window.innerWidth < 720;

function eventShapes(yref = "paper") {
  return bandEvents()
    .filter((e) => e.date >= DATA.dates[0])
    .map((e) => ({
      type: "line", xref: "x", yref, x0: e.date, x1: e.date, y0: 0, y1: 1,
      line: { color: css("--muted"), width: 1, dash: e.verified ? "solid" : "dot" },
      layer: "below",
    }));
}

function eventAnnotations() {
  if (narrow()) return [];
  return bandEvents()
    .filter((e) => e.date >= DATA.dates[0])
    .map((e, i) => ({
      x: e.date, y: i % 2 ? 0.86 : 1, xref: "x", yref: "paper", xanchor: "left", yanchor: "top",
      text: `${monthTag(e.date)}<br>${TAGS[e.action]}${e.verified ? "" : "<br>(unverified)"}`,
      showarrow: false, align: "left", xshift: 3,
      font: { size: 10.5, color: e.verified ? css("--ink-2") : css("--critical") },
    }));
}

// ---------- renderers ----------
function renderMain(index, band) {
  const x = DATA.dates;
  const outside = index.map((v, i) => (Math.abs(band.deviation[i]) > state.band ? v : null));
  const lo = Math.min(...band.lower, ...index), hi = Math.max(...band.upper, ...index);
  const traces = [
    { x, y: band.lower, mode: "lines", line: { color: css("--axis"), width: 1 }, hoverinfo: "skip", showlegend: false },
    { x, y: band.upper, mode: "lines", fill: "tonexty", fillcolor: css("--band"), line: { color: css("--axis"), width: 1 },
      name: `Estimated band (±${state.band}%)`, hoverinfo: "skip" },
    { x, y: band.center, mode: "lines", line: { color: css("--muted"), width: 1.2, dash: "dot" },
      name: "Estimated centre", hovertemplate: "Centre %{y:.2f}<extra></extra>" },
    { x, y: index, mode: "lines", line: { color: css("--s1"), width: 2 }, name: "Shadow S$NEER",
      customdata: band.deviation, hovertemplate: "<b>%{y:.2f}</b> index · <b>%{customdata:+.2f}%</b> vs centre<extra></extra>" },
    { x, y: outside, mode: "lines", line: { color: css("--serious"), width: 2.4 }, name: "Outside estimated band",
      hoverinfo: "skip", connectgaps: false },
  ];
  const layout = baseLayout({
    hovermode: "x unified",
    shapes: eventShapes(),
    annotations: eventAnnotations(),
    yaxis: Object.assign(baseLayout().yaxis, {
      range: [lo - 0.5, hi + (hi - lo) * (narrow() ? 0.05 : 0.32)],
      title: { text: `Index, ${monthTag(x[0])} = 100`, font: { size: 11, color: css("--muted") } },
    }),
  });
  Plotly.react("chart-main", traces, layout, PLOT_CONFIG);
}

function renderPosition(band) {
  const half = state.band;
  const lim = Math.max(half + 1, Math.ceil(Math.max(...band.deviation.map(Math.abs))) + 0.5);
  const x = DATA.dates;
  const zone = (y0, y1, color) => ({ type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0, y1,
    fillcolor: color, line: { width: 0 }, layer: "below" });
  const label = (y, text) => ({ x: 1, y, xref: "paper", yref: "y", xanchor: "left", xshift: 6, text, showarrow: false,
    font: { size: 11, color: css("--muted") } });
  const traces = [{
    x, y: band.deviation, mode: "lines", line: { color: css("--s1"), width: 1.8 }, name: "Deviation from centre",
    hovertemplate: "<b>%{y:+.2f}%</b> vs centre<extra></extra>", showlegend: false,
  }];
  const layout = baseLayout({
    hovermode: "x unified",
    showlegend: false,
    margin: { l: 52, r: narrow() ? 16 : 84, t: 8, b: 32 },
    shapes: [
      zone(half, lim, css("--outside-zone")), zone(-lim, -half, css("--outside-zone")),
      zone(0, half, css("--band")), zone(-half, 0, css("--band-upper")),
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: css("--muted"), width: 1, dash: "dot" } },
      ...eventShapes(),
    ],
    annotations: narrow() ? [] : [label((half + lim) / 2, "above band"), label(half / 2, "upper half"),
      label(-half / 2, "lower half"), label(-(half + lim) / 2, "below band")],
    yaxis: Object.assign(baseLayout().yaxis, { range: [-lim, lim], ticksuffix: "%", dtick: 1 }),
  });
  Plotly.react("chart-pos", traces, layout, PLOT_CONFIG);
}

function renderWeights() {
  const w = DATA.weights;
  const order = [...w.currencies].sort((a, b) => w.ols[a].coef - w.ols[b].coef);
  const coef = order.map((c) => w.ols[c].coef);
  const ci = order.map((c) => 1.96 * w.ols[c].se);
  const assumed = order.map((c) => w.assumed[c]);
  const traces = [
    { x: assumed, y: order, mode: "markers", name: "Hand-set guess",
      marker: { symbol: "diamond", size: 9, color: css("--muted"), line: { color: css("--surface"), width: 2 } },
      hovertemplate: "%{y} hand-set: <b>%{x:.1%}</b><extra></extra>" },
    { x: coef, y: order, mode: "markers", name: "Inferred (95% interval)",
      error_x: { type: "data", array: ci, color: css("--s1"), thickness: 1.5, width: 0 },
      marker: { size: 10, color: css("--s1"), line: { color: css("--surface"), width: 2 } },
      hovertemplate: "%{y} inferred: <b>%{x:.1%}</b><extra></extra>" },
  ];
  const connectors = order.map((c, i) => ({ type: "line", xref: "x", yref: "y", x0: assumed[i], x1: coef[i],
    y0: c, y1: c, line: { color: css("--grid"), width: 2 }, layer: "below" }));
  const layout = baseLayout({
    hovermode: "closest", shapes: connectors,
    margin: { l: 44, r: 16, t: 8, b: 40 },
    xaxis: Object.assign(baseLayout().xaxis, { tickformat: ".0%", rangemode: "tozero" }),
    yaxis: Object.assign(baseLayout().yaxis, { gridcolor: "rgba(0,0,0,0)" }),
  });
  Plotly.react("chart-weights", traces, layout, PLOT_CONFIG);

  // Accessible table view (textContent only).
  const table = $("weights-table");
  table.textContent = "";
  const head = table.createTHead().insertRow();
  ["Currency", "Inferred (full)", "95% interval", "Inferred (latest 2y)", "Hand-set"].forEach((h) => {
    const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
  });
  const body = table.createTBody();
  [...order].reverse().forEach((c) => {
    const r = body.insertRow();
    const o = w.ols[c];
    [c, (o.coef * 100).toFixed(1) + "%", `${((o.coef - 1.96 * o.se) * 100).toFixed(1)} to ${((o.coef + 1.96 * o.se) * 100).toFixed(1)}%`,
      (w.latest[c] * 100).toFixed(1) + "%", (w.assumed[c] * 100).toFixed(1) + "%"].forEach((v, i) => {
      const td = r.insertCell(); td.textContent = v; if (i) td.className = "num";
    });
  });
}

function spread(values, minGap) {
  // Push end-of-line labels apart so none overlap; keep the group centred.
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const pos = order.map((o) => o[0]);
  for (let i = 1; i < pos.length; i++) pos[i] = Math.max(pos[i], pos[i - 1] + minGap);
  const shift = pos.reduce((s, p, i) => s + p - order[i][0], 0) / pos.length;
  const out = new Array(values.length);
  order.forEach((o, i) => { out[o[1]] = pos[i] - shift; });
  return out;
}

function renderRolling() {
  const w = DATA.weights;
  // Fixed set and fixed colours: the five largest full-sample weights.
  const top = [...w.currencies].sort((a, b) => w.full[b] - w.full[a]).slice(0, 5);
  const colors = ["--s1", "--s2", "--s3", "--s4", "--s5"].map(css);
  const x = w.rolling.dates;
  const last = top.map((c) => w.rolling.series[c][x.length - 1]);
  const labelY = spread(last, 0.022);
  const traces = top.map((c, i) => ({
    x, y: w.rolling.series[c], mode: "lines", name: c, line: { color: colors[i], width: 2 },
    hovertemplate: `${c} <b>%{y:.1%}</b><extra></extra>`,
  }));
  const annotations = narrow() ? [] : top.map((c, i) => ({
    x: x[x.length - 1], y: labelY[i], xref: "x", yref: "y", xanchor: "left", xshift: 6,
    text: `${c} ${(last[i] * 100).toFixed(0)}%`, showarrow: false, font: { size: 11, color: css("--ink-2") },
  }));
  const layout = baseLayout({
    hovermode: "x unified", annotations,
    margin: { l: 44, r: narrow() ? 16 : 72, t: 8, b: 40 },
    xaxis: Object.assign(baseLayout().xaxis, { range: [x[0], x[x.length - 1]] }),
    yaxis: Object.assign(baseLayout().yaxis, { tickformat: ".0%", rangemode: "tozero" }),
  });
  Plotly.react("chart-roll", traces, layout, PLOT_CONFIG);
  $("roll-sub").textContent = `Rolling ${DATA.fit.window_weeks}-week window. Each point uses the data ending on that date. ` +
    "The five largest full-sample weights are shown.";
}

function renderSlope() {
  const drift = DATA.fit.rolling_drift;
  const start = drift.dates[0], end = drift.dates[drift.dates.length - 1];
  const xs = [start], ys = [slopeAt(start)];
  for (const e of bandEvents()) if (e.date > start && e.date <= end) { xs.push(e.date); ys.push(state.slopes[e.date]); }
  xs.push(end); ys.push(slopeAt(end));
  const traces = [
    { x: xs, y: ys, mode: "lines", line: { shape: "hv", color: css("--s1"), width: 2 }, name: "Your slope assumption",
      hovertemplate: "Assumed <b>%{y:.2f}%</b> p.a.<extra></extra>" },
    { x: drift.dates, y: drift.values, mode: "lines", line: { color: css("--s2"), width: 1.6 },
      name: "Regression drift (rolling)", hovertemplate: "Drift <b>%{y:.2f}%</b> p.a.<extra></extra>" },
  ];
  const layout = baseLayout({
    hovermode: "x unified",
    margin: { l: 52, r: 16, t: 8, b: 40 },
    yaxis: Object.assign(baseLayout().yaxis, { ticksuffix: "%", zeroline: true, zerolinecolor: css("--axis") }),
  });
  Plotly.react("chart-slope", traces, layout, PLOT_CONFIG);
}

function renderKpis(band) {
  const gap = band.gap, half = state.band;
  const inBand = Math.abs(gap) <= half;
  $("headline").textContent = `The Singapore dollar is ${zoneText(gap)} (${fmtPct(gap, 1)} vs estimated centre)`;
  $("k-pos").textContent = fmtPct(gap);
  const sub = $("k-pos-sub");
  sub.textContent = "";
  const s = document.createElement("span");
  s.className = "status " + (inBand ? "ok" : "warn");
  s.textContent = inBand ? `✓ inside ±${half}% band` : `⚠ outside ±${half}% band`;
  sub.appendChild(s);
  $("k-room").textContent = gap <= half ? `${(half - gap).toFixed(2)}%` : `${(gap - half).toFixed(2)}% over`;
  $("k-room-sub").textContent = gap <= half ? "before the estimated ceiling" : "above the estimated ceiling";
  $("k-slope").textContent = `${slopeAt(DATA.dates[DATA.dates.length - 1]).toFixed(2)}% p.a.`;
  $("k-out").textContent = `${(band.share * 100).toFixed(0)}%`;
  $("k-out-sub").textContent = `of days since ${monthTag(DATA.dates[0])}. Lower means a better model fit`;
}

function render() {
  const index = DATA.modes[state.mode].index;
  const band = computeBand(index);
  renderKpis(band);
  renderMain(index, band);
  renderPosition(band);
  renderSlope();
  $("meta").textContent = `Data to ${longDate(DATA.meta.last_date)} · updated ${DATA.meta.generated_utc} · ` +
    `basket: ${DATA.modes[state.mode].label.toLowerCase()}`;
  window.__model = { gap: band.gap, share: band.share, mode: state.mode, band: state.band };
}

function renderStatic() {
  renderWeights();
  renderRolling();
}

// ---------- controls ----------
function buildSlopeTable() {
  const body = $("slope-rows");
  body.textContent = "";
  const mode = DATA.modes[state.mode];
  const events = bandEvents();
  // The regime in force at the start of the sample belongs to the last event before it.
  const lastBefore = events.filter((e) => e.date <= DATA.dates[0]).pop();
  for (const e of events) {
    const key = e === lastBefore ? "start" : (e.date > DATA.dates[0] ? e.date : null);
    const implied = key ? mode.implied_slopes[key] : null;
    const obs = key ? mode.regime_obs[key] : 0;
    const tr = body.insertRow();
    tr.className = "slope-row";

    const tdDate = tr.insertCell(); tdDate.className = "num"; tdDate.textContent = longDate(e.date);

    const tdMove = tr.insertCell();
    tdMove.textContent = MOVE_LABELS[e.action] + " ";
    if (e.off_cycle) { const t = document.createElement("span"); t.className = "tag"; t.textContent = "off-cycle"; tdMove.appendChild(t); }
    if (!e.verified) { const t = document.createElement("span"); t.className = "tag unverified"; t.textContent = "unverified"; tdMove.appendChild(t); }

    const tdNote = tr.insertCell(); tdNote.className = "note-col"; tdNote.textContent = e.note;

    const tdInput = tr.insertCell(); tdInput.className = "num";
    const range = document.createElement("input");
    Object.assign(range, { type: "range", min: -1, max: 5, step: 0.25, value: state.slopes[e.date] });
    range.setAttribute("aria-label", `Slope after ${e.date}, percent per year`);
    const num = document.createElement("input");
    Object.assign(num, { type: "number", min: -1, max: 5, step: 0.25, value: state.slopes[e.date] });
    num.setAttribute("aria-label", `Slope after ${e.date}, percent per year (number)`);
    const update = (v) => {
      const val = Math.min(5, Math.max(-1, parseFloat(v)));
      if (!Number.isFinite(val)) return;
      state.slopes[e.date] = val; range.value = val; num.value = val;
      render();
    };
    range.addEventListener("input", () => update(range.value));
    num.addEventListener("change", () => update(num.value));
    tdInput.append(range, " ", num);

    const tdImplied = tr.insertCell(); tdImplied.className = "num";
    if (implied === null || implied === undefined) {
      tdImplied.textContent = "–";
    } else {
      tdImplied.textContent = fmtPct(implied, 2, false) + " ";
      if (obs < MIN_REGIME_OBS) {
        const t = document.createElement("span"); t.className = "hint"; t.textContent = `(${obs} days, noisy)`;
        tdImplied.appendChild(t);
      }
    }
  }
}

function resetState() {
  state.mode = DATA.meta.default_mode;
  state.band = DATA.meta.default_band_pct;
  state.slopes = {};
  for (const e of bandEvents()) state.slopes[e.date] = e.slope_after;
  $("mode").value = state.mode;
  $("band").value = state.band;
  $("band-out").textContent = `±${state.band}%`;
}

function initControls() {
  const sel = $("mode");
  for (const [key, m] of Object.entries(DATA.modes)) {
    const opt = document.createElement("option"); opt.value = key; opt.textContent = m.label; sel.appendChild(opt);
  }
  sel.addEventListener("change", () => { state.mode = sel.value; buildSlopeTable(); render(); });
  $("band").addEventListener("input", (ev) => {
    state.band = parseFloat(ev.target.value);
    $("band-out").textContent = `±${state.band}%`;
    render();
  });
  $("reset").addEventListener("click", () => { resetState(); buildSlopeTable(); render(); });
}

function checkAgainstPython() {
  // On default settings the browser must reproduce the Python numbers.
  const ref = DATA.meta.python_check[state.mode];
  const band = computeBand(DATA.modes[state.mode].index);
  const ok = Math.abs(band.gap - ref.latest_deviation_pct) < 0.01 && Math.abs(band.share - ref.share_outside_band) < 0.002;
  console[ok ? "info" : "warn"](`[shadow-neer] browser vs python (${state.mode}): gap ${band.gap.toFixed(4)} vs ` +
    `${ref.latest_deviation_pct}, share ${band.share.toFixed(4)} vs ${ref.share_outside_band} -> ${ok ? "MATCH" : "MISMATCH"}`);
  return ok;
}

// ---------- boot ----------
async function main() {
  try {
    if (typeof Plotly === "undefined") throw new Error("The chart library failed to load. Check your connection and reload.");
    const res = await fetch("data.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`Could not load data.json (HTTP ${res.status}).`);
    DATA = await res.json();
  } catch (err) {
    const box = $("error"); box.style.display = "block"; box.textContent = err.message;
    $("headline").textContent = "Data unavailable";
    return;
  }
  $("k-next").textContent = DATA.meta.next_review;
  $("fit-note").textContent = `Basket fit: the inferred weights explain ${(DATA.fit.full_r2 * 100).toFixed(0)}% of weekly SGD moves ` +
    `(R² ${DATA.fit.full_r2.toFixed(3)}). The regression implies a full-sample slope of ${DATA.fit.full_implied_slope_pct.toFixed(2)}% p.a.`;
  initControls();
  resetState();
  window.__checkOk = checkAgainstPython();
  buildSlopeTable();
  render();
  renderStatic();

  let timer = null;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => { render(); renderStatic(); }, 200); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { render(); renderStatic(); });
}

main();
