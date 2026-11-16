"use strict";

const LABEL_TEXT = { outbreak: "Outbreak", not_outbreak: "Not an outbreak", unsure: "Unsure", skip: "Skipped" };
const EVENT_AXIS = {
  CASES: "Weekly cases",
  HOSPITALIZATIONS: "Weekly hospitalizations",
  PERCENT_UNWEIGHTED: "% ILI visits (unweighted)",
};
const MINE_PAGE = 25;

const state = {
  outbreak: null,     // the series on screen
  label: null,
  confidence: null,
  shownAt: 0,         // performance.now() when the series appeared
  history: [],        // ids of series shown before this one, for Back
  fromQueue: true,    // false when revisiting an earlier series
  mineOffset: 0,
  sessionCount: 0,    // labels (not skips) submitted since the page loaded
  busy: false,
  pendingNext: false, // an answer was saved, but the next series has not loaded
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const r = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (r.status === 401) { location.href = "/login"; throw new Error("not logged in"); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${r.status})`);
  }
  return data;
}

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("err", isError);
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const c of children) node.append(c);
  return node;
}

function fmtNumber(v) {
  if (v === null || v === undefined) return "–";
  return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

// ---------- progress ----------
async function refreshMe() {
  const me = await api("/api/me");
  $("who").textContent = me.display_name;
  $("admin-link").hidden = !me.is_admin;
  const parts = [];
  if (me.labeled) parts.push(`You've labeled ${me.labeled.toLocaleString()} series`);
  if (state.sessionCount) parts.push(`${state.sessionCount} this session`);
  $("progress-text").textContent = parts.length ? `${parts.join(" · ")}. Thank you!` : "";
}

// ---------- chart & metadata ----------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function renderMeta(o) {
  const meta = $("meta");
  meta.replaceChildren(
    el("div", { className: "title", textContent: `${o.disease} — ${o.location}` }),
    ...[
      ["Measure", o.event.replaceAll("_", " ").toLowerCase()],
      ["Window", `${o.start_date} → ${o.end_date}`],
      ["Weeks shown", String(o.values.length)],
      ["Total", fmtNumber(o.total)],
      ["ID", String(o.id)],
    ].map(([k, v]) => el("span", {}, `${k}: `, el("b", { textContent: v }))),
  );
}

function renderChart(o) {
  const surface = cssVar("--surface-1");
  const ink = cssVar("--text-secondary");
  const grid = cssVar("--grid");
  const series = cssVar("--series-1");
  const axisTitle = EVENT_AXIS[o.event] || o.event;
  const narrow = $("chart").clientWidth < 500;
  const trace = {
    x: o.dates,
    y: o.values,
    type: "scatter",
    mode: "lines+markers",
    name: axisTitle,
    line: { color: series, width: 2, shape: "linear" },
    marker: { color: series, size: narrow && o.values.length > 30 ? 6 : 8, line: { color: surface, width: 2 } },
    connectgaps: false,
    hovertemplate: "Week ending %{x|%Y-%m-%d}<br>%{y:,.4~f}<extra></extra>",
  };
  const axis = { gridcolor: grid, linecolor: grid, zeroline: false, color: ink, tickfont: { size: 12 } };
  const layout = {
    margin: { l: 64, r: 16, t: 8, b: 48 },
    paper_bgcolor: surface,
    plot_bgcolor: surface,
    font: { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", color: ink },
    hovermode: "x",
    hoverlabel: { bgcolor: surface, bordercolor: grid, font: { color: cssVar("--text-primary") } },
    showlegend: false,
    xaxis: { ...axis, type: "date", title: { text: "Week ending", font: { size: 13 } }, showspikes: true,
             tickangle: 0, nticks: narrow ? 4 : undefined,
             spikemode: "across", spikethickness: 1, spikecolor: ink, spikedash: "solid" },
    yaxis: { ...axis, title: { text: axisTitle, font: { size: 13 } }, rangemode: "tozero" },
  };
  Plotly.react("chart", [trace], layout, { responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] });
  $("chart").setAttribute("aria-label",
    `${axisTitle} for ${o.disease} in ${o.location}, ${o.values.length} weeks from ${o.start_date}`);

  const rows = o.values.map((v, i) =>
    el("tr", {}, el("td", { textContent: o.dates[i] }), el("td", { className: "num", textContent: fmtNumber(v) })));
  $("values-table").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", { textContent: "Week ending" }), el("th", { className: "num", textContent: axisTitle }))),
    el("tbody", {}, ...rows),
  );
}

// ---------- form state ----------
function setLabel(label) {
  state.label = label;
  document.querySelectorAll(".label-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.label === label)));
  updateSubmit();
}

function setConfidence(c) {
  state.confidence = c;
  document.querySelectorAll(".conf-btn").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.conf) === c)));
  updateSubmit();
}

function updateSubmit() {
  const locked = state.busy || state.pendingNext || !state.outbreak;
  $("submit").disabled = locked || !state.label || !state.confidence;
  $("skip").disabled = locked;
  $("back").disabled = state.busy || state.history.length === 0;
  document.querySelectorAll(".label-btn, .conf-btn").forEach((b) => { b.disabled = locked; });
  $("comment").disabled = locked;
  $("resume").hidden = state.fromQueue && !state.pendingNext && !!state.outbreak;
  $("resume").disabled = state.busy;
  $("resume").textContent = state.pendingNext ? "Load next series" : "Continue annotating";
  $("annotate").setAttribute("aria-busy", String(state.busy));
}

function show(o, { fromQueue }) {
  state.outbreak = o;
  state.fromQueue = fromQueue;
  state.pendingNext = false;
  $("annotate").hidden = false;
  $("done").hidden = true;
  renderMeta(o);
  renderChart(o);
  const mine = o.my_annotation;
  if (mine && mine.label !== "skip") {
    setLabel(mine.label);
    setConfidence(mine.confidence);
  } else {
    setLabel(null);
    setConfidence(null);
  }
  $("comment").value = mine?.comment || "";
  const notice = $("revisit-notice");
  notice.hidden = !mine;
  if (mine) {
    notice.textContent = `You answered “${LABEL_TEXT[mine.label]}” on ${new Date(mine.created_at).toLocaleString()}. ` +
      "Submitting again records a new answer; the earlier one is kept in the history.";
  }
  state.shownAt = performance.now();
  setStatus("");
  updateSubmit();
}

// ---------- navigation ----------
async function loadNext() {
  const data = await api("/api/next");
  if (data.done) {
    state.outbreak = null;
    state.pendingNext = false;
    $("annotate").hidden = true;
    $("done").hidden = false;
    updateSubmit();
    return;
  }
  show(data.outbreak, { fromQueue: true });
}

async function loadById(id, { pushHistory = true } = {}) {
  const outbreak = await api(`/api/outbreaks/${id}`);
  if (pushHistory && state.outbreak && state.outbreak.id !== id) state.history.push(state.outbreak.id);
  show(outbreak, { fromQueue: false });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function withBusy(fn) {
  if (state.busy) return;
  state.busy = true;
  updateSubmit();
  try {
    await fn();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    state.busy = false;
    updateSubmit();
  }
}

function save(label) {
  if (state.pendingNext || (label !== "skip" && (!label || !state.confidence))) return;
  return withBusy(async () => {
    const o = state.outbreak;
    if (!o) return;
    setStatus("Saving…");
    const saved = await api("/api/annotations", {
      method: "POST",
      body: JSON.stringify({
        outbreak_id: o.id,
        label,
        confidence: label === "skip" ? null : state.confidence,
        comment: $("comment").value,
        time_spent_ms: Math.round(performance.now() - state.shownAt),
      }),
    });
    state.history.push(o.id);
    if (label !== "skip" && (!o.my_annotation || o.my_annotation.label === "skip")) state.sessionCount += 1;
    o.my_annotation = saved;
    state.pendingNext = true;
    // A failed refresh must never imply that the confirmed POST failed.
    const results = await Promise.allSettled([loadNext(), refreshMe(), loadMine()]);
    const failures = [];
    if (results[0].status === "rejected") failures.push("Could not load the next series. Use “Load next series” to retry.");
    if (results.slice(1).some((r) => r.status === "rejected")) failures.push("Progress or annotation history could not refresh; reload the page to update them.");
    setStatus(`Saved #${o.id} as “${LABEL_TEXT[label]}”. ${failures.join(" ")}`.trim(), failures.length > 0);
  });
}

// ---------- my annotations ----------
async function loadMine() {
  const data = await api(`/api/my-annotations?offset=${state.mineOffset}&limit=${MINE_PAGE}`);
  const tbody = $("mine").querySelector("tbody");
  if (!data.items.length) {
    tbody.replaceChildren(el("tr", {}, el("td", { colSpan: 7, textContent: "Nothing yet." })));
  } else {
    tbody.replaceChildren(...data.items.map((a) => {
      const tr = el("tr", { className: "clickable", tabIndex: 0, title: "Open to review or change" },
        el("td", { textContent: a.outbreak_id }),
        el("td", { textContent: a.disease }),
        el("td", { textContent: a.location }),
        el("td", { textContent: a.start_date }),
        el("td", {}, el("span", { className: `pill ${a.label}`, textContent: LABEL_TEXT[a.label] })),
        el("td", { className: "num", textContent: a.confidence ?? "–" }),
        el("td", { textContent: new Date(a.created_at).toLocaleString() }),
      );
      const open = () => withBusy(() => loadById(a.outbreak_id));
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.repeat) { e.preventDefault(); e.stopPropagation(); open(); }
      });
      return tr;
    }));
  }
  const end = Math.min(state.mineOffset + MINE_PAGE, data.total);
  $("mine-info").textContent = data.total ? `${state.mineOffset + 1}–${end} of ${data.total}` : "";
  $("mine-prev").disabled = state.mineOffset === 0;
  $("mine-next").disabled = end >= data.total;
}

// ---------- wiring ----------
document.querySelectorAll(".label-btn").forEach((b) => b.addEventListener("click", () => setLabel(b.dataset.label)));
document.querySelectorAll(".conf-btn").forEach((b) => b.addEventListener("click", () => setConfidence(Number(b.dataset.conf))));
$("submit").addEventListener("click", () => save(state.label));
$("skip").addEventListener("click", () => save("skip"));
$("resume").addEventListener("click", () => withBusy(async () => {
  await loadNext();
  setStatus("");
}));
$("back").addEventListener("click", () => withBusy(async () => {
  const id = state.history.at(-1);
  if (id !== undefined) {
    await loadById(id, { pushHistory: false });
    state.history.pop();
  }
}));
$("mine-prev").addEventListener("click", () => { state.mineOffset = Math.max(0, state.mineOffset - MINE_PAGE); loadMine(); });
$("mine-next").addEventListener("click", () => { state.mineOffset += MINE_PAGE; loadMine(); });
$("logout").addEventListener("click", async () => {
  await fetch("/api/logout", { method: "POST" });
  location.href = "/login";
});

// Enter submits, unless focus is on a control whose own Enter action matters (Back, Skip, a table row…).
function focusedOtherControl(target) {
  if (target.closest?.(".label-btn, .conf-btn")) return false;
  return target instanceof HTMLButtonElement || target instanceof HTMLAnchorElement || target.closest?.("tr.clickable");
}

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.altKey) return;
  const typing = e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement;
  if (e.repeat || state.busy || state.pendingNext) {
    if (!typing && !e.ctrlKey && /^(o|n|u|[1-5]|s|b|enter)$/i.test(e.key)) e.preventDefault();
    return;
  }
  if (typing) {
    if (e.key === "Enter" && e.ctrlKey && !$("submit").disabled) { e.preventDefault(); save(state.label); }
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.ctrlKey || !state.outbreak) return;
  const k = e.key.toLowerCase();
  if (k === "o") setLabel("outbreak");
  else if (k === "n") setLabel("not_outbreak");
  else if (k === "u") setLabel("unsure");
  else if (/^[1-5]$/.test(k)) setConfidence(Number(k));
  else if (k === "s" && !$("skip").disabled) save("skip");
  else if (k === "b" && !$("back").disabled) $("back").click();
  else if (k === "enter" && !$("submit").disabled && !focusedOtherControl(e.target)) save(state.label);
  else return;
  e.preventDefault();
});

// Re-color the chart when the OS theme flips.
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.outbreak && renderChart(state.outbreak));

withBusy(async () => {
  await refreshMe();
  const requested = Number(new URLSearchParams(location.search).get("id"));
  const results = await Promise.allSettled([requested ? loadById(requested) : loadNext(), loadMine()]);
  const failure = results.find((r) => r.status === "rejected");
  if (failure) throw failure.reason;
});
