"use strict";

// Segment review: the whole series, the method's proposed outbreak windows, and the expert's
// decision on each (accept / adjust / reject), plus outbreaks the expert adds.

const EVENT_AXIS = {
  CASES: "Weekly cases",
  HOSPITALIZATIONS: "Weekly hospitalizations",
  PERCENT_UNWEIGHTED: "% ILI visits (unweighted)",
};
const STATE_TEXT = { pending: "Undecided", accept: "Accepted", adjust: "Adjusted", reject: "Rejected", added: "Added" };
const MINE_PAGE = 25;
const DAY = 86400000;

const state = {
  runId: null,
  data: null,        // payload from the API: run, series, proposals, my_review
  items: [],         // [{proposed_id|null, verdict|null, start_idx, end_idx, orig_start, orig_end}]
  confidence: null,
  shownAt: 0,
  fromQueue: true,
  mineOffset: 0,
  busy: false,
  pendingNext: false,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const r = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  if (r.status === 401) { location.href = "/login"; throw new Error("not logged in"); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${r.status})`);
  }
  return data;
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("err", isError);
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const c of children) node.append(c);
  return node;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmtNumber(v) {
  if (v === null || v === undefined) return "–";
  return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

// ---------- week index <-> date ----------
const dates = () => state.data.series.dates;
const nWeeks = () => dates().length;
const t0 = () => Date.parse(dates()[0]);
function idxOfTime(ms) {
  return Math.max(0, Math.min(nWeeks() - 1, Math.round((ms - t0()) / (7 * DAY))));
}
function idxOfDate(s) { return idxOfTime(Date.parse(s)); }

// ---------- items ----------
function itemState(it) {
  if (it.proposed_id === null) return "added";
  return it.verdict || "pending";
}

function initItems(d) {
  const props = d.proposals;
  const prev = d.my_review && !d.my_review.skipped ? d.my_review.items : [];
  const byId = new Map(prev.filter((i) => i.proposed_id !== null).map((i) => [i.proposed_id, i]));
  state.items = props.map((p) => {
    const mine = byId.get(p.id);
    return {
      proposed_id: p.id, verdict: mine ? mine.verdict : null,
      start_idx: mine && mine.start_idx !== null ? mine.start_idx : p.start_idx,
      end_idx: mine && mine.end_idx !== null ? mine.end_idx : p.end_idx,
      orig_start: p.start_idx, orig_end: p.end_idx, proposal: p,
    };
  });
  for (const i of prev.filter((x) => x.proposed_id === null)) {
    state.items.push({ proposed_id: null, verdict: "added", start_idx: i.start_idx, end_idx: i.end_idx });
  }
  sortItems();
}

function sortItems() { state.items.sort((a, b) => a.start_idx - b.start_idx || a.end_idx - b.end_idx); }

function setBounds(it, start, end) {
  if (start > end) [start, end] = [end, start];
  it.start_idx = start;
  it.end_idx = end;
  if (it.proposed_id !== null) {
    const moved = start !== it.orig_start || end !== it.orig_end;
    if (moved) it.verdict = "adjust";
    else if (it.verdict === "adjust") it.verdict = "accept";
  }
}

// ---------- rendering ----------
function renderMeta() {
  const s = state.data.series;
  const n = state.data.proposals.length;
  $("meta").replaceChildren(
    el("div", { className: "title", textContent: `${s.disease} — ${s.location}` }),
    ...[
      ["Measure", s.event.replaceAll("_", " ").toLowerCase()],
      ["Weeks", `${s.dates[0]} → ${s.dates.at(-1)} (${s.values.length})`],
      ["Proposed outbreaks", String(n)],
      ["Run", state.data.run.name],
      ["Series", String(s.id)],
    ].map(([k, v]) => el("span", {}, `${k}: `, el("b", { textContent: v }))),
  );
}

function colorOf(st) {
  return { pending: cssVar("--seg-pending"), accept: cssVar("--seg-accept"), adjust: cssVar("--seg-accept"),
    added: cssVar("--seg-added"), reject: cssVar("--seg-reject") }[st];
}

function bandX(i0, i1) {
  // A week's value is plotted at its week-ending date; the band covers half a week either side.
  return [new Date(t0() + (i0 * 7 - 3.5) * DAY).toISOString(), new Date(t0() + (i1 * 7 + 3.5) * DAY).toISOString()];
}

function shapesAndNotes() {
  const shapes = [];
  const notes = [];
  state.items.forEach((it, k) => {
    const st = itemState(it);
    const c = colorOf(st);
    const [x0, x1] = bandX(it.start_idx, it.end_idx);
    shapes.push({
      type: "rect", xref: "x", yref: "paper", x0, x1, y0: 0, y1: 1, editable: st !== "reject",
      fillcolor: c, opacity: st === "reject" ? 0.08 : 0.16, layer: "below",
      line: { color: c, width: 1.5, dash: st === "pending" ? "dash" : "solid" },
      name: `seg-${k}`,
    });
    const p = it.proposal;
    if (p && st !== "reject") {
      if (p.core_start_idx !== null && p.core_end_idx !== null) {
        const [c0, c1] = bandX(p.core_start_idx, p.core_end_idx);
        shapes.push({ type: "rect", xref: "x", yref: "paper", x0: c0, x1: c1, y0: 0, y1: 1, editable: false,
          fillcolor: c, opacity: 0.12, layer: "below", line: { width: 0 } });
      }
      if (p.peak_idx !== null) {
        const xp = dates()[p.peak_idx];
        shapes.push({ type: "line", xref: "x", yref: "paper", x0: xp, x1: xp, y0: 0, y1: 1, editable: false,
          line: { color: c, width: 1, dash: "dot" } });
      }
    }
    notes.push({ x: dates()[Math.round((it.start_idx + it.end_idx) / 2)], y: 1, xref: "x", yref: "paper",
      yanchor: "bottom", text: String(k + 1), showarrow: false, font: { color: c, size: 12 } });
  });
  return { shapes, notes };
}

function renderChart() {
  if (typeof Plotly === "undefined") {
    // The chart library failed to load (blocked CDN). The table below still works.
    $("chart").textContent = "The chart could not load. You can still review using the dates in the table below.";
    return;
  }
  const s = state.data.series;
  const surface = cssVar("--surface-1");
  const ink = cssVar("--text-secondary");
  const grid = cssVar("--grid");
  const axisTitle = EVENT_AXIS[s.event] || s.event;
  const keepRange = state.keepRange;
  const { shapes, notes } = shapesAndNotes();
  const axis = { gridcolor: grid, linecolor: grid, zeroline: false, color: ink, tickfont: { size: 12 } };
  const layout = {
    margin: { l: 64, r: 16, t: 22, b: 48 },
    paper_bgcolor: surface, plot_bgcolor: surface,
    font: { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", color: ink },
    hovermode: "x", showlegend: false, shapes, annotations: notes,
    hoverlabel: { bgcolor: surface, bordercolor: grid, font: { color: cssVar("--text-primary") } },
    xaxis: { ...axis, type: "date", title: { text: "Week ending", font: { size: 13 } }, ...(keepRange ? { range: keepRange } : {}) },
    yaxis: { ...axis, title: { text: axisTitle, font: { size: 13 } }, rangemode: "tozero" },
    uirevision: s.id,
  };
  const trace = {
    x: s.dates, y: s.values, type: "scatter", mode: s.values.length > 150 ? "lines" : "lines+markers",
    line: { color: cssVar("--series-1"), width: 1.8 }, marker: { size: 5, color: cssVar("--series-1") },
    connectgaps: false, hovertemplate: "Week ending %{x|%Y-%m-%d}<br>%{y:,.4~f}<extra></extra>",
  };
  Plotly.react("chart", [trace], layout, {
    responsive: true, displaylogo: false, edits: { shapePosition: true },
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  });
  if (!state.relayoutBound) {
    $("chart").on("plotly_relayout", onRelayout);
    state.relayoutBound = true;
  }
  $("chart").setAttribute("aria-label",
    `${axisTitle} for ${s.disease} in ${s.location}, ${s.values.length} weeks, ${state.items.length} outbreak windows`);
}

function onRelayout(ev) {
  if (!state.data) return;
  if (ev["xaxis.range[0]"] !== undefined) state.keepRange = [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
  else if (ev["xaxis.autorange"]) state.keepRange = null;
  // Shape edits arrive as "shapes[k].x0" / "shapes[k].x1" (and y, which we ignore).
  const touched = new Map();
  for (const [key, v] of Object.entries(ev)) {
    const m = /^shapes\[(\d+)\]\.(x0|x1)$/.exec(key);
    if (m) touched.set(Number(m[1]), { ...(touched.get(Number(m[1])) || {}), [m[2]]: v });
  }
  if (!touched.size) return;
  const shapes = $("chart").layout.shapes;
  for (const [si, upd] of touched) {
    const name = shapes[si]?.name || "";
    const k = Number(name.replace("seg-", ""));
    const it = state.items[k];
    if (!name.startsWith("seg-") || !it) continue;
    const x0 = Date.parse(upd.x0 ?? shapes[si].x0) + 3.5 * DAY;
    const x1 = Date.parse(upd.x1 ?? shapes[si].x1) - 3.5 * DAY;
    setBounds(it, idxOfTime(x0), idxOfTime(x1));
  }
  sortItems();
  renderAll();
}

function dateInput(it, which) {
  const input = el("input", { type: "date", min: dates()[0], max: dates().at(-1), step: 7,
    value: dates()[which === "start" ? it.start_idx : it.end_idx], disabled: itemState(it) === "reject" || locked() });
  input.setAttribute("aria-label", which === "start" ? "Start week" : "End week");
  input.addEventListener("change", () => {
    if (!input.value) return;
    const i = idxOfDate(input.value);
    if (which === "start") setBounds(it, i, it.end_idx);
    else setBounds(it, it.start_idx, i);
    sortItems();
    renderAll();
  });
  return input;
}

function renderTable() {
  const tbody = $("segs").querySelector("tbody");
  if (!state.items.length) {
    tbody.replaceChildren(el("tr", {}, el("td", { colSpan: 5,
      textContent: "The method found no outbreak in this series. Add any you see, or submit as is." })));
    return;
  }
  tbody.replaceChildren(...state.items.map((it, k) => {
    const st = itemState(it);
    const btn = (text, verdict) => {
      const b = el("button", { textContent: text, disabled: locked() });
      b.setAttribute("aria-pressed", String(it.verdict === verdict || (verdict === "accept" && it.verdict === "adjust")));
      b.addEventListener("click", () => {
        if (verdict === "accept") {
          it.verdict = it.start_idx !== it.orig_start || it.end_idx !== it.orig_end ? "adjust" : "accept";
        } else it.verdict = verdict;
        renderAll();
      });
      return b;
    };
    const decide = el("div", { className: "decide" }, el("span", { className: "state", textContent: STATE_TEXT[st] }));
    if (it.proposed_id !== null) {
      decide.append(btn("Accept", "accept"), btn("Reject", "reject"));
      if (st === "adjust") {
        const reset = el("button", { textContent: "Reset", title: "Back to the proposed dates", disabled: locked() });
        reset.addEventListener("click", () => { setBounds(it, it.orig_start, it.orig_end); sortItems(); renderAll(); });
        decide.append(reset);
      }
    } else {
      const rm = el("button", { textContent: "Remove", disabled: locked() });
      rm.addEventListener("click", () => { state.items.splice(k, 1); renderAll(); });
      decide.append(rm);
    }
    const sw = el("i", { className: `sw ${st === "adjust" ? "accept" : st}` });
    return el("tr", { className: st },
      el("td", { className: "seg-n" }, sw, String(k + 1)),
      el("td", {}, dateInput(it, "start")),
      el("td", {}, dateInput(it, "end")),
      el("td", { className: `num${it.end_idx - it.start_idx + 1 > 60 ? " too-long" : ""}`,
        textContent: String(it.end_idx - it.start_idx + 1),
        title: it.end_idx - it.start_idx + 1 > 60 ? "Longer than the 60 weeks a benchmark row can hold; split it if it covers several waves." : "" }),
      el("td", {}, decide));
  }));
}

function undecided() { return state.items.filter((it) => itemState(it) === "pending").length; }
function locked() { return state.busy || state.pendingNext || !state.data; }

function updateControls() {
  const lock = locked();
  const unusable = $("unusable").checked;
  const left = undecided();
  $("submit").disabled = lock || !state.confidence || (!unusable && left > 0);
  $("todo").textContent = !state.data ? "" : unusable ? "Series marked unusable."
    : left ? `${left} window${left > 1 ? "s" : ""} still need a decision.` : "";
  for (const id of ["skip", "add", "accept-rest", "unusable", "comment"]) $(id).disabled = lock;
  $("accept-rest").disabled = lock || left === 0;
  document.querySelectorAll(".conf-btn").forEach((b) => {
    b.disabled = lock;
    b.setAttribute("aria-pressed", String(Number(b.dataset.conf) === state.confidence));
  });
  $("resume").hidden = state.fromQueue && !state.pendingNext && !!state.data;
  $("resume").textContent = state.pendingNext ? "Load next series" : "Continue reviewing";
  $("review").setAttribute("aria-busy", String(state.busy));
}

function renderAll() {
  renderChart();
  renderTable();
  updateControls();
}

function show(d, { fromQueue }) {
  state.data = d;
  state.fromQueue = fromQueue;
  state.pendingNext = false;
  state.keepRange = null;
  $("review").hidden = false;
  $("done").hidden = true;
  initItems(d);
  const mine = d.my_review;
  state.confidence = mine && !mine.skipped ? mine.confidence : null;
  $("unusable").checked = !!(mine && mine.series_unusable);
  $("comment").value = mine?.comment || "";
  const notice = $("revisit-notice");
  notice.hidden = !mine;
  if (mine) {
    notice.textContent = `You reviewed this series on ${new Date(mine.created_at).toLocaleString()}` +
      `${mine.skipped ? " (skipped)" : ""}. Submitting again records a new review; the earlier one is kept.`;
  }
  renderMeta();
  renderAll();
  state.shownAt = performance.now();
  setStatus("");
}

// ---------- data flow ----------
async function loadRuns() {
  const runs = await api("/api/seg/runs");
  const sel = $("run");
  sel.replaceChildren(...runs.map((r) => el("option", { value: r.id,
    textContent: `${r.name} (${r.reviewed_by_me}/${r.n_series} reviewed)` })));
  const wanted = Number(new URLSearchParams(location.search).get("run"));
  state.runId = runs.some((r) => r.id === wanted) ? wanted : runs[0]?.id ?? null;
  if (state.runId) sel.value = String(state.runId);
  return runs;
}

async function refreshMe() {
  const me = await api("/api/me");
  $("who").textContent = me.display_name;
  $("admin-link").hidden = !me.is_admin;
}

async function loadNext() {
  const d = await api(`/api/seg/next?run_id=${state.runId}`);
  if (d.done) {
    state.data = null;
    state.pendingNext = false;
    $("review").hidden = true;
    $("done").hidden = false;
    updateControls();
    return;
  }
  show(d, { fromQueue: true });
}

async function loadById(id) {
  show(await api(`/api/seg/series/${id}?run_id=${state.runId}`), { fromQueue: false });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function withBusy(fn) {
  if (state.busy) return;
  state.busy = true;
  updateControls();
  try {
    await fn();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    state.busy = false;
    if (state.data) renderTable();
    updateControls();
  }
}

function save(skipped) {
  if (state.pendingNext || (!skipped && $("submit").disabled)) return;
  return withBusy(async () => {
    const d = state.data;
    if (!d) return;
    setStatus("Saving…");
    const unusable = $("unusable").checked;
    const items = skipped ? [] : state.items
      .filter((it) => !(unusable && itemState(it) === "pending"))
      .map((it) => ({ proposed_id: it.proposed_id, verdict: it.proposed_id === null ? "added" : it.verdict,
        start_idx: it.start_idx, end_idx: it.end_idx }));
    await api("/api/seg/reviews", { method: "POST", body: JSON.stringify({
      run_id: d.run.id, series_id: d.series.id, items, skipped, series_unusable: !skipped && unusable,
      confidence: skipped ? null : state.confidence, comment: $("comment").value,
      time_spent_ms: Math.round(performance.now() - state.shownAt),
    }) });
    state.pendingNext = true;
    const results = await Promise.allSettled([loadNext(), loadMine(), loadRuns()]);
    const failed = results[0].status === "rejected";
    setStatus(`Saved series ${d.series.id}${skipped ? " as skipped" : ""}.` +
      (failed ? " Could not load the next series. Use “Load next series” to retry." : ""), failed);
  });
}

async function loadMine() {
  if (!state.runId) return;
  const data = await api(`/api/seg/my-reviews?run_id=${state.runId}&offset=${state.mineOffset}&limit=${MINE_PAGE}`);
  const tbody = $("mine").querySelector("tbody");
  if (!data.items.length) {
    tbody.replaceChildren(el("tr", {}, el("td", { colSpan: 6, textContent: "Nothing yet." })));
  } else {
    tbody.replaceChildren(...data.items.map((r) => {
      const tr = el("tr", { className: "clickable", tabIndex: 0, title: "Open to review again" },
        el("td", { textContent: r.series_id }), el("td", { textContent: r.disease }),
        el("td", { textContent: r.location }), el("td", { textContent: r.summary }),
        el("td", { className: "num", textContent: r.confidence ?? "–" }),
        el("td", { textContent: new Date(r.created_at).toLocaleString() }));
      const open = () => withBusy(() => loadById(r.series_id));
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.repeat) { e.preventDefault(); e.stopPropagation(); open(); } });
      return tr;
    }));
  }
  const end = Math.min(state.mineOffset + MINE_PAGE, data.total);
  $("mine-info").textContent = data.total ? `${state.mineOffset + 1}–${end} of ${data.total}` : "";
  $("mine-prev").disabled = state.mineOffset === 0;
  $("mine-next").disabled = end >= data.total;
}

// ---------- wiring ----------
document.querySelectorAll(".conf-btn").forEach((b) => b.addEventListener("click", () => {
  state.confidence = Number(b.dataset.conf);
  updateControls();
}));
$("unusable").addEventListener("change", updateControls);
$("add").addEventListener("click", () => {
  // New window in the middle of what is on screen, about 12 weeks wide; the expert then drags it.
  const range = $("chart").layout?.xaxis?.range;
  const lo = range ? idxOfTime(Date.parse(range[0])) : 0;
  const hi = range ? idxOfTime(Date.parse(range[1])) : nWeeks() - 1;
  const mid = Math.round((lo + hi) / 2);
  const half = Math.max(1, Math.min(6, Math.floor((hi - lo) / 4)));
  state.items.push({ proposed_id: null, verdict: "added", start_idx: Math.max(0, mid - half), end_idx: Math.min(nWeeks() - 1, mid + half) });
  sortItems();
  renderAll();
  setStatus("Added a window. Drag its edges on the chart or edit its dates in the table.");
});
$("accept-rest").addEventListener("click", () => {
  for (const it of state.items) if (itemState(it) === "pending") it.verdict = "accept";
  renderAll();
});
$("submit").addEventListener("click", () => save(false));
$("skip").addEventListener("click", () => save(true));
$("resume").addEventListener("click", () => withBusy(async () => { await loadNext(); setStatus(""); }));
$("run").addEventListener("change", () => withBusy(async () => {
  state.runId = Number($("run").value);
  state.mineOffset = 0;
  history.replaceState(null, "", `?run=${state.runId}`);
  await Promise.all([loadNext(), loadMine()]);
}));
$("mine-prev").addEventListener("click", () => { state.mineOffset = Math.max(0, state.mineOffset - MINE_PAGE); loadMine(); });
$("mine-next").addEventListener("click", () => { state.mineOffset += MINE_PAGE; loadMine(); });
$("logout").addEventListener("click", async () => { await fetch("/api/logout", { method: "POST" }); location.href = "/login"; });

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.altKey || e.ctrlKey || e.repeat) return;
  const t = e.target;
  if (t instanceof HTMLTextAreaElement || t instanceof HTMLInputElement || t instanceof HTMLSelectElement) {
    if (e.key === "Escape") t.blur();
    return;
  }
  if (locked()) return;
  const k = e.key.toLowerCase();
  if (/^[1-5]$/.test(k)) { state.confidence = Number(k); updateControls(); }
  else if (k === "a" && !$("accept-rest").disabled) $("accept-rest").click();
  else if (k === "s") save(true);
  else if (k === "enter" && !$("submit").disabled && !(t instanceof HTMLButtonElement) && !t.closest?.("tr.clickable")) save(false);
  else return;
  e.preventDefault();
});

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.data && renderChart());

withBusy(async () => {
  await refreshMe();
  const runs = await loadRuns();
  if (!runs.length) {
    $("review").hidden = true;
    setStatus("No segmentation run is loaded yet. An admin can load one with annotation/scripts/load_segments.py.");
    return;
  }
  const requested = Number(new URLSearchParams(location.search).get("series"));
  await Promise.all([requested ? loadById(requested) : loadNext(), loadMine()]);
});
