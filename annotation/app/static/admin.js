"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (v) => (v === null || v === undefined ? "–" : typeof v === "number" ? v.toLocaleString() : String(v));
const share = (n, total) => {
  const pct = total ? (100 * n) / total : 0;
  return n > 0 && pct < 0.1 ? "<0.1%" : `${pct.toFixed(1)}%`;
};
const kappa = (k) => (k === null || k === undefined ? "–" : k.toFixed(3));

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const c of children) node.append(c);
  return node;
}

function table(id, headers, rows, numeric = []) {
  const th = headers.map((h, i) => el("th", { className: numeric.includes(i) ? "num" : "", textContent: h }));
  const body = rows.length
    ? rows.map((r) => el("tr", {}, ...r.map((v, i) => el("td", { className: numeric.includes(i) ? "num" : "", textContent: fmt(v) }))))
    : [el("tr", {}, el("td", { colSpan: headers.length, textContent: "No data yet." }))];
  $(id).replaceChildren(el("thead", {}, el("tr", {}, ...th)), el("tbody", {}, ...body));
}

const pct = (v) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(1)}%`);
const SERVED = { random: "random draw", uncertain: "model unsure", revisit: "revisit", lookup: "opened by link", unknown: "before tracking" };

function tiles(id, items) {
  $(id).replaceChildren(...items.map(([k, v, note]) => el("div", { className: "card stat" },
    el("div", { className: "v", textContent: fmt(v) }), el("div", { className: "k", textContent: k }),
    ...(note ? [el("div", { className: "conf-hint", textContent: note })] : []))));
}

async function loadModel() {
  const r = await fetch("/api/admin/model");
  if (!r.ok) return;
  const m = await r.json();
  const rd = m.readiness;
  const served = Object.entries(m.served).map(([k, v]) => `${v} ${SERVED[k] || k}`).join(", ");
  if (!m.run) {
    $("model-status").textContent = `Warming up: series are served at random until there are ${rd.min_labels} ` +
      `Outbreak / Not outbreak answers with at least ${rd.min_per_class} of each ` +
      `(now ${rd.n_outbreak} Outbreak, ${rd.n_not_outbreak} Not outbreak).`;
    $("retrain").disabled = !rd.ready;
    tiles("model-tiles", []);
    $("model-detail").hidden = true;
    return;
  }
  const run = m.run, met = run.metrics, rs = met.evaluation.random_sample, p = met.predicted;
  $("model-status").textContent = `Trained ${new Date(run.created_at).toLocaleString()} on ${run.n_votes} answers ` +
    `(${run.n_outbreaks} series) · ${run.model_type.replace("_", " ")} · retrains every ${m.retrain_every} new answers` +
    (m.training ? " · training now…" : "") + (served ? ` · labels so far: ${served}` : "");
  $("retrain").disabled = m.training;
  tiles("model-tiles", [
    ["Cross-validated AUC", met.cv_auc === null ? "–" : met.cv_auc.toFixed(3), "1 = perfect, 0.5 = chance"],
    ["Accuracy on random sample", pct(rs.accuracy), rs.n ? `${rs.n} randomly drawn series` : "no random-draw labels yet"],
    ["Outbreak share (random sample)", pct(rs.share_outbreak), "expert majority in the sampled eligible pool"],
    ["Predicted outbreaks", `${p.n_predicted_outbreak.toLocaleString()} / ${p.n.toLocaleString()}`],
    ["Uncertain series", p.n_uncertain.toLocaleString(), "model probability 0.25–0.75"],
    ["Fewer labels for easy series", m.early_stop.active ? "On" : "Off",
      m.early_stop.active ? "1 confirming label is enough when the model is confident"
                          : `turns on at ${m.early_stop.min_labels}+ answers and AUC ≥ ${m.early_stop.min_auc}`],
  ]);

  const max = Math.max(...p.histogram, 1);
  $("hist").replaceChildren(...p.histogram.map((c, i) => {
    const lo = (i / 10).toFixed(1), hi = ((i + 1) / 10).toFixed(1);
    return el("div", { className: "hist-col", title: `p ${lo}–${hi}: ${c.toLocaleString()} series` },
      el("span", { className: "hist-n", textContent: c ? c.toLocaleString() : "" }),
      el("div", { className: "hist-bar", style: `height:${(100 * c) / max}%` }));
  }));
  $("hist").setAttribute("aria-label", "Series per probability bin: " + p.histogram.join(", "));

  const tbody = m.most_uncertain.map((u) => {
    const tr = el("tr", { className: "clickable", tabIndex: 0 },
      ...[String(u.id), u.disease, u.location, u.event.replaceAll("_", " ").toLowerCase(), u.start_date, u.p_outbreak.toFixed(3), u.n_labels]
        .map((v, i) => el("td", { className: i >= 5 ? "num" : "", textContent: fmt(v) })));
    const open = () => window.open(`/?id=${u.id}`, "_blank");
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    return tr;
  });
  $("uncertain").replaceChildren(
    el("thead", {}, el("tr", {}, ...["ID", "Disease", "Location", "Measure", "Start", "P(outbreak)", "Labels"]
      .map((h, i) => el("th", { className: i >= 5 ? "num" : "", textContent: h })))),
    el("tbody", {}, ...(tbody.length ? tbody : [el("tr", {}, el("td", { colSpan: 7, textContent: "None." }))])));

  table("cv", ["Model", "AUC", "Accuracy", "Log loss", ""],
    Object.entries(met.cv).map(([name, c]) => [name.replace("_", " "), c.auc?.toFixed(3), pct(c.accuracy),
      c.log_loss?.toFixed(3), name === met.selected ? "selected" : ""]), [1, 2, 3]);
  table("features", ["Feature", "Weight"], met.top_features.map((f) => [f.feature, f.weight.toFixed(3)]), [1]);
  $("features-block").hidden = met.top_features.length === 0;
  table("history", ["Run", "When", "Answers", "Model", "CV AUC", "Random-sample accuracy (n)"],
    m.history.map((h) => [h.id, new Date(h.created_at).toLocaleString(), h.n_votes, h.model_type.replace("_", " "),
      h.cv_auc?.toFixed(3), h.random_sample?.n ? `${pct(h.random_sample.accuracy)} (${h.random_sample.n})` : "–"]), [2, 4]);
  $("model-detail").hidden = false;
}

$("retrain").addEventListener("click", async () => {
  $("retrain").disabled = true;
  $("model-status").textContent = "Training…";
  const r = await fetch("/api/admin/retrain", { method: "POST" });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    $("status").textContent = d.detail || `Retrain failed (${r.status})`;
  }
  await loadModel();
});

// ---------- users ----------
async function apiJson(path, method, body) {
  const r = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = data.detail;
    throw new Error(Array.isArray(d) ? d.map((e) => e.msg.replace(/^Value error, /, "")).join("; ") : d || `Failed (${r.status})`);
  }
  return data;
}

function userStatus(text, isError = false) {
  $("user-status").textContent = text;
  $("user-status").classList.toggle("err", isError);
}

function passwordRow(u, cols) {
  const input = el("input", { type: "password", placeholder: "new password (at least 8 characters)", autocomplete: "new-password" });
  const show = el("input", { type: "checkbox" });
  show.addEventListener("change", () => { input.type = show.checked ? "text" : "password"; });
  const save = el("button", { className: "primary", textContent: "Save" });
  const cancel = el("button", { textContent: "Cancel" });
  const row = el("tr", { className: "inline-edit" }, el("td", { colSpan: cols },
    el("div", { className: "row" }, el("span", { textContent: `New password for ${u.username}:` }), input,
      el("label", { className: "check" }, show, " Show"), save, cancel)));
  const submit = async () => {
    try {
      await apiJson(`/api/admin/users/${encodeURIComponent(u.username)}`, "PATCH", { password: input.value });
      userStatus(`Password updated for ${u.username}.`);
      await loadUsers();
    } catch (err) { userStatus(err.message, true); }
  };
  save.addEventListener("click", submit);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); if (e.key === "Escape") row.remove(); });
  cancel.addEventListener("click", () => row.remove());
  setTimeout(() => input.focus(), 0);
  return row;
}

async function loadUsers() {
  const r = await fetch("/api/admin/users");
  if (!r.ok) return;
  const users = await r.json();
  const headers = ["Username", "Name", "Affiliation", "Expertise", "Role", "Labeled", "Status", ""];
  const rows = users.map((u) => {
    const setPw = el("button", { className: "link", textContent: "Set password" });
    const toggle = el("button", { className: "link", textContent: u.is_active ? "Deactivate" : "Activate" });
    const tr = el("tr", { className: u.is_active ? "" : "inactive" },
      ...[u.username, u.display_name, u.affiliation, u.expertise, u.is_admin ? "admin" : "expert",
        u.n_labels, u.is_active ? "active" : "deactivated"]
        .map((v, i) => el("td", { className: i === 5 ? "num" : "", textContent: fmt(v) })),
      el("td", { className: "actions-cell" }, setPw, toggle));
    setPw.addEventListener("click", () => {
      const next = tr.nextElementSibling;
      if (next?.classList.contains("inline-edit")) { next.remove(); return; }
      tr.after(passwordRow(u, headers.length));
    });
    toggle.addEventListener("click", async () => {
      try {
        await apiJson(`/api/admin/users/${encodeURIComponent(u.username)}`, "PATCH", { is_active: !u.is_active });
        userStatus(`${u.username} ${u.is_active ? "deactivated (their labels are kept)" : "reactivated"}.`);
        await loadUsers();
      } catch (err) { userStatus(err.message, true); }
    });
    return tr;
  });
  $("users").replaceChildren(
    el("thead", {}, el("tr", {}, ...headers.map((h, i) => el("th", { className: i === 5 ? "num" : "", textContent: h })))),
    el("tbody", {}, ...rows));
}

$("show-password").addEventListener("change", (e) => {
  $("user-form").password.type = e.target.checked ? "text" : "password";
});
$("user-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  try {
    const u = await apiJson("/api/admin/users", "POST", {
      username: f.username.value, password: f.password.value, display_name: f.display_name.value,
      affiliation: f.affiliation.value, expertise: f.expertise.value, is_admin: f.is_admin.checked,
    });
    userStatus(`Created ${u.is_admin ? "admin" : "expert"} account “${u.username}”.`);
    f.reset();
    f.password.type = "password";
    await loadUsers();
  } catch (err) { userStatus(err.message, true); }
});

(async () => {
  const r = await fetch("/api/admin/stats");
  if (r.status === 401) { location.href = "/login"; return; }
  if (r.status === 403) { $("status").textContent = "This page is for admins only."; return; }
  const s = await r.json();
  const totalLabels = Object.entries(s.labels).filter(([k]) => k !== "skip").reduce((a, [, v]) => a + v, 0);

  $("tiles").replaceChildren(...[
    ["Outbreak series", s.total_outbreaks],
    [`Complete (≥${s.target_annotations} labels)`, s.complete],
    ["Labels (current)", totalLabels],
    ["Annotators", s.annotators.length],
    ["Overall Fleiss' kappa", kappa(s.agreement.overall.fleiss_kappa)],
  ].map(([k, v]) => el("div", { className: "card stat" }, el("div", { className: "v", textContent: fmt(v) }), el("div", { className: "k", textContent: k }))));

  table("coverage", ["Labels per series", "Series", "Share"],
    Object.entries(s.coverage).map(([n, c]) => [n, c, share(c, s.total_outbreaks)]), [1, 2]);

  table("annotators", ["User", "Outbreak", "Not outbreak", "Unsure", "Skipped", "Last active"],
    s.annotators.map((a) => [a.username, a.labels.outbreak || 0, a.labels.not_outbreak || 0, a.labels.unsure || 0,
      a.labels.skip || 0, a.last_active ? new Date(a.last_active).toLocaleString() : "–"]), [1, 2, 3, 4]);

  table("agreement", ["Disease", "Series with 2+ labels", "Unanimous", "Fleiss' kappa"],
    [["All", s.agreement.overall], ...Object.entries(s.agreement.by_disease)].map(([d, a]) =>
      [d, a.outbreaks_with_2plus_labels, a.unanimous, kappa(a.fleiss_kappa)]), [1, 2, 3]);
  await Promise.all([loadUsers(), loadModel()]);
})().catch((err) => { $("status").textContent = err.message; });
