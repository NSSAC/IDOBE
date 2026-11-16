// Workflow regressions using Node's built-in runner; no browser or npm dependencies.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");

const source = readFileSync(`${__dirname}/../app/static/app.js`, "utf8");

async function harness() {
  const elements = new Map();
  const handlers = {};
  class Element {
    value = "";
    disabled = false;
    hidden = false;
    textContent = "";
    listeners = {};
    classList = { toggle() {} };
    setAttribute() {}
    replaceChildren() {}
    append() {}
    closest() { return null; }
    querySelector() { return new Element(); }
    addEventListener(name, fn) { this.listeners[name] = fn; }
  }
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  let respond = () => ({});
  const requests = [];
  const context = vm.createContext({
    document: {
      getElementById: element,
      querySelectorAll: () => [],
      createElement: () => new Element(),
      addEventListener: (name, fn) => { handlers[name] = fn; },
    },
    fetch: async (path, options = {}) => {
      requests.push({ path, options });
      const data = await respond(path, options);
      return { ok: true, status: 200, json: async () => data };
    },
    location: { search: "" },
    window: { scrollTo() {} },
    performance: { now: () => 100 },
    matchMedia: () => ({ addEventListener() {} }),
    URLSearchParams,
    HTMLButtonElement: class {},
    HTMLAnchorElement: class {},
    HTMLInputElement: class {},
    HTMLTextAreaElement: class {},
  });
  vm.runInContext(source, context);
  // Startup fails harmlessly against the empty response; wait for its error handler.
  await new Promise(setImmediate);
  vm.runInContext("renderMeta = () => {}; renderChart = () => {};", context);
  const run = (js) => vm.runInContext(js, context);
  run('show({ id: 7, my_annotation: null }, { fromQueue: true }); setLabel("outbreak"); setConfidence(4);');
  return {
    run, element, handlers, requests,
    respond: (fn) => { respond = fn; },
  };
}

function normalResponse(path) {
  if (path === "/api/me") return { display_name: "Test", labeled: 1 };
  if (path.startsWith("/api/my-annotations")) return { items: [], total: 0 };
  if (path === "/api/annotations") return { label: "outbreak", confidence: 4 };
  if (path === "/api/next") return { done: true };
  throw new Error("unexpected request");
}

test("a successful save advances to a fresh form and records the previous series", async () => {
  const h = await harness();
  h.respond((path) => path === "/api/next"
    ? { done: false, outbreak: { id: 8, my_annotation: null } }
    : normalResponse(path));
  await h.run('save("outbreak")');
  assert.equal(h.run("state.outbreak.id"), 8);
  assert.equal(h.run("state.label"), null);
  assert.equal(h.run("state.confidence"), null);
  assert.equal(h.run("state.history.at(-1)"), 7);
  assert.equal(h.run("state.sessionCount"), 1);
  assert.equal(h.element("comment").disabled, false);
  assert.equal(h.element("resume").hidden, true);
  assert.match(h.element("status").textContent, /Saved #7/);
});

test("a confirmed save survives next-series failure and retries without another POST", async () => {
  const h = await harness();
  h.respond((path) => {
    if (path === "/api/next") throw new Error("offline");
    return normalResponse(path);
  });
  await h.run('save("outbreak")');
  assert.match(h.element("status").textContent, /Saved #7.*Could not load the next series/);
  assert.equal(h.element("submit").disabled, true);
  assert.equal(h.element("resume").hidden, false);
  assert.equal(h.element("comment").disabled, true);
  await h.run('save("outbreak")');
  assert.equal(h.requests.filter((r) => r.options.method === "POST").length, 1);
  h.respond(normalResponse);
  await h.element("resume").listeners.click();
  assert.equal(h.run("state.pendingNext"), false);
  assert.equal(h.element("done").hidden, false);
  assert.equal(h.requests.filter((r) => r.options.method === "POST").length, 1);
});

test("a failed POST preserves the answer and comment for retry", async () => {
  const h = await harness();
  h.element("comment").value = "Reporting artifact";
  h.respond(() => { throw new Error("offline"); });
  await h.run('save("outbreak")');
  assert.equal(h.run("state.pendingNext"), false);
  assert.equal(h.run("state.label"), "outbreak");
  assert.equal(h.element("comment").value, "Reporting artifact");
  assert.equal(h.element("submit").disabled, false);
  assert.equal(h.run("state.history.length"), 0);
});

test("history refresh failure is distinguished from a successful save and queue completion", async () => {
  const h = await harness();
  h.respond((path) => {
    if (path.startsWith("/api/my-annotations")) throw new Error("offline");
    return normalResponse(path);
  });
  await h.run('save("outbreak")');
  assert.equal(h.element("done").hidden, false);
  assert.match(h.element("status").textContent, /Saved #7.*history could not refresh/);
});

test("reviewing an answer exposes a way back to the queue", async () => {
  const h = await harness();
  h.run('show({ id: 8, my_annotation: null }, { fromQueue: false });');
  assert.equal(h.element("resume").hidden, false);
});

test("failed back navigation retains the current series and back history", async () => {
  const h = await harness();
  h.run("state.history.push(3)");
  h.respond(() => { throw new Error("offline"); });
  await h.element("back").listeners.click();
  assert.equal(h.run("state.outbreak.id"), 7);
  assert.equal(h.run("state.history.at(-1)"), 3);
});

test("held shortcuts and shortcuts during a request cannot change the answer", async () => {
  const h = await harness();
  for (const busy of [false, true]) {
    h.run(`state.busy = ${busy}`);
    let prevented = false;
    h.handlers.keydown({
      key: "n", repeat: !busy, target: h.element("submit"),
      preventDefault() { prevented = true; },
    });
    assert.equal(h.run("state.label"), "outbreak");
    assert.equal(prevented, true);
  }
});
