// Charts of the page: the constant-policy plane, the construction figures, the sealed
// panel, the matched controls and the prompt ablations.  Values come from the data
// files; positions use the printed values, so a chart and its table never disagree.

import {
  DATA,
  centreIn,
  clear,
  el,
  hideTooltip,
  planeAxes,
  scale,
  showTooltip,
  spreadLabels,
  svg,
  tipRow,
} from "./lib.js";

const CHECK = "M3 8.5 6.5 12 13 4.5";

/** Redraw a chart whenever its container changes width. */
function responsive(container, draw) {
  let width = 0;
  const observer = new ResizeObserver(() => {
    const next = Math.round(container.clientWidth);
    if (next && next !== width) {
      width = next;
      draw(width);
    }
  });
  observer.observe(container);
}

function panelRows() {
  return DATA.results.panel.rows;
}

function column(name) {
  return DATA.results.panel.columns.indexOf(name);
}

/** Parse a printed rate ('.426', '+.008', '1.000') into a number. */
function num(text) {
  return Number(String(text).replace("+", ""));
}

/** Parse a printed exact count such as '12,288/12,288' or '4 of 16 positions' into a share. */
function share(text) {
  const match = String(text).replace(/,/g, "").match(/(\d+)\s*(?:\/|of)\s*(\d+)/);
  return Number(match[1]) / Number(match[2]);
}

function checkIcon() {
  return svg("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" }, [svg("path", { d: CHECK })]);
}

/** A meter row: name, bar and printed figure; optional mark labelled at the bar's own value. */
function meterRow(name, note, figure, value, { mark = null } = {}) {
  const bar = el("div", { class: "bar", role: "img", "aria-label": `${name}: ${figure}` }, [
    el("div", { class: "fill", style: `width:${(value * 100).toFixed(2)}%` }),
  ]);
  if (mark) {
    bar.append(
      el("div", { class: "mark", style: `left:calc(${(mark.at * 100).toFixed(2)}% - 1px)` }),
      el("div", { class: "mark-label", style: `left:${(mark.at * 100).toFixed(2)}%`, text: mark.text }),
    );
  }
  return el("li", { class: mark ? "chance" : null }, [
    el("div", { class: "gate-name" }, [name, note ? el("small", { text: note }) : null]),
    bar,
    el("div", { class: "figure" }, [checkIcon(), el("span", { text: figure })]),
  ]);
}

// --- constant policy ----------------------------------------------------------------

export function initConstant() {
  const policy = DATA.results.constant_policy;
  const grid = policy.grid;
  const rows = panelRows();
  const slider = document.getElementById("alpha");
  const output = document.getElementById("alpha-out");
  const plane = document.getElementById("constant-plane");
  const tiles = document.getElementById("constant-tiles");
  const meter = document.getElementById("bja-meter");
  const figure = document.getElementById("constant-figure");
  const oracle = rows["exact-oracle"].values;
  const ticks = DATA.results.axes.unit;
  slider.max = String(grid.length - 1);
  slider.value = String(Math.round((grid.length - 1) / 2));

  const formulas = document.getElementById("constant-formulas");
  ["AR = FCR = α", policy.bda_formula, policy.bja_formula].forEach((text) =>
    formulas.append(el("div", { text })),
  );

  const names = { AR: "AR", FCR: "FCR", BDA: "BDA", BJA: "BJA" };
  const tileNodes = {};
  for (const key of Object.keys(names)) {
    const value = el("dd", { "data-metric": key });
    const note = el("small", { text: `exact oracle ${oracle[column(key)]}` });
    tiles.append(el("div", {}, [el("dt", { text: names[key] }), value, note]));
    tileNodes[key] = value;
  }

  // The AR-FCR plane: the diagonal of constant policies and the oracle corner.
  const size = 320;
  const pad = { left: 46, right: 14, top: 14, bottom: 40 };
  clear(plane);
  plane.setAttribute("viewBox", `0 0 ${size} ${size}`);
  const { x, y } = planeAxes(plane, {
    size,
    pad,
    ticks,
    xTitle: "FCR: answers when underdetermined",
    yTitle: "AR: answers when answerable",
  });
  plane.append(
    svg("path", { class: "region", d: `M${x(0)} ${y(0)} L${x(1)} ${y(1)} L${x(0)} ${y(1)} Z` }),
    svg("line", { class: "diagonal", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }),
  );
  const ox = x(num(oracle[column("FCR")]));
  const oy = y(num(oracle[column("AR")]));
  plane.append(
    svg("rect", { class: "oracle", x: ox - 6, y: oy - 6, width: 12, height: 12, transform: `rotate(45 ${ox} ${oy})` }),
    svg("text", { class: "oracle-label", x: ox + 12, y: oy + 18, text: `exact oracle (BJA ${oracle[column("BJA")]})` }),
  );
  const point = svg("circle", { class: "policy", r: 7 });
  const pointLabel = svg("text", { class: "policy-label" });
  plane.append(point, pointLabel);

  // The BJA meter with the two constants and the oracle.
  const mw = 420;
  const scaleX = scale([0, 1], [8, mw - 8]);
  clear(meter);
  meter.setAttribute("viewBox", `0 0 ${mw} 84`);
  const fill = svg("rect", { class: "fill", x: scaleX(0), y: 30, height: 12, rx: 3 });
  const valueLabel = svg("text", { class: "value-label", y: 22 });
  meter.append(svg("rect", { class: "track", x: scaleX(0), y: 30, width: scaleX(1) - scaleX(0), height: 12, rx: 3 }), fill);
  const refs = [
    { row: "always-answer", text: "always ANSWER", y: 60, anchor: "middle" },
    { row: "always-abstain", text: "always ABSTAIN", y: 78, anchor: "middle" },
    { row: "exact-oracle", text: "oracle", y: 60, cls: "oracle", anchor: "end" },
  ];
  for (const ref of refs) {
    const value = rows[ref.row].values[column("BJA")];
    const rx = scaleX(num(value));
    meter.append(
      svg("line", { class: `ref ${ref.cls || ""}`, x1: rx, x2: rx, y1: 26, y2: ref.y - 12 }),
      svg("text", { class: "ref-label", x: rx, y: ref.y, "text-anchor": ref.anchor, text: `${ref.text} ${value}` }),
    );
  }
  meter.append(valueLabel);

  function update() {
    const entry = grid[Number(slider.value)];
    output.textContent = entry.alpha;
    for (const key of Object.keys(names)) tileNodes[key].textContent = entry[key];
    const ar = num(entry.AR);
    const fcr = num(entry.FCR);
    point.setAttribute("cx", x(fcr));
    point.setAttribute("cy", y(ar));
    const left = fcr > 0.6;
    pointLabel.setAttribute("x", x(fcr) + (left ? -12 : 12));
    pointLabel.setAttribute("y", y(ar) + 4);
    pointLabel.setAttribute("text-anchor", left ? "end" : "start");
    pointLabel.textContent = `α = ${entry.alpha}`;
    const bja = num(entry.BJA);
    fill.setAttribute("width", Math.max(0, scaleX(bja) - scaleX(0)));
    valueLabel.setAttribute("x", scaleX(bja));
    valueLabel.setAttribute("text-anchor", bja > 0.85 ? "end" : "middle");
    valueLabel.textContent = entry.BJA;
    figure.dataset.alpha = entry.alpha;
  }
  slider.addEventListener("input", update);
  update();
}

// --- construction ------------------------------------------------------------------

export function initConstruction() {
  const printed = DATA.construction.printed;
  const ex = DATA.example;

  // One real route group: query roles by evidence states.
  const group = ex.route_group;
  const bitOf = Object.fromEntries(ex.features.map((feature) => [feature.name, feature.bit]));
  const routeGrid = document.getElementById("route-grid");
  routeGrid.append(el("div", { class: "head", text: "query role \\ evidence state" }));
  const exampleIds = new Set(ex.queries.map((query) => query.id));
  group.cells[0].forEach((cell, index) => {
    const bits = Object.entries(cell.bits)
      .map(([name, bit]) => `${bitOf[name]}=${bit}`)
      .sort()
      .join(" ");
    routeGrid.append(el("div", { class: "head state" }, [el("b", { text: `state ${"ABCD".charAt(index)}` }), bits]));
  });
  group.cells.forEach((line, roleIndex) => {
    const role = group.roles[roleIndex];
    const query = ex.queries.find((item) => item.feature === role);
    const asked = query.task === "lexicon" ? `lexical query: meaning of ${query.query.form}` : "grammar query";
    routeGrid.append(
      el("div", { class: "role" }, [
        el("code", { text: bitOf[role] + " " + role }),
        el("span", { text: asked }),
        el("span", { class: "unit", text: "one four-cell unit" }),
      ]),
    );
    for (const cell of line) {
      const answer = cell.decision === "ANSWER";
      routeGrid.append(
        el("div", { class: `cell ${answer ? "answer" : "abstain"}${exampleIds.has(cell.id) ? " example" : ""}` }, [
          el("span", { class: `tag ${answer ? "answer" : "abstain"}`, text: cell.decision }),
          answer ? el("span", { class: "payload", text: cell.answer }) : null,
          exampleIds.has(cell.id) ? el("span", { class: "unit", text: "worked example" }) : null,
        ]),
      );
    }
  });
  document.getElementById("route-caption").textContent =
    `Route group ${group.id}, ${group.rows} rows from the release. Each column is a same-state pair: identical program and evidence, opposite gold actions. Each row is a four-cell unit: two ANSWER rows with different payloads and two ABSTAIN rows. Ringed: the two rows of the worked example.`;

  // The scale hierarchy (Table 1).
  const ladder = document.getElementById("ladder");
  const levels = [
    ["records", "Scored records", "every route of every template", "Repeated measurement views; never treated as iid examples"],
    ["same_state_query_pairs", "Complementary query pairs", "same program/evidence state, two queries", "Query-conditioned identifiability contrast"],
    ["four_cell_units", "Four-cell evaluator units", "two ANSWER and two ABSTAIN cells", "Exact all-four endpoint"],
    ["route_groups", "Paired route groups", "two linked four-cell units", "Exact all-eight endpoint"],
    ["templates", "Semantic-template clusters", `${printed.mechanism_families} mechanism families × ${printed.templates_per_family}`, "Primary family-stratified resampling unit"],
    ["support_templates", "Support templates", "fixed and semantically disjoint from evaluation", "Packing/leakage gate; never scored"],
  ];
  for (const [key, level, composition, role] of levels) {
    const cls = key === "templates" ? "primary" : key === "support_templates" ? "unscored" : null;
    ladder.append(
      el("li", { class: cls }, [
        el("span", { class: "count", "data-printed": key, text: printed[key] }),
        el("span", { class: "level" }, [level, el("small", { text: composition })]),
        el("span", { class: "role", text: role }),
      ]),
    );
  }

  // The release gates (Table 2).
  const gates = document.getElementById("gates");
  const groups = [
    ["Correctness", [
      ["In-generator exact oracle", "decision, payload, posterior and certificate", "gate.in_generator_oracle"],
      ["Independent clean-room oracle", "standard library only, imports no SILICA code", "gate.cleanroom_oracle"],
      ["Prompt-only solver", "reads only the rendered user message", "gate.prompt_only_solver"],
    ]],
    ["Constant sizes and balance", [
      [`Declared class size ${printed.declared_class_size}`, "recomputed from the prompt", "gate.declared_class_size"],
      [`Posterior size ${printed.posterior_size}`, "recomputed from the prompt", "gate.posterior_size"],
      ["Gold action balance", "ANSWER / ABSTAIN rows", "gate.gold_action_balance"],
      ["Same-state opposite-action pairs", "every state queried both ways", "gate.same_state_opposite_action_pairs"],
    ]],
    ["Shortcut resistance (exact held-out balanced decision accuracy)", [
      ["Every frozen deterministic shortcut", "query-masked controls", "gate.deterministic_shortcuts", true],
      ["Word and character TF-IDF probes", "held-out template and held-out family", "gate.learned_probes", true],
    ]],
    ["Uniqueness and separation", [
      ["Unique rendered prompt hashes", null, "gate.unique_prompt_hashes"],
      ["Minimum ternary-signature distance", "between evaluation templates, and to support templates", "gate.minimum_signature_distance"],
    ]],
  ];
  for (const [title, items] of groups) {
    gates.append(el("li", { class: "group", text: title }));
    for (const [name, note, key, chance] of items) {
      const figure = printed[key];
      const value = share(figure);
      gates.append(meterRow(name, note, figure, value, { mark: chance ? { at: value, text: "chance" } : null }));
    }
  }
}

// --- sealed panel ------------------------------------------------------------------

function panelTooltip(key) {
  const panel = DATA.results.panel;
  const row = panel.rows[key];
  const sub = panel.sub_gate.includes(key);
  return [
    el("strong", { text: row.label + (sub ? " †" : "") }),
    ...panel.columns.map((name, index) => tipRow(name, row.values[index] ?? "")),
  ];
}

/** A labelled scatter on the AR-FCR plane with a label column and leader lines. */
function scatter(root, wrap, tip, points, { width, title }) {
  const ticks = DATA.results.axes.unit;
  const labelWidth = Math.min(170, Math.max(118, width * 0.36));
  const plot = Math.min(width - labelWidth, 420);
  const size = plot;
  const pad = { left: 42, right: 10, top: 12, bottom: 40 };
  clear(root);
  root.setAttribute("viewBox", `0 0 ${plot + labelWidth} ${size}`);
  root.setAttribute("width", plot + labelWidth);
  root.style.maxWidth = `${plot + labelWidth}px`;
  const { x, y } = planeAxes(root, {
    size,
    pad,
    ticks,
    xTitle: "FCR",
    yTitle: "AR",
  });
  root.append(svg("line", { class: "diagonal", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }));
  if (title) root.setAttribute("aria-description", title);
  const labelX = plot + 16;
  const columnPoints = points.filter((point) => !point.inline);
  const spread = spreadLabels(columnPoints.map((point) => y(point.y)), 17, pad.top + 6, size - pad.bottom);
  const placed = points.map((point) => (point.inline ? y(point.y) : spread[columnPoints.indexOf(point)]));
  points.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    const ly = placed[index];
    const group = svg("g", {
      class: "target",
      tabindex: 0,
      role: "button",
      "aria-label": point.aria,
      "data-key": point.key,
    });
    group.append(
      svg("path", { class: "leader", d: `M${px + 6} ${py} L${plot + 2} ${ly} L${labelX - 4} ${ly}` }),
      point.shape === "diamond"
        ? svg("rect", { class: "oracle", x: px - 6, y: py - 6, width: 12, height: 12, transform: `rotate(45 ${px} ${py})` })
        : svg("circle", { class: `point ${point.cls || ""}`, cx: px, cy: py, r: 5 }),
      svg("circle", { class: "focus-ring", cx: px, cy: py, r: 9 }),
      svg("circle", { class: "hit", cx: px, cy: py, r: 12 }),
    );
    if (point.inline) {
      group.querySelector(".leader").remove();
      group.append(svg("text", { class: "label", x: px + 12, y: py + 4, text: point.label }));
    } else {
      group.append(
        svg("text", { class: "label", x: labelX, y: ly + 4, text: point.label }),
        svg("rect", { class: "hit", x: labelX - 6, y: ly - 9, width: labelWidth - 12, height: 18 }),
      );
    }
    const show = () => {
      group.classList.add("hover");
      const [cx, cy] = centreIn(group.querySelector(".hit"), wrap);
      showTooltip(tip, wrap, cx, cy, point.tooltip());
    };
    const hide = () => {
      group.classList.remove("hover");
      hideTooltip(tip);
    };
    group.addEventListener("pointerenter", show);
    group.addEventListener("pointerleave", hide);
    group.addEventListener("focus", show);
    group.addEventListener("blur", hide);
    root.append(group);
  });
}

export function initPanel() {
  const panel = DATA.results.panel;
  const rows = panel.rows;
  const ar = column("AR");
  const fcr = column("FCR");
  const bja = column("BJA");
  const ci = column("BJA 95% CI");

  const names = panel.conditions.map((key) => rows[key].label);
  document.getElementById("panel-list").textContent = `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;

  const root = document.getElementById("panel-scatter");
  const wrap = root.parentElement;
  const tip = document.getElementById("scatter-tooltip");
  const points = panel.conditions.map((key) => {
    const sub = panel.sub_gate.includes(key);
    return {
      key,
      x: num(rows[key].values[fcr]),
      y: num(rows[key].values[ar]),
      label: rows[key].label + (sub ? " †" : ""),
      cls: sub ? "hollow" : "",
      aria: `${rows[key].label}: ${panel.columns.map((name, i) => `${name} ${rows[key].values[i]}`).join(", ")}`,
      tooltip: () => panelTooltip(key),
    };
  });
  const oracle = rows["exact-oracle"];
  points.push({
    key: "exact-oracle",
    x: num(oracle.values[fcr]),
    y: num(oracle.values[ar]),
    label: oracle.label,
    shape: "diamond",
    inline: true,
    aria: `${oracle.label}: AR ${oracle.values[ar]}, FCR ${oracle.values[fcr]}`,
    tooltip: () => panelTooltip("exact-oracle"),
  });
  responsive(wrap, (width) => scatter(root, wrap, tip, points, { width, title: "AR = FCR" }));

  // BJA ladder.
  const ladder = document.getElementById("bja-ladder");
  const order = [...panel.conditions].sort((a, b) => num(rows[b].values[bja]) - num(rows[a].values[bja]));
  responsive(ladder.parentElement, (width) => {
    const labelWidth = Math.min(150, width * 0.34);
    const valueWidth = 44;
    const rowHeight = 26;
    const top = 40;
    const height = top + order.length * rowHeight + 30;
    clear(ladder);
    ladder.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const x = scale([0, 1], [labelWidth + 8, width - valueWidth - 10]);
    const refs = [
      ["always-answer", "always ANSWER"],
      ["always-abstain", "always ABSTAIN"],
      ["exact-oracle", "oracle"],
    ];
    for (const tick of DATA.results.axes.unit) {
      ladder.append(
        svg("line", { class: "grid-line", x1: x(tick.value), x2: x(tick.value), y1: top - 6, y2: height - 24 }),
        svg("text", { class: "tick", x: x(tick.value), y: height - 8, "text-anchor": "middle", text: tick.text }),
      );
    }
    refs.forEach(([key, text], index) => {
      const value = rows[key].values[bja];
      const rx = x(num(value));
      ladder.append(
        svg("line", { class: `ref ${key === "exact-oracle" ? "oracle" : ""}`, x1: rx, x2: rx, y1: top - 6, y2: height - 24 }),
        svg("text", {
          class: "ref-label",
          x: rx,
          y: index === 1 ? 12 : 26,
          "text-anchor": key === "exact-oracle" ? "end" : "middle",
          text: `${text} ${value}`,
        }),
      );
    });
    order.forEach((key, index) => {
      const row = rows[key];
      const cy = top + index * rowHeight + rowHeight / 2;
      const sub = panel.sub_gate.includes(key);
      const [lo, hi] = row.parsed[ci];
      ladder.append(
        svg("text", { class: "row-label", x: 0, y: cy + 4, text: row.label + (sub ? " †" : "") }),
        svg("line", { class: "ci", x1: x(lo), x2: x(hi), y1: cy, y2: cy }),
        svg("circle", { class: `point ${sub ? "hollow" : ""}`, cx: x(num(row.values[bja])), cy, r: 5 }),
        svg("text", { class: "row-value", x: width, y: cy + 4, "text-anchor": "end", text: row.values[bja] }),
      );
    });
  });
  ladder.classList.add("plane");

  table(document.getElementById("panel-table"), panel.columns, [...panel.conditions, ...panel.references], (key) => {
    const row = rows[key];
    return {
      label: row.label + (panel.sub_gate.includes(key) ? " †" : ""),
      values: row.values,
      reference: panel.references.includes(key),
    };
  });
}

/** Fill a table from printed rows. */
function table(node, columns, keys, rowOf) {
  const head = el("tr", {}, [el("th", { scope: "col", text: "Condition" }), ...columns.map((name) => el("th", { scope: "col", text: name }))]);
  const body = keys.map((key) => {
    const row = rowOf(key);
    return el("tr", { class: row.reference ? "reference" : null }, [
      el("td", { class: "label", text: row.label }),
      ...row.values.map((value) => el("td", { text: value ?? "" })),
    ]);
  });
  node.append(el("thead", {}, [head]), el("tbody", {}, body));
}

// --- matched controls --------------------------------------------------------------

export function initControls() {
  const matched = DATA.results.matched;
  const rows = matched.rows;
  const columns = matched.columns;
  const panel = panelRows();
  const index = (name) => columns.indexOf(name);
  const chance = num(panel["always-answer"].values[column("CPA")]);
  const constants = [panel["always-answer"], panel["always-abstain"]].map((row) => num(row.values[column("BJA")]));
  const container = document.getElementById("controls-cards");
  const figure = container.closest("figure");
  const off = document.getElementById("factor-off");
  const on = document.getElementById("factor-on");

  const metrics = [
    { name: "AR − FCR", column: "AR-FCR", domain: [-1, 1], zero: 0 },
    { name: "CPA", column: "CPA", domain: [0, 1], refs: [chance] },
    { name: "BJA", column: "BJA", domain: [0, 1], refs: constants },
    { name: `Groups ${columns[index("<8>")].replace("<", "⟨").replace(">", "⟩")}`, column: "<8>", domain: [0, 1] },
    { name: "Tokens", column: "Tok.", text: true },
  ];

  const cards = matched.switched.map((pair) => {
    const condition = el("p", { class: "condition" });
    const lines = metrics.map((metric) => {
      const value = el("span", { class: "value", "data-metric": metric.column });
      const fill = el("div", { class: "fill" });
      const bar = el("div", { class: "bar", "aria-hidden": "true" }, metric.text ? [] : [fill]);
      if (metric.zero !== undefined) {
        const at = scale(metric.domain, [0, 100])(metric.zero);
        bar.append(el("div", { class: "zero", style: `left:${at}%` }));
      }
      for (const ref of metric.refs || []) {
        bar.append(el("div", { class: "ref", style: `left:${scale(metric.domain, [0, 100])(ref)}%` }));
      }
      const line = el("div", { class: `metric${metric.text ? " tokens" : ""}` }, [el("span", { class: "name", text: metric.name }), bar, value]);
      return { metric, value, fill, line };
    });
    const card = el("article", { class: "control-card", "data-off": pair.off, "data-on": pair.on }, [
      el("h3", { text: pair.model }),
      condition,
      ...lines.map((item) => item.line),
    ]);
    container.append(card);
    return { pair, card, condition, lines };
  });

  function render(state) {
    figure.dataset.state = state;
    off.setAttribute("aria-pressed", String(state === "off"));
    on.setAttribute("aria-pressed", String(state === "on"));
    for (const { pair, card, condition, lines } of cards) {
      const key = pair[state];
      const row = rows[key];
      condition.textContent = row.label.slice(pair.model.length).trim();
      card.dataset.condition = key;
      const groups = row.parsed[index("<8>")];
      card.classList.toggle("moved", state === "on" && groups > 0);
      for (const { metric, value, fill } of lines) {
        const printedValue = row.values[index(metric.column)];
        value.textContent = printedValue;
        if (metric.text) continue;
        const parsed = row.parsed[index(metric.column)];
        const at = scale(metric.domain, [0, 100]);
        const start = metric.zero !== undefined ? Math.min(at(0), at(parsed)) : 0;
        const end = metric.zero !== undefined ? Math.max(at(0), at(parsed)) : at(parsed);
        fill.style.left = `${start}%`;
        fill.style.width = `${Math.max(end - start, 0.6)}%`;
      }
    }
  }
  off.addEventListener("click", () => render("off"));
  on.addEventListener("click", () => render("on"));
  render("off");

  table(
    document.getElementById("matched-table"),
    columns,
    Object.keys(rows),
    (key) => ({ label: rows[key].label, values: rows[key].values }),
  );
}

// --- prompt ablations and the size of the decision --------------------------------

export function initAblations() {
  const ablations = DATA.results.ablations;
  const numbers = DATA.results.numbers;
  const container = document.getElementById("ablation-planes");
  for (const [model, data] of Object.entries(ablations.models)) {
    const bySignal = [...data.arms].sort((a, b) => num(a.signal) - num(b.signal));
    const signal = `${bySignal[0].signal} to ${bySignal[bySignal.length - 1].signal}`;
    const wrap = el("div", { class: "plane-wrap" });
    const root = svg("svg", { class: "plane scatter", role: "group", "aria-label": `${data.label}: prompt arms on the AR-FCR plane` });
    const tip = el("div", { class: "tooltip", role: "status", hidden: true });
    wrap.append(root, tip);
    container.append(
      el("div", { class: "ablation-plane" }, [
        el("h3", { text: data.label }),
        el("p", { class: "summary" }, [
          "AR ranges over ",
          el("b", { text: data.recall_range }),
          "; AR − FCR stays between ",
          el("b", { text: signal }),
        ]),
        wrap,
      ]),
    );
    const points = data.arms.map((arm) => ({
      key: arm.arm,
      x: arm.x,
      y: arm.y,
      label: arm.arm,
      aria: `${data.label}, ${arm.arm}: AR ${arm.AR}, AR − FCR ${arm.signal}`,
      tooltip: () => [el("strong", { text: `${data.label}, ${arm.arm}` }), tipRow("AR", arm.AR), tipRow("AR − FCR", arm.signal)],
    }));
    responsive(wrap, (width) => scatter(root, wrap, tip, points, { width, title: null }));
  }

  // Operator evaluations: the stated procedure against the minimal formula, one cell each.
  const printed = DATA.construction.printed;
  const ops = document.getElementById("ops");
  const universe = Number(printed.base_class_size.replace(/,/g, ""));
  const operators = Number(printed["structure.operator_count_median"]);
  const minimal = Number(printed["structure.minimal_formula_max"]);
  const rowsOf = [
    {
      big: printed["structure.stated_procedure_median"],
      text: `operator evaluations (median) for the stated procedure: ${printed.base_class_size} assignments × ${printed["structure.operator_count_median"]} operators`,
      cols: universe,
      rows: operators,
      cls: "",
    },
    {
      big: `≤ ${printed["structure.minimal_formula_max"]}`,
      text: `operations for the minimal decision formula on the three revealed bits (median ${printed["structure.minimal_formula_median"]})`,
      cols: minimal,
      rows: 1,
      cls: "min",
    },
  ];
  const holders = rowsOf.map((row) => {
    const holder = el("div");
    ops.append(el("div", { class: "ops-row" }, [el("p", { class: "ops-label" }, [el("b", { text: row.big }), row.text]), holder]));
    return holder;
  });
  responsive(ops, (width) => {
    const available = Math.min(width >= 640 ? width - 280 : width, 768);
    const cell = available / universe;
    rowsOf.forEach((row, index) => {
      const holder = holders[index];
      clear(holder);
      const w = row.cols * cell;
      const h = row.rows * cell;
      const id = `ops-pattern-${index}`;
      const pattern = svg("pattern", { id, width: cell, height: cell, patternUnits: "userSpaceOnUse" }, [
        svg("rect", { width: cell * 0.78, height: cell * 0.78, class: `ops-cells ${row.cls}` }),
      ]);
      const outline = row.cls === "min" ? 4 : 0;
      holder.append(
        svg(
          "svg",
          {
            viewBox: `${-outline} ${-outline} ${available + 2 * outline} ${Math.max(h, cell) + 2 * outline}`,
            role: "img",
            "aria-label": `${row.big} ${row.text}`,
          },
          [
            svg("defs", {}, [pattern]),
            svg("rect", { x: 0, y: 0, width: w, height: h, fill: `url(#${id})` }),
            outline
              ? svg("rect", { class: "ops-outline", x: -3, y: -3, width: w + 6, height: h + 6, rx: 2 })
              : null,
          ],
        ),
      );
    });
  });

  // Guard probes.
  const probes = document.getElementById("guard-probes");
  const blind = printed["structure.guard_blind_probe"];
  const given = printed["structure.guard_given_probe"];
  probes.append(
    meterRow("Guard-blind probe", "every evidence- and query-side feature except the guard value", blind, num(blind), {
      mark: { at: num(blind), text: "chance" },
    }),
    meterRow("Guard-given probe", "the same features plus the guard value", given, num(given)),
  );
}
