// The version-space grid of the worked example: the hero miniature and the stepper.
// Every cell's state comes from example.json, computed by site/build_data.py.

import { DATA, clear, el, hideTooltip, motion, reducedMotion, showTooltip, svg, tipRow } from "./lib.js";

const CELL = 20;
const GAP = 2;
const MARGIN = { top: 20, left: 20, right: 4, bottom: 4 };

/** The stepper's stages, derived from the example data. */
function stages() {
  const ex = DATA.example;
  const [lexical, grammar] = ex.queries;
  const stages = [
    { key: "universe", label: "Universe", field: null },
    { key: "program", label: "Program", field: "program" },
    ...ex.facts.map((fact, index) => ({
      key: `fact-${index}`,
      label: `${fact.form} → ${fact.semantic}`,
      field: "facts",
      fact: index,
    })),
    { key: "query-a", label: `Query: ${lexical.query.form}`, field: "queries", query: 0 },
    { key: "query-b", label: `Query: ${grammar.query.feature}`, field: "queries", query: 1 },
  ];
  return stages;
}

/** How many filtering stages there are (universe, program, facts). */
function filterStages() {
  return DATA.example.counts.length;
}

/** The compatible-language count shown at a stage. */
function stageCount(stage) {
  const counts = DATA.example.counts;
  return counts[Math.min(stage, counts.length - 1)];
}

/** One-letter glyph for a payload: its first letter, upper-cased. */
function glyph(answer) {
  return String(answer).charAt(0).toUpperCase();
}

/** Draw the grid once; returns an updater that restyles the cells for a stage. */
export function drawGrid(root, { glyphs = true, labels = true } = {}) {
  const ex = DATA.example;
  const side = Number(ex.grid.side);
  const margin = labels ? MARGIN : { top: 2, left: 2, right: 2, bottom: 2 };
  const span = side * CELL;
  clear(root);
  root.setAttribute("viewBox", `0 0 ${margin.left + span + margin.right} ${margin.top + span + margin.bottom}`);
  if (labels) {
    const features = ex.features;
    const lexical = features.filter((item) => item.kind === "lexical");
    const grammar = features.filter((item) => item.kind === "grammar");
    root.append(
      svg("text", {
        class: "axis-label",
        x: margin.left,
        y: 12,
        text: `columns: lexical bits ${lexical[0].bit}–${lexical[lexical.length - 1].bit} →`,
      }),
      svg("text", {
        class: "axis-label",
        transform: `translate(12 ${margin.top}) rotate(90)`,
        text: `rows: grammar bits ${grammar[0].bit}–${grammar[grammar.length - 1].bit} →`,
      }),
    );
  }
  const layer = svg("g", { transform: `translate(${margin.left} ${margin.top})` });
  const cells = ex.cells.map((cell, index) => {
    const x = (index % side) * CELL;
    const y = Math.floor(index / side) * CELL;
    const rect = svg("rect", {
      class: "cell alive",
      x: x + GAP / 2,
      y: y + GAP / 2,
      width: CELL - GAP,
      height: CELL - GAP,
      rx: 3,
      "data-h": cell.h,
    });
    layer.append(rect);
    let text = null;
    if (glyphs && cell.answers) {
      text = svg("text", { class: "glyph", x: x + CELL / 2, y: y + CELL / 2 + 0.5, opacity: 0 });
      layer.append(text);
    }
    return { cell, rect, text, x, y };
  });
  const overlay = svg("g", { class: "overlay" });
  layer.append(overlay);
  root.append(layer);

  function update(stage) {
    const filters = filterStages();
    clear(overlay);
    for (const item of cells) {
      const alive = item.cell.last >= Math.min(stage, filters - 1);
      let cls = alive ? "cell alive" : "cell dead";
      let label = "";
      let tone = "";
      if (alive && stage >= filters && item.cell.answers) {
        const answer = item.cell.answers[stage - filters];
        if (stage === filters) {
          cls = "cell answer";
          tone = "on-fill";
        } else {
          const values = Object.keys(ex.split);
          const first = answer === values[0];
          cls = first ? "cell subject" : "cell verb";
          tone = first ? "on-fill" : "on-wash";
        }
        label = glyph(answer);
      }
      item.rect.setAttribute("class", cls);
      item.rect.dataset.state = cls.replace("cell ", "");
      if (item.text) {
        item.text.textContent = label;
        item.text.setAttribute("class", `glyph ${tone}`);
        item.text.setAttribute("opacity", label ? 1 : 0);
      }
    }
    if (labels && stage === filters + 1) {
      const witnesses = ex.queries[1].witnesses;
      witnesses.forEach((id, order) => {
        const item = cells.find((entry) => entry.cell.h === id);
        const answer = item.cell.answers[1];
        const ly = item.y + CELL / 2;
        const lx = item.x + CELL * 2.5;
        overlay.append(
          svg("rect", { class: "witness-ring", x: item.x, y: item.y, width: CELL, height: CELL, rx: 4 }),
          svg("line", { class: "witness-leader", x1: item.x + CELL, y1: ly, x2: lx - 4, y2: ly }),
          svg("rect", { x: lx - 2, y: ly - 8, width: CELL * 7.6, height: 16, rx: 3, fill: "var(--surface)" }),
          svg("text", { class: "witness-label", x: lx + 2, y: ly + 4, text: `${id} → ${answer}` }),
        );
        if (order === 0) item.rect.dataset.witness = "true";
      });
    }
  }
  return { update, cells };
}

/** The autoplaying hero miniature: universe → program → facts, looped, pausable. */
export function initHero() {
  const root = document.getElementById("hero-grid");
  const count = document.getElementById("hero-count");
  const label = document.getElementById("hero-label");
  const toggle = document.getElementById("hero-toggle");
  const figure = document.getElementById("hero-figure");
  const grid = drawGrid(root, { glyphs: false, labels: false });
  const ex = DATA.example;
  const last = ex.counts.length - 1;
  const names = stages().slice(0, last + 1).map((stage) => stage.label);
  document.getElementById("hero-sequence").textContent = ex.counts.join(" → ");
  let stage = last;
  let timer = null;

  function show(next) {
    stage = next;
    grid.update(stage);
    count.textContent = ex.counts[stage];
    label.textContent = stage === 0 ? "candidate languages" : `compatible after ${names[stage]}`;
    figure.dataset.state = String(stage);
  }
  function tick() {
    show(stage >= last ? 0 : stage + 1);
    timer = window.setTimeout(tick, stage === last ? 2600 : 1100);
  }
  function setPaused(paused) {
    toggle.dataset.paused = String(paused);
    toggle.querySelector(".chip-text").textContent = paused ? "Play" : "Pause";
    toggle.setAttribute("aria-label", paused ? "Play the animation" : "Pause the animation");
    window.clearTimeout(timer);
    timer = null;
    if (!paused) timer = window.setTimeout(tick, 900);
  }
  toggle.addEventListener("click", () => setPaused(toggle.dataset.paused !== "true"));
  motion.addEventListener("change", () => {
    if (reducedMotion()) {
      setPaused(true);
      show(last);
    }
  });
  show(reducedMotion() ? last : 0);
  setPaused(reducedMotion());
}

/** Describe one cell for the tooltip and the live region. */
function describe(entry) {
  const ex = DATA.example;
  const { cell } = entry;
  const filters = filterStages();
  const steps = stages();
  const bits = ex.features.map((feature, index) => `${feature.bit}=${cell.bits[index]}`).join(" ");
  let fate;
  if (cell.last === 0) fate = "outside the declared class (the program rejects it)";
  else if (cell.last < filters - 1) fate = `eliminated by ${steps[cell.last + 1].label}`;
  else fate = "compatible with the program and all three facts";
  const rows = [el("strong", { text: cell.h }), el("div", { text: bits }), el("div", { text: fate })];
  if (cell.answers) {
    ex.queries.forEach((query, index) => {
      const asked = query.task === "lexicon" ? `meaning of ${query.query.form}` : query.query.feature;
      rows.push(tipRow(asked, String(cell.answers[index])));
    });
  }
  return rows;
}

/** The interactive stepper of the worked example. */
export function initStepper() {
  const ex = DATA.example;
  const figure = document.getElementById("stepper");
  const root = document.getElementById("stepper-grid");
  const tip = document.getElementById("grid-tooltip");
  const wrap = root.parentElement;
  const scrub = document.getElementById("step-scrub");
  const play = document.getElementById("step-play");
  const prev = document.getElementById("step-prev");
  const next = document.getElementById("step-next");
  const track = document.getElementById("step-track");
  const countNode = document.getElementById("step-count");
  const caption = document.getElementById("step-caption");
  const steps = stages();
  const filters = filterStages();
  const grid = drawGrid(root);
  let stage = 0;
  let timer = null;
  let cursor = null;

  const buttons = steps.map((step, index) => {
    const button = el("button", { type: "button", text: step.label });
    button.addEventListener("click", () => go(index));
    track.append(el("li", {}, [button]));
    return button;
  });
  scrub.max = String(steps.length - 1);
  scrub.addEventListener("input", () => go(Number(scrub.value)));
  prev.addEventListener("click", () => go(stage - 1));
  next.addEventListener("click", () => go(stage + 1));
  play.addEventListener("click", () => setPlaying(play.dataset.playing !== "true"));

  fillPrompt();
  const readouts = buildReadouts();
  buildSurvivorTable();

  function captionFor(index) {
    const step = steps[index];
    const [lexical, grammar] = ex.queries;
    if (step.key === "universe") {
      return `All ${ex.grid.size} assignments of the ${ex.grid.features} feature bits: every candidate language of the base class, before anything is filtered.`;
    }
    if (step.key === "program") {
      return `The published Boolean program keeps ${ex.counts[1]} of the ${ex.grid.size} assignments: this is the declared class the record states in its prompt.`;
    }
    if (step.fact !== undefined) {
      const fact = ex.facts[step.fact];
      return `The fact ${fact.form} → ${fact.semantic} holds only where ${fact.bit} (${fact.feature}) = ${fact.value}, so ${ex.counts[step.fact + 2]} languages remain compatible.`;
    }
    const survivors = ex.counts[filters - 1];
    if (step.query === 0) {
      return `Query A asks for the meaning of ${lexical.query.form}. All ${survivors} compatible languages answer ${lexical.gold.answer}, so the gold action is ANSWER with the payload "${lexical.gold.answer}".`;
    }
    const parts = Object.entries(ex.split).map(([value, n]) => `${n} ${value}`);
    return `Query B asks for ${grammar.query.feature}. The same ${survivors} languages split ${parts.join(" and ")}, so the gold action is ABSTAIN; the evaluator stores two disagreeing witnesses, ${grammar.witnesses.join(" and ")}.`;
  }

  function go(index) {
    const target = Math.max(0, Math.min(steps.length - 1, index));
    if (target === stage && figure.dataset.ready === "true") return;
    stage = target;
    figure.dataset.step = String(stage);
    figure.dataset.ready = "true";
    scrub.value = String(stage);
    grid.update(stage);
    countNode.textContent = stageCount(stage);
    caption.textContent = captionFor(stage);
    root.setAttribute("aria-label", `Version space, ${steps[stage].label}: ${stageCount(stage)} languages compatible.`);
    buttons.forEach((button, index2) => {
      if (index2 === stage) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
      button.classList.toggle("done", index2 < stage);
    });
    prev.disabled = stage === 0;
    next.disabled = stage === steps.length - 1;
    highlight(steps[stage]);
    readouts(stage);
    if (cursor !== null) showCursor();
  }

  function highlight(step) {
    figure.querySelectorAll(".field").forEach((field) => {
      field.classList.toggle("active", field.dataset.stage === step.field);
    });
    figure.querySelectorAll("#facts-list li").forEach((item, index) => {
      item.classList.toggle("active", step.fact === index || (step.query !== undefined));
    });
    figure.querySelectorAll("#queries-list li").forEach((item, index) => {
      item.classList.toggle("active", step.query === index);
    });
  }

  function setPlaying(playing) {
    play.dataset.playing = String(playing);
    play.setAttribute("aria-label", playing ? "Pause" : "Play");
    window.clearTimeout(timer);
    if (!playing) return;
    if (stage === steps.length - 1) go(0);
    const advance = () => {
      if (stage >= steps.length - 1) {
        setPlaying(false);
        return;
      }
      go(stage + 1);
      timer = window.setTimeout(advance, 1700);
    };
    timer = window.setTimeout(advance, 900);
  }

  // Keyboard exploration of the grid: arrow keys move a cursor, Escape leaves.
  const side = Number(ex.grid.side);
  function showCursor() {
    const entry = grid.cells[cursor];
    root.querySelectorAll(".cursor").forEach((node) => node.remove());
    const layer = root.querySelector("g");
    layer.append(svg("rect", { class: "cursor", x: entry.x, y: entry.y, width: CELL, height: CELL, rx: 4 }));
    const rect = entry.rect.getBoundingClientRect();
    const box = wrap.getBoundingClientRect();
    showTooltip(tip, wrap, rect.left - box.left + rect.width, rect.top - box.top + rect.height / 2, describe(entry));
  }
  root.addEventListener("keydown", (event) => {
    const moves = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: side, ArrowUp: -side };
    if (event.key === "Escape") {
      cursor = null;
      root.querySelectorAll(".cursor").forEach((node) => node.remove());
      hideTooltip(tip);
      return;
    }
    if (!(event.key in moves)) return;
    event.preventDefault();
    const size = ex.cells.length;
    cursor = cursor === null ? 0 : Math.max(0, Math.min(size - 1, cursor + moves[event.key]));
    showCursor();
  });
  root.addEventListener("blur", () => {
    cursor = null;
    root.querySelectorAll(".cursor").forEach((node) => node.remove());
    hideTooltip(tip);
  });
  root.addEventListener("pointermove", (event) => {
    const target = event.target.closest("rect.cell");
    if (!target) {
      hideTooltip(tip);
      return;
    }
    const entry = grid.cells.find((item) => item.rect === target);
    const box = wrap.getBoundingClientRect();
    showTooltip(tip, wrap, event.clientX - box.left, event.clientY - box.top, describe(entry));
  });
  root.addEventListener("pointerleave", () => hideTooltip(tip));

  go(0);
  return { go, steps };
}

/** Fill the "in the prompt" card: program, facts, queries. */
function fillPrompt() {
  const ex = DATA.example;
  const program = document.getElementById("program-infix");
  ex.program.infix.forEach((line) => program.append(el("div", { text: line })));
  const used = new Set(ex.program.infix.join(" ").match(/b\d/g));
  document.getElementById("program-legend").textContent = ex.features
    .filter((feature) => used.has(feature.bit))
    .map((feature) => `${feature.bit} ${feature.name}`)
    .join(" · ");
  document.getElementById("program-json").textContent = JSON.stringify(ex.program.json, null, 1);
  const facts = document.getElementById("facts-list");
  ex.facts.forEach((fact) => {
    facts.append(
      el("li", {}, [
        el("span", { class: "form", text: `${fact.form} → ${fact.semantic}` }),
        el("span", { class: "pin", text: `${fact.category}; holds only where ${fact.bit} = ${fact.value}` }),
      ]),
    );
  });
  // The decision rule, verbatim from the rendered user message.
  const rule = JSON.parse(ex.queries[0].messages[1].content).decision_rule;
  const ruleNode = document.getElementById("decision-rule");
  const actions = Object.entries(rule).sort(([a], [b]) => (a === "ANSWER" ? -1 : b === "ANSWER" ? 1 : 0));
  for (const [action, text] of actions) {
    ruleNode.append(
      el("p", {}, [el("span", { class: `tag ${action === "ANSWER" ? "answer" : "abstain"}`, text: action }), ` ${text}`]),
    );
  }
  const queries = document.getElementById("queries-list");
  ex.queries.forEach((query, index) => {
    const name = index === 0 ? "A" : "B";
    const text =
      query.task === "lexicon"
        ? `meaning of the ${query.query.category} ${query.query.form}`
        : `value of ${query.query.feature}`;
    queries.append(
      el("li", {}, [el("strong", { text: `Query ${name}` }), el("span", { class: "form", text }), el("code", { text: query.task })]),
    );
  });
}

/** Draw a strip of the compatible languages, coloured by one query's answer. */
function strip(queryIndex) {
  const ex = DATA.example;
  const survivors = ex.survivors;
  const size = 18;
  const node = svg("svg", {
    class: "strip vs-grid",
    viewBox: `0 0 ${survivors.length * size} ${size + 2}`,
    role: "img",
    "aria-label": `${survivors.length} compatible languages: ${survivors.map((item) => item.answers[queryIndex]).join(", ")}`,
  });
  const values = Object.keys(ex.split);
  const witnesses = ex.queries[queryIndex].witnesses;
  survivors.forEach((item, index) => {
    const answer = item.answers[queryIndex];
    let cls = "cell answer";
    let tone = "on-fill";
    if (queryIndex === 1) {
      cls = answer === values[0] ? "cell subject" : "cell verb";
      tone = answer === values[0] ? "on-fill" : "on-wash";
    }
    node.append(svg("rect", { class: cls, x: index * size + 1, y: 1, width: size - 2, height: size - 2, rx: 3 }));
    node.append(svg("text", { class: `glyph ${tone}`, x: index * size + size / 2, y: size / 2 + 1.5, text: glyph(answer) }));
    if (witnesses.includes(item.h)) {
      node.append(svg("rect", { class: "witness-ring", x: index * size, y: 0, width: size, height: size, rx: 4 }));
    }
  });
  return node;
}

/** Build the evaluator-only readouts; returns an updater for the stage. */
function buildReadouts() {
  const ex = DATA.example;
  const container = document.getElementById("readouts");
  const filters = filterStages();
  const boxes = ex.queries.map((query, index) => {
    const name = index === 0 ? "A" : "B";
    const action = query.gold.decision;
    const asked = query.task === "lexicon" ? `meaning of ${query.query.form}` : query.query.feature;
    const body = el("div", { class: "readout-body" }, [
      strip(index),
      el("p", { class: "support", text: `Support: ${query.support_text}` }),
      query.witnesses.length
        ? el("p", { class: "support", text: `Disagreement witnesses: ${query.witnesses.join(", ")}` })
        : null,
      el("pre", {}, [el("code", { text: JSON.stringify(query.gold) })]),
    ]);
    const pending = el("p", { class: "pending-note", text: "Read off once the version space is final." });
    const box = el("div", { class: "readout pending", "data-query": name }, [
      el("div", { class: "readout-head" }, [
        el("span", { class: `tag ${action === "ANSWER" ? "answer" : "abstain"}`, text: action }),
        el("strong", { text: `Query ${name}` }),
        el("span", { text: asked }),
      ]),
      pending,
      body,
    ]);
    container.append(box);
    return { box, body, pending, index };
  });
  return (stage) => {
    for (const item of boxes) {
      const reached = stage >= filters + item.index;
      item.box.classList.toggle("pending", !reached);
      item.box.classList.toggle("active", stage === filters + item.index);
      item.body.hidden = !reached;
      item.pending.hidden = reached;
    }
  };
}

/** The table view of the compatible languages: one row per language, one column per bit. */
function buildSurvivorTable() {
  const ex = DATA.example;
  const table = document.getElementById("survivor-table");
  const pinned = new Set(ex.facts.map((fact) => fact.bit));
  const queried = ex.queries.map((query) => query.bit);
  const head = el("tr", {}, [
    el("th", { scope: "col", text: "language" }),
    ...ex.features.map((feature) => el("th", { scope: "col", title: feature.name, text: feature.bit })),
    ...ex.queries.map((query, index) => el("th", { scope: "col", text: `Query ${index === 0 ? "A" : "B"}` })),
  ]);
  const body = ex.survivors.map((item) =>
    el("tr", {}, [
      el("th", { scope: "row", text: item.h }),
      ...item.bits.map((bit, index) => {
        const name = ex.features[index].bit;
        let cls = pinned.has(name) ? "pinned" : "";
        if (name === queried[0]) cls += " queried-a";
        if (name === queried[1]) cls += " queried-b";
        return el("td", { class: cls.trim() || null, text: String(bit) });
      }),
      ...item.answers.map((answer) => el("td", { text: String(answer) })),
    ]),
  );
  table.append(el("thead", {}, [head]), el("tbody", {}, body));
  table.setAttribute(
    "aria-label",
    `The ${ex.survivors.length} languages compatible with the program and the facts; shaded columns are pinned by the facts.`,
  );
}
