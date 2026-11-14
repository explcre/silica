// The plain-language layer: the four step cards, glossary tooltips, the "You decide" quiz
// and the "Follow the logic" walkthrough.  Every value comes from example.json and
// results.json; the walkthrough's reasoning is computed by site/build_data.py.

import { DATA, clear, el, svg } from "./lib.js";
import { drawGrid, strip } from "./grid.js";

const ORDINALS = ["first", "second"];

/** "subject_initial" -> "subject first": a plain label for a declared grammar value. */
function plainValue(value) {
  return String(value).replace("_initial", " first").replace(/_/g, " ");
}

function tag(decision) {
  return el("span", { class: `tag ${decision === "ANSWER" ? "answer" : "abstain"}`, text: decision });
}

function survivorsCount() {
  const counts = DATA.example.counts;
  return counts[counts.length - 1];
}

/** The four step cards of the "Start here" section. */
export function initStart() {
  const ex = DATA.example;
  const sw = ex.switches;

  // Step one: the switches of one possible language, and 2 x 2 x ... x 2 = 2^n = N.
  const holder = document.getElementById("card-switches");
  const sample = ex.survivors[0];
  const groups = [
    ["vocabulary", sw.lexical],
    ["grammar", sw.grammar],
  ];
  const width = 260;
  const node = svg("svg", { viewBox: `0 0 ${width} 112`, class: "switches", role: "img" });
  node.setAttribute("aria-label", `${sw.count} on/off switches; one possible setting is shown`);
  groups.forEach(([label, items], row) => {
    const y = row * 56;
    node.append(svg("text", { class: "switch-group", x: 0, y: y + 10, text: label }));
    items.forEach((item, index) => {
      const x = index * (width / items.length) + 4;
      const on = sample.bits[Number(item.bit.slice(1))] === 1;
      node.append(
        svg("rect", { class: `switch-track${on ? " on" : ""}`, x, y: y + 18, width: 38, height: 20, rx: 10 }),
        svg("circle", { class: "switch-knob", cx: x + (on ? 28 : 10), cy: y + 28, r: 7 }),
        svg("text", { class: "switch-label", x: x + 19, y: y + 52, "text-anchor": "middle", text: item.bit }),
      );
    });
  });
  holder.append(node);
  const expr = document.getElementById("expr-switches");
  const two = sw.values_per_switch;
  expr.append(`${two} × ${two} × … × ${two} = ${two}`, el("sup", { text: sw.count }), ` = ${sw.languages}`);

  // Step two: the languages left after the rule and the facts.
  drawGrid(document.getElementById("card-grid"), { glyphs: false, labels: false }).update(ex.counts.length - 1);
  document.getElementById("expr-clues").textContent = ex.counts.join(" → ");

  // Step three: the two questions.
  const [lexical, grammar] = ex.queries;
  const order = sw.grammar.find((item) => item.feature === grammar.feature).values.map(plainValue);
  const bubbles = document.getElementById("card-bubbles");
  bubbles.append(
    el("li", {}, [`What does `, el("i", { text: lexical.query.form }), ` mean?`]),
    el("li", { text: `${order[0].charAt(0).toUpperCase()}${order[0].slice(1)} or ${order[1]}?` }),
  );
  const left = survivorsCount();
  document.getElementById("expr-question").textContent = `${left} languages left → ${left} answers to compare`;

  // Step four: one question's answers all agree, the other's split.
  const strips = document.getElementById("card-strips");
  ex.queries.forEach((query, index) => {
    strips.append(el("div", { class: "card-strip" }, [tag(query.gold.decision), strip(index)]));
  });
  const agree = lexical.support[0].count;
  const split = grammar.support.map((item) => item.count).join(" + ");
  document.getElementById("expr-decide").append(
    `${agree} of ${left} agree → `,
    tag("ANSWER"),
    ` · ${split} split → `,
    tag("ABSTAIN"),
  );
}

/** Tap, hover and focus tooltips on the first use of each glossary term. */
export function initGlossary() {
  const tip = document.getElementById("term-tip");
  document.body.append(tip);
  let current = null;
  const show = (term) => {
    const entry = document.getElementById(`gloss-${term.dataset.term}`);
    const text = entry.querySelector("dd").textContent;
    clear(tip).append(el("strong", { text: term.dataset.term }), ` ${text}`);
    tip.hidden = false;
    const box = term.getBoundingClientRect();
    const left = Math.min(window.scrollX + box.left, window.scrollX + document.documentElement.clientWidth - tip.offsetWidth - 8);
    tip.style.left = `${Math.max(window.scrollX + 8, left)}px`;
    tip.style.top = `${window.scrollY + box.bottom + 8}px`;
    term.setAttribute("aria-expanded", "true");
    current = term;
  };
  const hide = () => {
    tip.hidden = true;
    if (current) current.setAttribute("aria-expanded", "false");
    current = null;
  };
  document.querySelectorAll(".term").forEach((term) => {
    const entry = document.getElementById(`gloss-${term.dataset.term}`);
    const definition = entry.querySelector("dd");
    definition.id = `gloss-${term.dataset.term}-text`;
    term.setAttribute("tabindex", "0");
    term.setAttribute("role", "button");
    term.setAttribute("aria-expanded", "false");
    term.setAttribute("aria-describedby", definition.id);
    term.addEventListener("pointerenter", () => show(term));
    term.addEventListener("pointerleave", hide);
    term.addEventListener("focus", () => show(term));
    term.addEventListener("blur", hide);
    term.addEventListener("click", () => (current === term && !tip.hidden ? hide() : show(term)));
    term.addEventListener("keydown", (event) => {
      if (event.key === "Escape") hide();
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (current === term && !tip.hidden) hide();
        else show(term);
      }
    });
  });
}

/** The reason shown after a quiz choice, with the computed counts. */
function quizReason(index) {
  const ex = DATA.example;
  const query = ex.queries[index];
  const left = survivorsCount();
  if (query.gold.decision === "ANSWER") {
    return `All ${left} remaining languages say ${query.query.form} means ${query.gold.answer} (${query.support_text}), so the clues settle it: the answer is ${query.gold.answer}.`;
  }
  const parts = query.support.map((item) => `${item.count} ${plainValue(item.answer)}`).join(" and ");
  return `The ${left} remaining languages split ${parts}, so the clues don't settle it: the honest answer is "can't tell".`;
}

/** "You decide": pick an answer or "can't tell" for each of the two questions. */
export function initQuiz() {
  const ex = DATA.example;
  const container = document.getElementById("quiz-cards");
  ex.queries.forEach((query, index) => {
    const order = ex.switches.grammar.find((item) => item.feature === ex.queries[1].feature);
    const question =
      query.task === "lexicon"
        ? [`What does `, el("i", { text: query.query.form }), ` mean?`]
        : [`Does a sentence start with the subject or with the verb (${order.feature})?`];
    const feedback = el("p", { class: "quiz-feedback", "aria-live": "polite" });
    const options = [...ex.quiz[index].options, "ABSTAIN"];
    const buttons = options.map((option) => {
      const label = option === "ABSTAIN" ? "Can't tell" : query.task === "lexicon" ? option : plainValue(option);
      const button = el("button", { type: "button", class: "quiz-option", "data-option": option, "aria-pressed": "false", text: label });
      button.addEventListener("click", () => {
        // the gold row is read at click time, so the check always uses the data
        const gold = DATA.example.queries[index].gold;
        const expected = gold.decision === "ANSWER" ? gold.answer : "ABSTAIN";
        const right = option === expected;
        buttons.forEach((other) => other.setAttribute("aria-pressed", String(other === button)));
        card.dataset.choice = option;
        card.dataset.verdict = right ? "right" : "wrong";
        feedback.textContent = `${right ? "Right." : "Not quite."} ${quizReason(index)}`;
      });
      return button;
    });
    const card = el("div", { class: "quiz-card", "data-query": String(index) }, [
      el("p", { class: "quiz-question" }, [el("strong", { text: `Question ${index === 0 ? "A" : "B"}. ` }), ...question]),
      el("div", { class: "quiz-options", role: "group", "aria-label": `Answers to question ${index === 0 ? "A" : "B"}` }, buttons),
      feedback,
    ]);
    container.append(card);
  });
}

/** "Follow the logic": the computed deduction, one step at a time. */
export function initWalk() {
  const ex = DATA.example;
  const walk = ex.walkthrough;
  const section = document.getElementById("walk");
  const board = document.getElementById("walk-board");
  const text = document.getElementById("walk-text");
  const track = document.getElementById("walk-track");
  const prev = document.getElementById("walk-prev");
  const next = document.getElementById("walk-next");
  const left = survivorsCount();
  const [lexical] = ex.queries;
  const forced = walk.forced;
  const free = walk.free;
  const firing = walk.rules.findIndex((rule) => rule.fires);
  const silent = walk.rules.findIndex((rule) => rule.fires === false);
  const splitText = Object.entries(free.split).map(([value, n]) => `${n} ${plainValue(value)}`).join(" and ");

  document.getElementById("walk-lead").textContent =
    `No clue mentions ${forced.form}, yet the rule and the three facts force ${forced.form} to mean ${forced.meaning}. ` +
    `The rule leaves word order open, so the same ${left} languages split ${splitText}. Step through the reasoning.`;
  const symbols = document.getElementById("walk-symbols");
  symbols.append(el("div", { text: `G = ${walk.guard.symbolic}` }), ...walk.rules.map((rule) => el("div", { text: rule.symbolic })));

  // The switches the rule uses, with their plain meaning.
  const used = new Set([...walk.facts.map((fact) => fact.bit), forced.bit, free.bit]);
  const describe = (bit) => {
    const lex = ex.switches.lexical.find((item) => item.bit === bit);
    if (lex) return `${lex.forms[0]} / ${lex.forms[1]}: off means ${lex.forms[0]} = ${lex.meanings[0]}`;
    const gram = ex.switches.grammar.find((item) => item.bit === bit);
    return `${gram.feature}: off means ${plainValue(gram.values[0])}`;
  };
  const tiles = {};
  [...used].sort().forEach((bit) => {
    const value = el("span", { class: "tile-value", "data-bit": bit });
    board.append(el("li", { class: "walk-tile", "data-bit": bit }, [el("span", { class: "tile-bit", text: bit }), value, el("span", { class: "tile-name", text: describe(bit) })]));
    tiles[bit] = value;
  });

  const fact = (bit) => walk.facts.find((item) => item.bit === bit);
  const states = [
    {
      label: "The rule",
      body: () => [
        el("p", {}, [`The rule uses one shorthand, G: `, el("q", { text: walk.guard.plain }), `.`]),
        el(
          "ul",
          {},
          walk.rules.map((rule, index) =>
            el("li", { text: `The ${ORDINALS[index]} rule: if G is ${rule.when_guard ? "true" : "false"}, then ${rule.target} must be the same as ${rule.expression}.` }),
          ),
        ),
      ],
    },
    {
      label: "Plug in the facts",
      body: () => [
        el("p", { text: `The three facts set ${walk.facts.map((item) => `${item.bit} = ${item.value}`).join(", ")}. Put those into G:` }),
        el(
          "ul",
          {},
          walk.guard.parts.map((part) =>
            el("li", {
              text:
                `${part.plain}: ${part.plugged}. ` +
                (part.on === null ? "" : `${part.on} of the ${part.of} ${part.on === "1" ? "is" : "are"} on, an ${part.value ? "odd" : "even"} number. `) +
                `This part is ${part.value ? "true" : "false"}.`,
            }),
          ),
        ),
        el("p", { text: `So G is ${walk.guard.value ? "true" : "false"}.` }),
      ],
    },
    {
      label: "Which part switches on",
      body: () => [
        el("p", { text: `G is ${walk.guard.value ? "true" : "false"}, so the ${ORDINALS[firing]} rule applies.` }),
        el("p", { text: `The ${ORDINALS[silent]} rule only applies when G is ${walk.rules[silent].when_guard ? "true" : "false"}, so here it says nothing at all.` }),
      ],
    },
    {
      label: "What it forces",
      body: () => {
        const rule = walk.rules[firing];
        const axis = ex.switches.lexical.find((item) => item.bit === forced.bit);
        return [
          el("p", { text: `The ${ORDINALS[firing]} rule says ${rule.target} must be the same as ${rule.expression}.` }),
          el("p", { text: `Since ${rule.condition.bit} is ${rule.condition.value}, take ${rule.chosen.bit}, which is ${rule.chosen.value}. So ${forced.bit} = ${forced.value}.` }),
          el("p", {}, [
            `That means the ${axis.forms[0]} / ${axis.forms[1]} switch is ${forced.value === "1" ? "on" : "off"}, so ${forced.form} means ${forced.meaning}. All ${left} remaining languages agree (${forced.support}): `,
            tag(lexical.gold.decision),
          ]),
        ];
      },
    },
    {
      label: "What stays open",
      body: () => [
        el("p", {}, [
          `Nothing sets ${free.bit} (${free.feature}). The ${left} remaining languages split ${splitText}, so the clues don't settle it: `,
          tag(ex.queries[1].gold.decision),
        ]),
      ],
    },
  ];
  const buttons = states.map((state, index) => {
    const button = el("button", { type: "button", text: state.label });
    button.addEventListener("click", () => go(index));
    track.append(el("li", {}, [button]));
    return button;
  });
  prev.addEventListener("click", () => go(Number(section.dataset.walk) - 1));
  next.addEventListener("click", () => go(Number(section.dataset.walk) + 1));

  function go(index) {
    const stage = Math.max(0, Math.min(states.length - 1, index));
    section.dataset.walk = String(stage);
    clear(text).append(...states[stage].body());
    buttons.forEach((button, i) => {
      if (i === stage) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
      button.classList.toggle("done", i < stage);
    });
    prev.disabled = stage === 0;
    next.disabled = stage === states.length - 1;
    for (const [bit, node] of Object.entries(tiles)) {
      const known = fact(bit);
      let shown = "?";
      let state = "unknown";
      if (known && stage >= 1) {
        shown = known.value;
        state = "fact";
      } else if (bit === forced.bit && stage >= 3) {
        shown = forced.value;
        state = "forced";
      } else if (bit === free.bit && stage >= 4) {
        shown = "open";
        state = "open";
      }
      node.textContent = shown;
      node.parentElement.dataset.state = state;
    }
  }
  go(0);
}
