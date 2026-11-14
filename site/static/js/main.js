// Entry point: load the data files, bind every [data-k] value, and build the figures.

import { DATA, get } from "./lib.js";
import { initHero, initStepper } from "./grid.js";
import { initAblations, initConstant, initConstruction, initControls, initPanel } from "./charts.js";
import { initUse } from "./viewer.js";
import { initGlossary, initQuiz, initStart, initWalk } from "./plain.js";

const FILES = ["meta", "example", "construction", "results"];

async function load() {
  const parts = await Promise.all(
    FILES.map(async (name) => {
      const response = await fetch(`static/data/${name}.json`);
      if (!response.ok) throw new Error(`cannot load ${name}.json: ${response.status}`);
      return [name, await response.json()];
    }),
  );
  for (const [name, value] of parts) DATA[name] = value;
}

/** Fill every element that names a data value with that value, verbatim. */
function bind() {
  for (const node of document.querySelectorAll("[data-k]")) {
    const value = get(node.dataset.k);
    if (value === undefined) continue;
    node.textContent = typeof value === "string" ? value : JSON.stringify(value);
  }
}

async function main() {
  try {
    await load();
  } catch (error) {
    console.error(error);
    document.body.classList.add("data-missing");
    return;
  }
  bind();
  initHero();
  initStart();
  initQuiz();
  const stepper = initStepper();
  initWalk();
  initConstant();
  initConstruction();
  initPanel();
  initControls();
  initAblations();
  initUse();
  initGlossary();
  // handle for the browser checks: the data and the stepper
  window.SILICA = { data: DATA, stepper };
  document.documentElement.dataset.ready = "true";
}

main();
