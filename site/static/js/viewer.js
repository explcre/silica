// The "Use SILICA" section: commands, the record viewer, released responses, BibTeX.

import { DATA, clear, el } from "./lib.js";

/** Render any JSON value as a collapsible tree of native <details> elements. */
function tree(value, key = null, depth = 0) {
  const label = key === null ? [] : [el("span", { class: "key", text: JSON.stringify(key) }), ": "];
  if (value !== null && typeof value === "object") {
    const entries = Array.isArray(value) ? value.map((item, index) => [index, item]) : Object.entries(value);
    const open = Array.isArray(value) ? "[" : "{";
    const size = Array.isArray(value) ? `${entries.length} items` : `${entries.length} keys`;
    const node = el("details", depth < 1 ? { open: true } : {}, [
      el("summary", {}, [...label, open, el("span", { class: "meta", text: ` ${size} ` }), Array.isArray(value) ? "]" : "}"]),
    ]);
    for (const [childKey, child] of entries) {
      node.append(tree(child, Array.isArray(value) ? null : childKey, depth + 1));
    }
    return node;
  }
  const cls = typeof value === "string" ? "str" : "lit";
  return el("div", { class: "leaf" }, [...label, el("span", { class: cls, text: JSON.stringify(value) })]);
}

/** Copy a code block's text, with a fallback when the clipboard API is unavailable. */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    const area = el("textarea", { class: "visually-hidden", "aria-hidden": "true" });
    area.value = text;
    document.body.append(area);
    area.select();
    let done = false;
    try {
      done = document.execCommand("copy");
    } catch (inner) {
      done = false;
    }
    area.remove();
    return done;
  }
}

export function initUse() {
  const meta = DATA.meta;
  document.getElementById("cmd-install").textContent = meta.install;
  document.getElementById("cmd-quickstart").textContent = meta.quickstart;
  document.getElementById("cmd-reproduce").textContent = meta.reproduce.join("\n\n");
  document.getElementById("bibtex").textContent = meta.bibtex;
  const status = document.getElementById("copy-status");
  document.querySelectorAll("button.copy").forEach((button) => {
    button.addEventListener("click", async () => {
      const source = document.getElementById(button.dataset.copy);
      const done = await copyText(source.textContent);
      button.textContent = done ? "Copied" : "Copy failed";
      status.textContent = done ? "Copied to the clipboard." : "Copying is not available here; select the text instead.";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1600);
    });
  });

  // Record viewer: one item at a time, public row beside its gold record.
  const ex = DATA.example;
  const publicPane = document.getElementById("tree-public");
  const goldPane = document.getElementById("tree-gold");
  const prompt = document.getElementById("prompt-view");
  const buttons = [document.getElementById("item-answer"), document.getElementById("item-abstain")];
  function show(index) {
    buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
    clear(publicPane).append(tree(ex.public_rows[index]));
    clear(goldPane).append(tree(ex.gold_rows[index]));
    clear(prompt);
    const query = ex.queries[index];
    prompt.append(el("p", { class: "field-note", text: `Prompt hash ${query.prompt_hash}` }));
    for (const message of query.messages) {
      prompt.append(el("div", { class: "message" }, [el("p", { text: message.role }), el("pre", {}, [el("code", { text: message.content })])]));
    }
    document.querySelector(".viewer").dataset.item = query.id;
  }
  buttons.forEach((button, index) => button.addEventListener("click", () => show(index)));
  show(0);

  // Released responses to the two rows, scored by the release evaluator.
  const responses = ex.responses;
  const covering = responses.covering;
  document.getElementById("responses-note").textContent =
    `Released conditions that cover these two rows: ${covering.length} of ${responses.conditions_released}, ` +
    `${covering.map((item) => `${item.condition_id} (${item.model}, ${item.rows} rows)`).join("; ")}. ` +
    "Scored with the strict parser: a response is correct only with the gold decision and, for ANSWER, the canonical payload.";
  const table = document.getElementById("responses-table");
  const head = el(
    "tr",
    {},
    ["Condition", "Gold", "Response text", "Finish", "Tokens", "Valid", "Decision", "Correct"].map((name) =>
      el("th", { scope: "col", class: name === "Response text" ? "text" : null, text: name }),
    ),
  );
  const golds = Object.fromEntries(ex.queries.map((query) => [query.id, query.gold.decision]));
  const body = [];
  for (const condition of covering) {
    const ordered = [...condition.responses].sort(
      (a, b) => ex.queries.findIndex((q) => q.id === a.id) - ex.queries.findIndex((q) => q.id === b.id),
    );
    for (const response of ordered) {
      const gold = golds[response.id];
      body.push(
        el("tr", {}, [
          el("td", { class: "label", text: condition.condition_id }),
          el("td", {}, [el("span", { class: `tag ${gold === "ANSWER" ? "answer" : "abstain"}`, text: gold })]),
          el("td", { class: "text" }, [el("code", { text: response.response_text ?? "" })]),
          el("td", { text: response.finish_reason ?? "" }),
          el("td", { text: response.completion_tokens === null ? "" : String(response.completion_tokens) }),
          el("td", { text: response.format_valid ? "yes" : "no" }),
          el("td", { text: response.decision ?? "invalid" }),
          el("td", { text: response.correct ? "yes" : "no" }),
        ]),
      );
    }
  }
  table.append(el("thead", {}, [head]), el("tbody", {}, body));
}
