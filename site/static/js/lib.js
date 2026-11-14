// Shared helpers: data lookup, DOM and SVG builders, motion preference, tooltips.

const SVG_NS = "http://www.w3.org/2000/svg";

/** The page data, filled once by main.js: {meta, example, construction, results}. */
export const DATA = {};

/** Look up a "/"-separated path in the page data; log an error if it is missing. */
export function get(path) {
  const value = path.split("/").reduce((node, key) => (node == null ? undefined : node[key]), DATA);
  if (value === undefined) console.error(`missing data value: ${path}`);
  return value;
}

/** Create an HTML element with attributes and children (strings become text nodes). */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  setAttrs(node, attrs);
  append(node, children);
  return node;
}

/** Create an SVG element with attributes and children. */
export function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  setAttrs(node, attrs);
  append(node, children);
  return node;
}

function setAttrs(node, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "class") node.setAttribute("class", value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
}

function append(node, children) {
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
}

/** Remove every child of a node. */
export function clear(node) {
  while (node.firstChild) node.firstChild.remove();
  return node;
}

/** Whether the viewer asked for reduced motion (live). */
export const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
export function reducedMotion() {
  return motion.matches;
}

/** A linear scale from a domain to a range. */
export function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (value) => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

/** Show a tooltip inside a positioned wrapper, next to a point given in wrapper pixels. */
export function showTooltip(tip, wrapper, x, y, content) {
  clear(tip);
  append(tip, content);
  tip.hidden = false;
  const box = wrapper.getBoundingClientRect();
  const width = tip.offsetWidth;
  const height = tip.offsetHeight;
  let left = x + 14;
  if (left + width > box.width) left = Math.max(0, x - width - 14);
  let top = y - height - 10;
  if (top < 0) top = y + 14;
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

export function hideTooltip(tip) {
  tip.hidden = true;
}

/** A tooltip row: a label and a value. */
export function tipRow(label, value) {
  return el("div", { class: "row" }, [el("span", { text: label }), el("span", { text: value })]);
}

/** Map an SVG element's centre to pixels relative to a wrapper element. */
export function centreIn(node, wrapper) {
  const a = node.getBoundingClientRect();
  const b = wrapper.getBoundingClientRect();
  return [a.left + a.width / 2 - b.left, a.top + a.height / 2 - b.top];
}

/** Draw square plot axes with gridlines, ticks and titles; return the x and y scales. */
export function planeAxes(root, { size, pad, ticks, xTitle, yTitle }) {
  const x = scale([0, 1], [pad.left, size - pad.right]);
  const y = scale([0, 1], [size - pad.bottom, pad.top]);
  const axes = svg("g", { class: "axes" });
  for (const tick of ticks) {
    axes.append(
      svg("line", { class: "grid-line", x1: x(tick.value), x2: x(tick.value), y1: y(0), y2: y(1) }),
      svg("line", { class: "grid-line", x1: x(0), x2: x(1), y1: y(tick.value), y2: y(tick.value) }),
      svg("text", { class: "tick", x: x(tick.value), y: y(0) + 16, "text-anchor": "middle", text: tick.text }),
      svg("text", { class: "tick", x: x(0) - 7, y: y(tick.value) + 4, "text-anchor": "end", text: tick.text }),
    );
  }
  axes.append(
    svg("line", { class: "axis", x1: x(0), x2: x(1), y1: y(0), y2: y(0) }),
    svg("line", { class: "axis", x1: x(0), x2: x(0), y1: y(0), y2: y(1) }),
    svg("text", { class: "axis-title", x: (x(0) + x(1)) / 2, y: size - 4, "text-anchor": "middle", text: xTitle }),
    svg("text", {
      class: "axis-title",
      transform: `translate(12 ${(y(0) + y(1)) / 2}) rotate(-90)`,
      "text-anchor": "middle",
      text: yTitle,
    }),
  );
  root.append(axes);
  return { x, y };
}

/** Ticks for a 0..1 axis taken from printed reference values (no hand-typed numbers). */
export function unitTicks(low, high) {
  return [
    { value: 0, text: low },
    { value: 1, text: high },
  ];
}

/** Place labels in a column without overlap; returns the label y positions in input order. */
export function spreadLabels(targets, minGap, low, high) {
  const order = targets.map((y, index) => ({ y, index })).sort((a, b) => a.y - b.y);
  const placed = [];
  for (const item of order) {
    const previous = placed.length ? placed[placed.length - 1].y : -Infinity;
    placed.push({ index: item.index, y: Math.max(item.y, previous + minGap, low) });
  }
  const overflow = placed.length ? placed[placed.length - 1].y - high : 0;
  if (overflow > 0) {
    for (let i = placed.length - 1; i >= 0; i -= 1) {
      const limit = i === placed.length - 1 ? high : placed[i + 1].y - minGap;
      placed[i].y = Math.min(placed[i].y, limit);
    }
  }
  const result = new Array(targets.length);
  for (const item of placed) result[item.index] = item.y;
  return result;
}
