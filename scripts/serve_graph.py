#!/usr/bin/env python3
"""Serve a GraphBuilder Cytoscape JSON dump for browser visualization.

Usage:
    python3 scripts/serve_graph.py target/ir_dump/fractional_sumcheck_n256.cy.json [--port 8000]

Then open http://localhost:8000. The JSON file is re-read on every page
reload, so regenerating the dump and hitting F5 picks up the new graph.

Layout: node positions are computed offline. Two engines, both producing
top-to-bottom layered layouts for DAG viewing:

  * `dot` (Graphviz): full Sugiyama with crossing minimization — best
    quality, but O(|E|²) mincross makes it intractable above a few
    thousand edges (10+ minutes on the log_n=20 fractional-sumcheck
    fused graph).
  * `python-layered` (built-in): longest-path ranking + median-heuristic
    ordering, no dummy nodes for long edges. Runs in ~50ms on the same
    graph. Crossings-oblivious so the visual is messier than dot on
    small graphs, but usable on graphs where dot times out.

Results are cached in a `<dump>.pos.json` sidecar keyed on the dump's
mtime plus the layout knobs. `dot` requires Graphviz on PATH — install
with `apt install graphviz` or `brew install graphviz`.

Engine selection (`--engine auto`, the default) picks `dot` when the
graph has at most `--dot-max-edges` edges (default 2000) and
`python-layered` otherwise. Force `--engine dot` / `--engine
python-layered` to override.

If `dot` is missing (or `--no-layout` is passed) `python-layered` is
used regardless.

Layout knobs (all optional):

    --engine auto|dot|python-layered    layout engine (default auto)
    --dot-max-edges N                   `auto` uses dot when the graph
                                        has at most this many edges,
                                        python-layered otherwise
                                        (default 2000)
    --rankdir TB|LR|BT|RL               flow direction for dot only
                                        (default TB, top-to-bottom;
                                        python-layered is TB-only)
    --dup-leaves-threshold N            duplicate Const/Input nodes with
                                        more than N consumers (default 4;
                                        0 to disable) — one shadow copy
                                        per consumer, so the layout
                                        engine sees short local edges
                                        instead of one giant crossing
                                        bundle from a shared leaf
    --max-nodes N                       fall back to breadthfirst above N
                                        (post-duplication) nodes
    --no-layout                         skip layout entirely
"""

import argparse
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Graph IR viewer</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: monospace; }
  #cy { width: 100%; height: calc(100% - 28px); display: block; }
  #bar { height: 28px; line-height: 28px; padding: 0 8px; background: #222;
         color: #eee; font-size: 12px; }
  #bar .sw { display: inline-block; width: 10px; height: 10px; margin: 0 3px 0 12px;
             border-radius: 2px; vertical-align: middle; }
  #bar button { background: #333; color: #eee; border: 1px solid #555;
                border-radius: 3px; padding: 2px 8px; font: inherit; cursor: pointer;
                margin-left: 8px; }
  #bar button:hover { background: #444; }
  .panel { position: absolute; top: 40px; right: 12px; width: 520px;
           max-width: calc(100vw - 24px); max-height: calc(100vh - 60px);
           background: #fff; border: 1px solid #888; border-radius: 4px;
           box-shadow: 0 6px 20px rgba(0,0,0,0.25); display: none;
           flex-direction: column; z-index: 10; }
  .panel.open { display: flex; }
  .panel .hd { display: flex; align-items: center; justify-content: space-between;
               padding: 6px 10px; background: #eee; border-bottom: 1px solid #ccc;
               font-size: 12px; }
  .panel .hd .title { font-weight: bold; }
  .panel .hd .sub { color: #555; margin-left: 8px; font-weight: normal; }
  .panel .hd .close { cursor: pointer; user-select: none; padding: 0 6px;
                      color: #333; }
  .panel .hd .close:hover { color: #000; }
  .panel .body { padding: 8px 10px; overflow: auto; font-size: 11px;
                 position: relative; flex: 1; min-height: 0; }
  .panel .stats { display: grid; grid-template-columns: max-content auto;
                  column-gap: 12px; row-gap: 2px; margin-bottom: 8px; }
  .panel .stats .k { color: #555; }
  .panel h4 { margin: 8px 0 4px; font-size: 12px; }
  .panel pre { margin: 0; white-space: pre-wrap; word-break: break-word;
               background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 3px;
               padding: 6px 8px; }
  .panel table.freq { border-collapse: collapse; font-size: 11px; width: 100%; }
  .panel table.freq td, .panel table.freq th { border-bottom: 1px solid #eee;
                                                padding: 2px 6px; text-align: left; }
  .panel table.freq th { background: #f0f0f0; }
  .panel table.freq td.n { text-align: right; width: 3em; color: #555; }
  .panel table.shapes { border-collapse: collapse; font-size: 11px; }
  .panel table.shapes td, .panel table.shapes th { border-bottom: 1px solid #eee;
                                                    padding: 2px 8px 2px 0;
                                                    text-align: left; }
  .panel table.shapes th { color: #555; font-weight: normal; }
  .panel .chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 4px 0 8px; }
  .panel .chip { background: #eef4fb; border: 1px solid #b8d1ea; border-radius: 3px;
                 padding: 1px 6px; cursor: pointer; color: #14507b; font-size: 11px; }
  .panel .chip:hover { background: #d4e6f7; }
  .panel .chip .cid { color: #6a8fb0; margin-left: 4px; font-size: 10px; }
  /* Fusion history tree: an SVG binary tree (Reingold-Tilford-lite: leaves
     positioned left-to-right, internal nodes centered above children) with
     clickable nodes that populate a details pane underneath. The SVG lives
     inside a scrollable wrapper; mouse wheel zooms (cursor-anchored) by
     rescaling the SVG's width/height while its viewBox stays constant. */
  .panel .fh-svg-wrap { max-width: 100%; overflow: auto; max-height: 360px;
                        border: 1px solid #eee; background: #fafafa;
                        position: relative; cursor: grab; }
  .panel .fh-svg-wrap.panning { cursor: grabbing; user-select: none; }
  .panel .fh-svg-wrap .fh-hint { position: absolute; top: 4px; right: 6px;
                                 font-size: 10px; color: #888;
                                 background: #fafafae0; padding: 1px 5px;
                                 border-radius: 3px; pointer-events: none; }
  .panel .fh-svg { display: block; }
  .panel .fh-svg .fh-tn { cursor: pointer; }
  .panel .fh-svg .fh-tn rect { stroke-width: 1; }
  .panel .fh-svg .fh-tn.leaf rect { fill: #eef7ee; stroke: #59a14f; }
  .panel .fh-svg .fh-tn.fused rect { fill: #eef1f7; stroke: #4e79a7; }
  .panel .fh-svg .fh-tn:hover rect { stroke-width: 2; }
  .panel .fh-svg .fh-tn.selected rect { stroke: #d62728; stroke-width: 2; }
  .panel .fh-svg .fh-tn text { font: 10px monospace; fill: #111;
                                pointer-events: none; user-select: none; }
  .panel .fh-svg .fh-edge { stroke: #888; stroke-width: 1; fill: none; }
  .panel .fh-detail { margin-top: 8px; padding: 6px 8px;
                       border: 1px solid #ddd; border-radius: 3px;
                       background: #fff; min-height: 32px; }
  .panel .fh-detail .hint { color: #888; font-style: italic; }
  /* Stats panel: kernel-frequency rows clickable to reveal module IR
     in an inline detail pane below the tables. */
  .panel table.freq tr.clickable { cursor: pointer; }
  .panel table.freq tr.clickable:hover td { background: #f6f8fb; }
  .panel table.freq tr.clickable.selected td { background: #eef4fb; }
  .panel table.freq td .mod-key { color: #6a8fb0; margin-left: 6px;
                                   font-size: 10px; }
  .panel .sp-detail { margin-top: 8px; padding: 6px 8px;
                       border: 1px solid #ddd; border-radius: 3px;
                       background: #fff; min-height: 32px; }
  .panel .sp-detail .hint { color: #888; font-style: italic; }
</style>
</head>
<body>
<div id="bar">
  <span id="bar-text">loading…</span>
  <button id="stats-btn" style="display:none">Stats</button>
</div>
<div id="cy"></div>
<div id="node-panel" class="panel">
  <div class="hd">
    <span><span class="title" id="np-title"></span><span class="sub" id="np-sub"></span></span>
    <span class="close" data-target="node-panel">×</span>
  </div>
  <div class="body" id="np-body"></div>
</div>
<div id="stats-panel" class="panel">
  <div class="hd">
    <span class="title">Graph statistics</span>
    <span class="close" data-target="stats-panel">×</span>
  </div>
  <div class="body" id="sp-body"></div>
</div>
<script>
const TYPE_COLORS = {
  Kernel: "#4e79a7",
  BlackboxKernel: "#f28e2b",
  Const: "#59a14f",
  Memcpy: "#b07aa1",
  Memset: "#76b7b2",
  Input: "#edc948",
  Output: "#e15759",
};

const escHtml = s => String(s).replace(/[&<>"']/g, c =>
  ({"&":"&amp;", "<":"&lt;", ">":"&gt;", "\\"":"&quot;", "'":"&#39;"}[c]));

function openPanel(id) {
  document.querySelectorAll(".panel").forEach(p => {
    p.classList.toggle("open", p.id === id);
  });
}
function closePanel(id) {
  document.getElementById(id).classList.remove("open");
}
document.querySelectorAll(".panel .close").forEach(el => {
  el.addEventListener("click", () => closePanel(el.dataset.target));
});

// The active Cytoscape instance and the current data document — set once
// the graph loads. `focusNode` and the neighbor chips read them.
let CY = null;
let DATA = null;

function focusNode(id) {
  if (!CY) return;
  const ele = CY.$id(id);
  if (!ele || !ele.length) return;
  CY.elements().unselect();
  ele.select();
  CY.animate({ center: { eles: ele }, zoom: 1.5 }, { duration: 250 });
  const nd = ele.data();
  if (nd) renderNodePanel(nd, (DATA && DATA.modules) || {});
}

function shapeTable(shapes, heading) {
  if (!shapes || !shapes.length) return "";
  const rows = shapes.map((s, i) =>
    `<tr><td class="k">${i}</td><td><code>${escHtml(s)}</code></td></tr>`
  ).join("");
  return `<h4>${heading}</h4><table class="shapes">${rows}</table>`;
}

function chipRow(ids, names, heading) {
  if (!ids || !ids.length) return "";
  const chips = ids.map((id, i) => {
    const nm = (names && names[i]) || id;
    // Skip the redundant secondary label when the chip is already keyed
    // by its node id (used from the stats panel, which lists bare ids).
    const sub = nm === id ? "" : `<span class="cid">${escHtml(id)}</span>`;
    return `<span class="chip" data-focus="${escHtml(id)}">${escHtml(nm)}${sub}</span>`;
  }).join("");
  return `<h4>${heading}</h4><div class="chips">${chips}</div>`;
}

// Renders a kernel node's `param_bindings` map (name -> i64) as a small
// two-column table. `param_bindings` is emitted on Kernel nodes by
// `GraphBuilder::to_cytoscape_json`; each row lets the reader tie a
// symbol in the module HIR back to the concrete value bound for *this*
// node instantiation. Returns "" for missing / empty bindings.
function paramBindingsTable(bindings) {
  if (!bindings) return "";
  const entries = Object.entries(bindings);
  if (entries.length === 0) return "";
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  const rows = entries.map(([k, v]) =>
    `<tr><td class="k"><code>${escHtml(k)}</code></td>` +
    `<td><code>${escHtml(String(v))}</code></td></tr>`
  ).join("");
  return `<h4>Parameter bindings</h4><table class="shapes">${rows}</table>`;
}

// Lays out a FusionHistory tree using a simple Reingold-Tilford-lite:
// leaves get sequential x positions (0, 1, 2, ...); internal nodes are
// centered above the midpoint of their two children. y is depth from
// root. Both leaf and fused records carry name/ir/input_shapes/
// output_shapes so any node in the tree can be clicked for module
// details.
function layoutFusionTreeWithData(root) {
  const nodes = [];
  let nextLeafX = 0;
  function walk(h, depth) {
    let rec;
    if (h.kind === "leaf") {
      rec = { kind: "leaf", depth, x: nextLeafX++ };
    } else {
      const consumer = walk(h.consumer, depth + 1);
      const producer = walk(h.producer, depth + 1);
      rec = {
        kind: "fused", depth, x: (consumer.x + producer.x) / 2,
        consumer, producer,
      };
    }
    rec.name = h.name;
    rec.ir = h.ir;
    rec.input_shapes = h.input_shapes;
    rec.output_shapes = h.output_shapes;
    rec.id = "fh" + nodes.length;
    nodes.push(rec);
    return rec;
  }
  walk(root, 0);
  return nodes;
}

// SVG binary tree wrapped in a scrollable+zoomable container. Both leaves
// and fused internal nodes are clickable and carry IR + shapes (the Rust
// dumper snapshots the intermediate kernel at each fusion step).
function renderFusionTree(root) {
  const nodes = layoutFusionTreeWithData(root);
  const NW = 140, NH = 20, HGAP = 12, VGAP = 40;
  const maxDepth = Math.max(...nodes.map(n => n.depth));
  const nLeafPositions = Math.max(...nodes.map(n => n.x)) + 1;
  const width = Math.max(1, nLeafPositions * (NW + HGAP));
  const height = (maxDepth + 1) * VGAP + NH + 4;
  const nodeCX = n => n.x * (NW + HGAP) + NW / 2;
  const nodeTopY = n => n.depth * VGAP;
  const parts = [
    `<svg class="fh-svg" xmlns="http://www.w3.org/2000/svg" ` +
    `width="${width}" height="${height}" ` +
    `data-base-w="${width}" data-base-h="${height}" ` +
    `viewBox="0 0 ${width} ${height}">`,
  ];
  // Edges first so nodes overlay them.
  for (const n of nodes) {
    if (n.kind !== "fused") continue;
    const px = nodeCX(n), py = nodeTopY(n) + NH;
    for (const child of [n.consumer, n.producer]) {
      const cx = nodeCX(child), cy = nodeTopY(child);
      // Simple L-shape: down from parent, across, down into child.
      const midY = (py + cy) / 2;
      parts.push(
        `<path class="fh-edge" d="M ${px} ${py} L ${px} ${midY} ` +
        `L ${cx} ${midY} L ${cx} ${cy}"/>`
      );
    }
  }
  // Nodes.
  const truncate = (s, n) => s.length > n ? s.slice(0, n - 1) + "…" : s;
  for (const n of nodes) {
    const x = n.x * (NW + HGAP);
    const y = nodeTopY(n);
    parts.push(
      `<g class="fh-tn ${n.kind}" data-fh-id="${n.id}">` +
      `<title>${escHtml(n.name)}</title>` +
      `<rect x="${x}" y="${y}" width="${NW}" height="${NH}" rx="3"/>` +
      `<text x="${x + NW / 2}" y="${y + NH / 2 + 3}" text-anchor="middle">` +
      `${escHtml(truncate(n.name, 22))}</text>` +
      `</g>`
    );
  }
  parts.push("</svg>");
  const wrapHtml =
    `<div class="fh-svg-wrap">` +
    parts.join("") +
    `<span class="fh-hint">wheel = zoom · drag = pan · click = inspect</span>` +
    `</div>`;
  return {
    html: wrapHtml,
    detailHtml: `<div class="fh-detail"><span class="hint">` +
                `Click a node to inspect its module IR and shapes.</span></div>`,
    nodes,
  };
}

// Wires the SVG tree: cursor-anchored wheel zoom on the wrapper, and
// click a node → populate the details pane with its IR + shapes plus
// (for fused nodes) a consumer/producer/leaf-list summary.
function wireFusionTree(bodyDiv, nodes) {
  const wrap = bodyDiv.querySelector(".fh-svg-wrap");
  const svg = bodyDiv.querySelector(".fh-svg");
  const detail = bodyDiv.querySelector(".fh-detail");
  if (!wrap || !svg || !detail) return;
  const byId = new Map(nodes.map(n => [n.id, n]));

  // Wheel-to-zoom, cursor-anchored. We rescale the SVG's width/height
  // attributes while keeping viewBox constant — content scales, the
  // wrapper's scrollbars naturally accommodate the new element size,
  // and by adjusting scroll to compensate the point under the cursor
  // stays put.
  const baseW = parseFloat(svg.getAttribute("data-base-w"));
  const baseH = parseFloat(svg.getAttribute("data-base-h"));
  let zoom = 1;
  const MIN_Z = 0.15, MAX_Z = 6;
  wrap.addEventListener("wheel", ev => {
    ev.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    const contentX = wrap.scrollLeft + cx;
    const contentY = wrap.scrollTop + cy;
    const oldZoom = zoom;
    zoom *= Math.exp(-ev.deltaY * 0.0015);
    zoom = Math.min(MAX_Z, Math.max(MIN_Z, zoom));
    if (zoom === oldZoom) return;
    svg.setAttribute("width", baseW * zoom);
    svg.setAttribute("height", baseH * zoom);
    const factor = zoom / oldZoom;
    wrap.scrollLeft = contentX * factor - cx;
    wrap.scrollTop = contentY * factor - cy;
  }, { passive: false });

  // Drag-to-pan. mousedown anywhere in the wrap starts a pan session;
  // if the mouse moves past `PAN_THRESHOLD` pixels before mouseup we
  // switch into "grabbing" mode and translate the wrap's scroll. A short
  // click that never crosses the threshold falls through to the normal
  // node-click handler below; a drag that does move suppresses the
  // trailing `click` event so panning across a node doesn't accidentally
  // trigger its inspector.
  const PAN_THRESHOLD = 4;
  let panState = null;
  let suppressNextClick = false;
  wrap.addEventListener("mousedown", ev => {
    if (ev.button !== 0) return;
    panState = {
      startX: ev.clientX, startY: ev.clientY,
      scrollLeft: wrap.scrollLeft, scrollTop: wrap.scrollTop,
      moved: false,
    };
  });
  window.addEventListener("mousemove", ev => {
    if (!panState) return;
    const dx = ev.clientX - panState.startX;
    const dy = ev.clientY - panState.startY;
    if (!panState.moved && Math.hypot(dx, dy) > PAN_THRESHOLD) {
      panState.moved = true;
      wrap.classList.add("panning");
    }
    if (panState.moved) {
      wrap.scrollLeft = panState.scrollLeft - dx;
      wrap.scrollTop = panState.scrollTop - dy;
      ev.preventDefault();
    }
  });
  window.addEventListener("mouseup", () => {
    if (!panState) return;
    if (panState.moved) suppressNextClick = true;
    wrap.classList.remove("panning");
    panState = null;
  });
  // Capture-phase click interceptor: swallow the click that follows a
  // drag before it reaches the node-click handlers.
  wrap.addEventListener("click", ev => {
    if (suppressNextClick) {
      suppressNextClick = false;
      ev.stopPropagation();
      ev.preventDefault();
    }
  }, true);

  svg.querySelectorAll(".fh-tn").forEach(el => {
    el.addEventListener("click", () => {
      svg.querySelectorAll(".fh-tn.selected").forEach(s =>
        s.classList.remove("selected"));
      el.classList.add("selected");
      const rec = byId.get(el.dataset.fhId);
      if (!rec) return;
      const shapes = shapeTable(rec.input_shapes, "Inputs") +
                     shapeTable(rec.output_shapes, "Outputs");
      let ir;
      if (rec.ir) {
        ir = `<h4>Kernel IR</h4><pre>${escHtml(rec.ir)}</pre>`;
      } else if (rec.kind === "fused") {
        // The Rust emitter started attaching IR + shapes to intermediate
        // fused nodes; older dumps only have them on leaves.
        ir = `<h4>Kernel IR</h4><pre class="hint">(module IR is missing on ` +
             `this intermediate fused node — regenerate the .cy.json dump ` +
             `to pick up the new emitter.)</pre>`;
      } else {
        ir = "";
      }
      let header, extra = "";
      if (rec.kind === "leaf") {
        header = `<h4>${escHtml(rec.name)} <span class="hint">(leaf kernel)</span></h4>`;
      } else {
        // Collect leaves under this fusion for a quick provenance list.
        const leaves = [];
        const collect = n => {
          if (n.kind === "leaf") leaves.push(n.name);
          else { collect(n.consumer); collect(n.producer); }
        };
        collect(rec);
        header = `<h4>${escHtml(rec.name)} <span class="hint">` +
                 `(fused kernel · ${leaves.length} leaves)</span></h4>`;
        extra =
          `<div>consumer: <code>${escHtml(rec.consumer.name)}</code></div>` +
          `<div>producer: <code>${escHtml(rec.producer.name)}</code></div>` +
          `<h4>Leaves under this fusion</h4>` +
          `<div>${leaves.map(n => `<code>${escHtml(n)}</code>`).join(", ")}</div>`;
      }
      detail.innerHTML = header + shapes + ir + extra;
    });
  });
}

function renderNodePanel(data, modules) {
  const title = data.name || data.id;
  const type = data.type || "";
  document.getElementById("np-title").textContent = title;
  document.getElementById("np-sub").textContent = type ? "— " + type : "";
  const stats = [
    ["inputs", data.inputs],
    ["outputs", data.outputs],
    ["producers", data.producers],
    ["consumers", data.consumers],
  ].filter(([, v]) => v !== undefined);
  const statsHtml = stats.length
    ? `<div class="stats">${stats.map(([k, v]) =>
        `<span class="k">${k}</span><span>${v}</span>`).join("")}</div>`
    : "";
  const shapesHtml = shapeTable(data.input_shapes, "Inputs") +
                     shapeTable(data.output_shapes, "Outputs");
  const bindingsHtml = paramBindingsTable(data.param_bindings);
  const neighborsHtml = chipRow(data.producer_ids, data.producer_names, "Producers") +
                        chipRow(data.consumer_ids, data.consumer_names, "Consumers");
  const ir = data.ir || "";
  const irHtml = ir ? `<h4>Node IR</h4><pre>${escHtml(ir)}</pre>` : "";
  let moduleHtml = "";
  if (data.module && modules && modules[data.module] !== undefined) {
    moduleHtml = `<h4>Module IR — ${escHtml(data.module)}</h4>` +
                 `<pre>${escHtml(modules[data.module])}</pre>`;
  } else if (data.module) {
    moduleHtml = `<h4>Module IR — ${escHtml(data.module)}</h4>` +
                 `<pre>(module dump missing)</pre>`;
  }
  let fusionHtml = "";
  let fusionNodes = null;
  if (data.fusion_history) {
    const tree = renderFusionTree(data.fusion_history);
    fusionHtml = `<h4>Fusion history</h4>` + tree.html + tree.detailHtml;
    fusionNodes = tree.nodes;
  }
  const body = document.getElementById("np-body");
  body.innerHTML = statsHtml + shapesHtml + bindingsHtml + neighborsHtml +
                   irHtml + moduleHtml + fusionHtml;
  body.querySelectorAll(".chip[data-focus]").forEach(el => {
    el.addEventListener("click", () => focusNode(el.dataset.focus));
  });
  if (fusionNodes) wireFusionTree(body, fusionNodes);
  openPanel("node-panel");
}

function renderStatsPanel(data) {
  const modules = data.modules || {};
  const counts = {};
  // Kernel nodes group by module *key* (not name): same-name entries in
  // `data.modules` may be distinct modules (dedup suffixes them `#1`,
  // `#2`, ...); grouping by key keeps their bodies separate.
  const kernelGroups = {};   // { module_key: { name, ids: [node_id] } }
  const blackboxGroups = {}; // { name:       { ids: [node_id] } }
  for (const n of data.elements.nodes) {
    const nd = n.data;
    const t = nd.type || "?";
    counts[t] = (counts[t] || 0) + 1;
    if (t === "Kernel") {
      const key = nd.module || nd.name || nd.id;
      if (!kernelGroups[key]) {
        kernelGroups[key] = { name: nd.name || key, ids: [] };
      }
      kernelGroups[key].ids.push(nd.id);
    } else if (t === "BlackboxKernel") {
      const nm = nd.name || nd.id;
      if (!blackboxGroups[nm]) blackboxGroups[nm] = { ids: [] };
      blackboxGroups[nm].ids.push(nd.id);
    }
  }
  const summaryRows = ["Kernel", "BlackboxKernel", "Const", "Memcpy", "Memset", "Input", "Output"]
    .map(t => `<span class="k">${t}</span><span>${counts[t] || 0}</span>`).join("");
  const totalEdges = data.elements.edges.length;
  const totalNodes = data.elements.nodes.length;

  // Kernel frequency table — one row per distinct module. The `data-mkey`
  // attribute keys back into `kernelGroups` + `modules` for the click
  // handler below.
  const kernelEntries = Object.entries(kernelGroups)
    .sort((a, b) => b[1].ids.length - a[1].ids.length
                    || a[1].name.localeCompare(b[1].name));
  const kernelRowsHtml = kernelEntries.map(([key, g]) => {
    const suffix = key !== g.name
      ? `<span class="mod-key">${escHtml(key)}</span>` : "";
    return `<tr class="clickable" data-mkey="${escHtml(key)}">` +
           `<td class="n">${g.ids.length}</td>` +
           `<td>${escHtml(g.name)}${suffix}</td></tr>`;
  }).join("");
  const kernelTable = kernelRowsHtml
    ? `<h4>Kernel — ${counts.Kernel || 0} nodes, ${kernelEntries.length} distinct modules</h4>` +
      `<table class="freq"><thead><tr><th class="n">n</th><th>name</th></tr></thead>` +
      `<tbody>${kernelRowsHtml}</tbody></table>`
    : `<h4>Kernel — (none)</h4>`;

  // BlackboxKernel table — clickable to list nodes; no HIR body exists.
  const blackboxEntries = Object.entries(blackboxGroups)
    .sort((a, b) => b[1].ids.length - a[1].ids.length
                    || a[0].localeCompare(b[0]));
  const blackboxRowsHtml = blackboxEntries.map(([nm, g]) =>
    `<tr class="clickable" data-bname="${escHtml(nm)}">` +
    `<td class="n">${g.ids.length}</td><td>${escHtml(nm)}</td></tr>`
  ).join("");
  const blackboxTable = blackboxRowsHtml
    ? `<h4>BlackboxKernel — ${counts.BlackboxKernel || 0} nodes, ${blackboxEntries.length} distinct</h4>` +
      `<table class="freq"><thead><tr><th class="n">n</th><th>name</th></tr></thead>` +
      `<tbody>${blackboxRowsHtml}</tbody></table>`
    : `<h4>BlackboxKernel — (none)</h4>`;

  const spBody = document.getElementById("sp-body");
  spBody.innerHTML =
    `<div class="stats">` +
    `<span class="k">total nodes</span><span>${totalNodes}</span>` +
    `<span class="k">total edges</span><span>${totalEdges}</span>` +
    summaryRows +
    `</div>` +
    kernelTable + blackboxTable +
    `<div class="sp-detail"><span class="hint">` +
    `Click a row to show its body IR.</span></div>`;

  const detail = spBody.querySelector(".sp-detail");
  const clearSelection = () => spBody
    .querySelectorAll("tr.clickable.selected")
    .forEach(s => s.classList.remove("selected"));
  const wireChips = () => detail.querySelectorAll(".chip[data-focus]")
    .forEach(el => el.addEventListener("click", () => focusNode(el.dataset.focus)));

  spBody.querySelectorAll("tr[data-mkey]").forEach(tr => {
    tr.addEventListener("click", () => {
      clearSelection();
      tr.classList.add("selected");
      const key = tr.dataset.mkey;
      const grp = kernelGroups[key];
      const ir = modules[key];
      const header = `<h4>Module IR — ${escHtml(key)}` +
                     ` <span class="hint">(${grp.ids.length} node${grp.ids.length === 1 ? "" : "s"})</span></h4>`;
      const irHtml = ir !== undefined
        ? `<pre>${escHtml(ir)}</pre>`
        : `<pre class="hint">(module dump missing)</pre>`;
      const chips = chipRow(grp.ids, grp.ids, "Nodes");
      detail.innerHTML = header + irHtml + chips;
      wireChips();
    });
  });
  spBody.querySelectorAll("tr[data-bname]").forEach(tr => {
    tr.addEventListener("click", () => {
      clearSelection();
      tr.classList.add("selected");
      const nm = tr.dataset.bname;
      const grp = blackboxGroups[nm];
      const header = `<h4>${escHtml(nm)}` +
                     ` <span class="hint">(blackbox kernel — no HIR body)</span></h4>`;
      const chips = chipRow(grp.ids, grp.ids, "Nodes");
      detail.innerHTML = header + chips;
      wireChips();
    });
  });

  openPanel("stats-panel");
}

fetch("graph.json")
  .then(r => { if (!r.ok) throw new Error(r.status + " " + r.statusText); return r.json(); })
  .then(data => {
    DATA = data;
    // Preset layout consumes the {x, y} positions injected by the Python
    // side from dot's output. If they're missing (dot not on PATH, or
    // --no-layout was passed) fall back to cytoscape's built-in
    // breadthfirst — fast, near-linear, no dependency on any external
    // layout algorithm.
    const hasPos = data.elements.nodes.length && data.elements.nodes[0].position;
    const layoutName = hasPos ? "preset" : "breadthfirst";
    const layout = hasPos
      ? { name: "preset", fit: true }
      : { name: "breadthfirst", directed: true, spacingFactor: 1.0, grid: false };
    const big = data.elements.edges.length > 1500;
    const cy = cytoscape({
      container: document.getElementById("cy"),
      elements: data.elements,
      hideEdgesOnViewport: big,
      textureOnViewport: big,
      pixelRatio: big ? 1 : "auto",
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-wrap": "wrap",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": 9,
            shape: "round-rectangle",
            width: "label",
            height: "label",
            padding: "6px",
            "background-color": ele => TYPE_COLORS[ele.data("type")] || "#888",
            "background-opacity": 0.85,
            color: "#111",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 3,
            "border-color": "#d62728",
          },
        },
        {
          selector: "edge",
          style: {
            label: "data(label)",
            "curve-style": big ? "straight" : "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "data(color)",
            "target-arrow-color": "data(color)",
            width: 1.5,
            "arrow-scale": 0.8,
            "font-size": 8,
            "text-rotation": "autorotate",
            "text-background-color": "#fff",
            "text-background-opacity": 0.8,
            "min-zoomed-font-size": 8,
            color: "#333",
            opacity: "data(opacity)",
          },
        },
      ],
      layout,
      wheelSensitivity: 0.2,
    });
    CY = cy;
    // Length-based edge fade: long edges (typically the ones that span
    // many layers and cause the visual overload) decay exponentially
    // toward a floor opacity so they stay visible but don't dominate.
    // Scale is adaptive — pick the 75th-percentile edge length so that
    // long edges (top quartile) end up at ~half fade, regardless of
    // absolute graph size.
    const EDGE_OPACITY_FLOOR = 0.15;
    function applyEdgeOpacity() {
      const edges = cy.edges();
      if (edges.length === 0) return;
      const lengths = new Array(edges.length);
      edges.forEach((e, i) => {
        const sp = e.source().position();
        const tp = e.target().position();
        lengths[i] = Math.hypot(tp.x - sp.x, tp.y - sp.y);
      });
      const sorted = lengths.slice().sort((a, b) => a - b);
      const p75 = sorted[Math.floor(sorted.length * 0.75)];
      // scale so that a p75-length edge sits at opacity ~0.575 (half
      // between floor and 1). Guard against degenerate p75 = 0.
      const scale = p75 > 0 ? p75 / Math.LN2 : 1;
      cy.batch(() => {
        edges.forEach((e, i) => {
          const opacity = EDGE_OPACITY_FLOOR +
                          (1 - EDGE_OPACITY_FLOOR) * Math.exp(-lengths[i] / scale);
          e.data("opacity", opacity);
        });
      });
    }
    // Cytoscape may apply the initial layout synchronously (preset) or
    // asynchronously (breadthfirst). Run once now for the sync case;
    // layoutstop handles anything async.
    applyEdgeOpacity();
    cy.on("layoutstop", applyEdgeOpacity);
    cy.on("tap", "node", evt => renderNodePanel(evt.target.data(), data.modules || {}));
    cy.on("tap", (evt) => { if (evt.target === cy) closePanel("node-panel"); });
    const statsBtn = document.getElementById("stats-btn");
    statsBtn.style.display = "";
    statsBtn.addEventListener("click", () => {
      const p = document.getElementById("stats-panel");
      if (p.classList.contains("open")) closePanel("stats-panel");
      else renderStatsPanel(data);
    });
    const legend = Object.entries(TYPE_COLORS)
      .map(([t, c]) => `<span class="sw" style="background:${c}"></span>${t}`)
      .join("");
    document.getElementById("bar-text").innerHTML =
      `${cy.nodes().length} nodes, ${cy.edges().length} edges` +
      ` — layout: ${layoutName}` +
      ` — edges: <span style="color:#aaa">black=read</span>,` +
      ` <span style="color:#f66">red=modify</span> —${legend}`;
  })
  .catch(err => { document.getElementById("bar-text").textContent = "error: " + err; });
</script>
</body>
</html>
"""


def duplicate_high_fanout_leaves(doc: dict, threshold: int) -> int:
    """Splits Const / Input nodes with fanout > `threshold` into per-consumer
    shadow copies. Each shadow reuses the original node's data (name, ir,
    ...) but gets a unique id (`<orig>_c{n}`); edges from the original are
    rewritten to originate at the shadow. Layered layouts turn a
    high-fanout leaf into one long-edge bundle that spans the whole graph
    width — duplicating the leaf collapses that into many short local
    edges, dramatically reducing crossings and shrinking the layout
    engine's working set. Returns the number of shadow nodes injected
    (0 if nothing was eligible).
    """
    els = doc["elements"]
    nodes = els["nodes"]
    edges = els["edges"]
    fanout: dict[str, int] = {}
    for e in edges:
        s = e["data"]["source"]
        fanout[s] = fanout.get(s, 0) + 1
    node_by_id = {n["data"]["id"]: n for n in nodes}
    eligible = {
        n["data"]["id"]
        for n in nodes
        if n["data"].get("type") in ("Const", "Input")
        and fanout.get(n["data"]["id"], 0) > threshold
    }
    if not eligible:
        return 0
    new_nodes = [n for n in nodes if n["data"]["id"] not in eligible]
    new_edges = []
    counter: dict[str, int] = {}
    for e in edges:
        s = e["data"]["source"]
        if s in eligible:
            n = counter.get(s, 0)
            counter[s] = n + 1
            shadow_id = f"{s}_c{n}"
            orig = node_by_id[s]
            shadow = {"data": {**orig["data"], "id": shadow_id}}
            new_nodes.append(shadow)
            new_edge = {"data": {**e["data"], "source": shadow_id}}
            # Give shadow edges a unique id too, so cytoscape doesn't reject
            # them as duplicates when a leaf's shadows all feed the same
            # target through different original edges.
            new_edge["data"]["id"] = f"{e['data']['id']}_{shadow_id}"
            new_edges.append(new_edge)
        else:
            new_edges.append(e)
    els["nodes"] = new_nodes
    els["edges"] = new_edges
    return sum(counter.values())


def node_dims_px(data: dict) -> tuple[float, float]:
    """Node width/height in pixels, matching cytoscape's label-sized
    rendering: ~6px per char horizontally + 16px padding, ~12px per
    line vertically + 16px padding. Used by both DOT emission (after
    a /72 conversion to inches) and the Python layered layout."""
    label = data.get("label") or data["id"]
    lines = label.split("\n")
    max_chars = max((len(ln) for ln in lines), default=1)
    n_lines = len(lines)
    return (max(60.0, 6 * max_chars + 16), 16.0 + 12 * n_lines)


def emit_dot_source(doc: dict, rankdir: str) -> str:
    """Serialize the cytoscape doc as a Graphviz DOT source. Node widths
    and heights track the label — `fixedsize=true` so dot leaves enough
    room for cytoscape to draw the same node without overlap. Parallel
    edges between the same pair of endpoints are collapsed for layout
    (they would route separately, but only add crossing-minimization
    cost for our purposes)."""
    out = ["digraph G {"]
    out.append(f"  rankdir={rankdir};")
    # Spacing tuned so nodes don't collide when cytoscape renders labels
    # at font-size 9. Ranksep is generous because the labels wrap onto
    # two lines (name + type) and edge labels sit between ranks.
    out.append("  nodesep=0.25;")
    out.append("  ranksep=0.6;")
    # polyline is much faster than the default `spline` routing and
    # avoids dot's expensive edge-routing pass on graphs where we throw
    # the routes away and let cytoscape draw bezier curves anyway.
    out.append("  splines=polyline;")
    out.append('  node [shape=box, fixedsize=true];')
    for n in doc["elements"]["nodes"]:
        d = n["data"]
        w_px, h_px = node_dims_px(d)
        out.append(
            f'  "{d["id"]}" [width={w_px / 72:.3f}, height={h_px / 72:.3f}];'
        )
    seen: set[tuple[str, str]] = set()
    for e in doc["elements"]["edges"]:
        s = e["data"]["source"]
        t = e["data"]["target"]
        key = (s, t)
        if key in seen:
            continue
        seen.add(key)
        out.append(f'  "{s}" -> "{t}";')
    out.append("}")
    return "\n".join(out)


def python_layered_layout(doc: dict) -> dict[str, tuple[float, float]]:
    """Pure-Python layered layout for large DAGs. Longest-path ranking
    via Kahn's algorithm, then a few median-heuristic sweeps to order
    nodes within each rank (predecessor median top-down, successor
    median bottom-up), then width-aware x placement.

    Crossings-oblivious (no dummy-node insertion for long edges, no
    global optimization), but O(V + E · sweeps) — sub-second on graphs
    dot times out on. The result is a valid TB flow: every edge points
    from a smaller y to a larger y, which is what makes the graph read
    as a DAG.

    Any cycles in `doc` (there shouldn't be any — graph_ir is a strict
    DAG) get their edges dropped from ranking; those nodes settle at
    rank 0 and won't participate in the median ordering.
    """
    from collections import deque

    nodes = doc["elements"]["nodes"]
    edges_raw = doc["elements"]["edges"]
    node_ids = [n["data"]["id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    dims = [node_dims_px(nd["data"]) for nd in nodes]

    preds: list[list[int]] = [[] for _ in range(n)]
    succs: list[list[int]] = [[] for _ in range(n)]
    for e in edges_raw:
        s = idx.get(e["data"]["source"])
        t = idx.get(e["data"]["target"])
        if s is None or t is None or s == t:
            continue
        preds[t].append(s)
        succs[s].append(t)

    # Longest-path ranking. Kahn's peel: any node still with unprocessed
    # predecessors after the queue drains is part of (or downstream of)
    # a cycle; leave those at rank 0 rather than crash — the doc is
    # supposed to be acyclic but a defensive fallback keeps the viewer
    # useful even if that invariant slips.
    indeg = [len(p) for p in preds]
    rank = [0] * n
    q: deque[int] = deque(i for i in range(n) if indeg[i] == 0)
    while q:
        u = q.popleft()
        for v in succs[u]:
            if rank[v] < rank[u] + 1:
                rank[v] = rank[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    max_rank = max(rank) if rank else 0
    layers: list[list[int]] = [[] for _ in range(max_rank + 1)]
    for i, r in enumerate(rank):
        layers[r].append(i)

    # Median-heuristic ordering. Initial positions from insertion order.
    # Top-down: reorder each rank by median position of predecessors in
    # the previous rank. Bottom-up: same but with successors. Four
    # sweeps is enough for our graphs to converge to a stable ordering.
    pos = [0] * n
    for layer in layers:
        for j, ni in enumerate(layer):
            pos[ni] = j

    def resort(layer: list[int], neighbors: list[list[int]], r_neighbor: int) -> list[int]:
        keyed = []
        for ni in layer:
            neigh_pos = sorted(pos[nb] for nb in neighbors[ni] if rank[nb] == r_neighbor)
            if neigh_pos:
                m = len(neigh_pos)
                key = (
                    neigh_pos[m // 2]
                    if m % 2 == 1
                    else (neigh_pos[m // 2 - 1] + neigh_pos[m // 2]) / 2
                )
            else:
                key = pos[ni]
            keyed.append((key, ni))
        keyed.sort()
        return [ni for _, ni in keyed]

    for _ in range(4):
        for r in range(1, len(layers)):
            layers[r] = resort(layers[r], preds, r - 1)
            for j, ni in enumerate(layers[r]):
                pos[ni] = j
        for r in range(len(layers) - 2, -1, -1):
            layers[r] = resort(layers[r], succs, r + 1)
            for j, ni in enumerate(layers[r]):
                pos[ni] = j

    # X placement: pack each layer L→R by node width + a gap. Then
    # center each layer horizontally so narrow ranks float in the
    # middle of the canvas rather than left-aligning. Y is per-rank
    # multiples of RANKSEP.
    NODESEP = 20.0
    RANKSEP = 60.0
    layer_widths = []
    for layer in layers:
        w = 0.0
        for ni in layer:
            w += dims[ni][0] + NODESEP
        layer_widths.append(max(0.0, w - NODESEP))
    max_w = max(layer_widths) if layer_widths else 0.0
    positions: dict[str, tuple[float, float]] = {}
    for r, layer in enumerate(layers):
        offset = (max_w - layer_widths[r]) / 2
        xoff = offset
        y = r * RANKSEP
        for ni in layer:
            w, _ = dims[ni]
            positions[node_ids[ni]] = (xoff + w / 2, y)
            xoff += w + NODESEP
    return positions


def parse_dot_positions(text: str) -> dict[str, tuple[float, float]]:
    """Extract `{node_id: (x, y_screen)}` from `dot -Tjson0` output. Dot's
    y-axis points up (graphics convention); we flip using the graph
    bounding box (`bb` = "x0,y0,x1,y1") so downstream consumers get
    screen coordinates (y increases downward). Coordinates come back in
    points (72 per inch); we pass them straight to cytoscape, which
    treats them as pixels at zoom 1.0 — a 1:1 pt→px mapping is close
    enough that dot's spacing choices survive without extra scaling."""
    j = json.loads(text)
    bb = j.get("bb", "0,0,0,0").split(",")
    max_y = float(bb[3]) if len(bb) == 4 else 0.0
    positions: dict[str, tuple[float, float]] = {}
    for obj in j.get("objects", []):
        name = obj.get("name")
        pos = obj.get("pos")
        if not name or not pos:
            continue
        parts = pos.split(",")
        if len(parts) != 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        positions[name] = (x, max_y - y)
    return positions


def pick_engine(auto_engine: str, n_edges: int, dot_max_edges: int) -> str:
    """Resolve `auto` to `dot` when the (post-duplication) edge count
    fits under the threshold and `python-layered` otherwise. dot's
    mincross is O(|E|²) per iteration, so the crossover is
    edge-count-driven, not node-count."""
    if auto_engine == "auto":
        return "dot" if n_edges <= dot_max_edges else "python-layered"
    return auto_engine


class LayoutCache:
    """Layout positions for a dump file, cached in memory and in a
    `<dump>.pos.json` sidecar. Keyed on `(dump_mtime, params_hash)` so
    layout knobs changing (engine, rankdir, dup-leaves threshold) force
    a recompute."""

    def __init__(
        self,
        json_path: Path,
        auto_engine: str,
        rankdir: str,
        dup_leaves_threshold: int,
        max_nodes: int,
        dot_max_edges: int,
        use_layout: bool,
    ):
        self.json_path = json_path
        self.sidecar = Path(str(json_path) + ".pos.json")
        self.auto_engine = auto_engine
        self.rankdir = rankdir
        self.dup_leaves_threshold = dup_leaves_threshold
        self.max_nodes = max_nodes
        self.dot_max_edges = dot_max_edges
        self.use_layout = use_layout
        self.lock = threading.Lock()
        self.mtime: float | None = None
        self.body = b""  # serialized graph.json response

    def _params_hash(self, engine: str) -> str:
        # rankdir only affects dot; omit it from the python-layered
        # cache key so switching --rankdir doesn't force a recompute
        # for graphs handled by the built-in.
        params = {
            "engine": engine,
            "dup_leaves_threshold": self.dup_leaves_threshold,
        }
        if engine == "dot":
            params["rankdir"] = self.rankdir
        return json.dumps(params, sort_keys=True)

    def _resolve_engine(self, n_edges: int) -> str:
        """Auto → concrete engine, with a fallback to python-layered
        when dot is requested but not installed. Warns once via the
        caller's log line."""
        engine = pick_engine(self.auto_engine, n_edges, self.dot_max_edges)
        if engine == "dot" and shutil.which("dot") is None:
            print(
                "note: `dot` not on PATH — falling back to python-layered.\n"
                "Install Graphviz with `apt install graphviz` or "
                "`brew install graphviz` for the higher-quality layered "
                "layout on small graphs.",
                file=sys.stderr,
            )
            engine = "python-layered"
        return engine

    def _compute_positions(
        self, doc: dict, n_nodes: int, n_edges: int
    ) -> dict[str, tuple[float, float]] | None:
        mtime = self.json_path.stat().st_mtime
        engine = self._resolve_engine(n_edges)
        params_hash = self._params_hash(engine)
        if self.sidecar.is_file():
            try:
                cached = json.loads(self.sidecar.read_text())
                if (
                    cached.get("mtime") == mtime
                    and cached.get("params_hash") == params_hash
                ):
                    return {k: (v[0], v[1]) for k, v in cached["positions"].items()}
            except (json.JSONDecodeError, KeyError):
                pass
        if not self.use_layout:
            return None
        if n_nodes > self.max_nodes:
            print(
                f"note: {n_nodes} nodes exceeds --max-nodes "
                f"({self.max_nodes}); skipping layout and falling back to "
                "the in-browser breadthfirst layout. Bump the threshold or "
                "pass --no-layout to silence.",
                file=sys.stderr,
            )
            return None
        why = ""
        if self.auto_engine == "auto":
            cmp_op = "<=" if engine == "dot" else ">"
            why = (
                f" (auto: {n_edges} edges {cmp_op} "
                f"--dot-max-edges={self.dot_max_edges})"
            )
        print(
            f"computing {engine} layout for {self.json_path}{why} ...",
            flush=True,
        )
        t0 = time.time()
        if engine == "python-layered":
            positions = python_layered_layout(doc)
        else:
            dot_src = emit_dot_source(doc, self.rankdir)
            try:
                r = subprocess.run(
                    ["dot", "-Tjson0"],
                    input=dot_src,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
            except FileNotFoundError:
                # Race with dot being removed between the `which` check
                # in _resolve_engine and here — bail on this attempt.
                return None
            if r.returncode != 0:
                print(f"dot layout failed: {r.stderr.strip()}", file=sys.stderr)
                return None
            positions = parse_dot_positions(r.stdout)
        self.sidecar.write_text(
            json.dumps(
                {
                    "mtime": mtime,
                    "params_hash": params_hash,
                    # JSON has no tuples; store as [x, y] and reconstruct
                    # on load.
                    "positions": {k: [v[0], v[1]] for k, v in positions.items()},
                }
            )
        )
        print(f"{engine} layout done in {time.time() - t0:.2f}s", flush=True)
        return positions

    def graph_json(self) -> bytes:
        with self.lock:
            mtime = self.json_path.stat().st_mtime
            if mtime == self.mtime:
                return self.body
            doc = json.loads(self.json_path.read_bytes())
            n_before = len(doc["elements"]["nodes"])
            e_before = len(doc["elements"]["edges"])
            n_shadows = 0
            if self.dup_leaves_threshold > 0:
                n_shadows = duplicate_high_fanout_leaves(
                    doc, self.dup_leaves_threshold
                )
            if n_shadows:
                print(
                    f"duplicated {n_shadows} leaf fan-outs (Const/Input "
                    f"nodes with >{self.dup_leaves_threshold} consumers): "
                    f"{n_before} nodes / {e_before} edges -> "
                    f"{len(doc['elements']['nodes'])} nodes / "
                    f"{len(doc['elements']['edges'])} edges",
                    flush=True,
                )
            positions = self._compute_positions(
                doc,
                len(doc["elements"]["nodes"]),
                len(doc["elements"]["edges"]),
            )
            if positions:
                for n in doc["elements"]["nodes"]:
                    p = positions.get(n["data"]["id"])
                    if p:
                        n["position"] = {"x": p[0], "y": p[1]}
            self.body = json.dumps(doc).encode()
            self.mtime = mtime
            return self.body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path, help="Cytoscape .cy.json dump to serve")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "dot", "python-layered"],
        help="Layout engine. `auto` (default) picks dot for small graphs "
        "and the built-in python-layered otherwise — see --dot-max-edges. "
        "Force one explicitly to bypass the heuristic.",
    )
    ap.add_argument(
        "--dot-max-edges",
        type=int,
        default=2000,
        help="Edge-count threshold for the `auto` engine choice: at or "
        "below this many edges (post-duplication) use dot, otherwise "
        "python-layered. Default 2000 — dot's mincross is O(|E|²) per "
        "iteration and takes 10+ minutes past ~5000 edges on typical "
        "IR graphs.",
    )
    ap.add_argument(
        "--rankdir",
        default="TB",
        choices=["TB", "LR", "BT", "RL"],
        help="Graphviz layered flow direction for dot only (default TB, "
        "top-to-bottom). Ignored by python-layered (TB-only).",
    )
    ap.add_argument(
        "--dup-leaves-threshold",
        type=int,
        default=4,
        help="Duplicate any Const/Input node with more than this many "
        "consumers, injecting one shadow copy per consumer so the layout "
        "engine sees short local edges instead of a giant crossing "
        "bundle. Set to 0 to disable. Default 4.",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=25000,
        help="Skip layout and fall back to the in-browser breadthfirst "
        "layout when the (possibly duplicated) graph exceeds this many "
        "nodes. Default 25000.",
    )
    ap.add_argument(
        "--no-layout",
        action="store_true",
        help="Skip Graphviz unconditionally (useful for huge dumps where "
        "the in-browser breadthfirst layout is fine).",
    )
    args = ap.parse_args()
    if not args.json_path.is_file():
        sys.exit(f"error: {args.json_path} does not exist")

    cache = LayoutCache(
        args.json_path,
        auto_engine=args.engine,
        rankdir=args.rankdir,
        dup_leaves_threshold=args.dup_leaves_threshold,
        max_nodes=args.max_nodes,
        dot_max_edges=args.dot_max_edges,
        use_layout=not args.no_layout,
    )
    cache.graph_json()  # warm the layout before serving

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body, ctype = INDEX_HTML.encode(), "text/html; charset=utf-8"
            elif self.path == "/graph.json":
                body, ctype = cache.graph_json(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            pass

    with http.server.ThreadingHTTPServer(("localhost", args.port), Handler) as srv:
        print(f"serving {args.json_path} at http://localhost:{args.port}")
        srv.serve_forever()


if __name__ == "__main__":
    main()
