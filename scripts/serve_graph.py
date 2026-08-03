#!/usr/bin/env python3
"""Serve a GraphBuilder Cytoscape JSON dump for browser visualization.

Usage:
    python3 scripts/serve_graph.py target/ir_dump/fractional_sumcheck_n256.cy.json [--port 8000]

Then open http://localhost:8000. The JSON file is re-read on every page
reload, so regenerating the dump and hitting F5 picks up the new graph.

Layout: if `node` and the `elkjs` package are available, node positions are
computed server-side with ELK's layered algorithm (Sugiyama-style crossing
minimization) and served as a preset layout — this handles thousands of
nodes, where the in-browser layered layouts (dagre, klay) freeze the tab.
The result is cached in a `<dump>.pos.json` sidecar keyed on the dump's
mtime and the layout knobs (direction, wrapping, dup-leaves threshold).

Install elkjs with one of:

    npm install elkjs                          # resolved via ./node_modules
    npm install --prefix scripts elkjs         # resolved next to this script
    python3 scripts/serve_graph.py --node-modules /path/to/node_modules ...

Without elkjs the viewer falls back to the (fast, crossing-oblivious)
breadthfirst layout. Client-side overrides: ?layout=breadthfirst|dagre|klay.

Layout knobs (all optional, defaults are tuned for graphs up to ~10k nodes):

    --direction DOWN|RIGHT|UP|LEFT      flow direction (default DOWN)
    --wrap-layers                       wrap over-wide layers (opt-in)
    --dup-leaves-threshold N            duplicate Const/Input nodes with
                                        more than N consumers (default 4;
                                        0 to disable) — one shadow copy
                                        per consumer, so ELK sees short
                                        local edges instead of a giant
                                        crossing bundle from a shared leaf
    --node-heap-mb N                    V8 heap for the ELK subprocess
                                        (default 16384) — the default 4 GB
                                        OOMs on graphs of a few thousand
                                        nodes with heavy edge density
    --max-elk-nodes N                   fall back to breadthfirst above N
                                        (post-duplication) nodes; ELK's
                                        crossing minimization is quadratic
                                        in the widest layer
    --no-elk                            skip ELK entirely
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Reads the cy.json elements from argv[2], runs ELK layered layout, prints
# {node_id: [center_x, center_y]} to stdout.
#
# Options (via env):
#   ELK_DIRECTION       flow direction: DOWN|RIGHT|UP|LEFT (default DOWN)
#   ELK_ASPECT_RATIO    target width/height ratio (default 1.6)
#   ELK_WRAP            if "1", wrap over-wide layers into stacked sub-rows
#   ELK_QUALITY         quality/speed preset: fast|balanced|nice
#                        - fast:     SIMPLE placement, no crossing min, no
#                                    post-compaction — sub-second on ~10k
#                                    nodes, edges are messy.
#                        - balanced: LINEAR_SEGMENTS placement, LAYER_SWEEP
#                                    crossing min. Default.
#                        - nice:     BRANDES_KOEPF placement (4-alignment
#                                    straight-edge LP), LAYER_SWEEP, plus
#                                    EDGE_LENGTH post-compaction. Slowest.
#
# `nice` was the old default — it's the prettiest but on 2k+ node graphs
# is several × slower than `balanced` because BRANDES_KOEPF re-runs four
# independent placement alignments and picks the best.
ELK_JS = """
const fs = require("fs");
const ELK = require("elkjs");
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8")).elements;
const direction = process.env.ELK_DIRECTION || "DOWN";
const aspectRatio = process.env.ELK_ASPECT_RATIO || "1.6";
const wrap = process.env.ELK_WRAP === "1";
const quality = process.env.ELK_QUALITY || "balanced";
const seen = new Set();
const edges = data.edges.filter(e => {
  const k = e.data.source + ">" + e.data.target;
  if (seen.has(k)) return false;
  seen.add(k);
  return true;
});
const dim = n => {
  const lines = (n.data.label || n.data.id).split("\\n");
  const w = Math.max(...lines.map(l => l.length));
  return { width: Math.max(60, 6 * w + 16), height: 16 + 12 * lines.length };
};
// The three presets swap layering strategy, node placement, crossing
// minimization, and edge routing. NETWORK_SIMPLEX layering is a full LP
// over all nodes — on 2k+ node graphs it dominates the run time. Switch
// to LONGEST_PATH for the fast presets: linear-time, gives deeper but
// visually reasonable layers.
const PRESETS = {
  fast: {
    layering: "LONGEST_PATH",
    placement: "SIMPLE",
    crossingMin: "NONE",
    edgeRouting: "POLYLINE",
    postCompaction: null,
  },
  balanced: {
    layering: "LONGEST_PATH",
    placement: "LINEAR_SEGMENTS",
    crossingMin: "LAYER_SWEEP",
    edgeRouting: "POLYLINE",
    postCompaction: null,
  },
  nice: {
    layering: "NETWORK_SIMPLEX",
    placement: "BRANDES_KOEPF",
    crossingMin: "LAYER_SWEEP",
    edgeRouting: "ORTHOGONAL",
    postCompaction: "EDGE_LENGTH",
  },
};
const preset = PRESETS[quality] || PRESETS.balanced;
const layoutOptions = {
  "elk.algorithm": "layered",
  "elk.direction": direction,
  "elk.aspectRatio": aspectRatio,
  "elk.layered.thoroughness": "1",
  "elk.layered.layering.strategy": preset.layering,
  "elk.layered.crossingMinimization.strategy": preset.crossingMin,
  // O(n^2) per layer per iteration — OFF for any preset we support.
  "elk.layered.crossingMinimization.greedySwitch.type": "OFF",
  "elk.layered.crossingMinimization.greedySwitchHierarchical.type": "OFF",
  "elk.layered.nodePlacement.strategy": preset.placement,
  "elk.edgeRouting": preset.edgeRouting,
  // Merges parallel edges between the same two layers for layout
  // purposes (bundled for placement; still drawn separately). A huge
  // win on graphs with high-fanout hubs (each hub's outgoing bundle
  // gets treated as one edge for routing).
  "elk.layered.mergeEdges": "true",
  "elk.spacing.nodeNode": "20",
  "elk.layered.spacing.nodeNodeBetweenLayers": "40",
};
if (preset.placement === "BRANDES_KOEPF") {
  layoutOptions["elk.layered.nodePlacement.bk.fixedAlignment"] = "BALANCED";
}
if (preset.postCompaction) {
  layoutOptions["elk.layered.compaction.postCompaction.strategy"] = preset.postCompaction;
}
if (wrap) {
  // MULTI_EDGE wraps whenever the aspect-ratio target would be exceeded,
  // pulling parts of over-wide layers into stacked sub-rows.
  layoutOptions["elk.layered.wrapping.strategy"] = "MULTI_EDGE";
  layoutOptions["elk.layered.wrapping.additionalEdgeSpacing"] = "20";
}
const graph = {
  id: "root",
  layoutOptions,
  children: data.nodes.map(n => ({ id: n.data.id, ...dim(n) })),
  edges: edges.map((e, i) => ({ id: "le" + i, sources: [e.data.source], targets: [e.data.target] })),
};
const t0 = Date.now();
new ELK().layout(graph).then(g => {
  const pos = {};
  for (const c of g.children) pos[c.id] = [Math.round(c.x + c.width / 2), Math.round(c.y + c.height / 2)];
  process.stderr.write(`elk ${quality} preset done in ${((Date.now() - t0) / 1000).toFixed(1)}s\\n`);
  process.stdout.write(JSON.stringify(pos));
}).catch(e => { console.error(String(e)); process.exit(1); });
"""

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Graph IR viewer</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<script src="https://unpkg.com/klayjs@0.4.1/klay.js"></script>
<script src="https://unpkg.com/cytoscape-klay@3.1.4/cytoscape-klay.js"></script>
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

// "elk" means server-computed preset positions (crossing-minimized layered
// layout). The in-browser layered layouts (dagre, klay) freeze the tab on
// graphs beyond a few hundred edges; breadthfirst is near-linear.
const LAYOUTS = {
  elk: { name: "preset", fit: true },
  breadthfirst: { name: "breadthfirst", directed: true, spacingFactor: 1.0, grid: false },
  dagre: { name: "dagre", rankDir: "TB", nodeSep: 20, rankSep: 50 },
  klay: { name: "klay", klay: { direction: "DOWN" } },
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
    return `<span class="chip" data-focus="${escHtml(id)}">${escHtml(nm)}` +
           `<span class="cid">${escHtml(id)}</span></span>`;
  }).join("");
  return `<h4>${heading}</h4><div class="chips">${chips}</div>`;
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
  body.innerHTML = statsHtml + shapesHtml + neighborsHtml +
                   irHtml + moduleHtml + fusionHtml;
  body.querySelectorAll(".chip[data-focus]").forEach(el => {
    el.addEventListener("click", () => focusNode(el.dataset.focus));
  });
  if (fusionNodes) wireFusionTree(body, fusionNodes);
  openPanel("node-panel");
}

function renderStatsPanel(data) {
  const counts = {};
  const byName = {}; // { type: { name: count } }
  for (const n of data.elements.nodes) {
    const t = n.data.type || "?";
    counts[t] = (counts[t] || 0) + 1;
    if (t === "Kernel" || t === "BlackboxKernel") {
      const nm = n.data.name || n.data.id;
      byName[t] = byName[t] || {};
      byName[t][nm] = (byName[t][nm] || 0) + 1;
    }
  }
  const summaryRows = ["Kernel", "BlackboxKernel", "Const", "Memcpy", "Memset", "Input", "Output"]
    .map(t => `<span class="k">${t}</span><span>${counts[t] || 0}</span>`).join("");
  const totalEdges = data.elements.edges.length;
  const totalNodes = data.elements.nodes.length;
  const freqTable = t => {
    const rows = Object.entries(byName[t] || {})
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([nm, c]) => `<tr><td class="n">${c}</td><td>${escHtml(nm)}</td></tr>`)
      .join("");
    if (!rows) return `<h4>${t} — (none)</h4>`;
    const distinct = Object.keys(byName[t] || {}).length;
    return `<h4>${t} — ${counts[t] || 0} nodes, ${distinct} distinct</h4>` +
           `<table class="freq"><thead><tr><th class="n">n</th><th>name</th></tr></thead>` +
           `<tbody>${rows}</tbody></table>`;
  };
  document.getElementById("sp-body").innerHTML =
    `<div class="stats">` +
    `<span class="k">total nodes</span><span>${totalNodes}</span>` +
    `<span class="k">total edges</span><span>${totalEdges}</span>` +
    summaryRows +
    `</div>` +
    freqTable("Kernel") + freqTable("BlackboxKernel");
  openPanel("stats-panel");
}

fetch("graph.json")
  .then(r => { if (!r.ok) throw new Error(r.status + " " + r.statusText); return r.json(); })
  .then(data => {
    DATA = data;
    const hasPos = data.elements.nodes.length && data.elements.nodes[0].position;
    const layoutName = new URLSearchParams(location.search).get("layout")
      || (hasPos ? "elk" : "breadthfirst");
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
      layout: LAYOUTS[layoutName] || LAYOUTS.breadthfirst,
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
    // asynchronously (breadthfirst/dagre/klay). Run once now for the
    // sync case; layoutstop handles anything async plus later relayouts.
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
      ` — layout: ${layoutName} (?layout=${Object.keys(LAYOUTS).join("|")})` +
      ` — edges: <span style="color:#aaa">black=read</span>,` +
      ` <span style="color:#f66">red=modify</span> —${legend}`;
  })
  .catch(err => { document.getElementById("bar-text").textContent = "error: " + err; });
</script>
</body>
</html>
"""


def node_path_env(extra: Path | None) -> str:
    candidates = [extra] if extra else []
    candidates += [Path.cwd() / "node_modules", Path(__file__).parent / "node_modules"]
    paths = [str(c) for c in candidates if c and (c / "elkjs").is_dir()]
    if os.environ.get("NODE_PATH"):
        paths.append(os.environ["NODE_PATH"])
    return os.pathsep.join(paths)


def duplicate_high_fanout_leaves(doc: dict, threshold: int) -> int:
    """Splits Const / Input nodes with fanout > `threshold` into per-consumer
    shadow copies. Each shadow reuses the original node's data (name, ir,
    ...) but gets a unique id (`<orig>_c{n}`); edges from the original are
    rewritten to originate at the shadow. Layered layouts turn a
    high-fanout leaf into one long-edge bundle that spans the whole graph
    width — duplicating the leaf collapses that into many short local
    edges, dramatically reducing crossings and shrinking ELK's working
    set. Returns the number of shadow nodes injected (0 if nothing was
    eligible).
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


class LayoutCache:
    """ELK positions for a dump file, cached in memory and in a
    `<dump>.pos.json` sidecar. Keyed on `(dump_mtime, params_hash)` so
    layout knobs changing (direction, wrap, dup-leaves threshold, …)
    force a recompute."""

    def __init__(
        self,
        json_path: Path,
        node_modules: Path | None,
        node_heap_mb: int,
        direction: str,
        wrap: bool,
        quality: str,
        dup_leaves_threshold: int,
        max_elk_nodes: int,
        use_elk: bool,
    ):
        self.json_path = json_path
        self.sidecar = Path(str(json_path) + ".pos.json")
        self.node_heap_mb = node_heap_mb
        self.direction = direction
        self.wrap = wrap
        self.quality = quality
        self.dup_leaves_threshold = dup_leaves_threshold
        self.max_elk_nodes = max_elk_nodes
        self.use_elk = use_elk
        # `NODE_OPTIONS=--max-old-space-size=N` bumps V8's heap ceiling —
        # ELK's crossing-minimization matrices exceed the default 4 GB on
        # any graph with a few thousand nodes per layer.
        prior_opts = os.environ.get("NODE_OPTIONS", "")
        node_opts = f"--max-old-space-size={node_heap_mb}"
        if prior_opts:
            node_opts = f"{prior_opts} {node_opts}"
        self.env = {
            **os.environ,
            "NODE_PATH": node_path_env(node_modules),
            "NODE_OPTIONS": node_opts,
            "ELK_DIRECTION": direction,
            "ELK_WRAP": "1" if wrap else "0",
            "ELK_QUALITY": quality,
        }
        self.params_hash = json.dumps(
            {
                "direction": direction,
                "wrap": wrap,
                "quality": quality,
                "dup_leaves_threshold": dup_leaves_threshold,
            },
            sort_keys=True,
        )
        self.lock = threading.Lock()
        self.mtime = None
        self.body = b""  # serialized graph.json response
        self.available = use_elk and self._elk_available()
        if use_elk and not self.available:
            print(
                "note: node/elkjs not found - falling back to the in-browser "
                "breadthfirst layout.\nFor a crossing-minimized layout run "
                "`npm install elkjs` (or `npm install --prefix scripts elkjs`, "
                "or pass --node-modules).",
                file=sys.stderr,
            )

    def _elk_available(self) -> bool:
        try:
            return (
                subprocess.run(
                    ["node", "-e", "require('elkjs')"],
                    env=self.env,
                    capture_output=True,
                    timeout=30,
                ).returncode
                == 0
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _compute_positions(self, transformed_path: Path, n_nodes: int) -> dict | None:
        mtime = self.json_path.stat().st_mtime
        if self.sidecar.is_file():
            try:
                cached = json.loads(self.sidecar.read_text())
                if (
                    cached.get("mtime") == mtime
                    and cached.get("params_hash") == self.params_hash
                ):
                    return cached["positions"]
            except (json.JSONDecodeError, KeyError):
                pass
        if not self.available:
            return None
        if n_nodes > self.max_elk_nodes:
            print(
                f"note: {n_nodes} nodes exceeds --max-elk-nodes "
                f"({self.max_elk_nodes}); skipping ELK and falling back to "
                "the in-browser breadthfirst layout. Bump the threshold or "
                "pass --no-elk to silence.",
                file=sys.stderr,
            )
            return None
        print(f"computing ELK layout for {self.json_path} ...", flush=True)
        t0 = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(ELK_JS)
            script = f.name
        try:
            r = subprocess.run(
                ["node", script, str(transformed_path)],
                env=self.env,
                capture_output=True,
                timeout=1800,
            )
        finally:
            os.unlink(script)
        if r.returncode != 0:
            print(f"ELK layout failed: {r.stderr.decode().strip()}", file=sys.stderr)
            return None
        positions = json.loads(r.stdout)
        self.sidecar.write_text(
            json.dumps(
                {"mtime": mtime, "params_hash": self.params_hash, "positions": positions}
            )
        )
        print(f"ELK layout done in {time.time() - t0:.1f}s", flush=True)
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
            # ELK reads its input from a file — write the (possibly
            # leaf-duplicated) doc to a temp file so the JS side sees the
            # same graph the browser will render.
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False
            ) as f:
                json.dump({"elements": doc["elements"]}, f)
                transformed_path = Path(f.name)
            try:
                positions = self._compute_positions(
                    transformed_path, len(doc["elements"]["nodes"])
                )
            finally:
                os.unlink(transformed_path)
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
        "--node-modules",
        type=Path,
        default=None,
        help="node_modules directory containing elkjs",
    )
    ap.add_argument(
        "--node-heap-mb",
        type=int,
        default=16384,
        help="V8 heap ceiling for the ELK subprocess in MB (default 16384). "
        "ELK's crossing-minimization matrices blow past Node's default 4 GB "
        "on graphs with a few thousand nodes.",
    )
    ap.add_argument(
        "--direction",
        default="DOWN",
        choices=["DOWN", "RIGHT", "UP", "LEFT"],
        help="ELK layered flow direction (default DOWN).",
    )
    ap.add_argument(
        "--wrap-layers",
        action="store_true",
        help="Wrap over-wide layers into stacked sub-rows using ELK's "
        "MULTI_EDGE wrapping. Off by default: the wrap adds vertical "
        "sub-layers that can obscure the layer structure.",
    )
    ap.add_argument(
        "--elk-quality",
        default="fast",
        choices=["fast", "balanced", "nice"],
        help="ELK layout quality/speed trade-off:\n"
        "  `fast` (default): LONGEST_PATH layering + SIMPLE placement + no "
        "crossing minimization + POLYLINE edge routing. ~70s on a 2700-node "
        "graph. Best choice for anything above ~1000 nodes.\n"
        "  `balanced`: LONGEST_PATH layering + LINEAR_SEGMENTS placement + "
        "LAYER_SWEEP crossing min. LAYER_SWEEP is O(widest_layer²) per "
        "iteration and can take 15+ minutes on high-fanout graphs — use only "
        "for graphs of a few hundred nodes.\n"
        "  `nice`: NETWORK_SIMPLEX layering + BRANDES_KOEPF placement + "
        "LAYER_SWEEP crossing min + EDGE_LENGTH post-compaction + ORTHOGONAL "
        "edge routing. Prettiest, but elkjs has been observed to hit its JS "
        "stack limit on graphs above ~1500 nodes.",
    )
    ap.add_argument(
        "--dup-leaves-threshold",
        type=int,
        default=4,
        help="Duplicate any Const/Input node with more than this many "
        "consumers, injecting one shadow copy per consumer so ELK sees "
        "short local edges instead of a giant crossing bundle. Set to 0 "
        "to disable. Default 4.",
    )
    ap.add_argument(
        "--max-elk-nodes",
        type=int,
        default=20000,
        help="Skip ELK and fall back to the in-browser breadthfirst layout "
        "when the (possibly duplicated) graph exceeds this many nodes. "
        "LAYER_SWEEP crossing minimization is quadratic in the widest "
        "layer, so past ~20k nodes it will OOM even with a big heap.",
    )
    ap.add_argument(
        "--no-elk",
        action="store_true",
        help="Skip ELK unconditionally (useful for huge dumps where the "
        "in-browser breadthfirst layout is fine).",
    )
    args = ap.parse_args()
    if not args.json_path.is_file():
        sys.exit(f"error: {args.json_path} does not exist")

    cache = LayoutCache(
        args.json_path,
        args.node_modules,
        node_heap_mb=args.node_heap_mb,
        direction=args.direction,
        wrap=args.wrap_layers,
        quality=args.elk_quality,
        dup_leaves_threshold=args.dup_leaves_threshold,
        max_elk_nodes=args.max_elk_nodes,
        use_elk=not args.no_elk,
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
