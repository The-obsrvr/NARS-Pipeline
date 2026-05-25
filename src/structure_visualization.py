"""
BAS Visualizer — Bipolar Argument Structure graph visualization  (Step 4 output)

Reads a BAS JSONL file (from bas_assembler.py) and generates an interactive
HTML graph for each conversation using pyvis.

Node colours:
    Gold        — P_central (root / central proposition)
    Light blue  — regular argumentative unit
    Orange      — synthetic node (should not occur, but handled defensively)

Edge colours:
    Green       — support relation
    Red         — attack relation
    Dashed      — synthetic edge (repair mode)

Usage:
    # Visualise all conversations in a JSONL file → one HTML per conv_id:
    python bas_visualizer.py --input bas_repair.jsonl --output-dir ./vis

    # Visualise a single conversation by conv_id:
    python bas_visualizer.py --input bas_repair.jsonl --conv-id t3_abc123

    # Visualise both repair and no-repair side by side in one HTML:
    python bas_visualizer.py \\
        --input    bas_repair.jsonl \\
        --input-nr bas_no_repair.jsonl \\
        --output-dir ./vis

Requirements:
    pip install pyvis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("bas_visualizer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ─── Colour scheme ────────────────────────────────────────────────────────────

NODE_COLOR_ROOT      = "#FFD700"   # gold      — P_central
NODE_COLOR_DEFAULT   = "#ADD8E6"   # light blue — regular node
NODE_COLOR_SYNTHETIC = "#FFA500"   # orange    — defensive fallback

EDGE_COLOR_SUPPORT   = "#2ca02c"   # green
EDGE_COLOR_ATTACK    = "#d62728"   # red


# ─── Core graph builder ───────────────────────────────────────────────────────

def _strength_lines(node: dict, strategies: list[str]) -> str:
    """
    Build HTML lines for initial and final (acceptability) strength values
    for the given strategies. Only includes strategies present on the node.
    """
    lines = []
    for s in strategies:
        init  = node.get(f"initial_strength_{s}")
        final = node.get(f"acceptability_{s}")
        if init is not None or final is not None:
            parts = [f"<b>{s.upper()}</b>"]
            if init  is not None: parts.append(f"init={init:.4f}")
            if final is not None: parts.append(f"final={final:.4f}")
            lines.append("  ".join(parts))
    return ("<br>".join(lines)) if lines else ""


def build_pyvis_graph(bas: dict, title: str = "",
                      show_strengths: bool = False,
                      strength_strategy: str = "s1") -> "Network":
    """
    Build a pyvis Network from a single BAS dict.

    BAS dict shape (from bas_assembler.py / strength_initializer.py /
    gradual_semantics.py):
        conv_id  — str
        mode     — "repair" | "no_repair"
        nodes    — list of {id, text, is_root?, in_degree, out_degree,
                            post_id, speaker_id, global_idx,
                            initial_strength_s*, acceptability_s*}
        edges    — list of {source, target, relation, synthetic}
        summary  — {root_id, ...}

    Args:
        show_strengths:     If True, display initial/final strength values in
                            tooltips and scale node size by final strength.
        strength_strategy:  Which strategy key to use for node size scaling
                            (e.g. "s1", "s2", "s3", "s4"). Only used when
                            show_strengths=True.
    """
    from pyvis.network import Network

    conv_id = bas.get("conv_id", "unknown")
    mode    = bas.get("mode",    "unknown")
    root_id = bas.get("summary", {}).get("root_id")

    # Detect which strategies are present across all nodes
    all_strategies = []
    for s in ["s1", "s2", "s3", "s4"]:
        if any(f"initial_strength_{s}" in n or f"acceptability_{s}" in n
               for n in bas.get("nodes", [])):
            all_strategies.append(s)

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        heading=title or f"{conv_id}  [{mode}]",
    )

    net.barnes_hut(
        gravity=-20000,
        central_gravity=0.3,
        spring_length=110,
        spring_strength=0.02,
        damping=0.08,
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for node in bas.get("nodes", []):
        nid      = node["id"]
        is_root  = node.get("is_root", False) or (nid == root_id)
        text     = node.get("text", "")
        in_deg   = node.get("in_degree",  0)
        out_deg  = node.get("out_degree", 0)
        speaker  = node.get("speaker_id", "")
        post_id  = node.get("post_id",    "")

        color = NODE_COLOR_ROOT if is_root else NODE_COLOR_DEFAULT

        # Base tooltip
        tooltip = (
            f"<b>{nid}</b>{'  ★ P_central' if is_root else ''}<br>"
            f"<i>{text[:300]}{'…' if len(text) > 300 else ''}</i><br><br>"
            f"in_degree={in_deg}  out_degree={out_deg}<br>"
            f"speaker={speaker}  post={post_id}"
        )

        # Optional strength section in tooltip
        if show_strengths and all_strategies:
            strength_html = _strength_lines(node, all_strategies)
            if strength_html:
                tooltip += f"<br><br><b>Strengths</b><br>{strength_html}"

        # Node size: scale by final acceptability score of chosen strategy
        # when show_strengths is enabled, otherwise fixed size
        base_size = 30 if is_root else 20
        if show_strengths:
            final_val = node.get(f"acceptability_{strength_strategy}",
                        node.get(f"initial_strength_{strength_strategy}"))
            if final_val is not None:
                # Scale: min 15, max 50, proportional to strength value in [0,1]
                size = 15 + int(final_val * 35)
            else:
                size = base_size
        else:
            size = base_size

        net.add_node(
            nid,
            label=nid,
            title=tooltip,
            color=color,
            shape="circle",
            font={"size": 14},
            size=size,
            borderWidth=3 if is_root else 1,
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    for edge in bas.get("edges", []):
        src       = edge["source"]
        tgt       = edge["target"]
        relation  = edge.get("relation",  "support")
        synthetic = edge.get("synthetic", False)

        color  = EDGE_COLOR_SUPPORT if relation == "support" else EDGE_COLOR_ATTACK
        dashes = synthetic   # dashed line for synthetic repair edges
        label  = ("~" if synthetic else "") + relation   # ~ prefix for synthetic

        net.add_edge(
            src, tgt,
            color=color,
            arrows="to",
            width=1 if synthetic else 2,
            dashes=dashes,
            label=label,
            font={"size": 10, "color": color},
            title=f"{'[synthetic] ' if synthetic else ''}{relation}",
        )

    return net


# ─── HTML generation ──────────────────────────────────────────────────────────

def graph_to_html(bas: dict, title: str = "",
                  show_strengths: bool = False,
                  strength_strategy: str = "s1") -> str:
    net = build_pyvis_graph(bas, title=title,
                            show_strengths=show_strengths,
                            strength_strategy=strength_strategy)
    return net.generate_html()


def side_by_side_html(repair_bas: dict, no_repair_bas: dict, conv_id: str,
                      show_strengths: bool = False,
                      strength_strategy: str = "s1") -> str:
    """
    Generate a single HTML page with repair and no-repair graphs side by side.
    """
    repair_html    = graph_to_html(repair_bas,    title=f"{conv_id} [repair]",
                                   show_strengths=show_strengths,
                                   strength_strategy=strength_strategy)
    no_repair_html = graph_to_html(no_repair_bas, title=f"{conv_id} [no_repair]",
                                   show_strengths=show_strengths,
                                   strength_strategy=strength_strategy)

    # Strip outer HTML boilerplate from the second graph — embed as iframes
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{conv_id} — BAS comparison</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; }}
    h2   {{ text-align: center; padding: 8px; background: #f0f0f0; margin: 0; }}
    .row {{ display: flex; height: calc(100vh - 40px); }}
    .col {{ flex: 1; border: 1px solid #ccc; overflow: hidden; }}
    .col h3 {{ margin: 0; padding: 6px 12px;
               background: #e8e8e8; font-size: 14px; }}
    iframe {{ width: 100%; height: calc(100% - 32px); border: none; }}
  </style>
</head>
<body>
  <h2>{conv_id}</h2>
  <div class="row">
    <div class="col">
      <h3>Repair mode</h3>
      <iframe srcdoc="{_escape(repair_html)}"></iframe>
    </div>
    <div class="col">
      <h3>No-repair mode</h3>
      <iframe srcdoc="{_escape(no_repair_html)}"></iframe>
    </div>
  </div>
</body>
</html>"""


def _escape(html: str) -> str:
    """Escape HTML string for use in srcdoc attribute."""
    return html.replace("&", "&amp;").replace('"', "&quot;")


# ─── JSONL loading helpers ────────────────────────────────────────────────────

def load_jsonl(path: Path) -> dict[str, dict]:
    """Load a BAS JSONL file → {conv_id: bas_dict}."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bas = json.loads(line)
            result[bas["conv_id"]] = bas
    log.info("Loaded %d conversations from %s", len(result), path)
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise BAS graphs from bas_assembler.py output"
    )
    parser.add_argument("--input",       "-i", required=True,
                        help="Path to BAS JSONL file (repair or no-repair)")
    parser.add_argument("--input-nr",    "-n", default=None,
                        help="Path to no-repair BAS JSONL — enables side-by-side view")
    parser.add_argument("--output-dir",  "-o", default="./bas_vis",
                        help="Directory to write HTML files (default: ./bas_vis)")
    parser.add_argument("--conv-id",           "-c", default=None,
                        help="Visualise a single conversation by conv_id")
    parser.add_argument("--show-strengths",    action="store_true",
                        help="Show initial and final strength values in tooltips "
                             "and scale node size by strength")
    parser.add_argument("--strength-strategy", default="s1",
                        choices=["s1", "s2", "s3", "s4"],
                        help="Strategy to use for node size scaling (default: s1)")
    parser.add_argument("--verbose",           "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary   = load_jsonl(Path(args.input))
    secondary = load_jsonl(Path(args.input_nr)) if args.input_nr else {}

    conv_ids = ([args.conv_id] if args.conv_id else list(primary.keys()))

    for conv_id in conv_ids:
        if conv_id not in primary:
            log.warning("conv_id %s not found in %s — skipping", conv_id, args.input)
            continue

        bas = primary[conv_id]

        if secondary and conv_id in secondary:
            html     = side_by_side_html(bas, secondary[conv_id], conv_id,
                                         show_strengths=args.show_strengths,
                                         strength_strategy=args.strength_strategy)
            out_path = output_dir / f"{conv_id}_comparison.html"
        else:
            html     = graph_to_html(bas,
                                     show_strengths=args.show_strengths,
                                     strength_strategy=args.strength_strategy)
            mode     = bas.get("mode", "bas")
            out_path = output_dir / f"{conv_id}_{mode}.html"

        out_path.write_text(html, encoding="utf-8")
        log.info("Saved → %s", out_path)

    log.info("Done. %d file(s) written to %s", len(conv_ids), output_dir)


if __name__ == "__main__":
    main()