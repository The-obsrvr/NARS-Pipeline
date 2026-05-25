"""
BAS Assembler — Bipolar Argument Structure assembly (Step 4)
Builds on llm_reasoner.py output.

Constructs G(A, E, P_central) where:
  A          = set of argumentative EDUs (nodes)
  E          = set of support / attack relations (edges)
  P_central  = central proposition — the opening-post EDU with the highest
               in-degree centrality that is also semantically closest to the
               known discussion topic

Two output modes are always produced in a single pass:
  repair    — disconnected components are stitched to P_central's component
              via synthetic sentiment-based edges, yielding a single unified
              graph that preserves all argument activity
  no-repair — only the connected subgraph containing P_central is retained;
              isolated components are discarded

Discussions with ≤ 3 argumentative units are removed from both outputs.

Usage:
    python bas_assembler.py --input reasoning_input.jsonl \\
                            --output-repair    bas_repair.jsonl \\
                            --output-no-repair bas_no_repair.jsonl
    python bas_assembler.py --export-dot bas.dot   # GraphViz DOT export
    python bas_assembler.py                        # built-in sample
"""

import logging
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

from sys_utils import (read_jsonl, log_progress, JSONLWriter, count_lines, SAMPLE_CONVERSATIONS, setup_logging)


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bas_assembler.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("bas_assembler")

# ─── Constants ────────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"   # only needed when connectivity repair runs
_ENCODER    = None

def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading encoder for connectivity repair")
        _ENCODER = SentenceTransformer(EMBED_MODEL)
    return _ENCODER

# 1 · Extraction helpers
def extract_nodes_and_edges(
    reasoning_output: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Walk the reasoning output and collect:
      nodes  — dict keyed by node-id ("edu_<global_idx>") for every
               argumentative EDU
      edges  — list of {source, target, relation, synthetic=False} dicts
               for every support/attack PAC pair where BOTH endpoints are
               argumentative

    Non-argumentative EDUs and neutral relations are silently dropped.
    """
    nodes: dict[str, dict] = {}
    raw_edges: list[dict]  = []

    for turn in reasoning_output.get("conversation", []):
        if turn.get("deleted"):
            # skip deleted posts
            continue
        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                continue

            # ── Drop non-argumentative EDUs ───────────────────────────────────
            if not edu.get("target_is_argumentative"):
                log.debug("  Dropping non-arg EDU [%s]: %.55s",
                          edu.get("global_idx"), edu.get("text", ""))
                continue

            g_idx   = edu["global_idx"]
            node_id = f"edu_{g_idx}"

            nodes[node_id] = {
                "id":          node_id,
                "global_idx":  g_idx,
                "post_id":     turn.get("post_id", ""),
                "speaker_id":  turn.get("speaker_id", ""),
                "text":        edu.get("text", ""),
                # degree counts filled later
                "in_degree":   0,
                "out_degree":  0,
            }

            # ── Collect support/attack edges ───────────────────────────────────
            for pac in edu.get("pacs", []):
                rel = pac.get("relation", "neutral")
                if rel not in ("support", "attack"):
                    continue

                src_g_idx = pac.get("source_global_idx")
                if src_g_idx is None:
                    continue

                # Directionality: source (PAC) → target (this EDU)
                raw_edges.append({
                    "source":    f"edu_{src_g_idx}",
                    "target":    node_id,
                    "relation":  rel,
                    "synthetic": False,
                })

    log.info("Extracted %d argumentative nodes, %d raw edges (before pruning)",
             len(nodes), len(raw_edges))
    return nodes, raw_edges


def promote_source_nodes(
    nodes:     dict[str, dict],
    raw_edges: list[dict],
    reasoning_output: dict,
) -> tuple[dict[str, dict], list[dict]]:
    """
    Instead of pruning edges whose source is not argumentative, promote
    those source EDUs into the node set — an EDU that is the source of a
    support/attack relation is argumentative by virtue of being engaged with,
    regardless of the LLM's direct classification of that EDU.

    Also deduplicates (source, target) pairs keeping the stronger relation
    (attack > support when both exist for the same pair).
    """
    RELATION_RANK = {"attack": 1, "support": 0}

    # Build a lookup of all EDUs in the conversation for promotion
    all_edus: dict[str, dict] = {}
    for turn in reasoning_output.get("conversation", []):
        if turn.get("deleted"):
            continue
        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                continue
            g_idx   = edu.get("global_idx")
            node_id = f"edu_{g_idx}"
            all_edus[node_id] = {
                "id":         node_id,
                "global_idx": g_idx,
                "post_id":    turn.get("post_id", ""),
                "speaker_id": turn.get("speaker_id", ""),
                "text":       edu.get("text", ""),
                "in_degree":  0,
                "out_degree": 0,
            }

    promoted = 0
    best: dict[tuple, dict] = {}

    for edge in raw_edges:
        src, tgt = edge["source"], edge["target"]

        # Promote source node if not already argumentative
        if src not in nodes:
            if src in all_edus:
                nodes[src] = all_edus[src]
                promoted += 1
                log.debug("Promoted source EDU %s to argumentative node", src)
            else:
                # Source EDU not found in conversation — skip edge
                continue

        key = (src, tgt)
        if key not in best or (
            RELATION_RANK.get(edge["relation"], 0) >
            RELATION_RANK.get(best[key]["relation"], 0)
        ):
            best[key] = edge

    edges = list(best.values())
    log.info("Promoted %d source EDUs to argumentative nodes → "
             "%d total nodes  %d edges", promoted, len(nodes), len(edges))
    return nodes, edges


# ═══════════════════════════════════════════════════════════════════════════════
# 2 · Degree annotation
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_degrees(
    nodes: dict[str, dict],
    edges: list[dict],
) -> None:
    """Fill in_degree / out_degree on each node in-place."""
    for node in nodes.values():
        node["in_degree"]  = 0
        node["out_degree"] = 0

    for edge in edges:
        if edge["source"] in nodes:
            nodes[edge["source"]]["out_degree"] += 1
        if edge["target"] in nodes:
            nodes[edge["target"]]["in_degree"]  += 1


# 3 · Connectivity repair

def _embed_nodes(node_ids: list[str], texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings for a list of texts."""
    encoder = get_encoder()
    return encoder.encode(texts, convert_to_numpy=True,
                          normalize_embeddings=True
                          ).astype(np.float32)


def repair_connectivity(
    nodes: dict[str, dict],
    edges: list[dict],
) -> list[dict]:
    """
    Guarantee that the argument graph is weakly connected.

    For each disconnected component:
      1. Find the best connection to the main component — the pair
         (minor_nid, main_nid) with the highest cosine similarity.
      2. Determine relation from polarity between the two nodes:
         sim ≥ 0 → SUPPORT, sim < 0 → ATTACK.
      3. Connect with conversational directionality (earlier → later).
    """
    if not nodes:
        return edges

    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    for e in edges:
        G.add_edge(e["source"], e["target"], relation=e["relation"])
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    if len(comps) == 1:
        log.info("Graph is already connected — no repair needed")
        return edges

    log.info("Graph has %d weakly-connected components — repairing", len(comps))

    node_id_list = list(nodes.keys())
    texts        = [nodes[nid]["text"] for nid in node_id_list]
    embs         = _embed_nodes(node_id_list, texts)
    id_to_emb    = {nid: embs[i] for i, nid in enumerate(node_id_list)}
    synthetic_edges: list[dict] = []
    main_component:  set[str]   = set(comps[0])

    for minor_comp in comps[1:]:
        # ── Step 1: find the best connection across the component boundary ────
        best_score = -1.0
        best_src   = None
        best_tgt   = None
        for minor_nid in minor_comp:
            for main_nid in main_component:
                score = float(id_to_emb[minor_nid] @ id_to_emb[main_nid])
                if score > best_score:
                    best_score = score
                    best_src   = minor_nid
                    best_tgt   = main_nid

        if best_src and best_tgt:
            # Relation determined by polarity between the two connected nodes:
            # positive similarity → support, negative → attack
            relation = "support" if best_score >= 0 else "attack"

            # ── Step 3: enforce conversational directionality ─────────────────
            if nodes[best_src]["global_idx"] > nodes[best_tgt]["global_idx"]:
                best_src, best_tgt = best_tgt, best_src

            syn_edge = {
                "source":    best_src,
                "target":    best_tgt,
                "relation":  relation,
                "synthetic": True,
            }
            synthetic_edges.append(syn_edge)
            main_component |= minor_comp
            log.info(
                "  Synthetic edge: %s → %s  relation=%s  sim=%.3f",
                best_src, best_tgt, relation, best_score,
            )

    if synthetic_edges:
        log.info("Added %d synthetic edge(s) to repair connectivity", len(synthetic_edges))

    return edges + synthetic_edges



# ═══════════════════════════════════════════════════════════════════════════════
# 4 · P_central identification via LLM
# ═══════════════════════════════════════════════════════════════════════════════

IDENTIFY_ROOT_SYSTEM_PROMPT = """You are an expert in argumentation analysis.

Identify the single EDU that is the central proposition (main claim/opinion) of the discussion.
Return ONLY valid JSON:
{"central_proposition_idx": <int>, "reasoning": "<one line>"}

central_proposition_idx is the 0-based index from the EDU list provided."""


def _call_ollama_for_root(
    prompt:     str,
    ollama_url: str,
    model:      str,
    timeout:    int = 120,
) -> Optional[dict]:
    """Minimal Ollama call for P_central identification. No thinking."""
    import requests, json as _json
    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": IDENTIFY_ROOT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "format":  "json",
        "think":   "low",
        "options": {
            "temperature": 0.0, #deterministic behavior
            "num_predict": 128,
            "num_ctx":     2048,
        },
    }
    try:
        resp    = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return _json.loads(content)
    except Exception as exc:
        log.warning("LLM root identification failed: %s", exc)
        return None


def identify_root_llm(
    nodes:             dict[str, dict],
    edges:             list[dict],
    title:             str,
    opening_post_text: str,
    ollama_url:        str,
    model:             str,
) -> Optional[str]:
    """
    Use an LLM to identify P_central from the opening-post EDUs.

    Sends: CMV title + numbered opening-post EDUs.
    Expects: the index of the single main claim EDU.
    Falls back to heuristic identify_root() on any failure.
    """
    from collections import defaultdict

    if not nodes:
        return None

    # Collect opening-post EDUs (lowest post_id group by global_idx)
    post_groups: dict[str, list[str]] = defaultdict(list)
    for nid, n in nodes.items():
        post_groups[n["post_id"]].append(nid)

    opening_post_id = min(
        post_groups.keys(),
        key=lambda pid: min(nodes[nid]["global_idx"] for nid in post_groups[pid]),
    )
    op_nids = sorted(post_groups[opening_post_id],
                     key=lambda nid: nodes[nid]["global_idx"])

    if not op_nids:
        log.warning("No opening-post EDUs — falling back to heuristic")
        return identify_root(nodes, edges)

    # Build minimal prompt
    edu_lines = "\n".join(
        f"[{i}] {nodes[nid]['text']}" for i, nid in enumerate(op_nids)
    )
    prompt = (
        f"TOPIC: {title}\n\n"
        f"OPENING POST EDUs:\n{edu_lines}\n\n"
        f"Which EDU is the central proposition (main claim)?"
    )

    result = _call_ollama_for_root(prompt, ollama_url, model)

    if result is None:
        log.warning("LLM root call failed — falling back to heuristic")
        return identify_root(nodes, edges)

    selected = result.get("central_proposition_idx")
    log.info(
        "LLM P_central: idx=%s  reasoning=%s",
        selected, result.get("reasoning", ""),
    )

    if selected is not None and 0 <= int(selected) < len(op_nids):
        root_id = op_nids[int(selected)]
        log.info("P_central → %s  text=%.80s", root_id, nodes[root_id]["text"])
        return root_id

    log.warning("LLM returned invalid idx=%s — falling back to heuristic", selected)
    return identify_root(nodes, edges)


def identify_root(
    nodes:        dict[str, dict],
    edges:        list[dict],
    topic_text:   Optional[str] = None,
) -> Optional[str]:
    """
    Identify P_central — the central proposition of the discussion.

    Per the BAS definition, P_central is the EDU originating from the opening
    post (lowest post_id / earliest global_idx group) that simultaneously:
      1. Exhibits one of the highest in-degree centrality scores among all nodes
      2. Is semantically closest to the known discussion topic

    Selection procedure:
      a. Restrict candidates to EDUs from the opening post (minimum global_idx
         post among all nodes).
      b. Among those candidates, rank by in-degree (descending).
      c. Take the top-k (k=3) by in-degree; break ties by semantic similarity
         to topic_text when available, otherwise by lowest global_idx.
      d. If the opening post has no argumentative EDUs, fall back to the
         globally highest in-degree node.

    If the graph has no edges at all, the first argumentative node
    (lowest global_idx) is designated root.
    """
    if not nodes:
        return None

    if not edges:
        root_id = min(nodes.keys(), key=lambda nid: nodes[nid]["global_idx"])
        log.info("No edges — designating earliest node as root: %s", root_id)
        return root_id

    annotate_degrees(nodes, edges)

    # ── Identify opening-post candidates ──────────────────────────────────────
    # Group nodes by post_id; the opening post is the group whose earliest
    # global_idx is the smallest overall.
    from collections import defaultdict
    post_groups: dict[str, list[str]] = defaultdict(list)
    for nid, n in nodes.items():
        post_groups[n["post_id"]].append(nid)

    opening_post_id = min(
        post_groups.keys(),
        key=lambda pid: min(nodes[nid]["global_idx"] for nid in post_groups[pid])
    )
    candidates = post_groups[opening_post_id]

    if not candidates:
        # Fallback: use all nodes
        candidates = list(nodes.keys())
        log.info("No opening-post EDUs found — using all nodes as candidates")

    # ── Rank candidates by in-degree, take top-k ──────────────────────────────
    TOP_K = 3
    candidates_sorted = sorted(
        candidates,
        key=lambda nid: nodes[nid]["in_degree"],
        reverse=True,
    )
    top_candidates = candidates_sorted[:TOP_K]

    # ── Break ties with semantic similarity to topic ───────────────────────────
    if topic_text and len(top_candidates) > 1:
        try:
            encoder = get_encoder()
            topic_emb = encoder.encode([topic_text], convert_to_numpy=True,
                                       normalize_embeddings=True)[0]
            cand_texts = [nodes[nid]["text"] for nid in top_candidates]
            cand_embs  = encoder.encode(cand_texts, convert_to_numpy=True,
                                        normalize_embeddings=True)
            sims = cand_embs @ topic_emb
            root_id = top_candidates[int(np.argmax(sims))]
            log.info(
                "P_central selected by in-degree + topic similarity: %s  "
                "in_degree=%d  sim=%.3f  text=%.60s",
                root_id, nodes[root_id]["in_degree"],
                float(np.max(sims)), nodes[root_id]["text"],
            )
            return root_id
        except Exception as exc:
            log.warning("Semantic similarity for root selection failed (%s) — "
                        "falling back to in-degree + earliest idx", exc)

    # ── Fallback: highest in-degree, then earliest global_idx ─────────────────
    root_id = max(
        top_candidates,
        key=lambda nid: (nodes[nid]["in_degree"], -nodes[nid]["global_idx"]),
    )
    log.info(
        "P_central identified: %s  in_degree=%d  text=%.60s",
        root_id, nodes[root_id]["in_degree"], nodes[root_id]["text"],
    )
    return root_id


# ═══════════════════════════════════════════════════════════════════════════════
# 5 · BAS summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(
    nodes:   dict[str, dict],
    edges:   list[dict],
    root_id: Optional[str],
) -> dict:
    support_edges   = [e for e in edges if e["relation"] == "support"]
    attack_edges    = [e for e in edges if e["relation"] == "attack"]
    synthetic_edges = [e for e in edges if e.get("synthetic")]

    # Speaker breakdown
    speaker_counts = Counter(n["speaker_id"] for n in nodes.values())

    return {
        "total_nodes":       len(nodes),
        "total_edges":       len(edges),
        "support_edges":     len(support_edges),
        "attack_edges":      len(attack_edges),
        "synthetic_edges":   len(synthetic_edges),
        "root_id":           root_id,
        "root_text":         nodes[root_id]["text"] if root_id else None,
        "speaker_breakdown": dict(speaker_counts),
    }


def extract_central_subgraph(
    nodes:   dict[str, dict],
    edges:   list[dict],
    root_id: str,
) -> tuple[dict[str, dict], list[dict]]:
    """
    No-repair mode: retain only the weakly-connected component that contains
    P_central (root_id). All other components are discarded.
    Returns filtered (nodes, edges).
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    for e in edges:
        G.add_edge(e["source"], e["target"])

    # Find the component containing root_id
    for comp in nx.weakly_connected_components(G):
        if root_id in comp:
            central_comp = comp
            break
    else:
        log.warning("root_id %s not found in any component — returning full graph", root_id)
        return nodes, edges

    discarded = len(nodes) - len(central_comp)
    if discarded:
        log.info(
            "No-repair: retaining central component (%d nodes), "
            "discarding %d node(s) in isolated components",
            len(central_comp), discarded,
        )

    filtered_nodes = {nid: nodes[nid] for nid in central_comp}
    filtered_edges = [e for e in edges
                      if e["source"] in central_comp and e["target"] in central_comp]
    return filtered_nodes, filtered_edges


MIN_ARGUMENTATIVE_UNITS = 3

def is_too_small(nodes: dict[str, dict]) -> bool:
    """Return True if the BAS has insufficient argument activity (≤ 3 units)."""
    return len(nodes) <= MIN_ARGUMENTATIVE_UNITS


# ═══════════════════════════════════════════════════════════════════════════════
# 6 · DOT / GraphViz export
# ═══════════════════════════════════════════════════════════════════════════════

def export_dot(bas: dict, path: Path) -> None:
    """
    Write a GraphViz DOT file for visualisation.
    Nodes are labelled with a short text snippet.
    Support edges = green solid; attack edges = red dashed.
    Root node is double-circled.
    """
    root_id = bas.get("summary", {}).get("root_id")
    lines   = ["digraph BAS {", '  rankdir=LR;', '  node [shape=box fontsize=10];']

    for node in bas["nodes"]:
        nid   = node["id"]
        label = node["text"][:40].replace('"', "'")
        shape = 'doublecircle' if nid == root_id else 'box'
        lines.append(f'  "{nid}" [label="{label}…" shape={shape}];')

    for edge in bas["edges"]:
        src   = edge["source"]
        tgt   = edge["target"]
        rel   = edge["relation"]
        synth = edge.get("synthetic", False)
        color = "green4" if rel == "support" else "red3"
        style = "dashed" if synth else "solid"
        lines.append(
            f'  "{src}" -> "{tgt}" [label="{rel}" color={color} style={style}];'
        )

    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("DOT file written to %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7 · Main assembly pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_bas(
    reasoning_output: dict,
    ollama_url:  Optional[str] = None,
    model:       str = "gpt-oss:20b",
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Full BAS assembly pipeline — produces both modes in a single pass.

    Args:
        reasoning_output: Output dict from llm_reasoner.py (contains thread_id,
                          title, conversation, and per-EDU reasoning results).
        ollama_url:       If provided, use LLM to identify P_central.
                          Falls back to heuristic if None or on failure.
        model:            Ollama model to use for P_central identification.

    Returns (repair_bas, no_repair_bas). Both are None if the graph is too
    small or has no argumentative nodes.
    """
    conv_id    = reasoning_output.get("conv_id", "unknown")
    thread_id  = reasoning_output.get("thread_id", "")
    is_delta   = reasoning_output.get("is_delta")
    title      = reasoning_output.get("title", "")
    topic_text = reasoning_output.get("topic")
    log.info("─── BAS Assembly thread_id=%s ──", thread_id)

    # ── Step 1: extract argumentative nodes and non-neutral edges ─────────────
    nodes, raw_edges = extract_nodes_and_edges(reasoning_output)

    if not nodes:
        log.warning("thread_id=%s — no argumentative nodes found, skipping", thread_id)
        return None, None

    # ── Step 2: promote source EDUs and deduplicate edges ────────────────────
    nodes, edges = promote_source_nodes(nodes, raw_edges, reasoning_output)

    # ── Step 3: first-pass degree annotation ──────────────────────────────────
    annotate_degrees(nodes, edges)

    # ── Step 4: identify P_central ────────────────────────────────────────────
    # Extract opening post text for LLM prompt
    opening_post_text = ""
    conversation = reasoning_output.get("conversation", [])
    if conversation:
        opening_post_text = conversation[0].get("text", "")

    if ollama_url and title:
        root_id = identify_root_llm(
            nodes, edges,
            title=title,
            opening_post_text=opening_post_text,
            ollama_url=ollama_url,
            model=model,
        )
    else:
        root_id = identify_root(nodes, edges, topic_text=topic_text)

    if root_id and root_id in nodes:
        nodes[root_id]["is_root"] = True

    # ══ REPAIR mode ═══════════════════════════════════════════════════════════
    repair_edges = repair_connectivity(nodes, edges)
    annotate_degrees(nodes, repair_edges)
    repair_summary = compute_summary(nodes, repair_edges, root_id)

    if is_too_small(nodes):
        log.info("thread_id=%s — only %d argumentative units (≤ %d), discarding",
                 thread_id, len(nodes), MIN_ARGUMENTATIVE_UNITS)
        return None, None   # too small for both modes

    repair_bas = {
        "conv_id":   conv_id,
        "thread_id": thread_id,
        "is_delta":  is_delta,
        "mode":      "repair",
        "nodes":     list(nodes.values()),
        "edges":     repair_edges,
        "summary":   repair_summary,
    }
    log.info(
        "repair BAS — %d nodes  %d edges  (support=%d  attack=%d  synthetic=%d)  root=%s",
        repair_summary["total_nodes"], repair_summary["total_edges"],
        repair_summary["support_edges"], repair_summary["attack_edges"],
        repair_summary["synthetic_edges"], root_id,
    )

    # ══ NO-REPAIR mode ════════════════════════════════════════════════════════
    # Use original (non-synthetic) edges and extract only P_central's component
    nr_nodes, nr_edges = extract_central_subgraph(
        {nid: dict(n) for nid, n in nodes.items()},  # shallow copy to avoid mutation
        [e for e in edges if not e.get("synthetic")],
        root_id,
    )
    annotate_degrees(nr_nodes, nr_edges)
    nr_summary = compute_summary(nr_nodes, nr_edges, root_id)

    if is_too_small(nr_nodes):
        log.info(
            "thread_id=%s — no-repair central component has only %d units (≤ %d), "
            "discarding both modes to keep repair and no-repair counts equal",
            thread_id, len(nr_nodes), MIN_ARGUMENTATIVE_UNITS,
        )
        return None, None
    no_repair_bas = {
        "conv_id":   conv_id,
        "thread_id": thread_id,
        "is_delta":  is_delta,
        "mode":      "no_repair",
        "nodes":     list(nr_nodes.values()),
        "edges":     nr_edges,
        "summary":   nr_summary,
    }
    log.info(
        "no-repair BAS — %d nodes  %d edges  (support=%d  attack=%d)  root=%s",
        nr_summary["total_nodes"], nr_summary["total_edges"],
        nr_summary["support_edges"], nr_summary["attack_edges"], root_id,
    )

    return repair_bas, no_repair_bas


def run_on_jsonl(
    input_path:        Path,
    output_repair:     Path,
    output_no_repair:  Path,
    ollama_url:        Optional[str] = None,
    model:             str = "gpt-oss:20b",
) -> None:
    total = count_lines(input_path)
    log.info("BAS assembly starting — %d conversations  llm_root=%s",
             total, ollama_url or "heuristic")

    skipped_small   = 0
    written_repair  = 0
    written_no_rep  = 0

    with JSONLWriter(output_repair) as wr_repair, \
         JSONLWriter(output_no_repair) as wr_no_rep:
        for line_no, conv in read_jsonl(input_path):
            log_progress(line_no, total, conv.get("thread_id", ""), "BAS", log)
            repair_bas, no_repair_bas = assemble_bas(conv,
                                                     ollama_url=ollama_url,
                                                     model=model)

            if repair_bas is None:
                skipped_small += 1
                continue

            wr_repair.write(repair_bas)
            written_repair += 1

            if no_repair_bas is not None:
                wr_no_rep.write(no_repair_bas)
                written_no_rep += 1

    log.info(
        "Finished.  repair→%s (%d written)  no_repair→%s (%d written)  "
        "discarded (too small)=%d",
        output_repair, written_repair,
        output_no_repair, written_no_rep,
        skipped_small,
    )


# 9 · CLI
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble Bipolar Argument Structures (repair + no-repair) "
                    "from LLM reasoner output"
    )
    parser.add_argument("--input",            "-i", help="Path to llm_reasoner JSONL output")
    parser.add_argument("--output-repair",    "-r", help="Path to save repair-mode BAS JSONL",
                        default="bas_repair.jsonl")
    parser.add_argument("--output-no-repair", "-n", help="Path to save no-repair-mode BAS JSONL",
                        default="bas_no_repair.jsonl")
    parser.add_argument("--export-dot",       "-d", help="Path to export GraphViz DOT file")
    parser.add_argument("--ollama-url",       default=None,
                        help="Ollama API URL for LLM-based P_central identification "
                             "(e.g. http://127.0.0.1:11434/api/chat). "
                             "If omitted, falls back to heuristic.")
    parser.add_argument("--model",            default="gpt-oss:20b",
                        help="Ollama model for P_central identification (default: gpt-oss:20b)")
    parser.add_argument("--verbose",          "-v", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--log-file", "-l", default="bas_assembler.log",
                        help="Log file path (default: bas_assembler.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    if args.input:
        run_on_jsonl(
            Path(args.input),
            Path(args.output_repair),
            Path(args.output_no_repair),
            ollama_url=args.ollama_url,
            model=args.model,
        )
    else:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee
        import pac_selector as ps
        import llm_reasoner as lr
        with JSONLWriter(Path(args.output_repair)) as wr_repair, \
             JSONLWriter(Path(args.output_no_repair)) as wr_no_rep:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id", ""), "BAS", log)
                repair_bas, no_repair_bas = assemble_bas(
                    lr.run_reasoning(ps.select_all_pacs(ee.extract_edus(conv))),
                    ollama_url=args.ollama_url,
                    model=args.model,
                )
                if repair_bas:
                    wr_repair.write(repair_bas)
                if no_repair_bas:
                    wr_no_rep.write(no_repair_bas)

if __name__ == "__main__":
    main()