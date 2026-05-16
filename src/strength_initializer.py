"""
Strength Initializer — A priori strength initialization  (Step 5)
Reads bas_repair.jsonl and/or bas_no_repair.jsonl (from bas_assembler.py);
writes initialized_bas_repair.jsonl and initialized_bas_no_repair.jsonl.

  UI (s1) — Uniform Initialization: all nodes assigned strength 1.0
  GI (s2) — Graph-based Initialization: PageRank centrality
  TI (s3) — Topic-based Initialization: argument quality model fine-tuned
             on Webis Args.me corpus (falls back to topic cosine similarity
             when model is not yet trained)
  HI (s4) — Hybrid Initialization: distance-weighted combination of TI and GI
             α(d) = d / d_max  →  strength = α·TI + (1-α)·GI
             Peripheral nodes (large d) weighted toward TI;
             central nodes (small d) weighted toward GI.
             Distance measured on undirected graph from P_central (root).

Usage:
    python src/strength_initializer.py \
    --input-repair      bas_repair.jsonl \
    --input-no-repair   bas_no_repair.jsonl \
    --output-repair     initialized_repair.jsonl \
    --output-no-repair  initialized_no_repair.jsonl \
    --strategy          all \
    --quality-model     ./quality_model

"""
import logging
import argparse
import math
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import networkx as nx

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines,
    SAMPLE_CONVERSATIONS, setup_logging
)

log = logging.getLogger("strength_initializer")

STRATEGY_MAP = {"ui": "s1", "gi": "s2", "ti": "s3", "hi": "s4"}
ALL_STRATEGIES = ["ui", "gi", "ti", "hi"]


# def strategy_centrality(bas, measure="harmonic"):
#     log.info("[S2] Centrality  measure=%s", measure)
#     G = bas_to_digraph(bas)
#     if not G.nodes:
#         return bas
#     if measure == "harmonic":
#         raw = _harmonic(G)
#     elif measure == "betweenness":
#         raw = dict(nx.betweenness_centrality(G, normalized=True))
#     else:
#         try:
#             raw = dict(nx.pagerank(G, alpha=0.85))
#         except nx.PowerIterationFailedConvergence:
#             raw = dict(nx.degree_centrality(G))
#
#     ns = _normalise(raw)
#     ew = {(e["source"], e["target"]): (ns.get(e["source"],0) + ns.get(e["target"],0)) / 2
#           for e in bas["edges"]}
#     return apply_strengths(bas, ns, ew, "s2")


# class ArgumentQualityModel:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         from sentence_transformers import SentenceTransformer
#         self.encoder = SentenceTransformer(model_name)
#         self.dim     = self.encoder.get_sentence_embedding_dimension()
#         self.W       = None
#         self.b       = 0.0
#         self._fitted = False
#
#     def _fmt(self, texts, topic):
#         return [f"[TOPIC] {topic} [ARG] {t}" for t in texts] if topic else texts
#
#     def _sig(self, x):
#         return np.where(x >= 0,
#                         1.0 / (1.0 + np.exp(-x)),
#                         np.exp(x) / (1.0 + np.exp(x)))
#
#     def score(self, texts, topic=None):
#         if not self._fitted:
#             raise RuntimeError("Model not fitted")
#         embs = self.encoder.encode(self._fmt(texts, topic),
#                                    convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
#         return self._sig(embs @ self.W + self.b).flatten()
#
#     def fit(self, texts, scores, topic=None, epochs=50, lr=0.01, l2=1e-4):
#         X = self.encoder.encode(self._fmt(texts, topic),
#                                 convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
#         y = np.array(scores, dtype=np.float32)
#         self.W = np.random.randn(self.dim).astype(np.float32) * 0.01
#         self.b = 0.0
#         for ep in range(1, epochs + 1):
#             p    = self._sig(X @ self.W + self.b)
#             errs = p - y
#             loss = float(np.mean(errs**2)) + l2 * float(np.dot(self.W, self.W))
#             self.W -= lr * ((X.T @ errs) / len(y) + 2 * l2 * self.W)
#             self.b -= lr * float(np.mean(errs))
#             if ep % 10 == 0:
#                 log.info("[S3] epoch %3d  MSE=%.5f", ep, loss)
#         self._fitted = True
#
#     def save(self, path):
#         path = Path(path); path.mkdir(parents=True, exist_ok=True)
#         np.save(path / "W.npy", self.W)
#         (path / "b.txt").write_text(str(self.b))
#
#     def load(self, path):
#         path = Path(path)
#         self.W = np.load(path / "W.npy")
#         self.b = float((path / "b.txt").read_text())
#         self._fitted = True

# ─── Helper functions
def _normalise(values: dict) -> dict:
    """Min-max normalise to [0, 1]. Returns all 1.0 if range is zero."""
    v = list(values.values())
    lo, hi = min(v), max(v)
    if math.isclose(lo, hi):
        return {k: 1.0 for k in values}
    return {k: (val - lo) / (hi - lo) for k, val in values.items()}


def _bas_to_digraph(bas: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in bas.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in bas.get("edges", []):
        G.add_edge(edge["source"], edge["target"],
                   relation=edge["relation"],
                   weight=edge.get("weight", 1.0)
                   )
    return G


def _edge_weight_from_nodes(ns: dict, edges: list) -> dict:
    """Edge weights are uniform 1.0 for this study — only node strengths vary."""
    return {(e["source"], e["target"]): 1.0 for e in edges}


def _apply_strengths(
        bas: dict, node_strengths: dict,
        edge_weights: dict, suffix: str
        ) -> dict:
    """Write initial_strength_{suffix} and initial_weight_{suffix} into bas in-place."""
    for node in bas["nodes"]:
        node[f"initial_strength_{suffix}"] = round(
            float(node_strengths.get(node["id"], 1.0)), 6
            )
    for edge in bas["edges"]:
        edge[f"initial_weight_{suffix}"] = round(
            float(edge_weights.get((edge["source"], edge["target"]), 1.0)), 6
            )
    return bas


# ─── Strategy implementations ─────────────────────────────────────────────────

def strategy_ui(bas: dict) -> dict:
    """UI — Uniform Initialization: all nodes and edges assigned strength 1.0."""
    log.info("[UI/s1] Uniform initialization")
    ids = [n["id"] for n in bas["nodes"]]
    pairs = [(e["source"], e["target"]) for e in bas["edges"]]
    return _apply_strengths(
        bas,
        {nid: 1.0 for nid in ids},
        {p: 1.0 for p in pairs},
        "s1",
        )


def strategy_gi(bas: dict) -> dict:
    """
    GI — Graph-based Initialization using PageRank centrality.
    Captures structural importance of argumentative units irrespective of
    their semantic value.
    """
    log.info("[GI/s2] Graph-based initialization (PageRank)")
    G = _bas_to_digraph(bas)
    if not G.nodes:
        return bas

    try:
        raw = dict(nx.pagerank(G, alpha=0.85))
    except nx.PowerIterationFailedConvergence:
        log.warning("[GI/s2] PageRank failed to converge — falling back to degree centrality")
        raw = dict(nx.degree_centrality(G))

    ns = _normalise(raw)
    ew = _edge_weight_from_nodes(ns, bas["edges"])
    return _apply_strengths(bas, ns, ew, "s2")


class ArgumentQualityModel:
    """
    Inference wrapper for the fine-tuned DeBERTa-v3-small pairwise quality model.
    Loads artefacts saved by train_quality_model.py.

    At inference, each argument is paired with the discussion topic
    (central proposition) as a cross-encoder:
        [CLS] topic [SEP] argument_text [SEP]

    The model outputs a score in [0, 1] representing argument quality
    relative to the topic.
    """

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoTokenizer, AutoModel
        import torch.nn as nn

        config = json.loads((model_dir / "config.json").read_text())
        self.max_len = config.get("max_len", 512)
        self.score_min = config.get("score_min", 0.0)
        self.score_max = config.get("score_max", 1.0)
        base_model = config.get("base_model", "microsoft/deberta-v3-small")
        dropout = config.get("hyperparams", {}).get("dropout", 0.1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir / "tokenizer"), local_files_only=True
            )
        # Load architecture from config, then restore weights from pytorch_model.bin
        from transformers import AutoConfig
        enc_config = AutoConfig.from_pretrained(
            str(model_dir / "tokenizer"), local_files_only=True
            )
        self._encoder = AutoModel.from_config(enc_config)
        hidden = self._encoder.config.hidden_size
        self._dropout = nn.Dropout(dropout)
        self._regressor = nn.Linear(hidden, 1)
        self._sigmoid = nn.Sigmoid()

        state = torch.load(
            str(model_dir / "pytorch_model.bin"),
            map_location="cpu",
            )
        self._encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in state.items()
             if k.startswith("encoder.")}, strict=True,
            )
        self._regressor.load_state_dict(
            {k[len("regressor."):]: v for k, v in state.items()
             if k.startswith("regressor.")}, strict=True,
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder.eval()
        self._regressor.eval()
        self._encoder.to(self.device)
        self._regressor.to(self.device)

    def score(self, texts: list, topic: str = None) -> np.ndarray:
        """
        Score a list of argument texts against the central proposition (topic).
        If topic is None, texts are scored without a proposition prefix — less
        accurate but still usable as a fallback.
        """
        import torch
        queries = [topic] * len(texts) if topic else [""] * len(texts)
        with torch.no_grad():
            enc = self.tokenizer(
                queries,
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
                )
            enc = {k: v.to(self.device) for k, v in enc.items()
                   if k != "token_type_ids"}
            out = self._encoder(**enc)
            cls = self._dropout(out.last_hidden_state[:, 0, :])
            scores = self._sigmoid(self._regressor(cls)).squeeze(-1)
            return scores.cpu().numpy().astype(np.float32)


def strategy_ti(
        bas: dict, model_path: str = None,
        topic: str = None,
        model_name: str = "all-MiniLM-L6-v2"
        ) -> dict:
    """
    TI — Topic-based Initialization.
    Loads the DeBERTa quality model saved by train_quality_model.py when
    model_path is provided and valid; falls back to cosine similarity to
    topic embedding otherwise.
    """
    log.info("[TI/s3] Topic-based initialization  model=%s  topic=%s",
             model_path or "fallback", topic or "none"
             )
    nids = [n["id"] for n in bas["nodes"]]
    texts = [n["text"] for n in bas["nodes"]]
    if not nids:
        return bas

    model_dir = Path(model_path) if model_path else None
    if model_dir and (model_dir / "pytorch_model.bin").exists():
        qm = ArgumentQualityModel(model_dir)
        raw = qm.score(texts, topic=topic)
        log.info("[TI/s3] Using fine-tuned DeBERTa quality model")
    elif topic:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model_name)
        topic_e = enc.encode([topic], normalize_embeddings=True)[0]
        raw = (enc.encode(texts, normalize_embeddings=True) @ topic_e
               ).astype(np.float32)
        log.info("[TI/s3] DeBERTa model not found — using topic cosine similarity fallback")
    else:
        log.warning("[TI/s3] No model and no topic — defaulting to uniform 1.0")
        raw = np.ones(len(nids), dtype=np.float32)

    ns = _normalise({nid: float(sc) for nid, sc in zip(nids, raw)})
    ew = _edge_weight_from_nodes(ns, bas["edges"])
    return _apply_strengths(bas, ns, ew, "s3")


def strategy_hi(
        bas: dict, model_path: str = None,
        topic: str = None,
        model_name: str = "all-MiniLM-L6-v2"
        ) -> dict:
    """
    HI — Hybrid Initialization.

    TI is the base — semantic quality is never penalised.
    GI is an additive bonus weighted by proximity to P_central:

        HI(a) = normalise( TI(a) + α(d) · GI(a) )
        α(d)  = 1 - d / d_max   (α=1 at root, α=0 at periphery)

    Rationale:
      - A strong semantic score (TI) is always a strong signal — preserved in full
      - Structural importance (GI) is a bonus for nodes close to P_central,
        capturing implicitly influential or poorly constructed but central arguments
      - Peripheral nodes (large d) receive negligible GI bonus — structural
        position far from the root is less meaningful for the central discourse
      - A node with poor TI but high GI near the root still gets a meaningful
        boost, capturing structurally relevant but weakly constructed arguments
    """
    log.info("[HI/s4] Hybrid initialization (TI base + proximity-weighted GI bonus)")

    gi_key = "initial_strength_s2"
    ti_key = "initial_strength_s3"

    gi_present = all(gi_key in n for n in bas["nodes"])
    ti_present = all(ti_key in n for n in bas["nodes"])

    if not gi_present:
        bas = strategy_gi(deepcopy(bas))
    if not ti_present:
        bas = strategy_ti(deepcopy(bas), model_path, topic, model_name)

    gi_scores = {n["id"]: n[gi_key] for n in bas["nodes"]}
    ti_scores = {n["id"]: n[ti_key] for n in bas["nodes"]}

    # Compute undirected distances from P_central
    root_id = bas.get("summary", {}).get("root_id")
    G_und = _bas_to_digraph(bas).to_undirected()

    if root_id and root_id in G_und:
        lengths = dict(nx.single_source_shortest_path_length(G_und, root_id))
    else:
        log.warning("[HI/s4] root_id '%s' not found — GI bonus disabled", root_id)
        lengths = {}

    reachable = [d for nid, d in lengths.items() if nid in gi_scores]
    d_max = max(reachable) if reachable else 1

    # HI(a) = TI(a) + α(d) · GI(a),  α(d) = 1 - d/d_max
    ns: dict[str, float] = {}
    for nid in gi_scores:
        d = lengths.get(nid, None)
        alpha = (1.0 - d / d_max) if d is not None and d_max > 0 else 0.0
        ns[nid] = ti_scores[nid] + alpha * gi_scores[nid]

    ns = _normalise(ns)
    ew = _edge_weight_from_nodes(ns, bas["edges"])
    return _apply_strengths(bas, ns, ew, "s4")


# ─── Main initialization orchestrator ─────────────────────────────────────────

def initialize_strengths(
        bas: dict, strategies: list,
        model_path: str = None,
        topic: str = None,
        model_name: str = "all-MiniLM-L6-v2"
        ) -> dict:
    """
    Run requested strategies on a single BAS dict.

    Topic is read from bas["summary"]["root_text"] if not explicitly provided.
    Returns a lean output dict containing only fields needed by Step 6.
    """
    resolved = ALL_STRATEGIES if "all" in strategies else [
        s for s in strategies if s in ALL_STRATEGIES
        ]

    # Extract topic from the central proposition (root_text) if not passed in
    if topic is None:
        topic = bas.get("summary", {}).get("root_text")

    # Work on a copy; HI may internally call GI/TI so process in order
    result = deepcopy(bas)
    for strat in resolved:
        if strat == "ui":
            result = strategy_ui(result)
        elif strat == "gi":
            result = strategy_gi(result)
        elif strat == "ti":
            result = strategy_ti(result, model_path, topic, model_name)
        elif strat == "hi":
            result = strategy_hi(result, model_path, topic, model_name)

    # ── Override root node strength to 1.0 across all strategies ─────────────
    root_id = bas.get("summary", {}).get("root_id")
    if root_id:
        for node in result["nodes"]:
            if node["id"] == root_id:
                for strat in resolved:
                    suffix = STRATEGY_MAP[strat]
                    key = f"initial_strength_{suffix}"
                    if key in node and node[key] != 1.0:
                        log.info(
                            "Root %s initial_strength_%s overridden: %.4f → 1.0",
                            root_id, suffix, node[key],
                            )
                        node[key] = 1.0
                break

    # ── Ensure root is influentiable by gradual semantics ─────────────────────
    # If the root has zero incoming edges, the iterative update p^(t) = W+^T s^(t)
    # will always be zero for the root — the discourse can never affect P_central.
    # Fix: reverse all direct outgoing edges from the root so they point inward,
    # making the root's children sources of influence rather than targets.
    # Relation type is preserved — support stays support, attack stays attack —
    # since the semantic relationship is symmetric in this context.
    if root_id:
        edges = result["edges"]
        incoming_ids = {e["source"] for e in edges if e["target"] == root_id}
        outgoing = [e for e in edges if e["source"] == root_id]

        if not incoming_ids and outgoing:
            log.info(
                "Root %s has 0 incoming edges — reversing %d outgoing edge(s) "
                "to make it influentiable by gradual semantics",
                root_id, len(outgoing),
                )
            for edge in outgoing:
                log.info(
                    "  Reversing: %s → %s (%s)  →  %s → %s (%s)",
                    edge["source"], edge["target"], edge["relation"],
                    edge["target"], edge["source"], edge["relation"],
                    )
                edge["source"], edge["target"] = edge["target"], edge["source"]

    # ── Build lean output — only what Step 6 (gradual_semantics) needs ────────
    strength_keys = [f"initial_strength_{STRATEGY_MAP[s]}" for s in resolved]
    weight_keys = [f"initial_weight_{STRATEGY_MAP[s]}" for s in resolved]

    lean_nodes = [
        {
            "id": n["id"],
            **{k: n[k] for k in strength_keys if k in n},
            }
        for n in result["nodes"]
        ]
    lean_edges = [
        {
            "source": e["source"],
            "target": e["target"],
            "relation": e["relation"],
            "synthetic": e.get("synthetic", False),  # needed by gradual_semantics weight matrices
            **{k: e[k] for k in weight_keys if k in e},
            }
        for e in result["edges"]
        ]

    return {
        "thread_id": result.get("thread_id", "unknown"),
        "conv_id": result.get("conv_id", "unknown"),
        "is_delta": result.get("is_delta", False),
        "mode": result.get("mode", "unknown"),
        "summary": result.get("summary", {}),
        "nodes": lean_nodes,
        "edges": lean_edges,
        }


# ─── JSONL batch runner ───────────────────────────────────────────────────────

def run_on_jsonl(
        input_path: Path, output_path: Path, strategies: list,
        model_path: str, model_name: str
        ) -> None:
    total = count_lines(input_path)
    log.info("Strength init starting — %d conversations  input=%s", total, input_path)
    with JSONLWriter(output_path) as writer:
        for line_no, bas in read_jsonl(input_path):
            log_progress(line_no, total, bas.get("thread_id", ""), "INIT", log)
            writer.write(initialize_strengths(bas, strategies, model_path,
                                              model_name=model_name
                                              )
                         )
    log.info("Finished. Output → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strength initializer — BAS JSONL in / initialized BAS JSONL out"
        )
    parser.add_argument("--input-repair", "-r",
                        help="Path to bas_repair.jsonl (from bas_assembler)"
                        )
    parser.add_argument("--input-no-repair", "-n",
                        help="Path to bas_no_repair.jsonl (from bas_assembler)"
                        )
    parser.add_argument("--output-repair",
                        default="initialized_bas_repair.jsonl"
                        )
    parser.add_argument("--output-no-repair",
                        default="initialized_bas_no_repair.jsonl"
                        )
    parser.add_argument("--strategy", "-s", nargs="+",
                        choices=["ui", "gi", "ti", "hi", "all"], default=["all"],
                        help="UI=uniform  GI=graph/PageRank  TI=topic-quality  HI=hybrid"
                        )
    parser.add_argument("--quality-model", help="Path to fine-tuned DeBERTa model dir "
                                                "(trained by train_quality_model.py)"
                        )
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="Sentence encoder for cosine-similarity fallback"
                        )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log-file", "-l", default="strength_initializer.log",
                        help="Log file path (default: strength_initializer.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    # ── Inference mode ────────────────────────────────────────────────────────
    if args.input_repair:
        run_on_jsonl(Path(args.input_repair), Path(args.output_repair),
                     args.strategy, args.quality_model, args.model_name
                     )

    if args.input_no_repair:
        run_on_jsonl(Path(args.input_no_repair), Path(args.output_no_repair),
                     args.strategy, args.quality_model, args.model_name
                     )

    if not args.input_repair and not args.input_no_repair:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee, pac_selector as ps
        import llm_reasoner as lr, bas_assembler as ba
        with JSONLWriter(Path(args.output_repair)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("conv_id", ""), "INIT", log)
                repair_bas, _ = ba.assemble_bas(
                    lr.run_reasoning(ps.select_all_pacs(ee.extract_edus(conv)))
                    )
                if repair_bas:
                    writer.write(initialize_strengths(
                        repair_bas, args.strategy, args.quality_model,
                        args.topic, args.model_name,
                        )
                        )

if __name__ == "__main__":
    main()
