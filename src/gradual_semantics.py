"""
Gradual Argumentation Semantics  (Step 6)
Reads initialized_bas_repair.jsonl and/or initialized_bas_no_repair.jsonl;
writes semantics_output_repair.jsonl and semantics_output_no_repair.jsonl.

Usage:
    python gradual_semantics.py \\
        --input-repair    initialized_bas_repair.jsonl \\
        --input-no-repair initialized_bas_no_repair.jsonl \\
        --output-repair   semantics_output_repair.jsonl \\
        --output-no-repair semantics_output_no_repair.jsonl
    python gradual_semantics.py --lambda 0.7 --tau 0.5 --beta 0.05

"""

import logging
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines,
    SAMPLE_CONVERSATIONS, setup_logging
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gradual_semantics.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("gradual_semantics")

# Hyperparameters
DEFAULT_LAMBDA  = 0.5 # how much new update value affects current state
DEFAULT_TAU     = 0.3 # steepness of sigmoid (lower is steeper with any influence being amplified higher
DEFAULT_BETA    = 0.1 # retain 10% of initial strength, while 90% comes from the graph
DEFAULT_EPS     = 1e-6
DEFAULT_MAX_T   = 1000

SYNTHETIC_EDGE_WEIGHT = 0.7   # weight assigned to repair-mode synthetic edges
ATTACK_WEIGHT_MULTIPLIER = 3.0   # amplify attack edges (applied when --attack-boost enabled)

# ─── Matrix builders
def build_weight_matrices(node_ids, bas, strategy, row_norm=True, attack_boost=False):
    n       = len(node_ids)
    idx     = {nid: i for i, nid in enumerate(node_ids)}
    Wp      = np.zeros((n, n), dtype=np.float64)
    Wm      = np.zeros((n, n), dtype=np.float64)
    wkey    = f"initial_weight_{strategy}"
    for edge in bas.get("edges", []):
        src, tgt, rel = edge.get("source"), edge.get("target"), edge.get("relation")
        if src not in idx or tgt not in idx:
            continue
        if edge.get("synthetic", False):
            w = SYNTHETIC_EDGE_WEIGHT
        else:
            w = float(edge.get(wkey, edge.get("weight", 1.0)))
        i, j = idx[src], idx[tgt]
        if rel == "support":
            Wp[i, j] = w
        elif rel == "attack":
            Wm[i, j] = w * (ATTACK_WEIGHT_MULTIPLIER if attack_boost else 1.0)
    if row_norm:
        for j in range(n):
            cs = Wp[:, j].sum()
            if cs > 0: Wp[:, j] /= cs
            cs = Wm[:, j].sum()
            if cs > 0: Wm[:, j] /= cs
    return Wp, Wm



def build_b(node_ids, bas, strategy):
    """
    Initial strength of the nodes based on the initialization strategy.
    """
    skey = f"initial_strength_{strategy}"
    node_map = {n["id"]: n for n in bas.get("nodes", [])}
    # defaults to equal weight strategy.
    return np.array([float(node_map[nid].get(skey, node_map[nid].get("initial_strength_s1", 1.0)))
                     for nid in node_ids], dtype=np.float64
                    )


def sigma(x, tau):
    """
    Compute the gradual slope function
    """
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x / tau)),
                    np.exp(x / tau) / (1.0 + np.exp(x / tau))
                    )


def run_gradual_semantics(Wp, Wm, b, lam, tau, beta, eps, max_iter,
                          store_traj=False):
    """
    Implements Algorithm— Iterative Strength Propagation using Gradual Semantics:
      s^(0) = b
      p^(t) = (W+)^T s^(t)
      q^(t) = (W-)^T s^(t)
      u^(t) = β·b + p^(t) - q^(t)
      s̃^(t+1) = σ_τ(u^(t))
      s^(t+1) = (1-λ)·s^(t) + λ·s̃^(t+1)
      until ‖s^(t) - s^(t-1)‖_∞ < ε or t ≥ T_max
    """
    global t
    s    = b.copy()
    traj = [s.copy()] if store_traj else []
    conv, delta = False, float("inf")
    for t in range(1, max_iter + 1):
        p       = Wp.T @ s
        q       = Wm.T @ s
        u       = beta * b + p - q
        s_tilde = sigma(u, tau)
        s_new   = (1.0 - lam) * s + lam * s_tilde
        delta   = float(np.max(np.abs(s_new - s)))
        s       = s_new
        if store_traj:
            traj.append(s.copy())
        if delta < eps:
            conv = True
            log.debug("Converged at iter %d  Δ=%.2e", t, delta)
            break
    else:
        log.warning("No convergence after %d iters  Δ=%.2e", max_iter, delta)
    return {"s_star": s, "iterations": t, "converged": conv,
            "delta_final": delta, "trajectory": traj}


def _detect_strategies(bas):
    """Detect which initialisation strategies are present in the BAS nodes."""
    if not bas.get("nodes"):
        return []
    sample = bas["nodes"][0]
    return sorted(s for s in ["s1", "s2", "s3", "s4"]
                  if f"initial_strength_{s}" in sample)


# ─── Single-conversation entry point ─────────────────────────────────────────
def compute_gradual_semantics(bas, strategies, lam=DEFAULT_LAMBDA, tau=DEFAULT_TAU,
                               beta=DEFAULT_BETA, eps=DEFAULT_EPS,
                               max_iter=DEFAULT_MAX_T, row_norm=True, store_traj=False, attack_boost=False):
    """
    Run gradual argumentation semantics on a single BAS dict.
    Produces acceptability scores for each node under each strategy.
    Logs root node initial/final strength and |Δ| per strategy.
    """
    bas_out  = deepcopy(bas)
    node_ids = [n["id"] for n in bas_out["nodes"]]
    if not node_ids:
        return bas_out

    conv_id   = bas.get("conv_id", "?")
    thread_id = bas.get("thread_id", "?")
    root_id   = bas.get("summary", {}).get("root_id")

    available = _detect_strategies(bas_out)
    resolved  = available if "all" in strategies else [s for s in strategies if s in available]
    if not resolved:
        log.error("No requested strategies found in BAS")
        return bas_out

    bas_out.setdefault("semantics_meta", {})

    node_id_list = list(node_ids)

    for strat in resolved:
        Wp, Wm = build_weight_matrices(node_ids, bas_out, strat, row_norm, attack_boost)
        b      = build_b(node_ids, bas_out, strat)
        result = run_gradual_semantics(Wp, Wm, b, lam, tau, beta, eps, max_iter, store_traj)
        s      = result["s_star"]

        # Log root node initial vs final strength and |Δ|
        root_delta_str = ""
        if root_id and root_id in node_id_list:
            root_idx     = node_id_list.index(root_id)
            root_initial = float(b[root_idx])
            root_final   = float(s[root_idx])
            root_delta   = abs(root_final - root_initial)
            root_delta_str = (f"  root_init={root_initial:.4f}"
                              f"  root_final={root_final:.4f}"
                              f"  |Δ|={root_delta:.4f}")

        log.info("[%s] thread=%s  conv=%s  iters=%d  converged=%s  s*_mean=%.4f%s",
                 strat.upper(), thread_id, conv_id,
                 result["iterations"], result["converged"], s.mean(), root_delta_str)

        node_map = {n["id"]: n for n in bas_out["nodes"]}
        for nid, i in {nid: i for i, nid in enumerate(node_id_list)}.items():
            if nid in node_map:
                node_map[nid][f"acceptability_{strat}"] = float(round(s[i], 6))
                if result["trajectory"]:
                    node_map[nid][f"trajectory_{strat}"] = [float(round(arr[i], 6))
                                                             for arr in result["trajectory"]]

        bas_out["semantics_meta"][strat] = {
            "iterations":  result["iterations"],
            "converged":   result["converged"],
            "delta_final": float(result["delta_final"]),
        }
    return bas_out


# ─── JSONL batch runner
def run_on_jsonl(input_path, output_path, strategies, lam, tau, beta,
                 eps, max_iter, row_norm, store_traj):
    total = count_lines(input_path)
    log.info("Gradual semantics starting — %d conversations  input=%s", total, input_path)
    with JSONLWriter(output_path) as writer:
        for line_no, bas in read_jsonl(input_path):
            log_progress(line_no, total, bas.get("thread_id", ""), "SEM", log)
            writer.write(compute_gradual_semantics(bas, strategies, lam, tau,
                                                   beta, eps, max_iter, row_norm, store_traj))
    log.info("Finished. Output → %s", output_path)


def main():

    STRATEGY_MAP = {"ui": "s1", "gi": "s2", "ti": "s3", "hi": "s4"}
    ALL_STRATEGIES = ["ui", "gi", "ti", "hi"]


    parser = argparse.ArgumentParser(description="Gradual semantics — JSONL in / JSONL out")
    parser.add_argument("--input-repair",       "-r",
                        help="Path to initialized_bas_repair.jsonl")
    parser.add_argument("--input-no-repair",    "-n",
                        help="Path to initialized_bas_no_repair.jsonl")
    parser.add_argument("--output-repair",
                        default="semantics_output_repair.jsonl")
    parser.add_argument("--output-no-repair",
                        default="semantics_output_no_repair.jsonl")
    parser.add_argument("--strategy",          "-s", nargs="+",
                        choices=["ui", "gi", "ti", "hi", "all"], default=["all"])
    parser.add_argument("--lambda",            dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--tau",               type=float, default=DEFAULT_TAU)
    parser.add_argument("--beta",              type=float, default=DEFAULT_BETA)
    parser.add_argument("--eps",               type=float, default=DEFAULT_EPS)
    parser.add_argument("--max-iter",          type=int,   default=DEFAULT_MAX_T)
    parser.add_argument("--no-row-norm",       action="store_true")
    parser.add_argument("--attack-boost",      action="store_true",
                        help=f"Amplify attack edge weights by {ATTACK_WEIGHT_MULTIPLIER}x "
                             f"to balance them against the dominant support edges")
    parser.add_argument("--export-trajectory", action="store_true")
    parser.add_argument("--verbose",           "-v", action="store_true")
    parser.add_argument("--log-file", "-l", default="gradual_semantics.log",
                        help="Log file path (default: gradual_semantics.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    # Resolve strategy names to field suffixes
    resolved = (
        ["s1", "s2", "s3", "s4"] if "all" in args.strategy
        else [STRATEGY_MAP[s] for s in args.strategy if s in STRATEGY_MAP]
    )

    if args.input_repair:
        run_on_jsonl(Path(args.input_repair), Path(args.output_repair),
                     resolved, args.lam, args.tau, args.beta, args.eps,
                     args.max_iter, not args.no_row_norm, args.export_trajectory
                     )

    if args.input_no_repair:
        run_on_jsonl(Path(args.input_no_repair), Path(args.output_no_repair),
                     resolved, args.lam, args.tau, args.beta, args.eps,
                     args.max_iter, not args.no_row_norm, args.export_trajectory
                     )

    if not args.input_repair and not args.input_no_repair:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee, pac_selector as ps
        import llm_reasoner as lr, bas_assembler as ba
        import strength_initializer as si
        with JSONLWriter(Path(args.output_repair)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id", ""), "SEM", log)
                repair_bas, _ = ba.assemble_bas(
                    lr.run_reasoning(ps.select_all_pacs(ee.extract_edus(conv)))
                    )
                if repair_bas:
                    init = si.initialize_strengths(repair_bas, strategies=["all"])
                    writer.write(compute_gradual_semantics(
                        init, resolved, args.lam, args.tau, args.beta,
                        args.eps, args.max_iter, not args.no_row_norm,
                        args.export_trajectory,
                        )
                        )


if __name__ == "__main__":
    main()