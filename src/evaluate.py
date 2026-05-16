"""
Evaluation Script — Persuasiveness Detection Results  (Step 7 output)

Reads predictions JSONL file(s) produced by persuasiveness_detector.py and
computes Accuracy, F1-score, and AUC-ROC for each:
  - Threshold δ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
  - Initialization strategy (UI/GI/TI/HI) + Ensemble
  - BAS mode (repair / no-repair) side by side when both are provided

AUC-ROC uses |Δ(root)| (the absolute strength change) as the continuous score,
allowing threshold-free evaluation of ranking quality.

Usage:
    # Single mode:
    python evaluate.py --input predictions_repair.jsonl

    # Both modes side by side:
    python evaluate.py \
        --input-repair    predictions_repair.jsonl \
        --input-no-repair predictions_no_repair.jsonl

    # Save results to JSON:
    python evaluate.py --input predictions_repair.jsonl --output results.json

Requirements:
    pip install scikit-learn
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

log = logging.getLogger("evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

STRATEGIES = ["s1", "s2", "s3", "s4", "ensemble"]
STRATEGY_LABELS = {
    "s1": "UI", "s2": "GI", "s3": "TI", "s4": "HI", "ensemble": "ENS"
}


# ─── Loading

def load_predictions(path: Path) -> tuple[list[dict], dict, list[float]]:
    """
    Load predictions JSONL.
    Returns (records, eval_summary, thresholds).
    Skips the sentinel __eval__ line.
    Ground truth (is_delta) is read directly from each record if present.
    """
    records    = []
    eval_summ  = {}
    thresholds = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "__eval__" in obj:
                eval_summ  = obj["__eval__"]
                thresholds = [float(t) for t in obj.get("thresholds", [])]
            else:
                # Populate ground_truth from is_delta if not already set
                if obj.get("ground_truth") is None and obj.get("is_delta") is not None:
                    obj["ground_truth"] = bool(obj["is_delta"])
                records.append(obj)

    log.info("Loaded %d records from %s  thresholds=%s",
             len(records), path, thresholds)
    labelled = sum(1 for r in records if r.get("ground_truth") is not None)
    log.info("  labelled=%d  unlabelled=%d", labelled, len(records) - labelled)
    return records, eval_summ, thresholds


def load_ground_truth(path: Path, gt_key: str = "is_delta") -> dict[str, bool]:
    """
    Optionally load ground truth from original data JSONL, keyed by conv_id.
    Used as fallback when is_delta is not embedded in prediction records.
    """
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj     = json.loads(line)
            conv_id = obj.get("conv_id") or obj.get("thread_id", "").rsplit("_", 1)[0]
            gt_raw  = obj.get(gt_key)
            if conv_id and gt_raw is not None:
                gt[conv_id] = bool(gt_raw)
    log.info("Loaded %d ground truth labels from %s  (key=%s)", len(gt), path, gt_key)
    return gt


def inject_ground_truth(records: list[dict], gt: dict[str, bool]) -> list[dict]:
    """Inject ground truth labels into records by conv_id."""
    matched = unmatched = 0
    for rec in records:
        if rec.get("ground_truth") is None:
            conv_id = rec.get("conv_id")
            if conv_id and conv_id in gt:
                rec["ground_truth"] = gt[conv_id]
                matched += 1
            else:
                unmatched += 1
    if matched or unmatched:
        log.info("Ground truth injection: matched=%d  unmatched=%d", matched, unmatched)
    return records


# ─── Metrics ──────────────────────────────────────────────────────────────────

def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def f1(y_true: list, y_pred: list) -> float:
    tp = sum(t and p     for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def auc_roc(y_true: list, y_scores: list) -> float:
    """
    Compute AUC-ROC via trapezoidal rule (no sklearn dependency).
    y_scores are continuous values (|Δ| per strategy).
    """
    if len(set(y_true)) < 2:
        return float("nan")

    # Sort by score descending
    pairs     = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    n_pos     = sum(y_true)
    n_neg     = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = fp = 0
    prev_tp = prev_fp = 0
    prev_score = None
    auc = 0.0

    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp, prev_fp = tp, fp
        if label:
            tp += 1
        else:
            fp += 1
        prev_score = score

    auc += (fp - prev_fp) * (tp + prev_tp) / 2
    return auc / (n_pos * n_neg)


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate(records: list, thresholds: list) -> dict:
    """
    Compute Accuracy, F1, AUC-ROC for each (threshold, strategy) pair.

    AUC uses |Δ| as the continuous ranking score — computed once per strategy
    independent of threshold.

    Returns nested dict:
        results[threshold][strategy] = {accuracy, f1, auc, n}
    """
    labelled = [r for r in records if r.get("ground_truth") is not None]
    if not labelled:
        log.warning("No labelled records found — cannot compute metrics")
        return {}

    y_true_all = [r["ground_truth"] for r in labelled]

    # ── AUC scores (threshold-independent) ───────────────────────────────────
    auc_scores: dict[str, float] = {}
    for strat in STRATEGIES:
        if strat == "ensemble":
            # Use average |Δ| across all strategies as ensemble score
            scores = []
            for r in labelled:
                tr = r.get("thresholds", {})
                # Use first threshold's abs_delta values (threshold-independent)
                first_thresh = next(iter(tr.values()), {}) if tr else {}
                preds = first_thresh.get("predictions", {})
                vals  = [p["abs_delta"] for p in preds.values() if "abs_delta" in p]
                scores.append(sum(vals) / len(vals) if vals else 0.0)
        else:
            scores = []
            for r in labelled:
                tr    = r.get("thresholds", {})
                first = next(iter(tr.values()), {}) if tr else {}
                preds = first.get("predictions", {})
                scores.append(preds.get(strat, {}).get("abs_delta", 0.0))

        auc_scores[strat] = auc_roc(y_true_all, scores)

    # ── Per-threshold metrics ─────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for thresh in thresholds:
        tkey = str(thresh)
        results[tkey] = {}

        for strat in STRATEGIES:
            y_true, y_pred = [], []
            for r in labelled:
                tr    = r.get("thresholds", {}).get(tkey, {})
                if strat == "ensemble":
                    pred = tr.get("ensemble", {}).get("predicted")
                else:
                    pred = tr.get("predictions", {}).get(strat, {}).get("predicted")
                if pred is not None:
                    y_true.append(r["ground_truth"])
                    y_pred.append(pred)

            results[tkey][strat] = {
                "accuracy": round(accuracy(y_true, y_pred), 4),
                "f1":       round(f1(y_true, y_pred), 4),
                "auc":      round(auc_scores[strat], 4) if not math.isnan(auc_scores[strat]) else None,
                "n":        len(y_true),
            }

    return results


# ─── Printing ─────────────────────────────────────────────────────────────────

def print_results(results: dict, mode: str = "") -> None:
    header = f"  RESULTS{f'  [{mode}]' if mode else ''}"
    print(f"\n{'═'*78}")
    print(header)

    thresholds = sorted(results.keys(), key=float)
    strats     = STRATEGIES

    # Header row
    strat_hdrs = "  ".join(f"{STRATEGY_LABELS.get(s, s):>5}" for s in strats)
    print(f"\n  {'δ':<6}  {'Metric':<6}  {strat_hdrs}")
    print(f"  {'─'*72}")

    for tkey in thresholds:
        tr = results[tkey]
        for metric in ["accuracy", "f1", "auc"]:
            vals = "  ".join(
                f"{tr.get(s, {}).get(metric, 0.0) or 0.0:>5.3f}"
                for s in strats
            )
            print(f"  {float(tkey):<6.2f}  {metric.upper():<6}  {vals}")
        print(f"  {'─'*72}")

    # Strategy header legend
    print(f"\n  Strategies: " +
          "  ".join(f"{v}={k.upper()}" for k, v in STRATEGY_LABELS.items()))
    print()


def print_comparison(repair_results: dict, no_repair_results: dict) -> None:
    """Print repair vs no-repair F1 comparison table."""
    print(f"\n{'═'*78}")
    print("  REPAIR vs NO-REPAIR  (F1)")
    thresholds = sorted(repair_results.keys(), key=float)

    print(f"\n  {'δ':<6}  {'Strategy':<8}  {'Repair':>8}  {'No-Repair':>10}  {'Δ':>8}")
    print(f"  {'─'*50}")
    for tkey in thresholds:
        for strat in STRATEGIES:
            r_f1  = repair_results.get(tkey, {}).get(strat, {}).get("f1", 0.0)
            nr_f1 = no_repair_results.get(tkey, {}).get(strat, {}).get("f1", 0.0)
            diff  = r_f1 - nr_f1
            print(f"  {float(tkey):<6.2f}  {STRATEGY_LABELS.get(strat, strat):<8}  "
                  f"{r_f1:>8.4f}  {nr_f1:>10.4f}  {diff:>+8.4f}")
        print(f"  {'─'*50}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate persuasiveness detection results"
    )
    parser.add_argument("--input",            "-i",
                        help="predictions JSONL (single mode)")
    parser.add_argument("--input-repair",     "-r",
                        help="predictions_repair.jsonl")
    parser.add_argument("--input-no-repair",  "-n",
                        help="predictions_no_repair.jsonl")
    parser.add_argument("--ground-truth",     "-g", default=None,
                        help="Optional: original data JSONL with ground truth labels. "
                             "If omitted, is_delta is read directly from prediction records.")
    parser.add_argument("--gt-key",           default="is_delta",
                        help="Ground truth field name (default: is_delta)")
    parser.add_argument("--output",           "-o", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--verbose",          "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load external ground truth if provided
    gt = {}
    if args.ground_truth:
        gt = load_ground_truth(Path(args.ground_truth), args.gt_key)

    all_results = {}

    # ── Single input ──────────────────────────────────────────────────────────
    if args.input:
        records, _, thresholds = load_predictions(Path(args.input))
        if gt:
            records = inject_ground_truth(records, gt)
        results = evaluate(records, thresholds)
        print_results(results)
        all_results["single"] = results

    # ── Dual input (repair + no-repair) ───────────────────────────────────────
    if args.input_repair:
        r_records, _, r_thresholds = load_predictions(Path(args.input_repair))
        if gt:
            r_records = inject_ground_truth(r_records, gt)
        repair_results = evaluate(r_records, r_thresholds)
        print_results(repair_results, mode="repair")
        all_results["repair"] = repair_results

    if args.input_no_repair:
        nr_records, _, nr_thresholds = load_predictions(Path(args.input_no_repair))
        if gt:
            nr_records = inject_ground_truth(nr_records, gt)
        no_repair_results = evaluate(nr_records, nr_thresholds)
        print_results(no_repair_results, mode="no_repair")
        all_results["no_repair"] = no_repair_results

    if args.input_repair and args.input_no_repair:
        print_comparison(repair_results, no_repair_results)
        all_results["comparison"] = {
            tkey: {
                strat: {
                    "repair_f1":    repair_results.get(tkey, {}).get(strat, {}).get("f1"),
                    "no_repair_f1": no_repair_results.get(tkey, {}).get(strat, {}).get("f1"),
                }
                for strat in STRATEGIES
            }
            for tkey in sorted(repair_results.keys(), key=float)
        }

    if not args.input and not args.input_repair and not args.input_no_repair:
        parser.print_help()
        sys.exit(1)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        log.info("Results saved → %s", args.output)


if __name__ == "__main__":
    main()