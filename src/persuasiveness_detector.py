"""
Persuasiveness Detector  (Step 7 — final application)
Reads semantics_output.jsonl; writes predictions_output.jsonl.

Prediction rule:
  |Δ(root)| = |acceptability_sN(root) − initial_strength_sN(root)| ≥ threshold
  → predicted PERSUASIVE

Usage:
    python persuasiveness_detector.py --input semantics_output.jsonl
    python persuasiveness_detector.py --input semantics_output.jsonl \\
           --output predictions.jsonl --threshold 0.10
    python persuasiveness_detector.py --input semantics_output.jsonl \\
           --ground-truth-key delta_label --tune-threshold

"""
import logging
import argparse
import math
from pathlib import Path
from typing import Optional

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines,
    SAMPLE_CONVERSATIONS, setup_logging
)


log = logging.getLogger("persuasiveness_detector")

DEFAULT_THRESHOLD = 0.10
STRATEGIES        = ["s1", "s2", "s3", "s4"]


# ─── Root lookup
def get_root_node(bas: dict) -> Optional[dict]:
    root_id = bas.get("summary", {}).get("root_id")
    nodes   = bas.get("nodes", [])
    if root_id:
        return next((n for n in nodes if n["id"] == root_id), None)
    if nodes:
        return max(nodes, key=lambda n: (n.get("in_degree", 0), -n.get("global_idx", 0)))
    return None


# ─── Delta computation
def compute_deltas(root: dict) -> dict:
    deltas = {}
    for strat in STRATEGIES:
        init_k  = f"initial_strength_{strat}"
        final_k = f"acceptability_{strat}"
        if init_k not in root or final_k not in root:
            continue
        initial = float(root[init_k])
        final   = float(root[final_k])
        delta   = final - initial
        deltas[strat] = {
            "initial": round(initial, 6), "final": round(final, 6),
            "delta": round(delta, 6), "abs_delta": round(abs(delta), 6),
        }
    return deltas


def predict_persuasiveness(deltas: dict, threshold: float) -> dict:
    return {strat: {**d, "predicted": d["abs_delta"] >= threshold, "threshold": threshold}
            for strat, d in deltas.items()}


def ensemble_vote(predictions: dict) -> dict:
    if not predictions:
        return {"votes_persuasive": 0, "votes_total": 0, "predicted": False}
    votes_true = sum(1 for p in predictions.values() if p["predicted"])
    total      = len(predictions)
    return {"votes_persuasive": votes_true, "votes_total": total,
            "predicted": votes_true >= math.ceil(total / 2)}


# Single-conversation prediction
def predict_conversation(sem_output: dict, threshold: float,
                         ground_truth: Optional[bool] = None) -> dict:
    root = get_root_node(sem_output)
    if root is None:
        log.warning("thread_id=%s — no root node", sem_output.get("thread_id","?"))
        return {
            "thread_id": sem_output.get("thread_id"), "root_id": None, "root_text": None,
            "predictions": {}, "ensemble": ensemble_vote({}),
            "ground_truth": ground_truth, "correct": None,
        }

    deltas      = compute_deltas(root)
    predictions = predict_persuasiveness(deltas, threshold)
    ensemble    = ensemble_vote(predictions)

    if ground_truth is not None:
        for p in predictions.values():
            p["correct"] = p["predicted"] == ground_truth
        ensemble["correct"] = ensemble["predicted"] == ground_truth
    else:
        for p in predictions.values():
            p["correct"] = None
        ensemble["correct"] = None

    log.info("thread_id=%-15s  root=[%s]  ens=%s  gt=%s",
             sem_output.get("thread_id","?"), root.get("id"),
             ensemble["predicted"], ground_truth)
    for strat, p in predictions.items():
        log.debug("  [%s]  Δ=%+.4f  predicted=%s  correct=%s",
                  strat.upper(), p["delta"], p["predicted"], p.get("correct"))

    return {
        "thread_id":      sem_output.get("thread_id"),
        "root_id":      root.get("id"),
        "root_text":    root.get("text"),
        "predictions":  predictions,
        "ensemble":     ensemble,
        "ground_truth": ground_truth,
    }


# Evaluation
def binary_metrics(y_true: list, y_pred: list) -> dict:
    tp = sum(t and p     for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p  for t, p in zip(y_true, y_pred))
    fn = sum(t and not p  for t, p in zip(y_true, y_pred))
    n  = len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn - fp*fn) / denom if denom else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round((tp+tn)/n if n else 0, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "mcc": round(mcc, 4),
    }


def evaluate_dataset(records: list) -> dict:
    labelled    = [r for r in records if r.get("ground_truth") is not None]
    eval_results = {}
    for strat in STRATEGIES:
        yt = [r["ground_truth"] for r in labelled if strat in r.get("predictions",{})]
        yp = [r["predictions"][strat]["predicted"]
              for r in labelled if strat in r.get("predictions",{})]
        if yt:
            m = binary_metrics(yt, yp)
            eval_results[strat] = m
            log.info("[%s]  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  mcc=%.4f  n=%d",
                     strat.upper(), m["accuracy"], m["precision"],
                     m["recall"], m["f1"], m["mcc"], m["n"])
    yt  = [r["ground_truth"] for r in labelled]
    yp  = [r["ensemble"]["predicted"] for r in labelled]
    em  = binary_metrics(yt, yp)
    eval_results["ensemble"] = em
    log.info("[ENS]  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  mcc=%.4f  n=%d",
             em["accuracy"], em["precision"], em["recall"], em["f1"], em["mcc"], em["n"])
    return eval_results


def tune_threshold(records, metric="f1", strategy="ensemble"):
    candidates = [round(x * 0.01, 2) for x in range(1, 51)]
    labelled   = [r for r in records if r.get("ground_truth") is not None]
    if not labelled:
        return {}

    results = []
    for thresh in candidates:
        re_records = []
        for r in labelled:
            root_mock = {"id": r["root_id"], "text": r.get("root_text", "")}
            for strat, pred in r.get("predictions", {}).items():
                root_mock[f"initial_strength_{strat}"] = pred["initial"]
                root_mock[f"acceptability_{strat}"]    = pred["final"]
            sem_copy = {"nodes": [root_mock], "summary": {"root_id": r["root_id"]},
                        "thread_id": r["thread_id"]}
            re_records.append(predict_conversation(sem_copy, thresh, r["ground_truth"]))

        ev = evaluate_dataset(re_records)
        m  = ev.get(strategy, ev.get("ensemble", {})).get(metric, 0.0)
        results.append({"threshold": thresh, metric: m})

    best = max(results, key=lambda x: x[metric])
    log.info("Best threshold=%.3f  %s=%.4f", best["threshold"], metric, best[metric])
    return {"best_threshold": best["threshold"], f"best_{metric}": best[metric],
            "strategy": strategy, "metric": metric, "all": results}


#  JSONL batch runner
def run_on_jsonl(input_path, output_path, thresholds, gt_key, tune, tune_metric, tune_strategy):
    """
    Run persuasiveness detection for all thresholds in one pass.
    Each output record contains predictions keyed by threshold value.
    """
    total = count_lines(input_path)
    records = []
    log.info("Persuasiveness detection — %d conversations  thresholds=%s",
             total, thresholds
             )

    with JSONLWriter(output_path) as writer:
        for line_no, sem in read_jsonl(input_path):
            log_progress(line_no, total, sem.get("thread_id", ""), "PREDICT", log)
            gt_raw = sem.get(gt_key)
            gt: Optional[bool] = None
            if gt_raw is not None:
                gt = bool(gt_raw) if not isinstance(gt_raw, bool) else gt_raw

            root = get_root_node(sem)
            threshold_results = {}
            for thresh in thresholds:
                rec = predict_conversation(sem, thresh, gt)
                threshold_results[str(thresh)] = {
                    "predictions": rec["predictions"],
                    "ensemble": rec["ensemble"],
                    }

            record = {
                "conv_id":      sem.get("conv_id"),
                "thread_id":    sem.get("thread_id"),
                "is_delta":     gt,
                "mode":         sem.get("mode", "unknown"),
                "root_id":      root.get("id") if root else None,
                "root_text":    root.get("text") if root else None,
                "ground_truth": gt,
                "thresholds":   threshold_results,
                }
            records.append(record)
            writer.write(record)

        # Evaluation summary per threshold appended as sentinel
        eval_summary = {}
        for thresh in thresholds:
            thresh_records = []
            for r in records:
                tr = r["thresholds"].get(str(thresh), {})
                thresh_records.append({
                    "thread_id": r["thread_id"],
                    "ground_truth": r["ground_truth"],
                    "predictions": tr.get("predictions", {}),
                    "ensemble": tr.get("ensemble", {}),
                    }
                    )
            ev = evaluate_dataset(thresh_records)
            eval_summary[str(thresh)] = ev
            log.info("── δ=%.2f ──", thresh)
            print_evaluation(ev)

        writer.write({"__eval__": eval_summary, "thresholds": thresholds})

    log.info("Finished. Output → %s", output_path)

    if tune:
        tuning = tune_threshold(records, metric=tune_metric, strategy=tune_strategy)
        print(f"\n  Best threshold ({tune_strategy}/{tune_metric}): "
              f"{tuning['best_threshold']}  →  "
              f"{tune_metric}={tuning[f'best_{tune_metric}']:.4f}"
              )

    return records, eval_summary


# ─── Pretty print

def print_prediction(rec: dict) -> None:
    print(f"\n{'═'*70}")
    print(f"  thread_id : {rec.get('thread_id')}   root: [{rec.get('root_id')}]")
    print(f"  text    : \"{rec.get('root_text','')[:60]}\"")
    print(f"  GT      : {rec.get('ground_truth')}")
    print(f"  {'Strategy':<10}  {'Initial':>8}  {'Final':>8}  {'Δ':>8}  {'|Δ|':>8}  {'Predicted':<12}")
    for strat, p in rec.get("predictions",{}).items():
        print(f"  {strat.upper():<10}  {p['initial']:>8.4f}  {p['final']:>8.4f}  "
              f"{p['delta']:>+8.4f}  {p['abs_delta']:>8.4f}  "
              f"{'PERSUASIVE' if p['predicted'] else 'not-pers':<12}")
    ens = rec.get("ensemble",{})
    print(f"  {'ENSEMBLE':<10}  {'':>8}  {'':>8}  {'':>8}  {'':>8}  "
          f"{'PERSUASIVE' if ens.get('predicted') else 'not-pers':<12}  "
          f"({ens.get('votes_persuasive')}/{ens.get('votes_total')} votes)")


def print_evaluation(eval_results: dict) -> None:
    if not eval_results:
        return
    print(f"\n{'═'*70}")
    print("  EVALUATION METRICS")
    print(f"  {'Strategy':<10}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'MCC':>7}  {'N':>5}")
    print(f"  {'─'*65}")
    for strat in STRATEGIES + ["ensemble"]:
        if strat not in eval_results:
            continue
        m = eval_results[strat]
        print(f"  {strat.upper():<10}  {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  "
              f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['mcc']:>7.4f}  {m['n']:>5}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Persuasiveness detector — semantics JSONL in / predictions JSONL out"
    )
    parser.add_argument("--input-repair",     "-r",
                        help="semantics_output_repair.jsonl")
    parser.add_argument("--input-no-repair",  "-n",
                        help="semantics_output_no_repair.jsonl")
    parser.add_argument("--output-repair",
                        default="predictions_repair.jsonl")
    parser.add_argument("--output-no-repair",
                        default="predictions_no_repair.jsonl")
    parser.add_argument("--threshold",        type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="One or more δ thresholds (default: 0.1 0.3 0.5 0.7 0.9)")
    parser.add_argument("--ground-truth-key", default="is_delta",
                        help="Key holding GT label in each record (default: is_delta)")
    parser.add_argument("--tune-threshold",   action="store_true")
    parser.add_argument("--tune-metric",      default="f1",
                        choices=["accuracy", "precision", "recall", "f1", "mcc"])
    parser.add_argument("--tune-strategy",    default="ensemble",
                         choices=STRATEGIES + ["ensemble"])
    parser.add_argument("--verbose",          "-v", action="store_true")
    parser.add_argument("--log-file", "-l", default="bas_assembler1.log",
                        help="Log file path (default: bas_assembler1.log)"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    thresholds = sorted(set(args.threshold))
    args.tune_threshold = False

    if args.input_repair:
        run_on_jsonl(Path(args.input_repair), Path(args.output_repair),
                     thresholds, args.ground_truth_key,
                     args.tune_threshold, args.tune_metric, args.tune_strategy
                     )

    if args.input_no_repair:
        run_on_jsonl(Path(args.input_no_repair), Path(args.output_no_repair),
                     thresholds, args.ground_truth_key,
                     args.tune_threshold, args.tune_metric, args.tune_strategy
                     )

    if not args.input_repair and not args.input_no_repair:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee, pac_selector as ps
        import llm_reasoner as lr, bas_assembler as ba
        import strength_initializer as si, gradual_semantics as gs
        with JSONLWriter(Path(args.output_repair)) as writer:
            records = []
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS),
                             conv.get("thread_id", ""), "PREDICT", log
                             )
                repair_bas, _ = ba.assemble_bas(
                    lr.run_reasoning(ps.select_all_pacs(ee.extract_edus(conv)))
                    )
                if not repair_bas:
                    continue
                init = si.initialize_strengths(repair_bas, strategies=["all"])
                sem = gs.compute_gradual_semantics(init, strategies=["all"])
                root = get_root_node(sem)
                threshold_results = {}
                for thresh in thresholds:
                    rec = predict_conversation(sem, thresh)
                    threshold_results[str(thresh)] = {
                        "predictions": rec["predictions"],
                        "ensemble": rec["ensemble"],
                        }
                record = {
                    "thread_id": sem.get("thread_id"),
                    "root_id": root.get("id") if root else None,
                    "root_text": root.get("text") if root else None,
                    "ground_truth": None,
                    "thresholds": threshold_results,
                    }
                records.append(record)
                writer.write(record)
            writer.write({"__eval__": {}, "thresholds": thresholds})


if __name__ == "__main__":
    main()