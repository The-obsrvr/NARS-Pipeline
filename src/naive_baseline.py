import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("naive_baseline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

FEATURE_NAMES = ["attack_on_root", "n_attack", "support_ratio", "root_in_degree"]
SEED          = 42
TEST_RATIO    = 0.80


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(rec: dict) -> list:
    summary   = rec.get("summary", {})
    nodes     = rec.get("nodes", [])
    edges     = rec.get("edges", [])
    root_id   = summary.get("root_id")
    n_support = summary.get("support_edges", 0)
    n_attack  = summary.get("attack_edges",  0)
    total_rel = n_support + n_attack

    support_ratio  = n_support / total_rel if total_rel > 0 else 0.0
    attack_on_root = sum(
        1 for e in edges
        if e.get("target") == root_id and e.get("relation") == "attack"
    ) if root_id else 0

    root_in = 0
    if root_id:
        for n in nodes:
            if n.get("id") == root_id:
                root_in = n.get("in_degree", 0)
                break

    return [attack_on_root, n_attack, round(support_ratio, 6), root_in]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(path: Path):
    X, y, tids = [], [], []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec      = json.loads(line)
            is_delta = rec.get("is_delta")
            tid      = rec.get("thread_id")
            if is_delta is None or not tid:
                skipped += 1
                continue
            X.append(extract_features(rec))
            y.append(int(bool(is_delta)))
            tids.append(tid)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    log.info("Loaded %d records  skipped=%d  delta=%d  no_delta=%d",
             len(y), skipped, int(y.sum()), int((y == 0).sum()))
    return X, y, tids


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(X, y, tids, test_ratio, rng):
    idx_pos   = np.where(y == 1)[0]
    idx_neg   = np.where(y == 0)[0]
    n_test_pos = max(1, int(len(idx_pos) * test_ratio))
    n_test_neg = max(1, int(len(idx_neg) * test_ratio))
    test_pos  = rng.choice(idx_pos, n_test_pos, replace=False)
    test_neg  = rng.choice(idx_neg, n_test_neg, replace=False)
    test_idx  = np.concatenate([test_pos, test_neg])
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

    log.info("Split — train=%d (δ=%d no-δ=%d)  test=%d (δ=%d no-δ=%d)",
             len(train_idx), int(y[train_idx].sum()), int((y[train_idx]==0).sum()),
             len(test_idx),  int(y[test_idx].sum()),  int((y[test_idx]==0).sum()))

    return (X[train_idx], y[train_idx], [tids[i] for i in train_idx],
            X[test_idx],  y[test_idx],  [tids[i] for i in test_idx])


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Naive logistic regression baseline for persuasiveness detection"
    )
    parser.add_argument("--input",      "-i", required=True,
                        help="bas_repair.jsonl")
    parser.add_argument("--output-dir", "-o", default="outputs/baseline")
    parser.add_argument("--verbose",    "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Load and split
    X, y, tids = load_data(Path(args.input))
    X_tr, y_tr, tids_tr, X_te, y_te, tids_te = \
        stratified_split(X, y, tids, TEST_RATIO, rng)

    # Save thread ID splits
    (out / "train_thread_ids.txt").write_text("\n".join(tids_tr))
    (out / "test_thread_ids.txt").write_text("\n".join(tids_te))
    log.info("Saved splits → %s", out)

    # Standardise
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Train — class_weight='balanced' handles imbalance without oversampling
    clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=SEED, C=1.0)
    clf.fit(X_tr_s, y_tr)

    # Evaluate
    y_pred  = clf.predict(X_te_s)
    y_score = clf.predict_proba(X_te_s)[:, 1]
    acc     = accuracy_score(y_te, y_pred)
    f1      = f1_score(y_te, y_pred, zero_division=0)
    auc     = roc_auc_score(y_te, y_score) if len(np.unique(y_te)) > 1 else None

    log.info("Test — acc=%.4f  f1=%.4f  auc=%s",
             acc, f1, f"{auc:.4f}" if auc else "N/A")

    # Feature coefficients
    coefs = {name: round(float(c), 6)
             for name, c in zip(FEATURE_NAMES, clf.coef_[0])}
    coefs_sorted = dict(sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True))
    log.info("Coefficients: %s", coefs_sorted)

    # Save results
    results = {
        "test_metrics": {
            "accuracy": round(acc, 4),
            "f1":       round(f1,  4),
            "auc":      round(auc, 4) if auc else None,
            "n":        len(y_te),
        },
        "feature_coefficients": coefs_sorted,
        "class_distribution": {
            "total":    int(len(y)),
            "delta":    int(y.sum()),
            "no_delta": int((y == 0).sum()),
            "train_n":  int(len(y_tr)),
            "test_n":   int(len(y_te)),
        },
        "test_thread_ids":  tids_te,
        "train_thread_ids": tids_tr,
    }
    (out / "baseline_results.json").write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'═'*55}")
    print("  NAIVE BASELINE (Logistic Regression)")
    print(f"  Accuracy  {acc:.4f}")
    print(f"  F1        {f1:.4f}")
    print(f"  AUC       {auc:.4f}" if auc else "  AUC       N/A")
    print(f"  N test    {len(y_te)}")
    print(f"\n  test_thread_ids → {out}/test_thread_ids.txt")
    print()


if __name__ == "__main__":
    main()
