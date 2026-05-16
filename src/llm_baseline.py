"""
LLM Baseline — Persuasiveness Detection  (RQ3)

Investigates whether modern LLMs can independently detect persuasion without
explicit symbolic reasoning. Three zero-shot prompting settings:

  (1) TEXT      — raw discussion text only
  (2) TEXT+BAS  — discussion text + extracted argument structure (nodes + relations)
  (3) TEXT+BAS+STRENGTH — text + structure + a priori argument quality scores

Models: Qwen3.6:27B, Gemma4:26B (via Ollama)

Output per record:
  conv_id, thread_id, is_delta (ground truth), model, setting,
  reasoning (thinking), decision (yes/no), correct (bool)

Usage:
    python llm_baseline.py \\
        --raw-data     Data/sample_500.jsonl \\
        --bas-repair   bas_repair.jsonl \\
        --initialized  initialized_repair.jsonl \\
        --thread-ids   thread_ids.txt \\
        --output       llm_baseline_results.jsonl \\
        --model        qwen3.6:27b \\
        --ollama-url   http://127.0.0.1:11434/api/chat \\
        --settings     1 2 3

    # Run all settings on both models:
    python llm_baseline.py ... --model qwen3.6:27b gemma4:26b --settings 1 2 3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("llm_baseline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("llm_baseline.log", mode="w", encoding="utf-8"),
    ],
)

TIMEOUT          = 300
MAX_RETRIES      = 2
MAX_TOKENS_APPROX = 6000   # approximate word budget for discussion text (1 word ≈ 1.3 tokens)
DEFAULT_STRENGTH  = "s4"   # HI score by default; changeable via --strength-strategy


# ─── Delta leakage prevention & token budgeting ───────────────────────────────

import re

# Patterns that reveal the ground truth delta outcome
_DELTA_PATTERNS = [
    re.compile(r'\bConfirmed[:\s]+\d+\s+delta\s+awarded.*', re.IGNORECASE),
    re.compile(r'\bDelta\s+awarded\s+to\s+/u/\S+', re.IGNORECASE),
    re.compile(r'\bδ\s*awarded', re.IGNORECASE),
    re.compile(r'[∆Δ]\s*awarded', re.IGNORECASE),
    re.compile(r'\bgood\s+bot\b', re.IGNORECASE),       # common delta confirmation reply
    re.compile(r'\bDeltaBot\b', re.IGNORECASE),
    re.compile(r'\[\s*History\s*\]\s*\(/r/changemyview/wiki/user/', re.IGNORECASE),
]

# Inline delta symbols that may hint at a view change within text
_INLINE_DELTA = re.compile(r'(?<!\w)[∆Δ](?!\w)')


def strip_delta_signals(text: str) -> str:
    """
    Remove explicit delta award signals from a turn's text.
    Replaces delta confirmation sentences with empty string.
    Replaces inline delta symbols (Δ/∆) with '[EDIT]'.
    """
    for pat in _DELTA_PATTERNS:
        text = pat.sub("", text)
    text = _INLINE_DELTA.sub("[EDIT]", text)
    # Collapse excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def is_delta_bot_turn(turn: dict) -> bool:
    """Identify DeltaBot confirmation turns to drop entirely."""
    speaker = turn.get("speaker_id", "")
    text    = turn.get("text", "")
    if "deltabot" in speaker.lower():
        return True
    if re.search(r'Confirmed[:\s]+\d+\s+delta\s+awarded', text, re.IGNORECASE):
        return True
    return False


def _approx_words(text: str) -> int:
    return len(text.split())


def build_discussion_text(
    conversation:    list,
    argumentative_post_ids: set = None,
    max_words:       int = MAX_TOKENS_APPROX,
) -> str:
    """
    Build a cleaned, token-budgeted discussion string.

    Strategy:
      1. Drop deleted turns and DeltaBot confirmation turns entirely.
      2. Strip delta award signals from all remaining turns.
      3. If argumentative_post_ids is provided, prioritise turns whose
         post_id appears in the BAS (i.e. contains at least one argumentative EDU).
         Always include the original post (first turn).
      4. Truncate to max_words by dropping lowest-priority turns from the end.

    Returns a single string of formatted turns.
    """
    turns = [t for t in conversation if not t.get("deleted") and not is_delta_bot_turn(t)]

    # Clean delta signals from all turns
    cleaned = []
    for t in turns:
        text = strip_delta_signals(t.get("text", "")).strip()
        if text:
            cleaned.append({**t, "text": text})

    if not cleaned:
        return ""

    # Separate original post (always kept) from replies
    op      = [cleaned[0]]
    replies = cleaned[1:]

    # Prioritise turns with argumentative EDUs when BAS is available
    if argumentative_post_ids:
        priority   = [t for t in replies if t.get("post_id") in argumentative_post_ids]
        remainder  = [t for t in replies if t.get("post_id") not in argumentative_post_ids]
        ordered    = op + priority + remainder
    else:
        ordered = op + replies

    # Greedily include turns up to word budget
    selected = []
    words    = 0
    for turn in ordered:
        tw = _approx_words(turn["text"])
        if words + tw > max_words and selected:
            break
        selected.append(turn)
        words += tw

    # Format
    lines = []
    for t in selected:
        speaker = t.get("speaker_id", "Unknown")
        lines.append(f"[{speaker}]: {t['text']}")

    return "\n\n".join(lines)


def get_argumentative_post_ids(bas: dict) -> set:
    """Return set of post_ids that have at least one argumentative node in the BAS."""
    return {n.get("post_id") for n in bas.get("nodes", []) if n.get("post_id")}



SYSTEM_PROMPT = """You are an expert in argumentation and persuasion analysis.

You will be given a Reddit Change My View (CMV) discussion where Speaker 1 presents a view they hold.
Other speakers attempt to change Speaker 1's view through argumentation.

The CENTRAL PROPOSITION (P_central) — the core claim Speaker 1 is defending — has been explicitly identified and will be provided to you.

Your task: determine whether Speaker 1's central proposition was changed (persuaded) or not by the end of the discussion.

Output format — respond with ONLY this JSON structure, nothing else:
{
  "reasoning": "<your step-by-step reasoning>",
  "decision": "<yes or no>"
}

decision must be exactly "yes" (view changed) or "no" (view not changed)."""


def build_prompt_setting1(conversation: list, bas: dict,
                           max_words: int = MAX_TOKENS_APPROX) -> str:
    """Setting 1 — raw discussion text only (cleaned, token-budgeted)."""
    discussion  = build_discussion_text(conversation, max_words=max_words)
    root_text   = bas.get("summary", {}).get("root_text", "")
    return (
        f"CENTRAL PROPOSITION (P_central):\n{root_text}\n\n"
        f"DISCUSSION:\n{discussion}\n\n"
        f"Was Speaker 1's central proposition changed by the discussion? "
        f"Reason step by step, then give your decision."
    )


def build_prompt_setting2(conversation: list, bas: dict,
                           max_words: int = MAX_TOKENS_APPROX) -> str:
    """Setting 2 — discussion text + argument structure (nodes + relations)."""
    arg_post_ids = get_argumentative_post_ids(bas)
    discussion   = build_discussion_text(conversation, arg_post_ids, max_words=max_words)

    root_id   = bas.get("summary", {}).get("root_id")
    root_text = bas.get("summary", {}).get("root_text", "")
    nodes     = bas.get("nodes", [])
    edges     = bas.get("edges", [])

    node_lines = []
    for n in nodes:
        marker = " ★ [CENTRAL PROPOSITION]" if n["id"] == root_id else ""
        node_lines.append(f"  [{n['id']}]{marker}: {n.get('text', '')}")

    edge_lines = [
        f"  {e['source']} --{e['relation'].upper()}--> {e['target']}"
        for e in edges if not e.get("synthetic")
    ]

    structure = (
        f"Central Proposition (P_central):\n  {root_text}\n\n"
        f"Argumentative Units ({len(nodes)} nodes):\n" + "\n".join(node_lines) + "\n\n"
        f"Relations ({len(edge_lines)} edges):\n" + "\n".join(edge_lines)
    )

    return (
        f"CENTRAL PROPOSITION (P_central):\n{root_text}\n\n"
        f"DISCUSSION:\n{discussion}\n\n"
        f"EXTRACTED ARGUMENT STRUCTURE:\n{structure}\n\n"
        f"Given the discussion and its argument structure, was the central proposition "
        f"changed? Reason step by step, then give your decision."
    )


def build_prompt_setting3(conversation: list, bas: dict, initialized: dict,
                           strength_strategy: str = DEFAULT_STRENGTH,
                           max_words: int = MAX_TOKENS_APPROX) -> str:
    """Setting 3 — text + structure + a priori argument quality scores."""
    arg_post_ids = get_argumentative_post_ids(bas)
    discussion   = build_discussion_text(conversation, arg_post_ids, max_words=max_words)

    root_id   = bas.get("summary", {}).get("root_id")
    root_text = bas.get("summary", {}).get("root_text", "")
    edges     = bas.get("edges", [])

    skey = f"initial_strength_{strength_strategy}"
    strength_map = {
        n["id"]: n.get(skey, n.get("initial_strength_s1", 1.0))
        for n in initialized.get("nodes", [])
    }

    nodes = bas.get("nodes", [])
    node_lines = []
    for n in nodes:
        marker   = " ★ [CENTRAL PROPOSITION]" if n["id"] == root_id else ""
        strength = strength_map.get(n["id"], 1.0)
        node_lines.append(
            f"  [{n['id']}]{marker} (quality={strength:.3f}): {n.get('text', '')}"
        )

    edge_lines = [
        f"  {e['source']} --{e['relation'].upper()}--> {e['target']}"
        for e in edges if not e.get("synthetic")
    ]

    strategy_label = {
        "s1": "UI (uniform)", "s2": "GI (graph/PageRank)",
        "s3": "TI (topic quality)", "s4": "HI (hybrid)",
    }.get(strength_strategy, strength_strategy)

    structure = (
        f"Central Proposition (P_central):\n  {root_text}\n\n"
        f"Argumentative Units ({len(nodes)} nodes, "
        f"quality score ∈ [0,1] using {strategy_label}):\n"
        + "\n".join(node_lines) + "\n\n"
        f"Relations ({len(edge_lines)} edges):\n" + "\n".join(edge_lines)
    )

    return (
        f"DISCUSSION:\n{discussion}\n\n"
        f"EXTRACTED ARGUMENT STRUCTURE WITH QUALITY SCORES:\n{structure}\n\n"
        f"Quality scores indicate the semantic relevance of each argument to the "
        f"central proposition (higher = more relevant). Given the discussion, argument "
        f"structure, and quality scores, was the central proposition changed? "
        f"Reason step by step, then give your decision."
    )


# ─── Ollama call ──────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, ollama_url: str):
    """
    Call Ollama with the given prompt. Returns {reasoning, decision} or None.
    Think mode enabled for all models — string level for gpt-oss, boolean for others.
    """
    model_lower = model.lower()
    if "gpt" in model_lower:
        think_val = "low"      # gpt-oss uses string levels
    else:
        think_val = True       # Qwen3, Gemma4 use boolean

    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "format":  "json",
        "think":   think_val,
        "options": {
            "temperature": 0.1,
            "num_predict": 8032,
            "num_ctx":     16384,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp    = requests.post(ollama_url, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data    = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            thinking = data.get("message", {}).get("thinking", "")

            # Strip markdown fences
            content = content.lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed  = json.loads(content)

            decision = str(parsed.get("decision", "")).strip().lower()
            if decision not in ("yes", "no"):
                log.warning("Invalid decision '%s' — defaulting to 'no'", decision)
                decision = "no"

            reasoning = parsed.get("reasoning", thinking or "")
            return {"reasoning": reasoning, "decision": decision}

        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2)

    return None


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl_by_thread_id(path: Path) -> dict:
    """Load a JSONL file into {thread_id: record} dict."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("thread_id")
            if tid:
                result[tid] = obj
    log.info("Loaded %d records from %s", len(result), path)
    return result


def load_raw_by_thread_id(path: Path) -> dict:
    """
    Load original CMV JSONL into {thread_id: record}.
    Each thread_id is unique so no deduplication needed.
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("thread_id")
            if tid:
                result[tid] = obj
    log.info("Loaded %d raw conversations from %s", len(result), path)
    return result


def load_thread_ids(path: Path, valid_thread_ids: set = None) -> list:
    """
    Load thread IDs to process, one per line.
    If valid_thread_ids is provided, only keeps IDs present in CMV2BAS.
    """
    ids = [l.strip() for l in open(path).readlines() if l.strip()]
    if valid_thread_ids:
        filtered = [t for t in ids if t in valid_thread_ids]
        log.info("Loaded %d thread IDs from %s (%d filtered to CMV2BAS)",
                 len(ids), path, len(filtered))
        return filtered
    log.info("Loaded %d thread IDs from %s", len(ids), path)
    return ids


def get_thread_ids_from_bas(bas_data: dict) -> list:
    """
    Derive thread IDs directly from the BAS repair file (already keyed by thread_id).
    """
    ids = list(bas_data.keys())
    log.info("Derived %d thread IDs from BAS repair file", len(ids))
    return ids


# ─── Main runner ──────────────────────────────────────────────────────────────

def run(raw_data: dict, bas_data: dict, init_data: dict,
        thread_ids: list, models: list, settings: list,
        ollama_url: str, output_path: Path,
        strength_strategy: str = DEFAULT_STRENGTH,
        max_words: int = MAX_TOKENS_APPROX) -> None:

    # Load existing results for resume support
    completed = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = (obj.get("thread_id"), obj.get("model"), obj.get("setting"))
                completed.add(key)
        log.info("Resuming — %d already completed", len(completed))

    total = len(thread_ids) * len(models) * len(settings)
    done  = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for thread_id in thread_ids:
            conv_id = "_".join(thread_id.split("_")[:2]) if "_" in thread_id else thread_id

            raw  = raw_data.get(thread_id)
            bas  = bas_data.get(thread_id)
            init = init_data.get(thread_id)

            if not raw:
                log.warning("thread_id=%s — no raw data found, skipping", thread_id)
                continue
            if not bas:
                log.warning("thread_id=%s — no BAS found, skipping", thread_id)
                continue

            conversation = raw.get("conversation", [])
            # Read is_delta from BAS (thread-level ground truth, not conv-level)
            is_delta = bas.get("is_delta", raw.get("is_delta"))

            for model in models:
                for setting in settings:
                    key = (thread_id, model, setting)
                    if key in completed:
                        log.info("Skipping already completed: %s  model=%s  setting=%d",
                                 thread_id, model, setting)
                        done += 1
                        continue

                    done += 1
                    log.info("[%d/%d] thread=%s  model=%s  setting=%d",
                             done, total, thread_id, model, setting)

                    # Build prompt for this setting
                    if setting == 1:
                        prompt = build_prompt_setting1(conversation, bas=bas, max_words=max_words)
                    elif setting == 2:
                        prompt = build_prompt_setting2(conversation, bas, max_words=max_words)
                    elif setting == 3:
                        if not init:
                            log.warning("No initialized BAS for %s — skipping setting 3",
                                        thread_id)
                            continue
                        prompt = build_prompt_setting3(conversation, bas, init,
                                                       strength_strategy=strength_strategy,
                                                       max_words=max_words)
                    else:
                        continue

                    result = call_ollama(prompt, model, ollama_url)

                    if result is None:
                        log.error("Failed to get response for %s  model=%s  setting=%d",
                                  thread_id, model, setting)
                        result = {"reasoning": "", "decision": "no"}

                    correct = (result["decision"] == "yes") == bool(is_delta) \
                              if is_delta is not None else None

                    record = {
                        "thread_id":    thread_id,
                        "conv_id":      conv_id,
                        "is_delta":     is_delta,
                        "model":        model,
                        "setting":      setting,
                        "decision":     result["decision"],
                        "reasoning":    result["reasoning"],
                        "correct":      correct,
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()

                    log.info("  decision=%s  correct=%s  gt=%s",
                             result["decision"], correct, is_delta)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def _f1(y_true: list, y_pred: list) -> float:
    tp = sum(t and p     for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _auc_roc(y_true: list, y_scores: list) -> float:
    """Trapezoidal AUC-ROC — uses decision confidence proxy (1.0 for yes, 0.0 for no)."""
    if len(set(y_true)) < 2:
        return float("nan")
    pairs  = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    n_pos  = sum(y_true)
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = prev_tp = prev_fp = 0
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


def evaluate_results(output_path: Path) -> None:
    """
    Evaluate LLM baseline results using the same metrics as evaluate.py:
    Accuracy, F1, and AUC-ROC.
    Only records with ground truth (is_delta) are included.
    AUC uses binary decision as score (yes=1.0, no=0.0) — a proxy since
    LLMs output binary decisions rather than continuous scores.
    """
    import math
    from collections import defaultdict

    records = defaultdict(list)
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("is_delta") is not None:
                key = (obj["model"], obj["setting"])
                records[key].append(obj)

    if not records:
        log.warning("No labelled records found in %s", output_path)
        return

    setting_labels = {1: "TEXT", 2: "TEXT+BAS", 3: "TEXT+BAS+STR"}

    print(f"\n{'═'*78}")
    print("  LLM BASELINE RESULTS")
    print(f"  {'Model':<22}  {'Setting':<14}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'N':>5}")
    print(f"  {'─'*72}")

    for (model, setting), recs in sorted(records.items(),
                                         key=lambda x: (x[0][0], x[0][1])):
        y_true  = [bool(r["is_delta"])    for r in recs]
        y_pred  = [r["decision"] == "yes" for r in recs]
        y_score = [1.0 if r["decision"] == "yes" else 0.0 for r in recs]

        acc = _accuracy(y_true, y_pred)
        f1  = _f1(y_true, y_pred)
        auc = _auc_roc(y_true, y_score)
        auc_str = f"{auc:>7.4f}" if not math.isnan(auc) else "    N/A"

        label = setting_labels.get(setting, str(setting))
        print(f"  {model:<22}  {label:<14}  {acc:>7.4f}  {f1:>7.4f}  {auc_str}  {len(recs):>5}")

    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM baseline for persuasiveness detection (RQ3)"
    )
    parser.add_argument("--raw-data",    "-d", required=True,
                        help="Original CMV JSONL (e.g. Data/sample_500.jsonl)")
    parser.add_argument("--bas-repair",  "-b", required=True,
                        help="BAS repair JSONL (from bas_assembler.py)")
    parser.add_argument("--initialized", "-i", default=None,
                        help="Initialized BAS JSONL (from strength_initializer.py, "
                             "required for setting 3)")
    parser.add_argument("--thread-ids",  "-t", default=None,
                        help="Text file with thread IDs to process, one per line. "
                             "If omitted, all thread IDs from --bas-repair are used.")
    parser.add_argument("--output",      "-o", default="llm_baseline_results.jsonl")
    parser.add_argument("--model",       "-m", nargs="+",
                        default=["qwen3.6:27b"],
                        help="Ollama model name(s) (default: qwen3.6:27b)")
    parser.add_argument("--settings",    "-s", nargs="+", type=int,
                        default=[1, 2, 3], choices=[1, 2, 3],
                        help="Settings to run: 1=text, 2=text+bas, 3=text+bas+strength")
    parser.add_argument("--ollama-url",          default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--strength-strategy",   default=DEFAULT_STRENGTH,
                        choices=["s1", "s2", "s3", "s4"],
                        help="Strength strategy for setting 3 "
                             "(s1=UI, s2=GI, s3=TI, s4=HI; default: s4/HI)")
    parser.add_argument("--max-words",           type=int, default=MAX_TOKENS_APPROX,
                        help=f"Approximate word budget for discussion text "
                             f"(default: {MAX_TOKENS_APPROX})")
    parser.add_argument("--eval-only",           action="store_true",
                        help="Skip inference, evaluate existing output file")
    parser.add_argument("--verbose",             "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.eval_only:
        evaluate_results(Path(args.output))
        return

    # Load data — all keyed by thread_id
    raw_data  = load_raw_by_thread_id(Path(args.raw_data))
    bas_data  = load_jsonl_by_thread_id(Path(args.bas_repair))
    init_data = load_jsonl_by_thread_id(Path(args.initialized)) \
                if args.initialized else {}

    # Thread IDs — restricted to conversations in CMV2BAS (bas_repair)
    valid_thread_ids = set(bas_data.keys())
    if args.thread_ids:
        thread_ids = load_thread_ids(Path(args.thread_ids), valid_thread_ids)
    else:
        thread_ids = get_thread_ids_from_bas(bas_data)
    log.info("Processing %d thread IDs (all restricted to CMV2BAS)", len(thread_ids))

    run(
        raw_data          = raw_data,
        bas_data          = bas_data,
        init_data         = init_data,
        thread_ids        = thread_ids,
        models            = args.model,
        settings          = args.settings,
        ollama_url        = args.ollama_url,
        output_path       = Path(args.output),
        strength_strategy = args.strength_strategy,
        max_words         = args.max_words,
    )

    evaluate_results(Path(args.output))


if __name__ == "__main__":
    main()