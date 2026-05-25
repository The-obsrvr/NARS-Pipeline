"""
LLM Reasoner — Iterative argumentative relation classification (Step 3)

For every (target EDU, PAC source EDU) pair this module determines:
  (i)  Is the target EDU argumentative?
  (ii) Does the source EDU SUPPORT, ATTACK, or is NEUTRAL toward the target?
"""
import json
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

import requests

from sys_utils import (
read_jsonl, log_progress, JSONLWriter, count_lines, SAMPLE_CONVERSATIONS, setup_logging
    )

# Logging
log = logging.getLogger("llm_reasoner")

# ─── Initialization

OLLAMA_URL      = "http://localhost:11434/api/chat"
MODEL           = "qwen3.6:27b"
TIMEOUT         = 500
MAX_RETRIES     = 2
VALID_RELATIONS = {"support", "attack", "neutral"}
ENABLE_THINKING = True   # set to False via --no-think to disable Qwen3 thinking mode
# Thinking budget for models that support it (e.g. gpt-oss).
# Accepted values: "none" | "low" | "medium" | "high"  (None = omit the key)
THINKING_BUDGET: Optional[str] = None

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in argumentative discourse analysis.

DEFINITIONS:
- SUPPORT: The source EDU provides evidence, reasons, or elaboration that strengthens the target EDU's claim.
- ATTACK: The source EDU contradicts, undermines, rebuts, or weakens the target EDU's claim.
- NEUTRAL: The source EDU does not argue for or against the target EDU.

INSTRUCTIONS:
You will receive a batch of TARGET EDUs. For each target EDU and each of its SOURCE EDUs, classify the relation as SUPPORT, ATTACK, or NEUTRAL. Be critical — implicit relations (unstated premises, enthymemes) count.

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
{
  "batch": [
    {
      "target_idx": 0,
      "relations": [
        {"source_idx": 0, "relation": "support"},
        {"source_idx": 1, "relation": "attack"}
      ]
    }
  ]
}

target_idx is the 0-based position of the target EDU in the batch.
source_idx is the 0-based position in that target's SOURCE EDUs list.
"""


def build_user_prompt(
    target_text:   str,
    context_before: list[str],
    context_after:  list[str],
    source_edus:   list[str],
) -> str:
    """Single-target prompt — used only for tie-breaking on disputed pairs."""
    ctx_before_str = (
        "\n".join(f"  [{i}] {t}" for i, t in enumerate(context_before))
        if context_before else "  (none)"
    )
    ctx_after_str = (
        "\n".join(f"  [{i}] {t}" for i, t in enumerate(context_after))
        if context_after else "  (none)"
    )
    sources_str = "\n".join(
        f"  [{i}] {text}" for i, text in enumerate(source_edus)
    )
    return f"""=== TARGET EDU [0] ===
{target_text}

=== LOCAL CONTEXT (before target) ===
{ctx_before_str}

=== LOCAL CONTEXT (after target) ===
{ctx_after_str}

=== SOURCE EDUs (candidates to classify with respect to the target) ===
{sources_str}

Classify each SOURCE EDU's relation to the TARGET EDU and state whether the TARGET is argumentative.
"""


def build_batch_prompt(batch_items: list[dict]) -> str:
    """
    Build a single prompt covering a batch of target EDUs.

    Each batch_item contains:
        batch_position  : int         (0-based index within this batch)
        target_text     : str
        context_before  : list[str]
        context_after   : list[str]
        source_texts    : list[str]
    """
    sections = []
    for item in batch_items:
        pos       = item["batch_position"]
        ctx_b_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_before"]))
            if item["context_before"] else "  (none)"
        )
        ctx_a_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_after"]))
            if item["context_after"] else "  (none)"
        )
        sources_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["source_texts"]))
            if item["source_texts"] else "  (none)"
        )
        sections.append(
            f"=== TARGET EDU [{pos}] ===\n{item['target_text']}\n\n"
            f"=== LOCAL CONTEXT (before target [{pos}]) ===\n{ctx_b_str}\n\n"
            f"=== LOCAL CONTEXT (after target [{pos}]) ===\n{ctx_a_str}\n\n"
            f"=== SOURCE EDUs for target [{pos}] ===\n{sources_str}"
        )

    divider = "\n\n" + "─" * 60 + "\n\n"
    return divider.join(sections) + "\n\nClassify all target EDUs and their source candidates as specified."


TIEBREAK_SYSTEM_PROMPT = """You are an expert in argumentative discourse analysis acting as a tie-breaker.

DEFINITIONS:
- SUPPORT: The source EDU provides evidence, reasons, or elaboration that strengthens the target EDU's claim.
- ATTACK: The source EDU contradicts, undermines, rebuts, or weakens the target EDU's claim.
- NEUTRAL: The source EDU does not argue for or against the target EDU.

Your role is to resolve uncertainty — either because two previous attempts disagreed, or because both previous attempts failed to produce a valid response. Reason carefully and provide your best judgement.

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
{
  "batch": [
    {
      "target_idx": 0,
      "relations": [
        {"source_idx": 0, "relation": "support"},
        {"source_idx": 1, "relation": "attack"}
      ]
    }
  ]
}
"""


def build_tiebreak_prompt(tiebreak_items: list[dict], reason: str) -> str:
    """
    Build a tie-breaking prompt that tells the model why it is being invoked:
      reason = "disagreement"  — two calls produced conflicting labels
      reason = "failed"        — both calls failed to produce a valid response
    """
    if reason == "disagreement":
        preamble = (
            "Two independent analyses of the following EDUs produced conflicting results. "
            "Resolve the disagreement by providing your own careful judgement."
        )
    else:
        preamble = (
            "Two independent attempts to classify the following EDUs both failed to produce "
            "a valid response. Please analyse them carefully and provide your best judgement."
        )

    sections = []
    for item in tiebreak_items:
        pos       = item["batch_position"]
        ctx_b_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_before"]))
            if item["context_before"] else "  (none)"
        )
        ctx_a_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["context_after"]))
            if item["context_after"] else "  (none)"
        )
        sources_str = (
            "\n".join(f"  [{i}] {t}" for i, t in enumerate(item["source_texts"]))
            if item["source_texts"] else "  (none)"
        )
        sections.append(
            f"=== TARGET EDU [{pos}] ===\n{item['target_text']}\n\n"
            f"=== LOCAL CONTEXT (before target [{pos}]) ===\n{ctx_b_str}\n\n"
            f"=== LOCAL CONTEXT (after target [{pos}]) ===\n{ctx_a_str}\n\n"
            f"=== SOURCE EDUs for target [{pos}] ===\n{sources_str}"
        )

    divider = "\n\n" + "─" * 60 + "\n\n"
    body    = divider.join(sections)
    return f"{preamble}\n\n{body}\n\nClassify all target EDUs and their source candidates."


# Ollama client
def call_ollama(
    user_prompt:   str,
    temperature:   float,
    retries:       int = MAX_RETRIES,
    system_prompt: str = None,
) -> Optional[str]:
    """Single inference call; strips Qwen3 <think> block."""

    if "qwen3" in MODEL.lower():
        think_value = ENABLE_THINKING
    elif "gpt" in MODEL.lower():
        think_value = THINKING_BUDGET or False
    else:
        think_value = False

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "think":  think_value,   # controlled via --no-think CLI flag
        "options": {
            "temperature": temperature,
            "num_predict": 8192,
            "num_ctx":     16384,
        },
    }

    for attempt in range(1, retries + 1):
        try:
            log.debug("Ollama call  temp=%.2f  attempt=%d/%d", temperature, attempt, retries)
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            # Strip Qwen3 thinking block
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return raw
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_URL)
            return None
        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d/%d", attempt, retries)
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.error("Unexpected error: %s", exc)
            return None
    return None


# Response parsing
def parse_response(raw: str, n_sources: int) -> Optional[dict]:
    """
    Parse the model JSON response
    Returns None if parsing fails or the structure is invalid.
    """
    if raw is None:
        return None
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)

        # Handle both single-target {"target_is_argumentative": ..., "relations": [...]}
        # and batch response where only one entry exists
        if "batch" in data and isinstance(data["batch"], list) and data["batch"]:
            data = data["batch"][0]

        is_arg = bool(data.get("target_is_argumentative", False))
        rel_map: dict[int, str] = {}
        for entry in data.get("relations", []):
            idx = entry.get("source_idx")
            rel = str(entry.get("relation", "neutral")).lower().strip()
            if rel not in VALID_RELATIONS:
                rel = "neutral"
            if isinstance(idx, int) and 0 <= idx < n_sources:
                rel_map[idx] = rel

        return {"target_is_argumentative": is_arg,
                "relations": [rel_map.get(i, "neutral") for i in range(n_sources)]
                }

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Response parse failed: %s  raw=%.80s", exc, raw)
        return None


def parse_batch_response(raw: str, batch_items: list[dict]) -> Optional[list[dict]]:
    """
    Parse a multi-target batch response into a list of per-target result dicts.
    target_is_argumentative is no longer required from the model — it is inferred
    post-hoc from whether any support/attack relations exist.
    Returns None if parsing fails entirely.
    """
    if not raw or not raw.strip():
        log.warning("Empty batch response from model")
        return None

    try:
        cleaned   = re.sub(r"```(?:json)?|```", "", raw).strip()
        data      = json.loads(cleaned)
        n_targets = len(batch_items)

        if isinstance(data, dict) and "batch" in data:
            batch_list = data["batch"]
        elif isinstance(data, list):
            batch_list = data
        else:
            log.warning("Unexpected batch response structure: %s", type(data))
            return None

        results: list[Optional[dict]] = [None] * n_targets

        for entry in batch_list:
            t_idx = entry.get("target_idx")
            if not isinstance(t_idx, int) or not (0 <= t_idx < n_targets):
                continue

            n_srcs  = len(batch_items[t_idx]["source_texts"])
            rel_map : dict[int, str] = {}

            for rel_entry in entry.get("relations", []):
                s_idx = rel_entry.get("source_idx")
                rel   = str(rel_entry.get("relation", "neutral")).lower().strip()
                if rel not in VALID_RELATIONS:
                    rel = "neutral"
                if isinstance(s_idx, int) and 0 <= s_idx < n_srcs:
                    rel_map[s_idx] = rel

            results[t_idx] = {
                "relations": [rel_map.get(i, "unk") for i in range(n_srcs)],
            }

        # Fill any targets missing from the response with safe defaults
        for i in range(n_targets):
            if results[i] is None:
                n_srcs     = len(batch_items[i]["source_texts"])
                results[i] = {"relations": ["neutral"] * n_srcs}
                log.warning("Target idx %d missing from batch response — defaulting", i)

        return results  # type: ignore[return-value]

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning("Batch response parse failed: %s  raw=%.80s", exc, raw)
        return None


# ─── Self-consistency voting ──────────────────────────────────────────────────
def majority_vote(responses: list[dict], n_sources: int) -> dict:
    """Majority vote across relation labels for each source position."""
    relations = []
    for i in range(n_sources):
        votes = [r["relations"][i] for r in responses if i < len(r["relations"])]
        rel   = Counter(votes).most_common(1)[0][0] if votes else "neutral"
        relations.append(rel)
    return {"relations": relations}


# Context window builder
def get_context_window(
    all_edus:   list[str],
    target_idx: int,
    window:     int,
) -> tuple[list[str], list[str]]:
    """Return (edus_before, edus_after) within ±window positions."""
    before = all_edus[max(0, target_idx - window): target_idx]
    after  = all_edus[target_idx + 1: target_idx + 1 + window]
    return before, after


# Per-EDU inference with cascaded self-consistency
def _disputed_batch_pairs(
    res1: list[dict],
    res2: list[dict],
    batch_items: list[dict],
) -> list[tuple[int, int]]:
    """
    Compare two batch responses and return list of (target_idx, source_idx)
    tuples where the relation label differs between the two calls.
    """
    pair_disputes: list[tuple[int, int]] = []
    for t_idx in range(len(batch_items)):
        n_srcs = len(batch_items[t_idx]["source_texts"])
        for s_idx in range(n_srcs):
            if res1[t_idx]["relations"][s_idx] != res2[t_idx]["relations"][s_idx]:
                pair_disputes.append((t_idx, s_idx))
    return pair_disputes


def reason_for_edu_batch(
    batch_items:   list[dict],
    temp1:         float,
    temp2:         float,
    temp_tiebreak: float,
) -> list[dict]:
    """
    Run two inference calls on a batch of target EDUs.

    Both-failed case:  tie-break is triggered over the full batch with
                       reason="failed" so the model knows it is self-refining.
    Disagreement case: tie-break is triggered only over disputed pairs with
                       reason="disagreement".
    Agreed case:       return call-1 result directly.

    Returns a list of per-target result dicts aligned to batch_items order.
    """
    n_targets = len(batch_items)
    default   = [
        {"relations": ["neutral"] * len(item["source_texts"])}
        for item in batch_items
    ]

    # Skip LLM call if all targets have no sources
    if all(len(item["source_texts"]) == 0 for item in batch_items):
        return [{"relations": []} for _ in batch_items]

    prompt = build_batch_prompt(batch_items)

    # ── Call 1
    raw1  = call_ollama(prompt, temperature=temp1)
    res1  = parse_batch_response(raw1, batch_items)
    log.debug("Call-1 parsed: %s", res1 is not None)

    # ── Call 2
    raw2  = call_ollama(prompt, temperature=temp2)
    res2  = parse_batch_response(raw2, batch_items)
    log.debug("Call-2 parsed: %s", res2 is not None)

    valid = [r for r in [res1, res2] if r is not None]

    # ── Both failed — trigger tie-break over full batch as self-refinement
    if len(valid) == 0:
        log.warning("Both calls failed — invoking tie-break as self-refinement")
        tb_prompt = build_tiebreak_prompt(
            [{**item, "batch_position": i} for i, item in enumerate(batch_items)],
            reason="failed",
        )
        raw3 = call_ollama(tb_prompt, temperature=temp_tiebreak,
                           system_prompt=TIEBREAK_SYSTEM_PROMPT)
        res3 = parse_batch_response(raw3, batch_items)
        if res3 is not None:
            log.info("Self-refinement tie-break succeeded")
            return res3
        log.error("Self-refinement also failed — returning defaults")
        return default

    # ── Only one call succeeded
    if len(valid) == 1:
        log.warning("Only one successful call — using single result")
        return valid[0]

    # ── Both succeeded — check for pair-level disputes
    pair_disputes = _disputed_batch_pairs(valid[0], valid[1], batch_items)

    if not pair_disputes:
        log.debug("Full batch agreement — no tie-break needed")
        return valid[0]

    log.info("    Pair-level disagreement — disputed_pairs=%d", len(pair_disputes))

    # Build minimal tie-break sub-batch (disputed targets only)
    disputed_target_indices = sorted({t for t, _ in pair_disputes})
    tiebreak_items = [
        {**batch_items[t], "batch_position": new_pos}
        for new_pos, t in enumerate(disputed_target_indices)
    ]
    tb_prompt = build_tiebreak_prompt(tiebreak_items, reason="disagreement")
    raw3      = call_ollama(tb_prompt, temperature=temp_tiebreak,
                            system_prompt=TIEBREAK_SYSTEM_PROMPT)
    res3      = parse_batch_response(raw3, tiebreak_items)
    log.debug("Tie-break parsed: %s", res3 is not None)

    # Merge: start from call-1, patch in tie-break decisions for disputed pairs
    final = [dict(r) for r in valid[0]]

    if res3 is not None:
        for new_pos, orig_t_idx in enumerate(disputed_target_indices):
            disputed_sources = [s for t, s in pair_disputes if t == orig_t_idx]
            for s_idx in disputed_sources:
                if s_idx < len(res3[new_pos]["relations"]):
                    final[orig_t_idx]["relations"][s_idx] = res3[new_pos]["relations"][s_idx]
    else:
        # Tie-break itself failed — majority vote between call-1 and call-2
        log.warning("Tie-break call failed — majority vote for disputed pairs")
        for t_idx, s_idx in pair_disputes:
            votes = [valid[0][t_idx]["relations"][s_idx],
                     valid[1][t_idx]["relations"][s_idx]]
            final[t_idx]["relations"][s_idx] = Counter(votes).most_common(1)[0][0]

    return final


# ─── Main reasoning pass ──────────────────────────────────────────────────────

def run_reasoning(
    pac_output:     dict,
    batch_size:     int   = 10,
    context_window: int   = 2,
    temp1:          float = 0.1,
    temp2:          float = 0.2,
    temp_tiebreak:  float = 0.15,
) -> dict:
    """
    Collect all EDUs across all turns, group them into batches of batch_size,
    and run one LLM call (with self-consistency) per batch of EDUs.

    The batch unit is the TARGET EDU — each batch contains up to batch_size
    target EDUs, each with their own PAC source list. This matches the paper:
    "EDUs are processed in batches of up to 10 per call".
    """
    conversation = pac_output.get("conversation", [])

    # Build flat list of all EDU texts for context window lookup
    all_edu_texts: list[str] = []
    for turn in conversation:
        if turn.get("deleted"):
            continue
        for edu in turn.get("edus", []):
            all_edu_texts.append(
                edu.get("text", "") if isinstance(edu, dict) else str(edu)
            )

    thread_id = pac_output.get("thread_id", "?")
    N       = len(all_edu_texts)
    log.info(
        "Starting LLM reasoning — thread_id:%s  %d EDUs  batch_size=%d  context_window=±%d",
        thread_id, N, batch_size, context_window,
    )

    # Counters for summary
    total_argumentative = 0
    total_support       = 0
    total_attack        = 0
    total_neutral       = 0
    total_pacs          = 0

    # ── Collect all EDU dicts in order across turns ───────────────────────────
    all_edu_dicts: list[dict] = []
    for turn in conversation:
        if turn.get("deleted"):
            continue
        for edu in turn.get("edus", []):
            if isinstance(edu, dict):
                all_edu_dicts.append(edu)

    # ── Process in EDU-level batches ──────────────────────────────────────────
    # results_map: global_idx → result dict {target_is_argumentative, relations}
    results_map: dict[int, dict] = {}
    total_batches = (len(all_edu_dicts) + batch_size - 1) // batch_size

    for batch_start in range(0, len(all_edu_dicts), batch_size):
        batch_edus = all_edu_dicts[batch_start: batch_start + batch_size]
        batch_no   = batch_start // batch_size + 1

        log.info(
            "  thread_id=%s  batch [%d/%d]  EDUs=%d",
            thread_id, batch_no, total_batches, len(batch_edus),
        )

        # Build batch_items — one per target EDU in this batch
        batch_items: list[dict] = []
        for pos, edu in enumerate(batch_edus):
            g_idx        = edu.get("global_idx", 0)
            target_text  = edu.get("text", "")
            pacs         = edu.get("pacs", [])
            ctx_b, ctx_a = get_context_window(all_edu_texts, g_idx, context_window)

            batch_items.append({
                "batch_position": pos,
                "global_idx":     g_idx,
                "target_text":    target_text,
                "context_before": ctx_b,
                "context_after":  ctx_a,
                "source_texts":   [p["source_text"] for p in pacs],
                "pacs":           pacs,
            })

        t0            = time.time()
        batch_results = reason_for_edu_batch(
            batch_items   = batch_items,
            temp1         = temp1,
            temp2         = temp2,
            temp_tiebreak = temp_tiebreak,
        )
        elapsed = time.time() - t0
        log.info("  Batch %d done in %.1fs", batch_no, elapsed)

        for pos, result in enumerate(batch_results):
            g_idx = batch_items[pos]["global_idx"]
            results_map[g_idx] = result

    # ── Re-attach results back onto the conversation structure ─────────────────
    enriched_turns = []

    for turn in conversation:
        if turn.get("deleted"):
            enriched_turns.append(turn)
            continue

        enriched_edus = []

        for edu in turn.get("edus", []):
            if not isinstance(edu, dict):
                enriched_edus.append(edu)
                continue

            g_idx  = edu.get("global_idx", 0)
            pacs   = edu.get("pacs", [])
            result = results_map.get(g_idx)

            if result is None:
                enriched_edus.append({
                    **edu,
                    "target_is_argumentative": None,
                    "reasoning_skipped":       True,
                })
                continue

            relations = result["relations"]

            # Infer argumentativeness from relations — post-processing approach:
            # an EDU is argumentative if any of its PAC pairs are support or attack
            has_non_neutral = any(r in ("support", "attack") for r in relations)
            is_arg = has_non_neutral

            enriched_pacs = []
            for i, pac in enumerate(pacs):
                rel = relations[i] if i < len(relations) else "neutral"
                enriched_pacs.append({
                    **pac,
                    "relation":                rel,
                    "target_is_argumentative": is_arg,
                })

            rel_counts = Counter(p.get("relation", "neutral") for p in enriched_pacs)

            if is_arg:
                total_argumentative += 1
            total_support += rel_counts.get("support", 0)
            total_attack  += rel_counts.get("attack",  0)
            total_neutral += rel_counts.get("neutral", 0)
            total_pacs    += len(enriched_pacs)

            log.info(
                "  [EDU %03d] arg=%s  support=%d  attack=%d  neutral=%d",
                g_idx, is_arg,
                rel_counts.get("support", 0),
                rel_counts.get("attack",  0),
                rel_counts.get("neutral", 0),
            )

            enriched_edus.append({
                **edu,
                "pacs":                    enriched_pacs,
                "target_is_argumentative": is_arg,
                "relation_summary": {
                    "support": rel_counts.get("support", 0),
                    "attack":  rel_counts.get("attack",  0),
                    "neutral": rel_counts.get("neutral", 0),
                },
            })

        enriched_turns.append({**turn, "edus": enriched_edus})

    reasoning_summary = {
        "total_edus":            N,
        "argumentative_edus":    total_argumentative,
        "non_argumentative":     N - total_argumentative,
        "total_pacs_classified": total_pacs,
        "support_relations":     total_support,
        "attack_relations":      total_attack,
        "neutral_relations":     total_neutral,
        "batch_size":            batch_size,
        "context_window":        context_window,
        "temp1":                 temp1,
        "temp2":                 temp2,
        "temp_tiebreak":         temp_tiebreak,
    }

    log.info(
        "Reasoning complete — thread_id=%s  %d/%d argumentative EDUs  "
        "support=%d  attack=%d  neutral=%d",
        thread_id, total_argumentative, N,
        total_support, total_attack, total_neutral,
    )

    return {
        **pac_output,
        "conversation":      enriched_turns,
        "reasoning_summary": reasoning_summary,
    }


def load_completed_thread_ids(output_path: Path) -> set[str]:
    """
    Scan an existing output JSONL file and return the set of thread_ids that
    have already been processed (i.e. have a reasoning_summary present).
    Returns an empty set if the file does not exist.
    """
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    for _, record in read_jsonl(output_path):
        thread_id = record.get("thread_id")
        # Only count as done if reasoning actually ran (not just copied through)
        if thread_id and "reasoning_summary" in record:
            completed.add(thread_id)

    log.info("Resume: found %d already-completed conversations in %s",
             len(completed), output_path)
    return completed


def run_on_jsonl(input_path, output_path, batch_size, context_window,
                 temp1, temp2, temp_tiebreak):
    total    = count_lines(input_path)
    out_path = Path(output_path)

    # ── Resume: scan output file for already-processed thread_ids
    completed = load_completed_thread_ids(out_path)
    n_skip    = 0

    if completed:
        log.info(
            "Resuming — %d/%d conversations already done",
            len(completed), total,
        )
    else:
        log.info("LLM reasoning starting fresh — %d conversations", total)

    # ── Open in append mode when resuming, write mode when starting fresh ──────
    file_mode = "a" if completed else "w"
    log.info("Output file mode=%s → %s", file_mode, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, file_mode, encoding="utf-8") as out_f:
        for line_no, conv in read_jsonl(input_path):
            thread_id = conv.get("thread_id", "")

            if thread_id in completed:
                log.debug("Skipping already-completed thread_id=%s", thread_id)
                n_skip += 1
                continue

            log_progress(line_no, total, thread_id, "REASON", log)
            result = run_reasoning(conv, batch_size, context_window,
                                   temp1, temp2, temp_tiebreak)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()   # flush after each conversation — safe against crashes

    log.info(
        "Finished. Skipped=%d  Processed=%d  Output → %s",
        n_skip, total - n_skip, out_path,
    )


# CLI
def main() -> None:

    global MODEL, ENABLE_THINKING, OLLAMA_URL, THINKING_BUDGET

    parser = argparse.ArgumentParser(
        description="Iterative LLM reasoning for argumentative relation classification"
    )
    parser.add_argument("--input",          "-i", help="PAC selector JSON output")
    parser.add_argument("--output",         "-o", help="Path to save enriched JSON output")
    parser.add_argument("--model",          "-m", default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"Ollama API endpoint (default: {OLLAMA_URL})"
                        )
    parser.add_argument("--batch-size",     type=int,   default=10,
                        help="Number of target EDUs per LLM call (default: 10)")
    parser.add_argument("--context-window", type=int,   default=2,
                        help="±N EDUs of local context added to prompt (default: 2)")
    parser.add_argument("--temp1",          type=float, default=0.1,
                        help="Temperature for call 1 (default: 0.1)")
    parser.add_argument("--temp2",          type=float, default=0.2,
                        help="Temperature for call 2 (default: 0.2)")
    parser.add_argument("--temp-tiebreak",  type=float, default=0.15,
                        help="Temperature for tie-breaking call (default: 0.15)")
    parser.add_argument("--no-think",       action="store_true",
                        help="Disable thinking mode (faster but less reasoning depth)")
    parser.add_argument("--thinking-budget", default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Thinking budget for models that support it, e.g. gpt-oss "
                             "(none/low/medium/high).")
    parser.add_argument("--verbose",        "-v", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--log-file", "-l", default="llm_reasoner.log",
                        help="Log file path (default: llm_reasoner.log)")
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    MODEL = args.model
    OLLAMA_URL = args.ollama_url
    ENABLE_THINKING = not args.no_think
    THINKING_BUDGET = args.thinking_budget

    log.info("Model=%s  ollama_url=%s  thinking=%s  thinking_budget=%s  batch_size=%d  context_window=%d",
             MODEL, OLLAMA_URL, ENABLE_THINKING, THINKING_BUDGET, args.batch_size, args.context_window
             )

    # Load input — fall back to running step 2's sample inline
    if args.input:
        run_on_jsonl(Path(args.input), Path(args.output),
                     args.batch_size, args.context_window,
                     args.temp1, args.temp2, args.temp_tiebreak)
    else:
        log.info("No --input — chaining from built-in sample")
        import extract_edu as ee
        import pac_selector  as ps
        with JSONLWriter(Path(args.output)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("thread_id",""), "REASON", log)
                edu = ee.extract_edus(conv)
                pac = ps.select_all_pacs(edu)
                writer.write(run_reasoning(pac, args.batch_size, args.context_window,
                                           args.temp1, args.temp2, args.temp_tiebreak))


if __name__ == "__main__":
    main()