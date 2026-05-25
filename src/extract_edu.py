import json
import re
import time
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional
import requests

from sys_utils import (
    read_jsonl, JSONLWriter, log_progress, count_lines,
    SAMPLE_CONVERSATIONS, setup_logging
    )

# ─── Logging — configured after CLI args are parsed ───────────────────────────

log = logging.getLogger("edu_extractor")

# ─── Constants ────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:27b"
TIMEOUT = 500 # 2 minutes per paragraph call
MAX_RETRIES = 2

# ~4 chars per token; 2000 tokens × 4 = 8000 chars
MAX_TOKENS = 3000
CHARS_PER_TOKEN = 4

# EDU content filter
DELETED_RE = re.compile(r'\[removed\]|\[deleted\]', re.IGNORECASE)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a discourse analysis expert. Segment the given text into Elementary Discourse Units (EDUs).

RULES:
- Each EDU contains exactly one proposition or claim
- Split at clause boundaries and connectives (but, because, although, however, so, therefore)
- Preserve original wording exactly — do not paraphrase
- If a clause cannot be split further, return it as a single EDU

OUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:
{"edus": ["EDU 1", "EDU 2", "EDU 3"]}

EXAMPLE:
Input: I love coffee, but I avoid it because it affects my sleep. Climate change is real and it threatens our future.
Output: {"edus": ["I love coffee", "but I avoid it", "because it affects my sleep", "Climate change is real", "and it threatens our future"]}
"""


# ─── EDU filter ───────────────────────────────────────────────────────────────

def filter_edus(edus: list[str]) -> tuple[list[str], int]:
    """
    Remove EDUs that contain [deleted] or [removed] as a substring.
    Also strips empty/whitespace-only entries.
    Returns (filtered_list, n_removed).
    """
    result: list[str] = []
    removed: int = 0
    for edu in edus:
        if not edu.strip():
            removed += 1
            continue
        if DELETED_RE.search(edu):
            log.debug("  Dropping EDU: %.70s", edu)
            removed += 1
            continue
        result.append(edu)
    return result, removed


# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate_edus(edus: list[str]) -> list[str]:
    """
    Remove duplicate EDUs. Order-preserving; keeps first occurrence;
    comparison is case-insensitive.
    """
    seen: set[str] = set()
    result: list[str] = []
    for edu in edus:
        key = edu.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(edu)
    return result


# ─── Ollama client ────────────────────────────────────────────────────────────

def call_ollama(user_text: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Send a prompt to Ollama; return raw response string or None on failure."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            ],
        "stream": False,
        "format": "json",  # grammar-constrained JSON output
        "think": False,  # disable Qwen3 thinking mode at API level
        "options": {
            "temperature": 0.1,
            "num_predict": 8192,
            "num_ctx": 16432,
            },
        }
    for attempt in range(1, retries + 1):
        try:
            log.debug("Ollama request attempt %d/%d  chars=%d",
                      attempt, retries, len(user_text)
                      )
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            # Strip Qwen3 thinking block in case it slips through format=json
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            log.debug("Raw output (first 300 chars): %.300s", raw)
            return raw
        except requests.exceptions.ConnectionError:
            log.error("Cannot reach Ollama at %s — is it running?", OLLAMA_URL)
            return None
        except requests.exceptions.Timeout:
            log.warning("Timeout (attempt %d/%d)", attempt, retries)
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
        except Exception as exc:
            log.error("Unexpected error: %s", exc)
            return None
    return None


# ─── EDU parser ───────────────────────────────────────────────────────────────

def parse_edus(raw: str, fallback_text: str = "") -> list[str]:
    """
    Parse model output into a flat list of EDU strings.
    Falls back to the full input text as a single EDU on failure.
    """
    if not raw or not raw.strip():
        log.warning("Empty response — using text as single EDU")
        return [fallback_text.strip()] if fallback_text.strip() else []

    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "edus" in data:
            return [e.strip() for e in data["edus"] if e.strip()]
        if isinstance(data, list):
            return [e.strip() for e in data if isinstance(e, str) and e.strip()]
    except json.JSONDecodeError:
        pass

    try:
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            data = json.loads(m.group())
            if isinstance(data, dict) and "edus" in data:
                return [e.strip() for e in data["edus"] if e.strip()]
    except json.JSONDecodeError:
        pass

    log.warning("JSON parse failed — using text as single EDU")
    return [fallback_text.strip()] if fallback_text.strip() else []


# ─── EDU → turn attribution ───────────────────────────────────────────────────

def attribute_edus_to_turns(
        edus: list[str],
        batch: list[tuple[int, str]],
        ) -> dict[int, list[str]]:
    """
    Match each EDU to the turn whose text contains it (substring match).

    Strategy:
      For each EDU, check which turn text contains it as a substring
      (case-insensitive). Assign it to the first matching turn in batch order.
      If no turn contains the EDU (e.g. the model rephrased slightly),
      assign it to the turn with the highest character overlap ratio.

    Falls back to distributing unmatched EDUs proportionally across turns
    if overlap scoring also fails.

    Returns dict of turn_idx → [edus belonging to that turn].
    """
    result: dict[int, list[str]] = {t_idx: [] for t_idx, _ in batch}

    # Pre-normalise turn texts for matching
    turn_texts_lower = [(t_idx, txt.lower()) for t_idx, txt in batch]

    for edu in edus:
        edu_lower = edu.lower().strip()

        # 1. Exact substring match
        matched = False
        for t_idx, txt_lower in turn_texts_lower:
            if edu_lower in txt_lower:
                result[t_idx].append(edu)
                matched = True
                break

        if matched:
            continue

        # 2. Best overlap — find turn with most characters in common
        best_score = -1
        best_idx = batch[0][0]  # default to first turn
        for t_idx, txt_lower in turn_texts_lower:
            # Count shared words as a simple overlap score
            edu_words = set(edu_lower.split())
            turn_words = set(txt_lower.split())
            score = len(edu_words & turn_words)
            if score > best_score:
                best_score = score
                best_idx = t_idx

        log.debug("  EDU not found in any turn — assigned to turn %d by overlap: %.50s",
                  best_idx, edu
                  )
        result[best_idx].append(edu)

    return result


# ─── Paragraph (turn-batch) builder ───────────────────────────────────────────

def build_turn_batches(
        turns: list[dict],
        max_tokens: int = MAX_TOKENS,
        ) -> list[list[tuple[int, str]]]:
    """
    Group consecutive turns into batches where the combined text stays within
    max_tokens. Each batch is a list of (turn_idx, turn_text) pairs.
    A single turn exceeding max_tokens forms its own batch.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    for t_idx, turn in enumerate(turns):
        text = turn.get("text", "").strip()
        if not text:
            continue
        turn_chars = len(text)
        if current and current_chars + turn_chars > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append((t_idx, text))
        current_chars += turn_chars

    if current:
        batches.append(current)

    log.debug("Built %d turn-batch(es) from %d turns", len(batches), len(turns))
    return batches


def build_prompt(batch: list[tuple[int, str]]) -> str:
    """Concatenate all turn texts into one plain paragraph for the LLM."""
    return " ".join(text for _, text in batch)


# ─── Per-conversation logic ───────────────────────────────────────────────────

def extract_edus(conv: dict) -> dict:
    """
    Process one conversation by merging turns into token-bounded paragraphs.

    Strategy:
      1. Group consecutive turns into batches of ≤ MAX_TOKENS combined tokens.
      2. Concatenate turn texts into a single plain paragraph per batch.
      3. One LLM call per batch — model returns {"edus": [...]} for the paragraph.
      4. Post-generation: attribute each EDU to the turn whose text contains it
         (substring match → word-overlap fallback). No turn-labelling in prompt.
      5. Deduplicate and filter per turn.

    The LLM only sees a plain paragraph and returns a simple EDU list —
    no structured turn indexing required from the model.
    """
    turns = [t for t in conv.get("conversation", []) if not t.get("deleted")]
    total = len(turns)
    thread_id = conv.get("thread_id", "?")

    log.info("thread_id=%s  turns=%d  title=%.55s",
             thread_id, total, conv.get("title", "")
             )

    if total == 0:
        return {**conv,
                "conversation": [],
                "edu_summary": {"total_turns": 0, "total_edus": 0}
                }

    batches = build_turn_batches(turns, MAX_TOKENS)
    n_batches = len(batches)
    log.info("  %d batch(es) of ≤%d tokens", n_batches, MAX_TOKENS)

    # Accumulator: turn_idx → EDU list
    turn_edus: dict[int, list[str]] = {i: [] for i in range(total)}
    t0 = time.time()

    for b_idx, batch in enumerate(batches, 1):
        t_indices = [t for t, _ in batch]
        approx_tokens = sum(len(txt) for _, txt in batch) // CHARS_PER_TOKEN
        log.info("  batch [%d/%d]  turns=%s  ~%d tokens",
                 b_idx, n_batches, t_indices, approx_tokens
                 )

        paragraph = build_prompt(batch)
        raw = call_ollama(paragraph)
        edus = parse_edus(raw, fallback_text=paragraph)

        log.info("    → %d EDUs before attribution", len(edus))

        # Attribute EDUs back to their originating turns by text matching
        if len(batch) == 1:
            # Single turn — all EDUs belong to it
            t_idx = batch[0][0]
            turn_edus[t_idx].extend(edus)
        else:
            attributed = attribute_edus_to_turns(edus, batch)
            for t_idx, t_edus in attributed.items():
                turn_edus[t_idx].extend(t_edus)

    elapsed = time.time() - t0

    # ── Assemble enriched turn dicts ──────────────────────────────────────────
    enriched: list[dict] = []
    grand_total: int = 0

    for t_idx, turn in enumerate(turns):
        deduped = deduplicate_edus(turn_edus[t_idx])
        filtered, n_drop = filter_edus(deduped)

        if n_drop:
            log.debug("  turn %d: filtered %d EDU(s)", t_idx, n_drop)

        log.info("  turn [%d/%d] %s  post_id=%s  → %d EDUs",
                 t_idx + 1, total,
                 turn.get("speaker_id", "?"),
                 turn.get("post_id", ""),
                 len(filtered)
                 )

        enriched.append({**turn, "edus": filtered, "edu_count": len(filtered)})
        grand_total += len(filtered)

    log.info("thread_id=%s  done — %d EDUs across %d turns in %.1fs",
             thread_id, grand_total, total, elapsed
             )

    return {
        **conv,
        "conversation": enriched,
        "edu_summary": {"total_turns": total, "total_edus": grand_total},
        }


# ─── JSONL batch runner ───────────────────────────────────────────────────────

def load_completed_thread_ids(output_path: Path) -> set[str]:
    """
    Scan an existing output JSONL file and return the set of thread_ids that
    have already been successfully processed (have a non-empty edu_summary).
    Returns an empty set if the file does not exist.
    """
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    for _, record in read_jsonl(output_path):
        thread_id = record.get("thread_id")
        summary = record.get("edu_summary", {})
        # Only count as done if EDU extraction actually ran
        if thread_id and "total_edus" in summary:
            completed.add(thread_id)

    log.info("Resume: found %d already-completed conversations in %s",
             len(completed), output_path
             )
    return completed


def run_on_jsonl(input_path: Path, output_path: Path) -> None:
    total = count_lines(input_path)
    out_path = Path(output_path)

    # ── Resume: scan output file for already-processed thread_ids ───────────────
    completed = load_completed_thread_ids(out_path)
    n_skip = 0

    if completed:
        log.info(
            "Resuming — %d/%d conversations already done",
            len(completed), total,
            )
    else:
        log.info("EDU extraction starting fresh — %d conversations in %s",
                 total, input_path
                 )

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

            log_progress(line_no, total, thread_id, "EDU", log)
            result = extract_edus(conv)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()  # flush after each conversation — safe against crashes

    log.info(
        "Finished. Skipped=%d  Processed=%d  Output → %s",
        n_skip, total - n_skip, out_path,
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:

    global MODEL, TIMEOUT, MAX_TOKENS

    parser = argparse.ArgumentParser(description="EDU extractor — JSONL in / JSONL out")
    parser.add_argument("--input", "-i",
                        help="Input JSONL (one conversation per line)"
                        )
    parser.add_argument("--output", "-o", default="edu_output.jsonl",
                        help="Output JSONL (default: edu_output.jsonl)"
                        )
    parser.add_argument("--model", "-m", default=MODEL,
                        help=f"Ollama model name (default: {MODEL})"
                        )
    parser.add_argument("--log-file", "-l", default="edu_extractor.log",
                        help="Log file path (default: edu_extractor.log)"
                        )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens per paragraph (default: {MAX_TOKENS})"
                        )
    parser.add_argument("--timeout", type=int, default=TIMEOUT,
                        help=f"Per-request timeout seconds (default: {TIMEOUT})"
                        )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging"
                        )
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    MODEL = args.model
    TIMEOUT = args.timeout
    MAX_TOKENS = args.max_tokens

    log.info("Model=%s  max_tokens=%d  timeout=%ds  log=%s",
             MODEL, MAX_TOKENS, TIMEOUT, args.log_file
             )

    if args.input:
        run_on_jsonl(Path(args.input), Path(args.output))
    else:
        log.info("No --input — using built-in sample")
        with JSONLWriter(Path(args.output)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS),
                             conv.get("thread_id", ""), "EDU", log
                             )
                writer.write(extract_edus(conv))


if __name__ == "__main__":
    main()
