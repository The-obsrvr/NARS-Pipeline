import logging
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from sys_utils import (
read_jsonl, JSONLWriter, log_progress, count_lines, SAMPLE_CONVERSATIONS, setup_logging
    )


# Logging
log = logging.getLogger("pac_selector")

#Initialization
DEFAULT_MODEL     = "all-MiniLM-L6-v2"
DEFAULT_K         = 25
DEFAULT_THRESHOLD = 0.45
DEFAULT_TOP_N     = 10
DEFAULT_WINDOW    = 5

_ENCODER = None

def get_encoder(model_name: str):
    global _ENCODER
    if _ENCODER is None:
        log.info("Loading sentence-transformer: %s", model_name)
        _ENCODER = SentenceTransformer(model_name)
    return _ENCODER


# EDU index
def build_edu_index(conversation: list[dict]) -> list[dict]:
    """
    Flatten all EDUs from all (non-deleted) turns into a global ordered list.
    Each entry carries:
        global_idx  – position in the full document (0-based)
        post_id     – turn this EDU belongs to
        speaker_id  – speaker
        local_idx   – position within the turn (0-based)
        text        – the EDU string
    """
    index: list[dict] = []
    for turn in conversation:
        if turn.get("deleted"):
            # we skip deleted
            continue
        post_id    = turn.get("post_id", "")
        speaker_id = turn.get("speaker_id", "")
        for local_idx, edu_text in enumerate(turn.get("edus", [])):
            index.append({
                "global_idx": int(len(index)),
                "post_id":    post_id,
                "speaker_id": speaker_id,
                "local_idx":  int(local_idx),
                "text":       edu_text,
            })
    return index


# Embedding
def embed_edus(edu_index: list[dict], model_name: str) -> np.ndarray:
    """
    Encode all EDU texts and return a (N, D) float32 matrix.
    """
    encoder = get_encoder(model_name)
    texts = [e["text"] for e in edu_index]
    log.info("Encoding %d EDUs…", len(texts))
    t0 = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log.info("Encoding done in %.2fs  shape=%s", time.time() - t0, embeddings.shape)
    return embeddings.astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Returns (N, N) cosine similarity matrix (dot product between embeddings)
    """
    return embeddings @ embeddings.T   # shape (N, N)


def select_pacs_for_edu(
    target_idx:   int,
    sim_row:      np.ndarray,   # cosine similarities of target against all EDUs
    edu_index:    list[dict],
    k:            int,
    threshold:    float,
    top_n:        int,
    window:       int,
) -> list[dict]:
    """
    Select PACs for a single target EDU.

    Returns a list of dicts, each with:
        source_global_idx  – global index of the candidate EDU
        source_text        – candidate EDU text
        cosine_sim         – similarity score (None for window-only additions)
        selection_type     – "semantic" | "contextual"
    """
    N = len(edu_index)

    # 1. kNN candidates based on highest similarity (exclude self)
    all_indices  = np.argsort(sim_row)[::-1]
    knn_indices  = [i for i in all_indices if i != target_idx][:k]

    # ── 2. Cosine-similarity filter
    filtered = [
        i for i in knn_indices
        if sim_row[i] >= threshold
    ]

    # ── 3. Directionality: source must come AFTER target in the text ──────────
    filtered = [i for i in filtered if i > target_idx]

    # ── 4. Top-N semantic PACs ────────────────────────────────────────────────
    semantic_pacs_idx = filtered[:top_n]
    semantic_set      = set(semantic_pacs_idx)

    # ── 5. Contextual window: up to `window` subsequent EDUs not yet selected ─
    contextual_pacs_idx: list[int] = []
    cursor = target_idx + 1
    while len(contextual_pacs_idx) < window and cursor < N:
        if cursor not in semantic_set:
            contextual_pacs_idx.append(cursor)
        # update cursor to move to the next EDU
        cursor += 1

    # ── 6. Assemble result ────────────────────────────────────────────────────
    pacs: list[dict] = []
    for src_idx in semantic_pacs_idx:
        pacs.append({
            "source_global_idx": int(src_idx),
            "source_post_id":    edu_index[src_idx]["post_id"],
            "source_speaker_id": edu_index[src_idx]["speaker_id"],
            "source_local_idx":  int(edu_index[src_idx]["local_idx"]),
            "source_text":       edu_index[src_idx]["text"],
            "cosine_sim":        float(round(float(sim_row[src_idx]), 4)),
            "selection_type":    "semantic",
        })

    # add the local pairs that are likely!
    for src_idx in contextual_pacs_idx:
        pacs.append({
            "source_global_idx": int(src_idx),
            "source_post_id":    edu_index[src_idx]["post_id"],
            "source_speaker_id": edu_index[src_idx]["speaker_id"],
            "source_local_idx":  int(edu_index[src_idx]["local_idx"]),
            "source_text":       edu_index[src_idx]["text"],
            "cosine_sim":        float(round(float(sim_row[src_idx]), 4)),
            "selection_type":    "contextual",
        })

    return pacs


# Main PAC selection pass
def select_all_pacs(
    edu_output:   dict,
    model_name:   str   = DEFAULT_MODEL,
    k:            int   = DEFAULT_K,
    threshold:    float = DEFAULT_THRESHOLD,
    top_n:        int   = DEFAULT_TOP_N,
    window:       int   = DEFAULT_WINDOW,
) -> dict:
    """
    build EDU index → embed → similarity matrix → PAC selection per EDU.
    Returns an enriched version of edu_output with 'pacs' attached to each EDU
    and a top-level 'pac_summary'.
    """
    conversation = edu_output.get("conversation", [])
    edu_index    = build_edu_index(conversation)
    N            = len(edu_index)
    conv_id = edu_output.get("conv_id", "?")

    if N == 0:
        log.warning("No EDUs found — returning input unchanged")
        return edu_output

    # Embed
    embeddings = embed_edus(edu_index, model_name)

    # Full (N×N) cosine similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings)

    # Per-EDU PAC selection
    log.info("Selecting PACs  (k=%d  threshold=%.2f  top_n=%d  window=%d)…",
             k, threshold, top_n, window)

    # Build a lookup: global_idx → list position in edu_index
    total_pacs, pac_counts, enriched_flat = 0, [], []

    for g_idx, edu_entry in enumerate(edu_index):
        pacs = select_pacs_for_edu(
            target_idx = g_idx,
            sim_row    = sim_matrix[g_idx],
            edu_index  = edu_index,
            k          = k,
            threshold  = threshold,
            top_n      = top_n,
            window     = window,
        )
        total_pacs  += len(pacs)
        pac_counts.append(len(pacs))
        enriched_flat.append({**edu_entry, "pacs": pacs, "pac_count": len(pacs)})
        log.debug(
            "  EDU[%03d] %-50s → %d PACs (%d sem / %d ctx)",
            g_idx,
            edu_entry["text"][:48],
            len(pacs),
            sum(1 for p in pacs if p["selection_type"] == "semantic"),
            sum(1 for p in pacs if p["selection_type"] == "contextual"),
        )

    avg_pacs = total_pacs / N if N else 0
    log.info(
        "PAC selection done — %d total PACs  avg=%.1f/EDU  min=%d  max=%d",
        total_pacs, avg_pacs, min(pac_counts), max(pac_counts),
    )

    # ── Re-embed PAC lists back into the conversation structure ───────────────
    pac_map: dict[tuple, list] = {
        (e["post_id"], e["local_idx"]): e["pacs"]
        for e in enriched_flat
    }
    pac_count_map: dict[tuple, int] = {
        (e["post_id"], e["local_idx"]): e["pac_count"]
        for e in enriched_flat
    }
    # map back to the segment id (post + edu (local) id)
    g_idx_map = {(e["post_id"], e["local_idx"]): int(e["global_idx"]) for e in enriched_flat}

    enriched_turns = []
    for turn in conversation:
        if turn.get("deleted"):
            enriched_turns.append(turn)
            continue
        post_id = turn.get("post_id", "")
        enriched_edus = [
            {
                "global_idx": g_idx_map.get((post_id, li), 0),
                "text": edu_text,
                "pacs": pac_map.get((post_id, li), []),
                "pac_count": pac_count_map.get((post_id, li), 0),
                }
            for li, edu_text in enumerate(turn.get("edus", []))
            ]
        enriched_turns.append({**turn, "edus": enriched_edus})

    return {
        **edu_output,
        "conversation": enriched_turns,
        "pac_summary": {
            "model":           model_name,
            "k":               k,
            "threshold":       threshold,
            "top_n":           top_n,
            "window":          window,
            "total_edus":      N,
            "total_pacs":      total_pacs,
            "avg_pacs_per_edu": round(avg_pacs, 2),
            "min_pacs":        int(min(pac_counts)),
            "max_pacs":        int(max(pac_counts)),
        },
    }


# JSONL write
def run_on_jsonl(
    input_path:  Path,
    output_path: Path,
    model_name:  str   = DEFAULT_MODEL,
    k:           int   = DEFAULT_K,
    threshold:   float = DEFAULT_THRESHOLD,
    top_n:       int   = DEFAULT_TOP_N,
    window:      int   = DEFAULT_WINDOW,
) -> None:
    # Preload the encoder once before the loop
    get_encoder(model_name)
    total = count_lines(input_path)
    log.info("PAC selection starting — %d conversations", total)

    with JSONLWriter(output_path) as writer:
        for line_no, conv in read_jsonl(input_path):
            log_progress(line_no, total, conv.get("conv_id",""), "PAC", log)
            result = select_all_pacs(conv, model_name, k, threshold, top_n, window)
            writer.write(result)

    log.info("Finished. Output → %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select Probable Argumentative Candidates (PACs) for each EDU"
    )
    parser.add_argument("--input",     "-i", help="Path to EDU extractor JSON output")
    parser.add_argument("--output",    "-o", help="Path to save enriched JSON output")
    parser.add_argument("--model",     "-m", default=DEFAULT_MODEL,
                        help=f"Sentence-transformer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--k",         type=int,   default=DEFAULT_K,
                        help=f"kNN neighbourhood size (default: {DEFAULT_K})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--top-n",     type=int,   default=DEFAULT_TOP_N,
                        help=f"Max semantic PACs per EDU (default: {DEFAULT_TOP_N})")
    parser.add_argument("--window",    type=int,   default=DEFAULT_WINDOW,
                        help=f"Contextual window size (default: {DEFAULT_WINDOW})")
    parser.add_argument("--verbose",   "-v", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--log-file", "-l", default="edu_extractor.log",
                        help="Log file path (default: edu_extractor.log)")
    args = parser.parse_args()

    setup_logging(args.verbose, args.log_file)

    # Load input
    if args.input:
        run_on_jsonl(Path(args.input), Path(args.output),
                     args.model, args.k, args.threshold, args.top_n, args.window
                     )
    else:
        # Default to sample conversation
        log.info("No --input — using built-in sample")
        # Import edu_extractor to generate EDU output on the fly
        import extract_edu as ee
        get_encoder(args.model)
        with JSONLWriter(Path(args.output)) as writer:
            for i, conv in enumerate(SAMPLE_CONVERSATIONS, 1):
                log_progress(i, len(SAMPLE_CONVERSATIONS), conv.get("conv_id", ""), "PAC", log)
                edu_conv = ee.extract_edus(conv)
                writer.write(select_all_pacs(edu_conv, args.model,
                                             args.k, args.threshold, args.top_n, args.window
                                             ))


if __name__ == "__main__":
    main()
