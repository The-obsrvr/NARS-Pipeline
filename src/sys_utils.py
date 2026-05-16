
import json
import logging
import sys
from pathlib import Path
from typing import Generator, Iterable

import numpy as np

log = logging.getLogger("pipeline_io")


# ─── Readers ──────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> Generator[tuple[int, dict], None, None]:
    """
    Yield (line_number, conversation_dict) for every non-empty line in a
    JSONL file.  Malformed lines are logged and skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Line %d: JSON parse error (%s) — skipping", line_no, exc)


def count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file (for progress reporting)."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Writers
def write_jsonl(path: Path, conversations: Iterable[dict]) -> int:
    """
    Write an iterable of dicts to a JSONL file (one JSON object per line).
    Returns the number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            n += 1
    log.info("Wrote %d records to %s", n, path)
    return n


class JSONLWriter:
    """
    Streaming JSONL writer — keeps the file open across many conversations.
    Use as a context manager so the file is always closed on exit / error.

    Usage:
        with JSONLWriter(output_path) as writer:
            for conv in conversations:
                result = process(conv)
                writer.write(result)
    """

    def __init__(self, path: Path):
        self.path  = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._n    = 0

    def __enter__(self) -> "JSONLWriter":
        self._file = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, obj: dict) -> None:
        if self._file is None:
            raise RuntimeError("JSONLWriter must be used as a context manager")
        self._file.write(json.dumps(obj, ensure_ascii=False, cls=NumpyEncoder) + "\n")
        self._file.flush()   # flush after each record — safe against crashes
        self._n += 1

    def __exit__(self, *_) -> None:
        if self._file:
            self._file.close()
        log.info("Closed %s  (%d records written)", self.path, self._n)

    @property
    def records_written(self) -> int:
        return self._n


# ─── Progress helper ──────────────────────────────────────────────────────────

def log_progress(
    current:  int,
    total:    int,
    conv_id:  str = "",
    step:     str = "",
    logger:   logging.Logger = log,
) -> None:
    """Emit a consistent progress line that is easy to grep in logs."""
    pct    = 100.0 * current / total if total else 0.0
    id_str = f"  [{conv_id}]" if conv_id else ""
    step_s = f"[{step}] " if step else ""
    logger.info("%sConversation %d/%d (%.1f%%)%s", step_s, current, total, pct, id_str)


# ─── Built-in sample JSONL (two minimal conversations) ───────────────────────

SAMPLE_CONVERSATIONS: list[dict] = [
    {
        "thread_id": "t3_69cxuj_3",
        "conv_id":   "t3_69cxuj",
        "title":     "CMV: U.S. healthcare system",
        "conversation": [
            {
                "post_id":    "t3_69cxuj",
                "speaker_id": "Speaker 1",
                "text": (
                    "My biggest problem with Obamacare was the mandate. "
                    "I believe it is unconstitutional to tell people they must buy health care. "
                    "But there's a problem. "
                    "One can't have a program that provides health care for those with "
                    "pre-existing conditions without FORCING young people to buy into health insurance."
                ),
            },
            {
                "post_id":    "dh5mxav",
                "speaker_id": "Speaker 7",
                "text": (
                    "The US has incredibly high public funding of healthcare, more than Canada or the UK. "
                    "That's as a percent of GDP. "
                    "The US has a gold plated healthcare system, much of it paid for by tax dollars, "
                    "which fails to look after many of the people who really need healthcare."
                ),
            },
        ],
    },
    {
        "thread_id": "t3_abc123_1",
        "conv_id":   "t3_abc123",
        "title":     "CMV: Social media does more harm than good",
        "conversation": [
            {
                "post_id":    "t3_abc123",
                "speaker_id": "Speaker A",
                "text": (
                    "Social media is fundamentally harmful to society. "
                    "It spreads misinformation rapidly and reduces attention spans. "
                    "Studies show a strong correlation between heavy use and depression."
                ),
            },
            {
                "post_id":    "reply_001",
                "speaker_id": "Speaker B",
                "text": (
                    "Social media also enables marginalised communities to organise and find support. "
                    "The Arab Spring would not have happened without it. "
                    "The harm you describe comes from misuse, not from the platform itself."
                ),
            },
        ],
    },
]


def sample_jsonl_bytes() -> bytes:
    """Return the sample conversations as UTF-8 encoded JSONL bytes."""
    return b"\n".join(
        json.dumps(c, ensure_ascii=False).encode("utf-8")
        for c in SAMPLE_CONVERSATIONS
    )


def setup_logging(verbose: bool, log_file: str) -> None:
    """Initialise logging with a runtime-specified log file path."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )