"""Scores every {"en","hi"} step in a translated dataset by cross-lingual
embedding similarity (English source vs. Hindi translation), via the same
vLLM-backed `embed()` used everywhere else in this pipeline
(`data_gen/embeddings.py` - no separate embedding client here).

This is a coarse, topic-level correctness signal, not a precision one: it
reliably flags a translation that's about a different thing entirely (the
hallucination failure mode seen in some Fireworks batch output - see
stitch_fireworks_output.py's flagged-rows split), but can miss a
translation that's subtly wrong (a flipped sign, a swapped number, one
mistranslated term) while still reading as fluent, on-topic Hindi, since
embeddings track overall meaning, not fact-level accuracy. Treat low scores
as "worth a look," not as ground truth on correctness - and a real,
above-threshold score is not proof of correctness either.

Usage:
    uv run python -m data_gen.score_translation_similarity \
        --in_file reasoning_hi_from_fireworks.jsonl \
        --out_file reasoning_hi_from_fireworks_scored.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from data_gen import config
from data_gen.embeddings import embed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("score_translation_similarity")


def score_file(in_file: Path, out_file: Path, batch_size: int) -> None:
    """Scores every step's en/hi similarity and writes {"doc_id",
    "step_index", "en", "hi", "similarity"} rows, one per step.

    Args:
        in_file: A {"id", "steps": [{"en","hi"}, ...]} dataset.
        out_file: Where to write per-step similarity rows.
        batch_size: Passed through to embed() (per-call forward-pass cap).
    """
    docs = [json.loads(line) for line in open(in_file, encoding="utf-8")]
    steps = [(doc["id"], i, step) for doc in docs for i, step in enumerate(doc["steps"])]
    log.info(f"Loaded {len(docs)} documents, {len(steps)} steps")

    log.info("Embedding English sources...")
    en_vectors = embed(
        [s["en"] for _, _, s in steps],
        model=config.EMBEDDING_MODEL,
        backend=config.EMBEDDING_BACKEND,
        base_url=config.EMBEDDING_BASE_URL,
        batch_size=batch_size,
    )
    log.info("Embedding Hindi translations...")
    hi_vectors = embed(
        [s["hi"] for _, _, s in steps],
        model=config.EMBEDDING_MODEL,
        backend=config.EMBEDDING_BACKEND,
        base_url=config.EMBEDDING_BASE_URL,
        batch_size=batch_size,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for (doc_id, step_index, step), en_vec, hi_vec in zip(steps, en_vectors, hi_vectors):
            similarity = float(en_vec @ hi_vec)
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "step_index": step_index,
                        "en": step["en"],
                        "hi": step["hi"],
                        "similarity": similarity,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log.info(f"Wrote {len(steps)} scored rows to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_file", type=Path, required=True)
    parser.add_argument("--out_file", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=config.EMBED_BATCH_SIZE)
    args = parser.parse_args()
    score_file(args.in_file, args.out_file, args.batch_size)


if __name__ == "__main__":
    main()
