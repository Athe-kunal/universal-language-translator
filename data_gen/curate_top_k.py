"""Curates the top-K {"en","hi"} steps by EN-HI embedding similarity (see
`score_translation_similarity.py`) into a final {"id", "steps": [...]}
dataset, filtering at the step level (a document keeps whatever steps of
its own make the cut - see stitch_fireworks_output.py's docstring for why
that's fine for this training format).

Sorts by (-similarity, doc_id, step_index) rather than similarity alone, so
the cut is deterministic even across exact score ties (floating-point
equality between two different steps is rare but not impossible) - this
module verifies that determinism itself (--verify runs the selection twice
independently and diffs the results) rather than assuming a stable sort is
enough on its own.

Usage:
    uv run python -m data_gen.curate_top_k \
        --scored_file reasoning_hi_from_fireworks_scored.jsonl \
        --k 100000 \
        --out reasoning_hi_final_100k.jsonl
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("curate_top_k")


def select_top_k(rows: list[dict], k: int) -> list[dict]:
    """Deterministically selects the k highest-similarity rows.

    Args:
        rows: Scored rows (from score_translation_similarity.py).
        k: Number of rows to keep.

    Returns:
        The top-k rows, sorted by (-similarity, doc_id, step_index) - a
        total order (no ties), so the result is the same regardless of the
        input list's original order.
    """
    return sorted(rows, key=lambda r: (-r["similarity"], r["doc_id"], r["step_index"]))[:k]


def group_into_docs(rows: list[dict]) -> list[dict]:
    """Groups selected steps back into {"id", "steps": [{"en","hi"}, ...]}
    documents, each doc's steps in their original step_index order.

    Args:
        rows: Selected rows (each with doc_id, step_index, en, hi).

    Returns:
        One record per doc_id that has at least one selected step.
    """
    docs: dict[str, list[tuple[int, dict]]] = {}
    for r in rows:
        docs.setdefault(r["doc_id"], []).append((r["step_index"], {"en": r["en"], "hi": r["hi"]}))
    return [
        {"id": doc_id, "steps": [step for _, step in sorted(indexed, key=lambda pair: pair[0])]}
        for doc_id, indexed in docs.items()
    ]


def _file_hash(rows: list[dict]) -> str:
    """Stable content hash of a list of doc records, independent of dict
    key insertion order (json.dumps with sort_keys makes that irrelevant).
    """
    h = hashlib.sha256()
    for doc in rows:
        h.update(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def verify_determinism(rows: list[dict], k: int) -> bool:
    """Runs the full select -> group pipeline twice independently (on
    reversed copies of the input, so a bug that depends on input order
    would actually show up) and checks the outputs are byte-identical.

    Args:
        rows: Scored rows, as loaded from disk.
        k: Number of rows to select.

    Returns:
        True if both runs produced identical output.
    """
    run_a = group_into_docs(select_top_k(list(rows), k))
    run_b = group_into_docs(select_top_k(list(reversed(rows)), k))
    hash_a, hash_b = _file_hash(run_a), _file_hash(run_b)
    log.info(f"determinism check: run_a={hash_a[:12]} run_b={hash_b[:12]} match={hash_a == hash_b}")
    return hash_a == hash_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored_file", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    log.info(f"Loading {args.scored_file}...")
    rows = [json.loads(line) for line in open(args.scored_file, encoding="utf-8")]
    log.info(f"Loaded {len(rows)} scored steps")

    if not verify_determinism(rows, args.k):
        log.error("Determinism check FAILED - selection differs depending on input order. Not writing output.")
        sys.exit(1)

    top = select_top_k(rows, args.k)
    docs = group_into_docs(top)
    cutoff = top[-1]["similarity"]
    log.info(f"Selected {len(top)} steps ({len(docs)} documents), similarity cutoff={cutoff:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
