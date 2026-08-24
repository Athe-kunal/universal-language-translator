"""Builds the final nested EN-HI training dataset from translated documents.

Reads fully-reconstructed documents (`translated_reasoning_25k.jsonl`) and
their per-unit translations (`translated_reasoning_25k_units.jsonl`),
segments each document's chain-of-thought into coherent reasoning steps via
`segment_steps.segment_document`, and writes one nested record per document —
question + an ordered list of {en, hi} steps — split into train/val at the
document level (never split a document's steps across both, and never split
by unit either, to avoid leaking a chain of thought across the boundary).

Pure post-processing: no new LLM calls, no re-translation.

Usage:
    uv run python -m data_gen.build_training_dataset
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from loguru import logger

from data_gen.segment_steps import (
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_STEP_TOKENS,
    DEFAULT_MIN_STEP_TOKENS,
    DEFAULT_MIN_UNITS_FOR_SEMANTIC,
    DEFAULT_SEMANTIC_PERCENTILE,
    UnitRecord,
    segment_document,
)

# "⟦N⟧" byte-fallbacks into 7 opaque tokens on ModernBERT; "<placeholder-N>" is 5 clean ones.
_PLACEHOLDER_RE = re.compile(r"⟦(\d+)⟧")


def convert_placeholders(text: str) -> str:
    """Rewrites chunking's "⟦N⟧" placeholders into the SFT-facing "<placeholder-N>" form.

    Args:
        text: Step text (en or hi) containing zero or more "⟦N⟧" placeholders.

    Returns:
        The same text with every placeholder rewritten.
    """
    return _PLACEHOLDER_RE.sub(lambda m: f"<placeholder-{m.group(1)}>", text)

DEFAULT_DOCS_FILE = Path("translated_reasoning_25k.jsonl")
DEFAULT_UNITS_FILE = Path("translated_reasoning_25k_units.jsonl")
DEFAULT_TRAIN_FILE = Path("reasoning_translation_train.jsonl")
DEFAULT_VAL_FILE = Path("reasoning_translation_val.jsonl")
DEFAULT_VAL_FRACTION = 0.05


def load_doc_metadata(docs_file: Path) -> dict[str, dict]:
    """Loads question/source for every fully-reconstructed document.

    Args:
        docs_file: Path to `translated_reasoning_25k.jsonl`.

    Returns:
        Map of doc_id -> {"question": str, "source": str}.
    """
    meta = {}
    with open(docs_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta[row["id"]] = {"question": row["question"], "source": row["source"]}
    return meta


def load_units_for_docs(units_file: Path, doc_ids: set[str]) -> dict[str, list[UnitRecord]]:
    """Loads and groups units, restricted to a set of fully-completed documents.

    Args:
        units_file: Path to `translated_reasoning_25k_units.jsonl`.
        doc_ids: Only units whose doc_id is in this set are kept.

    Returns:
        Map of doc_id -> UnitRecords sorted by unit_id (document order).
    """
    by_doc: dict[str, list[UnitRecord]] = {}
    with open(units_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["doc_id"] not in doc_ids:
                continue
            by_doc.setdefault(row["doc_id"], []).append(
                UnitRecord(
                    doc_id=row["doc_id"],
                    source=row["source"],
                    unit_id=row["unit_id"],
                    kind=row["kind"],
                    en=row["en"],
                    hi=row.get("hi"),
                )
            )
    for units in by_doc.values():
        units.sort(key=lambda u: u.unit_id)
    return by_doc


def is_val_doc(doc_id: str, val_fraction: float) -> bool:
    """Deterministic, stable train/val assignment keyed on doc_id.

    Args:
        doc_id: The document's id (already an md5 hash of its question, so
            this is a second, independent hash to avoid correlating the
            split with any property of the id itself).
        val_fraction: Fraction of documents assigned to validation.

    Returns:
        True if this document belongs in the validation split.
    """
    digest = hashlib.md5(f"split:{doc_id}".encode()).hexdigest()
    return (int(digest, 16) % 10_000) < int(val_fraction * 10_000)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs_file", type=Path, default=DEFAULT_DOCS_FILE)
    parser.add_argument("--units_file", type=Path, default=DEFAULT_UNITS_FILE)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--val_fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--min_step_tokens", type=int, default=DEFAULT_MIN_STEP_TOKENS)
    parser.add_argument("--max_step_tokens", type=int, default=DEFAULT_MAX_STEP_TOKENS)
    parser.add_argument("--semantic_percentile", type=float, default=DEFAULT_SEMANTIC_PERCENTILE)
    parser.add_argument("--min_units_for_semantic", type=int, default=DEFAULT_MIN_UNITS_FOR_SEMANTIC)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default=DEFAULT_EMBEDDING_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading document metadata from {args.docs_file}")
    doc_meta = load_doc_metadata(args.docs_file)
    logger.info(f"{len(doc_meta)} fully-reconstructed documents")

    logger.info(f"Loading units from {args.units_file} (restricted to those documents)")
    by_doc = load_units_for_docs(args.units_file, set(doc_meta.keys()))
    logger.info(f"{sum(len(u) for u in by_doc.values())} units across {len(by_doc)} documents")

    args.train_file.parent.mkdir(parents=True, exist_ok=True)
    args.val_file.parent.mkdir(parents=True, exist_ok=True)

    train_docs = train_steps = val_docs = val_steps = 0
    with open(args.train_file, "w", encoding="utf-8") as train_f, open(
        args.val_file, "w", encoding="utf-8"
    ) as val_f:
        for doc_index, (doc_id, units) in enumerate(by_doc.items(), start=1):
            steps = segment_document(
                units,
                args.min_step_tokens,
                args.max_step_tokens,
                args.semantic_percentile,
                args.min_units_for_semantic,
                args.embedding_model,
                args.embedding_device,
            )
            meta = doc_meta[doc_id]
            record = {
                "doc_id": doc_id,
                "source": meta["source"],
                "question": meta["question"],
                "num_steps": len(steps),
                "steps": [
                    {
                        "step_index": s.step_index,
                        "boundary_reason": s.boundary_reason,
                        "token_count": s.token_count,
                        "en": convert_placeholders(s.en),
                        "hi": convert_placeholders(s.hi),
                        "has_missing_translation": s.has_missing_translation,
                    }
                    for s in steps
                ],
            }
            target = val_f if is_val_doc(doc_id, args.val_fraction) else train_f
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if target is val_f:
                val_docs += 1
                val_steps += len(steps)
            else:
                train_docs += 1
                train_steps += len(steps)
            if doc_index % 500 == 0:
                logger.info(f"processed {doc_index}/{len(by_doc)} documents")

    logger.info(
        f"train: {train_docs} documents, {train_steps} steps -> {args.train_file}"
    )
    logger.info(f"val: {val_docs} documents, {val_steps} steps -> {args.val_file}")


if __name__ == "__main__":
    main()
