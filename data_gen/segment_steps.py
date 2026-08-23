"""Groups translated chunking units into coherent reasoning "steps" for
process-reward-model-style training data.

Pure post-processing: reads the per-unit records already written by
`translate_reasoning.py` (`translated_reasoning_25k_units.jsonl`), regroups
each document's units (currently sentence/paragraph-level, sized for
translation-call accuracy, not for standalone training examples) into larger
step-level spans, and writes one record per step. No LLM calls.

Boundary detection, in priority order:
  1. Structural: a `kind == "heading"` unit always starts a new step.
  2. Discourse marker: a unit (openthoughts only) whose text opens with a
     reasoning-pivot phrase ("Wait,", "Actually,", "Hold on,", ...) starts a
     new step — these mark a deliberate shift in the reasoning, the same
     signal `chunking.py` already flags per-unit.
  3. Semantic fallback: for stretches with no structural/discourse signal,
     LaBSE-embeds each unit's English text and breaks where adjacent-unit
     cosine similarity drops into the document's own bottom percentile (a
     document-adaptive "meaning jump" detector, only once the current step
     already has >= --min_step_tokens, so it fine-tunes long stretches
     rather than fragmenting short ones).
  4. Hard cap: --max_step_tokens is a final safety net regardless of signal.

Usage:
    uv run python -m data_gen.segment_steps --units_file translated_reasoning_25k_units.jsonl
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tiktoken
from loguru import logger

from data_gen.translate_reasoning import DEFAULT_LABSE_MODEL, _get_labse_model

DEFAULT_UNITS_FILE = Path("translated_reasoning_25k_units.jsonl")
DEFAULT_OUTPUT_FILE = Path("reasoning_steps.jsonl")
DEFAULT_MIN_STEP_TOKENS = 80
DEFAULT_MAX_STEP_TOKENS = 600
DEFAULT_SEMANTIC_PERCENTILE = 20.0
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_MIN_UNITS_FOR_SEMANTIC = 4

_ENCODING = tiktoken.get_encoding("cl100k_base")
_DISCOURSE_MARKER_START_RE = re.compile(
    r"^\s*(wait|hold on|actually|hmm|that's not right|but wait|let me reconsider)\b",
    re.IGNORECASE,
)
# Sources whose CoT traces carry stream-of-consciousness discourse-marker
# texture, matching data_gen.chunking's _DISCOURSE_MARKER_SOURCES.
_DISCOURSE_MARKER_SOURCES = {"openthoughts", "opencodereasoning"}


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class UnitRecord:
    """One translated unit, as read back from `translated_reasoning_*_units.jsonl`."""

    doc_id: str
    source: str
    unit_id: str
    kind: str
    en: str
    hi: str | None


@dataclass
class Step:
    """A group of consecutive units forming one coherent reasoning step.

    Attributes:
        doc_id: Parent document id.
        source: "openthoughts" | "naturalreasoning".
        step_index: Contiguous index of this step within its document, from 0.
        unit_ids: The unit_ids merged into this step, in document order.
        boundary_reason: Why this step started ("doc_start", "heading",
            "discourse_marker", "semantic_break", or "token_cap").
        token_count: cl100k_base token count of the step's English text.
        en: Units' English text, joined with blank lines.
        hi: Units' Hindi text, joined with blank lines; a unit with a missing
            translation falls back to its English text.
        has_missing_translation: Whether any merged unit has `hi is None`.
    """

    doc_id: str
    source: str
    step_index: int
    unit_ids: list[str] = field(default_factory=list)
    boundary_reason: str = "doc_start"
    token_count: int = 0
    en: str = ""
    hi: str = ""
    has_missing_translation: bool = False


def load_units_by_doc(units_file: Path) -> dict[str, list[UnitRecord]]:
    """Reads a units JSONL file and groups records by document, in order.

    Args:
        units_file: Path to a `translated_reasoning_*_units.jsonl` file.

    Returns:
        Map of doc_id -> UnitRecords sorted by unit_id (zero-padded index,
        so lexicographic sort matches document order).
    """
    by_doc: dict[str, list[UnitRecord]] = {}
    with open(units_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
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


def _adjacent_similarities(units: list[UnitRecord], model_name: str, device: str) -> np.ndarray:
    """Cosine similarity between each pair of adjacent units' English text.

    Args:
        units: A document's units, in order.
        model_name: SentenceTransformer model name (LaBSE).
        device: Device to run the embedding model on.

    Returns:
        Array of length len(units) - 1; entry i is sim(units[i], units[i+1]).
    """
    model = _get_labse_model(model_name, device)
    embeddings = model.encode([u.en for u in units], normalize_embeddings=True)
    return np.array([float(embeddings[i] @ embeddings[i + 1]) for i in range(len(units) - 1)])


def _boundary_reason(
    unit: UnitRecord,
    prev_similarity: float | None,
    semantic_threshold: float | None,
    current_step_tokens: int,
    min_step_tokens: int,
    max_step_tokens: int,
) -> str | None:
    """Decides whether a new step should start right before `unit`.

    Args:
        unit: The candidate unit to start a new step at.
        prev_similarity: Cosine similarity between the previous unit and
            `unit`, or None if not computed for this document.
        semantic_threshold: This document's bottom-percentile similarity
            threshold, or None if the semantic fallback isn't applicable
            (too few units).
        current_step_tokens: Token count accumulated in the in-progress step
            so far (before adding `unit`).
        min_step_tokens: Floor below which only structural/discourse signals
            can trigger a break (the semantic fallback is suppressed).
        max_step_tokens: Hard cap; always triggers a break once reached,
            regardless of any other signal.

    Returns:
        A boundary reason string, or None to keep extending the current step.
    """
    if unit.kind == "heading":
        return "heading"
    if unit.source in _DISCOURSE_MARKER_SOURCES and _DISCOURSE_MARKER_START_RE.match(unit.en):
        return "discourse_marker"
    if current_step_tokens >= max_step_tokens:
        return "token_cap"
    if (
        current_step_tokens >= min_step_tokens
        and semantic_threshold is not None
        and prev_similarity is not None
        and prev_similarity <= semantic_threshold
    ):
        return "semantic_break"
    return None


def segment_document(
    units: list[UnitRecord],
    min_step_tokens: int,
    max_step_tokens: int,
    semantic_percentile: float,
    min_units_for_semantic: int,
    labse_model: str,
    embedding_device: str,
) -> list[Step]:
    """Segments one document's units into coherent PRM-style steps.

    Args:
        units: A document's units, in order (from `load_units_by_doc`).
        min_step_tokens: Passed through to `_boundary_reason`.
        max_step_tokens: Passed through to `_boundary_reason`.
        semantic_percentile: Bottom percentile of this document's own
            adjacent-similarity distribution treated as a "meaning jump".
        min_units_for_semantic: Minimum unit count for the semantic fallback
            to apply at all (too few units gives an unreliable percentile).
        labse_model: SentenceTransformer model name for the semantic fallback.
        embedding_device: Device to run the embedding model on.

    Returns:
        Steps in document order, reindexed contiguously from 0.
    """
    if not units:
        return []

    similarities: np.ndarray | None = None
    threshold: float | None = None
    if len(units) >= min_units_for_semantic:
        similarities = _adjacent_similarities(units, labse_model, embedding_device)
        threshold = float(np.percentile(similarities, semantic_percentile))

    steps: list[Step] = []
    current = [units[0]]
    current_reason = "doc_start"
    for i in range(1, len(units)):
        unit = units[i]
        current_tokens = sum(_count_tokens(u.en) for u in current)
        prev_similarity = float(similarities[i - 1]) if similarities is not None else None
        reason = _boundary_reason(
            unit, prev_similarity, threshold, current_tokens, min_step_tokens, max_step_tokens
        )
        if reason is not None:
            steps.append(_finalize_step(current, len(steps), current_reason))
            current = [unit]
            current_reason = reason
        else:
            current.append(unit)
    steps.append(_finalize_step(current, len(steps), current_reason))
    return steps


def _finalize_step(units: list[UnitRecord], step_index: int, boundary_reason: str) -> Step:
    """Builds a Step from a run of consecutive units.

    Args:
        units: Consecutive units to merge into one step.
        step_index: This step's index within its document.
        boundary_reason: Why this step started.

    Returns:
        The assembled Step.
    """
    en = "\n\n".join(u.en for u in units)
    hi = "\n\n".join(u.hi if u.hi is not None else u.en for u in units)
    return Step(
        doc_id=units[0].doc_id,
        source=units[0].source,
        step_index=step_index,
        unit_ids=[u.unit_id for u in units],
        boundary_reason=boundary_reason,
        token_count=_count_tokens(en),
        en=en,
        hi=hi,
        has_missing_translation=any(u.hi is None for u in units),
    )


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units_file", type=Path, default=DEFAULT_UNITS_FILE)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--min_step_tokens", type=int, default=DEFAULT_MIN_STEP_TOKENS)
    parser.add_argument("--max_step_tokens", type=int, default=DEFAULT_MAX_STEP_TOKENS)
    parser.add_argument(
        "--semantic_percentile",
        type=float,
        default=DEFAULT_SEMANTIC_PERCENTILE,
        help="Bottom percentile of each document's own adjacent-unit similarity "
        "distribution treated as a semantic-jump boundary (default: %(default)s).",
    )
    parser.add_argument("--labse_model", default=DEFAULT_LABSE_MODEL)
    parser.add_argument("--embedding_device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument(
        "--min_units_for_semantic",
        type=int,
        default=DEFAULT_MIN_UNITS_FOR_SEMANTIC,
        help="Minimum unit count in a document before the semantic fallback "
        "applies at all (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Loading units from {args.units_file}")
    by_doc = load_units_by_doc(args.units_file)
    logger.info(f"Loaded {sum(len(u) for u in by_doc.values())} units across {len(by_doc)} documents")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for doc_index, (_, units) in enumerate(by_doc.items(), start=1):
            steps = segment_document(
                units,
                args.min_step_tokens,
                args.max_step_tokens,
                args.semantic_percentile,
                args.min_units_for_semantic,
                args.labse_model,
                args.embedding_device,
            )
            for step in steps:
                f.write(
                    json.dumps(
                        {
                            "doc_id": step.doc_id,
                            "source": step.source,
                            "step_id": f"{step.doc_id}:step{step.step_index:03d}",
                            "step_index": step.step_index,
                            "unit_ids": step.unit_ids,
                            "boundary_reason": step.boundary_reason,
                            "token_count": step.token_count,
                            "en": step.en,
                            "hi": step.hi,
                            "has_missing_translation": step.has_missing_translation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            total_steps += len(steps)
            if doc_index % 200 == 0:
                logger.info(f"segmented {doc_index}/{len(by_doc)} documents ({total_steps} steps so far)")

    logger.info(f"Wrote {total_steps} steps from {len(by_doc)} documents to {args.output_file}")


if __name__ == "__main__":
    main()
