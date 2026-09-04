"""Groups chunking units into coherent reasoning "steps" for translation,
BEFORE any translation call — so `chunking.Unit`s sent to the translator are
whole steps, not small AST-level chunks. A step usually carries its own
local context (a discourse marker like "Wait" reads correctly because the
sentence before it is in the same translation unit), which chunk-level
translation structurally can't provide.

Boundary detection, in priority order:
  1. Structural: a `kind == "heading"` unit always starts a new step.
  2. Discourse marker: a unit (openthoughts/opencodereasoning only - see
     `_DISCOURSE_MARKER_SOURCES`) whose text opens with a
     reasoning-pivot phrase ("Wait,", "Actually,", "Hold on,", ...) starts a
     new step - these mark a deliberate shift in the reasoning, the same
     signal `chunking.py` already flags per-unit.
  3. Semantic fallback: for stretches with no structural/discourse signal,
     embeds each unit's English text and breaks where adjacent-unit
     cosine similarity drops into the document's own bottom percentile (a
     document-adaptive "meaning jump" detector, only once the current step
     already has >= --min_step_tokens, so it fine-tunes long stretches
     rather than fragmenting short ones).
  4. Hard cap: --max_step_tokens is a final safety net regardless of signal.
"""

import re

import numpy as np

from data_gen import embeddings
from data_gen.chunking import ChunkedDocument, Unit, _assert_invariants, _compute_stats, _merge_group

DEFAULT_MIN_STEP_TOKENS = 80
DEFAULT_MAX_STEP_TOKENS = 600
DEFAULT_SEMANTIC_PERCENTILE = 20.0
DEFAULT_EMBEDDING_MODEL = embeddings.DEFAULT_EMBEDDING_MODEL
DEFAULT_EMBEDDING_BACKEND = embeddings.DEFAULT_EMBEDDING_BACKEND
DEFAULT_EMBEDDING_BASE_URL = embeddings.DEFAULT_EMBEDDING_BASE_URL
DEFAULT_EMBEDDING_DEVICE = embeddings.DEFAULT_EMBEDDING_DEVICE
DEFAULT_MIN_UNITS_FOR_SEMANTIC = 4

_DISCOURSE_MARKER_START_RE = re.compile(
    r"^\s*(wait|hold on|actually|hmm|that's not right|but wait|let me reconsider)\b",
    re.IGNORECASE,
)
# Sources whose CoT traces carry stream-of-consciousness discourse-marker
# texture, matching data_gen.chunking's _DISCOURSE_MARKER_SOURCES.
_DISCOURSE_MARKER_SOURCES = {"openthoughts", "opencodereasoning"}


def _boundary_reason(
    kind: str,
    source: str,
    text: str,
    prev_similarity: float | None,
    semantic_threshold: float | None,
    current_step_tokens: int,
    min_step_tokens: int,
    max_step_tokens: int,
) -> str | None:
    """Decides whether a new step should start right before a candidate unit.

    Args:
        kind: The candidate unit's `kind` (e.g. "heading", "paragraph").
        source: "openthoughts" | "naturalreasoning" | "opencodereasoning".
        text: The candidate unit's English text.
        prev_similarity: Cosine similarity between the previous unit and
            this one, or None if not computed for this document.
        semantic_threshold: This document's bottom-percentile similarity
            threshold, or None if the semantic fallback isn't applicable
            (too few units).
        current_step_tokens: Token count accumulated in the in-progress step
            so far (before adding this unit).
        min_step_tokens: Floor below which only structural/discourse signals
            can trigger a break (the semantic fallback is suppressed).
        max_step_tokens: Hard cap; always triggers a break once reached,
            regardless of any other signal.

    Returns:
        A boundary reason string, or None to keep extending the current step.
    """
    if kind == "heading":
        return "heading"
    if source in _DISCOURSE_MARKER_SOURCES and _DISCOURSE_MARKER_START_RE.match(text):
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


def _split_translatable_runs(units: list[Unit]) -> list[tuple[str, list[Unit]]]:
    """Splits a document's units into ordered segments: a non-translatable
    unit passes through alone, and maximal contiguous runs of translatable
    units are grouped together (never merged across a non-translatable
    unit, e.g. a code/math block).

    Args:
        units: A document's units, in order.

    Returns:
        [("keep", [unit]), ("run", [unit, unit, ...]), ...] in document order.
    """
    segments: list[tuple[str, list[Unit]]] = []
    i = 0
    n = len(units)
    while i < n:
        if not units[i].translate:
            segments.append(("keep", [units[i]]))
            i += 1
            continue
        run = []
        while i < n and units[i].translate:
            run.append(units[i])
            i += 1
        segments.append(("run", run))
    return segments


def _group_run_into_steps(
    run: list[Unit],
    min_step_tokens: int,
    max_step_tokens: int,
    semantic_percentile: float,
    vectors: np.ndarray | None,
) -> list[list[Unit]]:
    """Applies the step-boundary decision to a contiguous run of
    translatable `chunking.Unit`s (pre-translation).

    Args:
        run: Consecutive translatable units (no code/math/rule units mixed
            in - see `_split_translatable_runs`).
        min_step_tokens: Passed through to `_boundary_reason`.
        max_step_tokens: Passed through to `_boundary_reason`.
        semantic_percentile: Percentile of `run`'s own adjacent-similarity
            distribution treated as a semantic-jump threshold.
        vectors: Precomputed (len(run), dim) embeddings for `run`'s units
            (see `regroup_chunked_documents`, which batches the embedding
            calls across many runs/documents rather than one call per run),
            or None to skip the semantic-break signal entirely (below
            min_units_for_semantic).

    Returns:
        Groups of consecutive units, each to become one merged step-unit.
    """
    if not run:
        return []

    similarities: np.ndarray | None = None
    threshold: float | None = None
    if vectors is not None:
        similarities = np.array([float(vectors[i] @ vectors[i + 1]) for i in range(len(run) - 1)])
        threshold = float(np.percentile(similarities, semantic_percentile))

    groups: list[list[Unit]] = []
    current = [run[0]]
    for i in range(1, len(run)):
        unit = run[i]
        # u.token_count is chunking.py's protected-text token count (what
        # actually gets sent to the translator) - counting text_raw here
        # would understate steps with number/code-dense units, since a
        # placeholder like "⟦0⟧" can cost more tokens than what it replaces.
        current_tokens = sum(u.token_count for u in current)
        prev_similarity = float(similarities[i - 1]) if similarities is not None else None
        reason = _boundary_reason(
            unit.kind,
            unit.source,
            unit.text_raw,
            prev_similarity,
            threshold,
            current_tokens,
            min_step_tokens,
            max_step_tokens,
        )
        if reason is not None:
            groups.append(current)
            current = [unit]
        else:
            current.append(unit)
    groups.append(current)
    return groups


def regroup_chunked_documents(
    docs: list[tuple[ChunkedDocument, str]],
    min_step_tokens: int = DEFAULT_MIN_STEP_TOKENS,
    max_step_tokens: int = DEFAULT_MAX_STEP_TOKENS,
    semantic_percentile: float = DEFAULT_SEMANTIC_PERCENTILE,
    min_units_for_semantic: int = DEFAULT_MIN_UNITS_FOR_SEMANTIC,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    embed_batch_size: int = embeddings.DEFAULT_EMBED_BATCH_SIZE,
) -> list[ChunkedDocument]:
    """Rebuilds many `ChunkedDocument`s with their units grouped into
    step-sized units, BEFORE translation - so each unit sent to the
    translator is a whole coherent reasoning step (heading/discourse-
    marker/semantic-break boundaries) rather than a small AST-level chunk.

    Every document's runs needing the semantic-break signal are embedded in
    ONE batched pass (chunked internally to `embed_batch_size` - see
    `embeddings.embed`) across the WHOLE input, not one `embeddings.embed`
    call per document: with thousands of documents, a call-per-document (or
    call-per-run) pattern is thousands of tiny forward passes dominated by
    per-call overhead: gathering all of them into a handful of large calls
    is what actually uses the GPU (or vLLM server) efficiently.

    Args:
        docs: (doc, text) pairs - each `doc` a `ChunkedDocument` from
            `chunk_document`, not yet translated, and `text` the full
            source document it was chunked from.
        min_step_tokens: Passed through to `_boundary_reason`.
        max_step_tokens: Passed through to `_boundary_reason`.
        semantic_percentile: Passed through to `_boundary_reason`'s threshold.
        min_units_for_semantic: Minimum run length before the semantic
            fallback applies at all.
        embedding_model: Passed through to `embeddings.embed`.
        embedding_backend: "vllm" or "local" - see `embeddings.embed`.
        embedding_base_url: ["vllm" only] Passed through to `embeddings.embed`.
        embedding_device: ["local" only] Passed through to `embeddings.embed`.
        embed_batch_size: Passed through to `embeddings.embed`.

    Returns:
        A new `ChunkedDocument` per input, in the same order, each with its
        units grouped into steps.
    """
    # Segment every document first (cheap, no embeddings yet), and collect
    # the texts of every run that will actually need the semantic signal
    # into one flat list, remembering which (doc_index, segment_index) each
    # slice of that flat list belongs to. Embeds text_protected (with
    # ⟦N⟧ placeholders), not text_raw - same "measure/consider the
    # protected form, not raw" convention as chunking.py's token caps
    # (Unit.token_count), so a unit's embedding length is judged on what it
    # actually is post-protection rather than the pre-protection original.
    all_segments: list[list[tuple[str, list[Unit]]]] = [_split_translatable_runs(doc.units) for doc, _ in docs]
    flat_texts: list[str] = []
    offsets: dict[tuple[int, int], tuple[int, int]] = {}  # (doc_i, seg_i) -> (start, end) in flat_texts
    for doc_i, segments in enumerate(all_segments):
        for seg_i, (kind, run) in enumerate(segments):
            if kind == "run" and len(run) >= min_units_for_semantic:
                start = len(flat_texts)
                flat_texts.extend(u.text_protected for u in run)
                offsets[(doc_i, seg_i)] = (start, len(flat_texts))

    all_vectors = embeddings.embed(
        flat_texts,
        model=embedding_model,
        backend=embedding_backend,
        base_url=embedding_base_url,
        device=embedding_device,
        batch_size=embed_batch_size,
    )

    regrouped_docs = []
    for doc_i, (doc, text) in enumerate(docs):
        result: list[Unit] = []
        for seg_i, (kind, run) in enumerate(all_segments[doc_i]):
            if kind == "keep":
                result.append(run[0])
                continue
            vectors = None
            if (doc_i, seg_i) in offsets:
                start, end = offsets[(doc_i, seg_i)]
                vectors = all_vectors[start:end]
            for group in _group_run_into_steps(run, min_step_tokens, max_step_tokens, semantic_percentile, vectors):
                result.append(_merge_group(group, text) if len(group) > 1 else group[0])

        for index, unit in enumerate(result):
            unit.index = index
            unit.unit_id = f"{unit.doc_id}:{index:04d}"

        gaps = [text[result[i].char_end : result[i + 1].char_start] for i in range(len(result) - 1)]
        regrouped = ChunkedDocument(
            doc_id=doc.doc_id,
            source=doc.source,
            units=result,
            gaps=gaps,
            original_length=doc.original_length,
            stats=_compute_stats(result),
        )
        _assert_invariants(text, regrouped)
        regrouped_docs.append(regrouped)

    return regrouped_docs
