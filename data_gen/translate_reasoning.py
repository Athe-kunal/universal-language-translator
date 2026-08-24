"""Translates chunked OpenThoughts3 / natural_reasoning documents to Hindi.

Samples `open-thoughts/OpenThoughts3-1.2M`, `facebook/natural_reasoning`, and
`nvidia/OpenCodeReasoning` (all uniform random) via `data_gen.sample_reasoning`,
chunks each `cot_answer` via
`data_gen.chunking.chunk_document`, translates every translatable unit
through an OpenAI-compatible vllm endpoint, and reconstructs the full Hindi
document. Prompts are Jinja templates under `data_gen/prompts/`, so wording
changes don't require touching this module.

A unit whose translation degenerates into a repeated-token/phrase loop (a
known decoding failure), drops or alters a protected placeholder, or scores
below `--similarity_floor` on cross-lingual embedding similarity (a
semantic-drift/hallucination signal — see the manual review that motivated
this: some broken translations pass both the repetition and placeholder
checks yet share almost no meaning with the source) is retried up to
`--max_retries` times at a higher temperature, with the previous broken
attempt shown to the model so it doesn't repeat the same mistake. The
erroneous attempt is kept in the output record for review, never silently
discarded.

Usage:
    uv run python data_gen/translate_reasoning.py --num_openthoughts3 100 --num_natural_reasoning 100
    uv run python data_gen/translate_reasoning.py --base_url http://localhost:8077/v1 --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import asyncio
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger
import httpx
from openai import AsyncOpenAI

from data_gen import segment_steps
from data_gen.chunking import ChunkedDocument, Unit, chunk_document, reconstruct, set_sat_device
from data_gen.datamodels import TranslationDataset
from data_gen.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_similarity
from data_gen.sample_reasoning import (
    load_done,
    sample_natural_reasoning,
    sample_opencodereasoning_shards,
    sample_openthoughts3_shards,
)

DEFAULT_BASE_URL = "http://localhost:8077/v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_FILE = Path("translated_reasoning.jsonl")
DEFAULT_UNITS_OUTPUT_FILE = Path("translated_reasoning_units.jsonl")
DEFAULT_TARGET_LANGUAGE = "Hindi"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_TEMPERATURE = 0.7
DEFAULT_MIN_TOKENS = 60
DEFAULT_MAX_TOKENS = 400
DEFAULT_CONCURRENCY = 48
DEFAULT_SIMILARITY_FLOOR = 0.5
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_SAT_DEVICE = "cpu"

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(loader=FileSystemLoader(_PROMPTS_DIR), trim_blocks=True, lstrip_blocks=True)

# A 1-5 token phrase repeated 4+ times in a row — a decoding-degeneration
# signature. Widened beyond single-token repetition after manual review
# turned up broken translations looping on a repeated bigram/phrase
# ("लिए गए लिए गए...", "एक दूसरे के बाद एक दूसरे के बाद...") that a
# single-token-only check missed entirely.
_REPETITION_RE = re.compile(r"(\S+(?:\s+\S+){0,4})(?:\s+\1){3,}")


def has_repetition(text: str) -> bool:
    """Whether `text` contains a short phrase (1-5 tokens) repeated 4+ times
    consecutively.

    Args:
        text: Candidate translation text.

    Returns:
        True if a degenerate repetition loop is detected.
    """
    return bool(_REPETITION_RE.search(text))


def render_translate_prompt(
    target_language: str = DEFAULT_TARGET_LANGUAGE, prior_context: str | None = None
) -> str:
    """Renders the base translation system prompt.

    Args:
        target_language: Language to translate into.
        prior_context: Tail end of the preceding unit's source text, shown
            for disambiguation only (e.g. so a discourse marker like "Wait"
            reads as a mid-reasoning interjection rather than in isolation).
            None for a document's first unit, where there's nothing before it.

    Returns:
        Rendered prompt text.
    """
    return _JINJA_ENV.get_template("translate.jinja").render(
        target_language=target_language, prior_context=prior_context
    )


def render_retry_prompt(
    previous_translation: str,
    reason: str,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    prior_context: str | None = None,
) -> str:
    """Renders the retry system prompt, showing the previous broken attempt.

    Args:
        previous_translation: The prior attempt that failed validation.
        reason: Human-readable description of what was wrong with it.
        target_language: Language to translate into.
        prior_context: Same as `render_translate_prompt`.

    Returns:
        Rendered prompt text.
    """
    return _JINJA_ENV.get_template("translate_retry.jinja").render(
        target_language=target_language,
        previous_translation=previous_translation,
        reason=reason,
        prior_context=prior_context,
    )


_PRIOR_CONTEXT_CHARS = 100


def get_prior_context(doc: ChunkedDocument, unit: Unit, max_chars: int = _PRIOR_CONTEXT_CHARS) -> str | None:
    """Gets up to the last `max_chars` characters of the unit immediately
    before this one in document order, as disambiguation context.

    Uses the immediately preceding unit regardless of its `kind`/`translate`
    status (even a code unit's tail can carry useful continuity), and always
    the original English (never a translation), since units are translated
    concurrently and an adjacent unit's Hindi output may not exist yet.

    Args:
        doc: The unit's parent ChunkedDocument.
        unit: The unit about to be translated.
        max_chars: Max characters of trailing context to include.

    Returns:
        The trailing context string, or None if this is the document's
        first unit (nothing precedes it).
    """
    if unit.index == 0:
        return None
    prior_text = doc.units[unit.index - 1].text_raw
    tail = prior_text[-max_chars:]
    # Trim to a clean word boundary rather than starting mid-word.
    space_idx = tail.find(" ")
    if 0 <= space_idx < len(tail) - 1:
        tail = tail[space_idx + 1 :]
    return tail.strip() or None


_PLACEHOLDER_RE = re.compile(r"⟦\d+⟧")


async def _validate(
    translation: str,
    expected_placeholders: list[str],
    en_text: str,
    similarity_floor: float,
    embedding_model: str,
    embedding_device: str,
) -> str | None:
    """Checks a translation attempt for known defect signatures.

    Args:
        translation: Candidate translation text.
        expected_placeholders: Sorted, deduped placeholder ids the unit's
            fingerprint expects (e.g. ["⟦0⟧", "⟦1⟧"]).
        en_text: The source text sent to the translator, for the embedding
            similarity check.
        similarity_floor: Minimum embedding cosine similarity to accept; a
            non-positive value disables the check entirely (skips loading
            the embedding model at all).
        embedding_model: SentenceTransformer model name for the similarity check.
        embedding_device: Device to run the embedding model on.

    Returns:
        None if the translation looks valid; otherwise a short human-readable
        reason, used both for logging and shown to the model on retry.
    """
    if has_repetition(translation):
        return "it got stuck repeating the same word or phrase over and over"
    found = sorted(set(_PLACEHOLDER_RE.findall(translation)))
    if found != expected_placeholders:
        return (
            f"it dropped or altered required placeholder tokens — expected {expected_placeholders}, "
            f"got {found}. Every placeholder token must appear exactly once, unchanged"
        )
    if similarity_floor > 0:
        similarity = await embedding_similarity(en_text, translation, embedding_model, embedding_device)
        if similarity < similarity_floor:
            return (
                "it does not accurately or completely convey the meaning of the source text "
                "(it may be incomplete, unrelated, or otherwise inaccurate)"
            )
    return None


@dataclass
class UnitTranslation:
    """Result of translating one unit, including retry provenance.

    Attributes:
        unit_id: The Unit's id.
        translation: Final, validated translation text, or None if every
            attempt (including after retries) still failed validation — the
            caller should fall back to the unit's original English text
            rather than use a known-broken translation.
        retries: Number of retries performed (0 = succeeded on the first try).
        had_defect: Whether any attempt failed validation (repetition or a
            dropped/altered placeholder).
        exhausted: True if every attempt, including the last, still failed
            validation when retries ran out.
        erroneous_translation: The last broken attempt, kept for review even
            when `translation` is None; None if the first attempt succeeded.
    """

    unit_id: str
    translation: str | None
    retries: int
    had_defect: bool
    exhausted: bool
    erroneous_translation: str | None


async def _write_jsonl(fh, write_lock: asyncio.Lock, record: dict) -> None:
    """Appends one JSON record to an open file handle, lock-serialized.

    Args:
        fh: Open file handle (append mode).
        write_lock: Lock shared by every concurrent writer to this handle.
        record: JSON-serializable record to write.
    """
    async with write_lock:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


async def translate_unit(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    unit: Unit,
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_temperature: float = DEFAULT_RETRY_TEMPERATURE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    prior_context: str | None = None,
) -> UnitTranslation:
    """Translates one unit, retrying on detected repetition, placeholder loss,
    or low cross-lingual embedding similarity to the source.

    Args:
        client: Async OpenAI-compatible client (pointed at the vllm server).
        sem: Concurrency-limiting semaphore.
        unit: The unit to translate (its `text_protected` is sent as-is).
        model: Model name to request.
        max_retries: Max retry attempts after an initial failed attempt.
        retry_temperature: Sampling temperature for retry attempts (the
            initial attempt uses temperature=0.0 for reproducibility).
        target_language: Language to translate into.
        similarity_floor: Minimum embedding cosine similarity to accept; a
            non-positive value disables the embedding check.
        embedding_model: SentenceTransformer model name for the similarity check.
        embedding_device: Device to run the embedding model on.
        prior_context: Tail of the preceding unit's source text, shown to
            the model for disambiguation only — see `get_prior_context`.

    Returns:
        The final UnitTranslation, with retry/erroneous-attempt provenance.
        `translation` is None if every attempt failed validation — the
        caller is expected to fall back to the original English for this
        unit rather than accept a known-broken translation.
    """
    expected_placeholders = sorted(set(span.placeholder for span in unit.spans))
    base_prompt = render_translate_prompt(target_language, prior_context)
    previous_translation: str | None = None
    translation = ""
    reason: str | None = None

    for attempt in range(max_retries + 1):
        system_prompt = (
            base_prompt
            if attempt == 0
            else render_retry_prompt(previous_translation, reason, target_language, prior_context)
        )
        async with sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": unit.text_protected},
                ],
                temperature=0.0 if attempt == 0 else retry_temperature,
                max_tokens=2048,
            )
        translation = (resp.choices[0].message.content or "").strip()
        reason = await _validate(
            translation,
            expected_placeholders,
            unit.text_protected,
            similarity_floor,
            embedding_model,
            embedding_device,
        )
        if reason is None:
            return UnitTranslation(
                unit_id=unit.unit_id,
                translation=translation,
                retries=attempt,
                had_defect=attempt > 0,
                exhausted=False,
                erroneous_translation=previous_translation,
            )
        previous_translation = translation

    return UnitTranslation(
        unit_id=unit.unit_id,
        translation=None,
        retries=max_retries,
        had_defect=True,
        exhausted=True,
        erroneous_translation=previous_translation,
    )


async def _translate_and_log_unit(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    unit: Unit,
    model: str,
    max_retries: int,
    retry_temperature: float,
    target_language: str,
    similarity_floor: float,
    embedding_model: str,
    embedding_device: str,
    doc_id: str,
    source: str,
    units_fh,
    write_lock: asyncio.Lock,
    prior_context: str | None = None,
) -> UnitTranslation:
    """Translates one unit and immediately appends its record to `units_fh`,
    so per-chunk results are visible on disk as soon as each unit finishes
    rather than only once the whole document is done.
    """
    result = await translate_unit(
        client,
        sem,
        unit,
        model,
        max_retries,
        retry_temperature,
        target_language,
        similarity_floor,
        embedding_model,
        embedding_device,
        prior_context,
    )
    if units_fh is not None:
        await _write_jsonl(
            units_fh,
            write_lock,
            {
                "doc_id": doc_id,
                "source": source,
                "unit_id": result.unit_id,
                "kind": unit.kind,
                "en": unit.text_raw,
                "hi": result.translation,
                "retries": result.retries,
                "had_defect": result.had_defect,
                "exhausted": result.exhausted,
                "erroneous_translation": result.erroneous_translation,
            },
        )
    return result


async def translate_document(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    doc: ChunkedDocument,
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_temperature: float = DEFAULT_RETRY_TEMPERATURE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    units_fh=None,
    write_lock: asyncio.Lock | None = None,
) -> dict[str, UnitTranslation]:
    """Translates every translatable unit in a document concurrently.

    Args:
        client: Async OpenAI-compatible client. Multi-GPU throughput is
            handled by vllm's own data-parallel serving (--data-parallel-size)
            behind this single endpoint, not by this module.
        sem: Concurrency-limiting semaphore, shared across the whole run.
        doc: A ChunkedDocument from `chunk_document`.
        model: Model name to request.
        max_retries: Max retry attempts per unit.
        retry_temperature: Sampling temperature for retry attempts.
        target_language: Language to translate into.
        similarity_floor: Minimum embedding cosine similarity to accept; a
            non-positive value disables the embedding check.
        embedding_model: SentenceTransformer model name for the similarity check.
        embedding_device: Device to run the embedding model on.
        units_fh: Open file handle to append per-unit records to as soon as
            each one finishes (incremental, chunk-level persistence). None
            to skip per-unit logging.
        write_lock: Lock shared by every concurrent writer to `units_fh`.
            Required if `units_fh` is given.

    Returns:
        Map of unit_id -> UnitTranslation, for translatable units only.
    """
    translatable = [unit for unit in doc.units if unit.translate]
    results = await asyncio.gather(
        *(
            _translate_and_log_unit(
                client,
                sem,
                unit,
                model,
                max_retries,
                retry_temperature,
                target_language,
                similarity_floor,
                embedding_model,
                embedding_device,
                doc.doc_id,
                doc.source,
                units_fh,
                write_lock,
                get_prior_context(doc, unit),
            )
            for unit in translatable
        )
    )
    return {result.unit_id: result for result in results}


def _chunk_one(
    row: TranslationDataset, source: str, min_tokens: int, max_tokens: int
) -> tuple[TranslationDataset, str, ChunkedDocument | None, str | None]:
    """Process-pool worker for chunk_all: chunks one document, returning any
    error as a string instead of raising, so one bad document can't take
    down a worker process.
    """
    try:
        doc = chunk_document(row.cot_answer, doc_id=row.id, source=source, min_tokens=min_tokens, max_tokens=max_tokens)
        return row, source, doc, None
    except Exception as e:
        return row, source, None, str(e)


def chunk_all(
    jobs: list[tuple[TranslationDataset, str]],
    min_tokens: int,
    max_tokens: int,
    log_every: int = 1000,
    max_workers: int | None = None,
) -> list[tuple[TranslationDataset, str, ChunkedDocument]]:
    """Pre-chunks every sampled document in parallel, before any translation
    task is created.

    This is a deliberate two-phase design: chunk_document is CPU-bound (AST
    parsing, tiktoken encoding, wtpsplit for oversized units). Two problems
    observed running it lazily inside each async translation task: (1) it
    serializes a large batch entirely through chunking before any HTTP
    traffic can flow (a 50K-document batch spent 35+ minutes at 0% vllm GPU
    utilization), and (2) it only used one CPU core even on a 128-core host
    (~470 docs/min single-threaded — ~91ms/doc of pure-Python AST/regex work
    that has no GPU equivalent, unlike wtpsplit). Chunking everything up
    front in a process pool fixes both: translation starts against
    already-chunked documents, and the embarrassingly-parallel-per-document
    work actually uses the machine's cores.

    Args:
        jobs: (row, source) pairs from the samplers.
        min_tokens: Passed through to chunk_document.
        max_tokens: Passed through to chunk_document.
        log_every: Log progress every this many completed documents.
        max_workers: Process pool size. None uses ProcessPoolExecutor's
            default (os.cpu_count()).

    Returns:
        (row, source, doc) triples. Not necessarily in `jobs` order (returned
        in completion order) — fine, since downstream translation dispatch
        doesn't depend on ordering either.
    """
    results = []
    failed = 0
    completed = 0
    # Force CPU for wtpsplit inside worker processes: each worker that hits
    # the split path would otherwise lazily load its own SaT model copy onto
    # whatever GPU the main process configured, and with many workers doing
    # this concurrently the GPU runs out of memory (observed: CUDA OOM with
    # 64 workers). Splits are rare, so losing GPU accel for them here (the
    # main process's async translation phase still uses GPU for its own
    # occasional splits) is a good trade for safe, uncontended parallelism.
    with ProcessPoolExecutor(max_workers=max_workers, initializer=set_sat_device, initargs=("cpu",)) as executor:
        futures = [executor.submit(_chunk_one, row, source, min_tokens, max_tokens) for row, source in jobs]
        for future in as_completed(futures):
            row, source, doc, error = future.result()
            completed += 1
            if error is not None:
                # One malformed document must never take down a batch of
                # tens of thousands — log and skip it.
                failed += 1
                logger.error(f"chunk_document failed for doc_id={row.id}: {error}")
            else:
                results.append((row, source, doc))
            if completed % log_every == 0:
                logger.info(f"chunked {completed}/{len(jobs)} documents ({failed} failed)")
    logger.info(f"chunked {len(results)}/{len(jobs)} documents ({failed} failed)")
    return results


async def process_document(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    row: TranslationDataset,
    source: str,
    doc: ChunkedDocument,
    args: argparse.Namespace,
    units_fh=None,
    write_lock: asyncio.Lock | None = None,
) -> dict:
    """Translates and reconstructs one already-chunked document into an output record.

    Args:
        client: Async OpenAI-compatible client.
        sem: Concurrency-limiting semaphore.
        row: The mapped TranslationDataset row `doc` was chunked from.
        source: "openthoughts" | "naturalreasoning".
        doc: The pre-chunked ChunkedDocument, from `chunk_all`.
        args: Parsed CLI arguments (model/retry knobs).
        units_fh: Open file handle for incremental per-unit records; see
            `translate_document`.
        write_lock: Lock shared by every concurrent writer to `units_fh`.

    Returns:
        A JSON-serializable record with the reconstructed Hindi document and
        per-unit translation provenance.
    """
    unit_results = await translate_document(
        client,
        sem,
        doc,
        args.model,
        args.max_retries,
        args.retry_temperature,
        args.target_language,
        args.similarity_floor,
        args.embedding_model,
        args.embedding_device,
        units_fh,
        write_lock,
    )
    # Only feed validated translations into reconstruct(); an exhausted unit
    # (translation=None) is omitted so reconstruct() falls back to its
    # original English text_raw instead of the whole document failing.
    translations = {
        unit_id: result.translation for unit_id, result in unit_results.items() if result.translation is not None
    }
    hi_cot_answer = reconstruct(doc, translations)

    def _field(unit_id: str, attr: str, default):
        result = unit_results.get(unit_id)
        return getattr(result, attr) if result is not None else default

    return {
        "id": row.id,
        "source": source,
        "question": row.question,
        "en_cot_answer": row.cot_answer,
        "hi_cot_answer": hi_cot_answer,
        "stats": doc.stats,
        "units": [
            {
                "unit_id": unit.unit_id,
                "kind": unit.kind,
                "translate": unit.translate,
                "en": unit.text_raw,
                "hi": _field(unit.unit_id, "translation", None),
                "retries": _field(unit.unit_id, "retries", 0),
                "had_defect": _field(unit.unit_id, "had_defect", False),
                "exhausted": _field(unit.unit_id, "exhausted", False),
                "erroneous_translation": _field(unit.unit_id, "erroneous_translation", None),
            }
            for unit in doc.units
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_openthoughts3", type=int, default=100)
    parser.add_argument("--num_natural_reasoning", type=int, default=100)
    parser.add_argument(
        "--num_opencodereasoning",
        type=int,
        default=0,
        help="Number of nvidia/OpenCodeReasoning rows to sample (default: %(default)s — "
        "opt-in, alongside the original two sources).",
    )
    parser.add_argument(
        "--opencodereasoning_config",
        default="split_0",
        choices=["split_0", "split_1"],
        help="Which OpenCodeReasoning HF config to sample from (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling, matching sample_reasoning.py "
        "(default: %(default)s) — same seed + same output_file lets a rerun resume and reuse "
        "cached results.",
    )
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--units_output_file",
        type=Path,
        default=DEFAULT_UNITS_OUTPUT_FILE,
        help="Per-unit records, appended incrementally as each chunk finishes translating "
        "(useful for watching progress live).",
    )
    parser.add_argument(
        "--base_url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible vllm endpoint. Multi-GPU throughput is handled by vllm's own "
        "data-parallel serving (--data-parallel-size) behind this single endpoint, not by this "
        "script (default: %(default)s).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target_language", default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_temperature", type=float, default=DEFAULT_RETRY_TEMPERATURE)
    parser.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--min_step_tokens",
        type=int,
        default=segment_steps.DEFAULT_MIN_STEP_TOKENS,
        help="Below this size, a step keeps extending regardless of semantic-break signal "
        "(structural/discourse-marker/token-cap boundaries still apply) (default: %(default)s).",
    )
    parser.add_argument(
        "--max_step_tokens",
        type=int,
        default=segment_steps.DEFAULT_MAX_STEP_TOKENS,
        help="Hard cap on step size, a final safety net regardless of boundary signal "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--semantic_percentile",
        type=float,
        default=segment_steps.DEFAULT_SEMANTIC_PERCENTILE,
        help="Bottom percentile of each document's own adjacent-unit similarity distribution "
        "treated as a semantic-jump step boundary (default: %(default)s).",
    )
    parser.add_argument(
        "--min_units_for_semantic",
        type=int,
        default=segment_steps.DEFAULT_MIN_UNITS_FOR_SEMANTIC,
        help="Minimum unit count in a translatable run before the semantic-break fallback "
        "applies at all (default: %(default)s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent in-flight requests to the vllm endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--similarity_floor",
        type=float,
        default=DEFAULT_SIMILARITY_FLOOR,
        help="Minimum embedding cosine similarity between source and translation to accept "
        "without retrying; <= 0 disables the embedding check entirely (default: %(default)s).",
    )
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding_device",
        default=DEFAULT_EMBEDDING_DEVICE,
        help="Device for the embedding model (e.g. cpu, cuda:0) — pick a GPU that isn't already "
        "busy serving the translation model (default: %(default)s).",
    )
    parser.add_argument(
        "--sat_device",
        default=DEFAULT_SAT_DEVICE,
        help="Device for wtpsplit's SaT sentence-splitting model, used by chunking's oversized-"
        "unit split fallback. CPU is ~20-40x slower than GPU for this model and is on the hot "
        "path for every oversized unit — set to a free GPU for any real batch run "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--chunk_workers",
        type=int,
        default=None,
        help="Process pool size for the pre-chunking pass (CPU-bound, embarrassingly parallel "
        "per document). None uses os.cpu_count().",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    set_sat_device(args.sat_device)
    # openai's default httpx client caps max_keepalive_connections at 100 and
    # connect timeout at 5s — with --concurrency in the hundreds, most
    # requests can't reuse a kept-alive connection and have to open a new
    # one, and under sustained high concurrency that can take longer than
    # 5s, producing spurious "Request timed out" errors even though the
    # vllm server itself is healthy and responding (observed: 67% document
    # failure rate at --concurrency 768 with the default client config).
    # Give the connection pool enough headroom for our own concurrency and a
    # much more forgiving connect timeout.
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency),
        timeout=httpx.Timeout(connect=60.0, read=600.0, write=600.0, pool=600.0),
    )
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", http_client=http_client)
    sem = asyncio.Semaphore(args.concurrency)

    done = load_done(args.output_file)
    logger.info(f"Loaded {len(done)} already-translated ids from {args.output_file}")

    ot3_rows = sample_openthoughts3_shards(args.num_openthoughts3, args.seed, done)
    nr_rows = sample_natural_reasoning(args.num_natural_reasoning, args.seed, done)
    jobs: list[tuple[TranslationDataset, str]] = [(row, "openthoughts") for row in ot3_rows] + [
        (row, "naturalreasoning") for row in nr_rows
    ]
    if args.num_opencodereasoning > 0:
        done_after_ot3_nr = done | {row.id for row in ot3_rows} | {row.id for row in nr_rows}
        ocr_rows = sample_opencodereasoning_shards(
            args.num_opencodereasoning, args.seed, done_after_ot3_nr, args.opencodereasoning_config
        )
        jobs += [(row, "opencodereasoning") for row in ocr_rows]

    logger.info(f"Translating {len(jobs)} documents ({len(done)} already done, skipped)")

    logger.info("Pre-chunking all documents (one-time CPU/GPU-bound pass before any translation traffic)...")
    chunked_jobs = chunk_all(jobs, args.min_tokens, args.max_tokens, max_workers=args.chunk_workers)

    logger.info("Regrouping chunks into whole reasoning steps for translation (embedding semantic-break pass)...")
    chunked_jobs = [
        (
            row,
            source,
            segment_steps.regroup_chunked_document(
                doc,
                row.cot_answer,
                args.min_step_tokens,
                args.max_step_tokens,
                args.semantic_percentile,
                args.min_units_for_semantic,
                args.embedding_model,
                args.embedding_device,
            ),
        )
        for row, source, doc in chunked_jobs
    ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.units_output_file.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    completed = 0
    with (
        open(args.output_file, "a", encoding="utf-8") as f,
        open(args.units_output_file, "a", encoding="utf-8") as units_fh,
    ):
        tasks = [
            process_document(client, sem, row, source, doc, args, units_fh, write_lock)
            for row, source, doc in chunked_jobs
        ]
        for coro in asyncio.as_completed(tasks):
            try:
                record = await coro
            except Exception as e:
                logger.error(f"document failed: {e}")
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1
            retried = sum(1 for u in record["units"] if u["retries"] > 0)
            exhausted = sum(1 for u in record["units"] if u["exhausted"])
            logger.info(
                f"[{completed}/{len(jobs)}] id={record['id']} source={record['source']} "
                f"units={len(record['units'])} retried={retried} exhausted={exhausted}"
            )

    logger.info(f"Wrote {completed} documents to {args.output_file} ({args.units_output_file} has per-unit records)")


if __name__ == "__main__":
    asyncio.run(main())
