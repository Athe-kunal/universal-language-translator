"""Translates chunked OpenThoughts3 / natural_reasoning documents to Hindi.

Samples `open-thoughts/OpenThoughts3-1.2M` (stratified by domain/source/
difficulty) and `facebook/natural_reasoning` (uniform) via
`data_gen.sample_reasoning`, chunks each `cot_answer` via
`data_gen.chunking.chunk_document`, translates every translatable unit
through an OpenAI-compatible vllm endpoint, and reconstructs the full Hindi
document. Prompts are Jinja templates under `data_gen/prompts/`, so wording
changes don't require touching this module.

A unit whose translation degenerates into a repeated-token/phrase loop (a
known decoding failure), drops or alters a protected placeholder, or scores
below `--similarity_floor` on LaBSE cross-lingual embedding similarity (a
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
import threading
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger
from openai import AsyncOpenAI

from data_gen.chunking import ChunkedDocument, Unit, chunk_document, reconstruct, set_sat_device
from data_gen.datamodels import TranslationDataset
from data_gen.sample_reasoning import (
    load_done,
    sample_natural_reasoning,
    sample_openthoughts3_streaming,
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
DEFAULT_LABSE_MODEL = "sentence-transformers/LaBSE"
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


def render_translate_prompt(target_language: str = DEFAULT_TARGET_LANGUAGE) -> str:
    """Renders the base translation system prompt.

    Args:
        target_language: Language to translate into.

    Returns:
        Rendered prompt text.
    """
    return _JINJA_ENV.get_template("translate.jinja").render(target_language=target_language)


def render_retry_prompt(
    previous_translation: str, reason: str, target_language: str = DEFAULT_TARGET_LANGUAGE
) -> str:
    """Renders the retry system prompt, showing the previous broken attempt.

    Args:
        previous_translation: The prior attempt that failed validation.
        reason: Human-readable description of what was wrong with it.
        target_language: Language to translate into.

    Returns:
        Rendered prompt text.
    """
    return _JINJA_ENV.get_template("translate_retry.jinja").render(
        target_language=target_language, previous_translation=previous_translation, reason=reason
    )


_PLACEHOLDER_RE = re.compile(r"⟦\d+⟧")

_labse_model = None
_labse_lock = threading.Lock()


def _get_labse_model(model_name: str, device: str):
    """Lazily loads and caches the LaBSE cross-lingual embedding model, so
    importing this module doesn't pay the model-load cost when validation
    is disabled (`similarity_floor <= 0`).

    Guarded by a real (non-async) lock: `_encode` runs inside
    `asyncio.to_thread`, so with concurrency > 1 many OS threads can reach
    this function on their first call simultaneously — without the lock,
    multiple threads racing to construct the same model onto the same
    device corrupts the load (observed as a "meta tensor" error).

    Args:
        model_name: SentenceTransformer model name.
        device: Device to load the model onto (e.g. "cpu", "cuda:0").

    Returns:
        A loaded `SentenceTransformer` instance.
    """
    global _labse_model
    if _labse_model is None:
        with _labse_lock:
            if _labse_model is None:
                from sentence_transformers import SentenceTransformer

                _labse_model = SentenceTransformer(model_name, device=device)
    return _labse_model


async def _embedding_similarity(en_text: str, hi_text: str, model_name: str, device: str) -> float:
    """Computes cosine similarity between source and translation embeddings.

    Runs the (synchronous, GPU-or-CPU-bound) encode call in a thread so it
    doesn't block the asyncio event loop other translation tasks are using.

    Args:
        en_text: Source text sent to the translator.
        hi_text: Candidate translation text.
        model_name: SentenceTransformer model name.
        device: Device to run the model on.

    Returns:
        Cosine similarity in [-1, 1] (in practice, [0, 1]-ish for real text).
    """

    def _encode() -> float:
        model = _get_labse_model(model_name, device)
        embeddings = model.encode([en_text, hi_text], normalize_embeddings=True)
        return float(embeddings[0] @ embeddings[1])

    return await asyncio.to_thread(_encode)


async def _validate(
    translation: str,
    expected_placeholders: list[str],
    en_text: str,
    similarity_floor: float,
    labse_model: str,
    embedding_device: str,
) -> str | None:
    """Checks a translation attempt for known defect signatures.

    Args:
        translation: Candidate translation text.
        expected_placeholders: Sorted, deduped placeholder ids the unit's
            fingerprint expects (e.g. ["⟦0⟧", "⟦1⟧"]).
        en_text: The source text sent to the translator, for the embedding
            similarity check.
        similarity_floor: Minimum LaBSE cosine similarity to accept; a
            non-positive value disables the check entirely (skips loading
            the embedding model at all).
        labse_model: SentenceTransformer model name for the similarity check.
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
        similarity = await _embedding_similarity(en_text, translation, labse_model, embedding_device)
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
    labse_model: str = DEFAULT_LABSE_MODEL,
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
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
        similarity_floor: Minimum LaBSE cosine similarity to accept; a
            non-positive value disables the embedding check.
        labse_model: SentenceTransformer model name for the similarity check.
        embedding_device: Device to run the embedding model on.

    Returns:
        The final UnitTranslation, with retry/erroneous-attempt provenance.
        `translation` is None if every attempt failed validation — the
        caller is expected to fall back to the original English for this
        unit rather than accept a known-broken translation.
    """
    expected_placeholders = sorted(set(span.placeholder for span in unit.spans))
    base_prompt = render_translate_prompt(target_language)
    previous_translation: str | None = None
    translation = ""
    reason: str | None = None

    for attempt in range(max_retries + 1):
        system_prompt = (
            base_prompt
            if attempt == 0
            else render_retry_prompt(previous_translation, reason, target_language)
        )
        async with sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": unit.text_protected},
                ],
                temperature=0.0 if attempt == 0 else retry_temperature,
                max_tokens=1024,
            )
        translation = (resp.choices[0].message.content or "").strip()
        reason = await _validate(
            translation,
            expected_placeholders,
            unit.text_protected,
            similarity_floor,
            labse_model,
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
    labse_model: str,
    embedding_device: str,
    doc_id: str,
    source: str,
    units_fh,
    write_lock: asyncio.Lock,
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
        labse_model,
        embedding_device,
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
    labse_model: str = DEFAULT_LABSE_MODEL,
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
    units_fh=None,
    write_lock: asyncio.Lock | None = None,
) -> dict[str, UnitTranslation]:
    """Translates every translatable unit in a document concurrently.

    Args:
        client: Async OpenAI-compatible client.
        sem: Concurrency-limiting semaphore, shared across the whole run.
        doc: A ChunkedDocument from `chunk_document`.
        model: Model name to request.
        max_retries: Max retry attempts per unit.
        retry_temperature: Sampling temperature for retry attempts.
        target_language: Language to translate into.
        similarity_floor: Minimum LaBSE cosine similarity to accept; a
            non-positive value disables the embedding check.
        labse_model: SentenceTransformer model name for the similarity check.
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
                labse_model,
                embedding_device,
                doc.doc_id,
                doc.source,
                units_fh,
                write_lock,
            )
            for unit in translatable
        )
    )
    return {result.unit_id: result for result in results}


def chunk_all(
    jobs: list[tuple[TranslationDataset, str]], min_tokens: int, max_tokens: int, log_every: int = 1000
) -> list[tuple[TranslationDataset, str, ChunkedDocument]]:
    """Pre-chunks every sampled document in a single synchronous pass, before
    any translation task is created.

    This is a deliberate two-phase design: chunk_document is CPU-bound (AST
    parsing, tiktoken encoding, wtpsplit for oversized units), and running it
    lazily inside each async translation task serializes a large batch
    entirely through chunking before any HTTP traffic can flow — observed:
    a 50K-document batch spent 35+ minutes at 0% vllm GPU utilization while
    chunking worked through its backlog, one task at a time, on the
    single-threaded event loop. Chunking everything up front instead (call
    `chunking.set_sat_device()` to a free GPU first — CPU wtpsplit inference
    is ~20-40x slower) means the translation phase starts against
    already-chunked documents and never blocks on chunking again.

    Args:
        jobs: (row, source) pairs from the samplers.
        min_tokens: Passed through to chunk_document.
        max_tokens: Passed through to chunk_document.
        log_every: Log progress every this many documents.

    Returns:
        (row, source, doc) triples, same order as `jobs`.
    """
    results = []
    for i, (row, source) in enumerate(jobs):
        doc = chunk_document(row.cot_answer, doc_id=row.id, source=source, min_tokens=min_tokens, max_tokens=max_tokens)
        results.append((row, source, doc))
        if (i + 1) % log_every == 0:
            logger.info(f"chunked {i + 1}/{len(jobs)} documents")
    logger.info(f"chunked {len(results)}/{len(jobs)} documents")
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
        args.labse_model,
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
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible stratified/uniform sampling, matching sample_reasoning.py "
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
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target_language", default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_temperature", type=float, default=DEFAULT_RETRY_TEMPERATURE)
    parser.add_argument("--min_tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--similarity_floor",
        type=float,
        default=DEFAULT_SIMILARITY_FLOOR,
        help="Minimum LaBSE cosine similarity between source and translation to accept "
        "without retrying; <= 0 disables the embedding check entirely (default: %(default)s).",
    )
    parser.add_argument("--labse_model", default=DEFAULT_LABSE_MODEL)
    parser.add_argument(
        "--embedding_device",
        default=DEFAULT_EMBEDDING_DEVICE,
        help="Device for the LaBSE model (e.g. cpu, cuda:0) — pick a GPU that isn't already "
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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    set_sat_device(args.sat_device)
    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    sem = asyncio.Semaphore(args.concurrency)

    done = load_done(args.output_file)
    logger.info(f"Loaded {len(done)} already-translated ids from {args.output_file}")

    ot3_rows = sample_openthoughts3_streaming(args.num_openthoughts3, args.seed, done)
    nr_rows = sample_natural_reasoning(args.num_natural_reasoning, args.seed, done)
    jobs: list[tuple[TranslationDataset, str]] = [(row, "openthoughts") for row in ot3_rows] + [
        (row, "naturalreasoning") for row in nr_rows
    ]

    logger.info(f"Translating {len(jobs)} documents ({len(done)} already done, skipped)")

    logger.info("Pre-chunking all documents (one-time CPU/GPU-bound pass before any translation traffic)...")
    chunked_jobs = chunk_all(jobs, args.min_tokens, args.max_tokens)

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
