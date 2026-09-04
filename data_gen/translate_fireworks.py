"""Chunks OpenThoughts3/natural_reasoning/OpenCodeReasoning documents into
translation-ready steps, then translates them via Fireworks - batch or
online, whichever `data_gen.config.USE_BATCH_API` selects.

Pipeline: sample the 3 sources -> chunk_document (markdown-aware AST
chunking, protection, oversized-unit splitting - all budgeted against
config.MAX_TOKENS) -> regroup_chunked_documents (merges chunks into coherent
steps, still capped at config.MAX_STEP_TOKENS) -> one translation request
per translatable step.

Always writes, under --output_dir:
  - chunked_docs.jsonl: the full chunked+regrouped ChunkedDocument per
    source document (doc_id, gaps, units...) - needed to splice
    translations back into whole documents via chunking.reconstruct().
    Also what resumability is based on (a doc already here is skipped).

Then, depending on config.USE_BATCH_API:
  - True: batch_requests_NNN.jsonl - Fireworks Batch Inference API rows
    ({"custom_id", "body": {"messages": ...}}), ready to upload as a
    Fireworks Dataset and reference from a BatchInferenceJob. No network
    calls made - see https://docs.fireworks.ai/guides/batch-inference for
    submitting them.
  - False: translated_units.jsonl - {"custom_id", "hi"} rows, translated
    directly via Fireworks' synchronous chat completions endpoint
    (FIREWORKS_API_KEY required in .env).

Usage:
    uv run python -m data_gen.translate_fireworks \
        --num_openthoughts3 1000 --num_natural_reasoning 1000
"""

import argparse
import asyncio
import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from data_gen import config
from data_gen.chunking import ChunkedDocument, Unit, chunk_document, set_sat_device
from data_gen.datamodels import TranslationDataset
from data_gen.openai_client import AsyncChatClient
from data_gen.sample_reasoning import (
    sample_natural_reasoning,
    sample_opencodereasoning_shards,
    sample_openthoughts3_shards,
)
from data_gen.segment_steps import regroup_chunked_documents

DEFAULT_OUTPUT_DIR = Path("fireworks_batch")
DEFAULT_MAX_REQUESTS_PER_FILE = 40_000
DEFAULT_ONLINE_CONCURRENCY = 32
# Hindi (Devanagari) can need more tokens than the English source did under
# Fireworks' tokenizer, so max_tokens is sized off the request, not a flat
# constant - this multiplier is a safety margin against truncation.
_OUTPUT_TOKEN_MULTIPLIER = 3
_MIN_OUTPUT_TOKENS = 256
_MAX_OUTPUT_TOKENS = 4096

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(loader=FileSystemLoader(_PROMPTS_DIR), trim_blocks=True, lstrip_blocks=True)
_PRIOR_CONTEXT_ALNUM_CHARS = 100


def render_translate_prompt(prior_context: str | None = None) -> str:
    """Renders the translation system prompt (data_gen/prompts/translate.jinja)."""
    return _JINJA_ENV.get_template("translate.jinja").render(
        target_language=config.TARGET_LANGUAGE, prior_context=prior_context
    )


def prior_context(doc: ChunkedDocument, unit: Unit, max_alnum_chars: int = _PRIOR_CONTEXT_ALNUM_CHARS) -> str | None:
    """Gets trailing context from the unit immediately before `unit` in
    document order, for disambiguation only (e.g. so a discourse marker like
    "Wait" reads as a mid-reasoning interjection rather than in isolation).

    Sized by alphanumeric character count rather than raw character count,
    so markdown noise (punctuation, list markers, whitespace) at the
    boundary doesn't eat into the budget of actually meaningful context.

    Uses the immediately preceding unit regardless of its `kind`/`translate`
    status (even a code unit's tail can carry useful continuity), and always
    the original English (never a translation) - translations don't exist
    yet at request-build time.

    Args:
        doc: `unit`'s parent ChunkedDocument.
        unit: The unit about to be translated.
        max_alnum_chars: Minimum alphanumeric characters of trailing context
            to include (the returned string can be slightly longer, since
            non-alphanumeric characters in the window aren't dropped).

    Returns:
        The trailing context string, trimmed to a clean word boundary, or
        None if this is the document's first unit (nothing precedes it).
    """
    if unit.index == 0:
        return None
    prior_text = doc.units[unit.index - 1].text_raw
    alnum_seen = 0
    start = len(prior_text)
    while start > 0 and alnum_seen < max_alnum_chars:
        start -= 1
        if prior_text[start].isalnum():
            alnum_seen += 1
    tail = prior_text[start:]
    # Trim to a clean word boundary rather than starting mid-word.
    space_idx = tail.find(" ")
    if 0 <= space_idx < len(tail) - 1:
        tail = tail[space_idx + 1 :]
    return tail.strip() or None


def load_done_doc_ids(chunked_docs_file: Path) -> set[str]:
    """Doc ids already chunked in a prior run, so a rerun can resume.

    Args:
        chunked_docs_file: Path to the (possibly not-yet-existing) chunked_docs.jsonl.

    Returns:
        Set of doc_ids already present.
    """
    if not chunked_docs_file.exists():
        return set()
    done = set()
    with open(chunked_docs_file, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _chunk_one(
    row: TranslationDataset, source: str
) -> tuple[TranslationDataset, str, ChunkedDocument | None, str | None]:
    """Process-pool worker: chunks one document, returning any error as a
    string instead of raising, so one bad document can't take down a worker.
    """
    try:
        doc = chunk_document(
            row.cot_answer, doc_id=row.id, source=source, min_tokens=config.MIN_TOKENS, max_tokens=config.MAX_TOKENS
        )
        return row, source, doc, None
    except Exception as e:
        return row, source, None, str(e)


def chunk_all(
    jobs: list[tuple[TranslationDataset, str]], max_workers: int | None = None, log_every: int = 1000
) -> list[tuple[TranslationDataset, str, ChunkedDocument]]:
    """Pre-chunks every sampled document in parallel (CPU-bound: AST
    parsing, tiktoken encoding, wtpsplit for oversized units).

    Args:
        jobs: (row, source) pairs from the samplers.
        max_workers: Process pool size. None uses os.cpu_count().
        log_every: Log progress every this many completed documents.

    Returns:
        (row, source, doc) triples, in completion order (not `jobs` order).
    """
    results = []
    failed = 0
    completed = 0
    # Force CPU inside worker processes: a worker that hits the wtpsplit
    # split path would otherwise lazily load its own SaT model copy onto
    # whatever GPU the main process configured, and many workers doing this
    # concurrently can OOM a shared GPU.
    with ProcessPoolExecutor(max_workers=max_workers, initializer=set_sat_device, initargs=("cpu",)) as executor:
        futures = [executor.submit(_chunk_one, row, source) for row, source in jobs]
        for future in as_completed(futures):
            row, source, doc, error = future.result()
            completed += 1
            if error is not None:
                failed += 1
                logger.error(f"chunk_document failed for doc_id={row.id}: {error}")
            else:
                results.append((row, source, doc))
            if completed % log_every == 0:
                logger.info(f"chunked {completed}/{len(jobs)} documents ({failed} failed)")
    logger.info(f"chunked {len(results)}/{len(jobs)} documents ({failed} failed)")
    return results


def build_requests(doc: ChunkedDocument) -> list[dict]:
    """One Fireworks batch request row per translatable unit in `doc`.

    Each request's system prompt is rendered per-unit, with up to
    `_PRIOR_CONTEXT_ALNUM_CHARS` of the immediately preceding unit's English
    text folded in as disambiguation context - see `prior_context`.

    Args:
        doc: A regrouped ChunkedDocument (see `regroup_chunked_documents`).

    Returns:
        Fireworks batch rows: {"custom_id", "body": {"messages", "max_tokens", "temperature"}}.
        Kept to exactly Fireworks' documented row schema - input-token
        metadata for outlier-spotting goes in a parallel file, see
        `build_request_meta`.
    """
    requests = []
    for unit in doc.units:
        if not unit.translate:
            continue
        max_tokens = min(
            _MAX_OUTPUT_TOKENS, max(_MIN_OUTPUT_TOKENS, unit.token_count * _OUTPUT_TOKEN_MULTIPLIER)
        )
        system_prompt = render_translate_prompt(prior_context(doc, unit))
        requests.append(
            {
                "custom_id": unit.unit_id,
                "body": {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": unit.text_protected},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            }
        )
    return requests


def build_request_meta(doc: ChunkedDocument) -> list[dict]:
    """One metadata row per translatable unit in `doc`, parallel to
    `build_requests`' rows (same custom_id) but kept out of the Fireworks
    batch rows themselves, whose schema Fireworks documents exactly as
    {"custom_id", "body"}.

    input_tokens is `chunking.py`'s protected-text token count (what's
    actually sent) - once translations come back, compare each row's output
    token count against this to flag suspiciously huge (repetition/
    hallucination) or tiny (truncated/empty) translations before they reach
    a training or eval set.

    Args:
        doc: A regrouped ChunkedDocument (see `regroup_chunked_documents`).

    Returns:
        {"custom_id", "doc_id", "source", "input_tokens"} rows.
    """
    return [
        {
            "custom_id": unit.unit_id,
            "doc_id": doc.doc_id,
            "source": doc.source,
            "input_tokens": unit.token_count,
        }
        for unit in doc.units
        if unit.translate
    ]


def summarize_input_tokens(meta_rows: list[dict]) -> dict:
    """Aggregate input-token stats across a whole run, for a fast
    order-of-magnitude sanity check (cost, and a wildly off total/max
    hinting at a chunking regression) without scanning request_meta.jsonl.

    Args:
        meta_rows: Rows from `build_request_meta`, across all documents.

    Returns:
        {"total_requests", "total_input_tokens", "min", "p50", "p90", "max"}
        ("min"/"p50"/"p90"/"max" over individual requests' input_tokens).
    """
    counts = sorted(row["input_tokens"] for row in meta_rows)
    n = len(counts)
    if n == 0:
        return {"total_requests": 0, "total_input_tokens": 0, "min": 0, "p50": 0, "p90": 0, "max": 0}
    return {
        "total_requests": n,
        "total_input_tokens": sum(counts),
        "min": counts[0],
        "p50": counts[n // 2],
        "p90": counts[int(n * 0.9)],
        "max": counts[-1],
    }


async def translate_online(requests: list[dict], concurrency: int, output_file: Path) -> None:
    """Translates every request directly via Fireworks' synchronous chat
    completions endpoint, concurrently, writing results as they land.

    No retry, no validation - config.USE_BATCH_API's whole point is to
    trade that off against not needing a batch job/poll/download cycle.

    Args:
        requests: Rows from `build_requests` ({"custom_id", "body": {...}}).
        concurrency: Max in-flight requests.
        output_file: Appended with one {"custom_id", "hi"} JSON line per
            completed request.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set in environment or .env")
    client = AsyncChatClient(
        base_url=config.FIREWORKS_ONLINE_BASE_URL, api_key=api_key, model=config.FIREWORKS_MODEL, concurrency=concurrency
    )
    write_lock = asyncio.Lock()
    completed = 0

    async def _one(req: dict) -> None:
        nonlocal completed
        hi = await client.complete(**req["body"])
        async with write_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"custom_id": req["custom_id"], "hi": hi}, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 500 == 0:
                logger.info(f"translated {completed}/{len(requests)}")

    await asyncio.gather(*(_one(req) for req in requests))
    logger.info(f"Wrote {completed} translations to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_openthoughts3", type=int, default=100)
    parser.add_argument("--num_natural_reasoning", type=int, default=100)
    parser.add_argument(
        "--num_opencodereasoning",
        type=int,
        default=0,
        help="Number of nvidia/OpenCodeReasoning rows to sample (default: %(default)s - opt-in).",
    )
    parser.add_argument("--opencodereasoning_config", default="split_0", choices=["split_0", "split_1"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling (default: %(default)s).")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_requests_per_file", type=int, default=DEFAULT_MAX_REQUESTS_PER_FILE)
    parser.add_argument(
        "--chunk_workers", type=int, default=None, help="Process pool size for chunking. None uses os.cpu_count()."
    )
    parser.add_argument(
        "--online_concurrency",
        type=int,
        default=DEFAULT_ONLINE_CONCURRENCY,
        help="[online mode only] Max in-flight requests to Fireworks (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunked_docs_file = args.output_dir / "chunked_docs.jsonl"

    done = load_done_doc_ids(chunked_docs_file)
    logger.info(f"Loaded {len(done)} already-chunked doc ids from {chunked_docs_file}")

    ot3_rows = sample_openthoughts3_shards(args.num_openthoughts3, args.seed, done)
    nr_rows = sample_natural_reasoning(args.num_natural_reasoning, args.seed, done)
    jobs: list[tuple[TranslationDataset, str]] = [(r, "openthoughts") for r in ot3_rows] + [
        (r, "naturalreasoning") for r in nr_rows
    ]
    if args.num_opencodereasoning > 0:
        done_after = done | {r.id for r in ot3_rows} | {r.id for r in nr_rows}
        ocr_rows = sample_opencodereasoning_shards(
            args.num_opencodereasoning, args.seed, done_after, args.opencodereasoning_config
        )
        jobs += [(r, "opencodereasoning") for r in ocr_rows]
    logger.info(f"Sampled {len(jobs)} new documents ({len(done)} already done, skipped)")

    logger.info("Chunking documents (parallel, CPU-bound pass)...")
    chunked = chunk_all(jobs, max_workers=args.chunk_workers)

    logger.info(f"Regrouping units into steps ({len(chunked)} documents, one batched embedding pass)...")
    regrouped_docs = regroup_chunked_documents(
        [(doc, row.cot_answer) for row, _source, doc in chunked],
        min_step_tokens=config.MIN_STEP_TOKENS,
        max_step_tokens=config.MAX_STEP_TOKENS,
        semantic_percentile=config.SEMANTIC_PERCENTILE,
        min_units_for_semantic=config.MIN_UNITS_FOR_SEMANTIC,
        embedding_model=config.EMBEDDING_MODEL,
        embedding_backend=config.EMBEDDING_BACKEND,
        embedding_base_url=config.EMBEDDING_BASE_URL,
        embedding_device=config.EMBEDDING_DEVICE,
        embed_batch_size=config.EMBED_BATCH_SIZE,
    )
    with open(chunked_docs_file, "a", encoding="utf-8") as f:
        for regrouped in regrouped_docs:
            f.write(json.dumps(dataclasses.asdict(regrouped), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(regrouped_docs)} chunked documents to {chunked_docs_file}")

    all_requests = []
    all_meta = []
    for doc in regrouped_docs:
        all_requests.extend(build_requests(doc))
        all_meta.extend(build_request_meta(doc))
    logger.info(f"Built {len(all_requests)} translation requests")

    meta_file = args.output_dir / "request_meta.jsonl"
    with open(meta_file, "a", encoding="utf-8") as f:
        for row in all_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Manifest is a fresh summary of *this run's* requests, not the
    # cumulative resumed total (unlike chunked_docs.jsonl/request_meta.jsonl,
    # which append) - it's meant to be eyeballed right after this run.
    summary = summarize_input_tokens(all_meta)
    manifest_file = args.output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({"documents": len(regrouped_docs), "input_tokens": summary}, f, indent=2)
    logger.info(
        f"Input tokens this run: total={summary['total_input_tokens']} "
        f"min={summary['min']} p50={summary['p50']} p90={summary['p90']} max={summary['max']} "
        f"(see {manifest_file} / {meta_file})"
    )

    if config.USE_BATCH_API:
        n_files = 0
        for i in range(0, len(all_requests), args.max_requests_per_file):
            batch = all_requests[i : i + args.max_requests_per_file]
            out_file = args.output_dir / f"batch_requests_{n_files:03d}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for req in batch:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(batch)} requests to {out_file}")
            n_files += 1
        logger.info(
            f"Done: {len(regrouped_docs)} documents, {len(all_requests)} requests, {n_files} batch file(s) "
            f"in {args.output_dir}"
        )
    else:
        output_file = args.output_dir / "translated_units.jsonl"
        asyncio.run(translate_online(all_requests, args.online_concurrency, output_file))
        logger.info(f"Done: {len(regrouped_docs)} documents, {len(all_requests)} requests translated online")


if __name__ == "__main__":
    main()
