"""Fixes two known defects in the translated units corpus, in place of a full
re-translation pass:

1. The "Wait" discourse marker was frequently mistranslated as a literal
   "देर" ("delay") variant instead of the natural interjection "रुको". This
   pattern is 100%-deterministically wrong whenever it occurs (verified by
   inspection — see RUN_NOTES / project memory), so it's corrected via a
   direct text substitution rather than re-generation, which only partially
   fixed it even after prompting for the correct register.
2. Units with no translation at all (`hi` is null / `exhausted`) — these
   have no shortcut; they're re-translated for real, from the pre-chunked
   `Unit` reconstructed via re-chunking the parent document (deterministic,
   since chunk_document(text, doc_id, source) always produces the same
   units for the same input).

Writes a corrected copy of the units file; the original is left untouched.

Usage:
    uv run python -m data_gen.fix_translations
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI

from data_gen.chunking import chunk_document
from data_gen.datamodels import TranslationDataset
from data_gen.translate_reasoning import (
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RETRY_TEMPERATURE,
    DEFAULT_SIMILARITY_FLOOR,
    DEFAULT_TARGET_LANGUAGE,
    translate_unit,
)

DEFAULT_UNITS_FILE = Path("translated_reasoning_25k_units.jsonl")
DEFAULT_DOCS_FILE = Path("translated_reasoning_25k.jsonl")
DEFAULT_OUTPUT_FILE = Path("translated_reasoning_25k_units_fixed.jsonl")
DEFAULT_CONCURRENCY = 128

_WAIT_EN_RE = re.compile(r"^\s*Wait\b")
_DER_VERB_PHRASE = r"(?:हो गई|हो गया|करो|करें|कर रहा हूँ|कर रहे हैं|कर लेते हैं|तक|लगेगी|नहीं)?"
_DER_PREFIX_RE = re.compile(r"^(\s*देर\s*" + _DER_VERB_PHRASE + r"\s*[,।!]?\s*){1,3}", re.UNICODE)


def is_wait_der_mistranslation(en: str, hi: str | None) -> bool:
    """Whether a unit is the known "Wait" -> "देर" mistranslation pattern.

    Args:
        en: The unit's source English text.
        hi: The unit's Hindi translation, if any.

    Returns:
        True if this unit matches the known-bad pattern.
    """
    return bool(hi) and bool(_WAIT_EN_RE.match(en)) and hi.strip().startswith("देर")


def fix_wait_der(hi: str) -> str:
    """Replaces the leading "देर..." mistranslation with the natural "रुको,".

    Args:
        hi: The unit's (mistranslated) Hindi text.

    Returns:
        The corrected Hindi text.
    """
    return _DER_PREFIX_RE.sub("रुको, ", hi.strip(), count=1)


def needs_retranslation(row: dict) -> bool:
    """Whether a unit has no usable translation at all.

    Args:
        row: A parsed line from the units JSONL.

    Returns:
        True if `hi` is missing or the unit was marked exhausted.
    """
    return row.get("hi") is None or bool(row.get("exhausted"))


def load_doc_texts(docs_file: Path, doc_ids: set[str]) -> dict[str, TranslationDataset]:
    """Loads the original English CoT text for a set of documents, so they
    can be deterministically re-chunked to recover a missing unit's
    protected/placeholder-annotated text.

    Args:
        docs_file: Path to `translated_reasoning_25k.jsonl`.
        doc_ids: Only documents in this set are loaded.

    Returns:
        Map of doc_id -> a minimal TranslationDataset-like row (id, question,
        cot_answer, source) sufficient to re-chunk.
    """
    out = {}
    with open(docs_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in doc_ids:
                out[row["id"]] = row
    return out


async def retranslate_units(
    unit_ids: set[str],
    docs_file: Path,
    base_url: str,
    model: str,
    concurrency: int,
    embedding_device: str,
) -> dict[str, str]:
    """Re-translates a set of units from scratch via the live vllm endpoint.

    Args:
        unit_ids: unit_ids needing a fresh translation.
        docs_file: Path to `translated_reasoning_25k.jsonl`, for source text.
        base_url: OpenAI-compatible vllm endpoint.
        model: Model name to request.
        concurrency: Max concurrent in-flight requests.
        embedding_device: Device for the embedding similarity check.

    Returns:
        Map of unit_id -> new Hindi translation (only for units that
        resolved successfully; still-failing units are omitted and keep
        their original None).
    """
    doc_ids = {uid.rsplit(":", 1)[0] for uid in unit_ids}
    doc_texts = load_doc_texts(docs_file, doc_ids)
    logger.info(f"re-chunking {len(doc_texts)} documents to recover {len(unit_ids)} missing units")

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency),
        timeout=httpx.Timeout(connect=60.0, read=600.0, write=600.0, pool=600.0),
    )
    client = AsyncOpenAI(base_url=base_url, api_key="none", http_client=http_client)
    sem = asyncio.Semaphore(concurrency)

    async def _fix_one(doc_id: str, row: dict):
        doc = chunk_document(
            row["cot_answer"] if "cot_answer" in row else row["en_cot_answer"],
            doc_id=doc_id,
            source=row["source"],
            min_tokens=DEFAULT_MIN_TOKENS,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        results = {}
        tasks = []
        target_units = [u for u in doc.units if u.unit_id in unit_ids]
        for unit in target_units:
            tasks.append(
                translate_unit(
                    client,
                    sem,
                    unit,
                    model,
                    DEFAULT_MAX_RETRIES,
                    DEFAULT_RETRY_TEMPERATURE,
                    DEFAULT_TARGET_LANGUAGE,
                    DEFAULT_SIMILARITY_FLOOR,
                    DEFAULT_EMBEDDING_MODEL,
                    embedding_device,
                )
            )
        outcomes = await asyncio.gather(*tasks)
        for outcome in outcomes:
            if outcome.translation is not None:
                results[outcome.unit_id] = outcome.translation
        return results

    all_results: dict[str, str] = {}
    completed = 0
    coros = [_fix_one(doc_id, row) for doc_id, row in doc_texts.items()]
    for coro in asyncio.as_completed(coros):
        try:
            results = await coro
        except Exception as e:
            logger.error(f"re-translation failed for a document: {e}")
            continue
        all_results.update(results)
        completed += 1
        if completed % 200 == 0:
            logger.info(f"re-translated {completed}/{len(doc_texts)} documents ({len(all_results)} units fixed so far)")

    logger.info(f"re-translation done: {len(all_results)}/{len(unit_ids)} units recovered")
    return all_results


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units_file", type=Path, default=DEFAULT_UNITS_FILE)
    parser.add_argument("--docs_file", type=Path, default=DEFAULT_DOCS_FILE)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--embedding_device", default=DEFAULT_EMBEDDING_DEVICE)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    logger.info(f"Scanning {args.units_file} for known-bad patterns")
    rows = []
    wait_der_count = 0
    missing_unit_ids: set[str] = set()
    with open(args.units_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            if is_wait_der_mistranslation(row["en"], row.get("hi")):
                wait_der_count += 1
            elif needs_retranslation(row):
                missing_unit_ids.add(row["unit_id"])
    logger.info(f"{wait_der_count} 'Wait'->'देर' mistranslations (fixed via substitution)")
    logger.info(f"{len(missing_unit_ids)} missing/exhausted units (need re-translation)")

    retranslated = await retranslate_units(
        missing_unit_ids, args.docs_file, args.base_url, args.model, args.concurrency, args.embedding_device
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fixed_wait = fixed_missing = 0
    with open(args.output_file, "w", encoding="utf-8") as out:
        for row in rows:
            if is_wait_der_mistranslation(row["en"], row.get("hi")):
                row["hi"] = fix_wait_der(row["hi"])
                row["had_defect"] = True
                fixed_wait += 1
            elif row["unit_id"] in retranslated:
                row["hi"] = retranslated[row["unit_id"]]
                row["exhausted"] = False
                row["had_defect"] = True
                fixed_missing += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(rows)} units to {args.output_file}")
    logger.info(f"fixed: {fixed_wait} wait/देर substitutions, {fixed_missing}/{len(missing_unit_ids)} re-translated")


if __name__ == "__main__":
    asyncio.run(main())
