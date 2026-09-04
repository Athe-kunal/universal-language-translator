"""
Translate OpenR1-Math-220k to Hindi using the OpenAI Batch API (via litellm).
Submits one batch job at a time: build → upload → create → poll → parse → write → next.

Outputs the same translation_chunked.jsonl format as translate_ds.py.
State is persisted in batch_jobs.jsonl so incomplete batches are resumed on restart.

Usage:
    uv run python data_gen/translate_ds_batch.py
    uv run python data_gen/translate_ds_batch.py --num_examples 500
    uv run python data_gen/translate_ds_batch.py --dry_run
    uv run python data_gen/translate_ds_batch.py --batch_size 20000
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import litellm
from datasets import load_dataset

from data_gen.openai_client import download_batch, poll_batch, submit_batch
from translate_ds import TRANSLATE_PROMPT, load_done, problem_id, split_paragraphs, strip_think

CACHE_FILE = Path("translation_chunked.jsonl")
JOBS_FILE = Path("batch_jobs.jsonl")
LOG_FILE = Path("translation_batch.log")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Cache / state helpers
# ---------------------------------------------------------------------------

def load_jobs() -> list[dict]:
    if not JOBS_FILE.exists():
        return []
    jobs = []
    with open(JOBS_FILE) as f:
        for line in f:
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return jobs


def append_job(job: dict) -> None:
    with open(JOBS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")


def update_job_status(batch_id: str, status: str, output_file_id: str | None = None) -> None:
    jobs = load_jobs()
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        for job in jobs:
            if job["batch_id"] == batch_id:
                job["status"] = status
                if output_file_id:
                    job["output_file_id"] = output_file_id
            f.write(json.dumps(job) + "\n")


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def build_requests(examples: list[tuple[int, dict]]) -> tuple[list[dict], dict]:
    """
    Build batch JSONL request dicts for all chunks of all examples.

    Custom ID format: {pid}_{p|s}_{chunk_index}
      p = problem chunk, s = solution chunk

    Returns (requests, chunk_map) where chunk_map[custom_id] = (pid, field, idx)
    so we can reconstruct entries after results come back.
    """
    requests = []
    chunk_map: dict[str, tuple[str, str, int]] = {}

    for _, ex in examples:
        pid = problem_id(ex["problem"])
        problem = ex["problem"]
        solution = strip_think(
            next(g for g, ok in zip(ex["generations"], ex["correctness_math_verify"]) if ok)
        )

        for field_key, text in (("p", problem), ("s", solution)):
            chunks = split_paragraphs(text)
            for idx, chunk in enumerate(chunks):
                cid = f"{pid}_{field_key}_{idx}"
                chunk_map[cid] = (pid, field_key, idx)
                requests.append({
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": TRANSLATE_PROMPT},
                            {"role": "user",   "content": chunk},
                        ],
                        "temperature": 0.0,
                    },
                })

    return requests, chunk_map


def build_example_meta(examples: list[tuple[int, dict]]) -> dict[str, dict]:
    """Store the original English text and chunk lists keyed by problem_id."""
    meta: dict[str, dict] = {}
    for _, ex in examples:
        pid = problem_id(ex["problem"])
        problem = ex["problem"]
        solution = strip_think(
            next(g for g, ok in zip(ex["generations"], ex["correctness_math_verify"]) if ok)
        )
        meta[pid] = {
            "problem": problem,
            "solution": solution,
            "problem_chunks_en": split_paragraphs(problem),
            "solution_chunks_en": split_paragraphs(solution),
        }
    return meta


# submit_batch / poll_batch / download_batch are shared with
# translate_fireworks.py's OpenAI-Batch-API path via data_gen.openai_client.

# ---------------------------------------------------------------------------
# Assemble final entries
# ---------------------------------------------------------------------------

def assemble_entries(
    examples: list[tuple[int, dict]],
    translations: dict[str, str],
    log: logging.Logger,
) -> list[dict]:
    meta = build_example_meta(examples)
    entries = []
    missing = 0

    for pid, m in meta.items():
        problem_chunks = []
        for idx, en in enumerate(m["problem_chunks_en"]):
            hi = translations.get(f"{pid}_p_{idx}", "")
            if not hi:
                missing += 1
            problem_chunks.append({"en": en, "hi": hi})

        solution_chunks = []
        for idx, en in enumerate(m["solution_chunks_en"]):
            hi = translations.get(f"{pid}_s_{idx}", "")
            if not hi:
                missing += 1
            solution_chunks.append({"en": en, "hi": hi})

        entries.append({
            "id": pid,
            "problem": m["problem"],
            "problem_chunks": problem_chunks,
            "solution": m["solution"],
            "solution_chunks": solution_chunks,
        })

    if missing:
        log.warning(f"{missing} chunks had no translation (empty or errored)")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(num_examples: int | None, batch_size: int, dry_run: bool, poll_interval: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    log = logging.getLogger("translate_batch")

    # --- Resume: re-poll any in-progress batches from a prior run ---
    jobs = load_jobs()
    for job in jobs:
        if job["status"] == "submitted":
            log.info(f"Resuming in-progress batch {job['batch_id']}…")
            try:
                output_file_id = await poll_batch(job["batch_id"], log, poll_interval)
                translations = await download_batch(output_file_id, log)

                # Reload examples for this job's problem IDs (already-done check skips them after)
                done = load_done()
                ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
                job_pids = set(job.get("problem_ids", []))
                resume_examples = [
                    (i, ex) for i, ex in enumerate(ds)
                    if any(ex["correctness_math_verify"])
                    and problem_id(ex["problem"]) in job_pids
                    and problem_id(ex["problem"]) not in done
                ]

                entries = assemble_entries(resume_examples, translations, log)
                with open(CACHE_FILE, "a", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                update_job_status(job["batch_id"], "completed", output_file_id)
                log.info(f"Resumed batch {job['batch_id']}: wrote {len(entries)} entries")
            except Exception as e:
                log.error(f"Failed to resume batch {job['batch_id']}: {e}")
                update_job_status(job["batch_id"], "failed")

    # --- Load remaining work ---
    done = load_done()
    log.info(f"Loaded {len(done)} already-translated entries from cache")

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    examples = [
        (i, ex) for i, ex in enumerate(ds)
        if any(ex["correctness_math_verify"]) and problem_id(ex["problem"]) not in done
    ]
    if num_examples is not None:
        examples = examples[:num_examples]
        log.info(f"Capped to {num_examples} examples")

    log.info(f"Queuing {len(examples)} examples ({len(done)} skipped)")

    # --- Slice into batches and process sequentially ---
    slices = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    log.info(f"Will submit {len(slices)} batch job(s) of up to {batch_size} requests each")

    if dry_run:
        total_requests = 0
        for sl in slices:
            reqs, _ = build_requests(sl)
            total_requests += len(reqs)
        print(f"[dry_run] Would submit {total_requests} requests in {len(slices)} batch job(s)")
        return

    for batch_num, sl in enumerate(slices, 1):
        log.info(f"--- Batch job {batch_num}/{len(slices)} ({len(sl)} examples) ---")
        requests, _ = build_requests(sl)
        pids = [problem_id(ex["problem"]) for _, ex in sl]

        batch_id = await submit_batch(requests, log)
        append_job({"batch_id": batch_id, "status": "submitted", "output_file_id": None, "problem_ids": pids})

        output_file_id = await poll_batch(batch_id, log, poll_interval)
        translations = await download_batch(output_file_id, log)

        entries = assemble_entries(sl, translations, log)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        update_job_status(batch_id, "completed", output_file_id)
        log.info(f"Batch job {batch_num}/{len(slices)} done — wrote {len(entries)} entries")

    log.info("All batch jobs complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--batch_size",   type=int, default=40_000, help="Max requests per batch job")
    parser.add_argument("--poll_interval", type=int, default=60,   help="Seconds between status polls")
    parser.add_argument("--dry_run", action="store_true",           help="Preview without submitting")
    args = parser.parse_args()
    asyncio.run(main(args.num_examples, args.batch_size, args.dry_run, args.poll_interval))
