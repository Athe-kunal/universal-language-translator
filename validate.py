"""
Run our trained checkpoint on N examples not yet in translation_cache.jsonl,
translating all texts in large parallel batches. Save to validation.jsonl.

Usage:
    uv run python validate.py
    uv run python validate.py --checkpoint .models/modernbert-translation/checkpoint-24942 --n 50 --batch_size 64
"""

import argparse
import asyncio
import json
from pathlib import Path

from datasets import load_dataset

from data_gen.translate_ds import problem_id, strip_think, split_paragraphs, load_done
from translate import ScriptArguments, SamplerConfig, load_pipeline, translate_batch


async def writer(queue: asyncio.Queue, out: Path) -> None:
    with open(out, "w", encoding="utf-8") as f:
        while True:
            entry = await queue.get()
            if entry is None:
                queue.task_done()
                break
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [saved] {entry['id']}")
            queue.task_done()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=".models/modernbert-translation-bs16/checkpoint-24942")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--out", default="validation.jsonl")
    args = parser.parse_args()

    out = Path(args.out)

    print(f"Loading model from {args.checkpoint} ...")
    model_args = ScriptArguments(model_name_or_path=args.checkpoint)
    _, tokenizer, sampler = load_pipeline(model_args)
    config = SamplerConfig(max_new_tokens=args.max_new_tokens)

    done = load_done()
    print(f"Cache: {len(done)} already translated — picking {args.n} unseen examples")

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    examples = []
    for ex in ds:
        if len(examples) >= args.n:
            break
        if not any(ex["correctness_math_verify"]):
            continue
        if problem_id(ex["problem"]) in done:
            continue
        examples.append(ex)

    # Build one flat list of all texts; track per-example slice so we can reassemble
    all_texts = []
    meta = []  # (problem, solution, start_idx, n_texts)
    for ex in examples:
        problem = ex["problem"]
        solution = strip_think(
            next(g for g, ok in zip(ex["generations"], ex["correctness_math_verify"]) if ok)
        )
        chunks = split_paragraphs(solution)
        texts = [problem] + chunks
        meta.append((problem, solution, len(all_texts), len(texts)))
        all_texts.extend(texts)

    total = len(all_texts)
    n_batches = (total + args.batch_size - 1) // args.batch_size
    print(f"{total} total texts → {n_batches} batch(es) of {args.batch_size}\n")

    write_queue: asyncio.Queue = asyncio.Queue()
    writer_task = asyncio.create_task(writer(write_queue, out))

    # Translate all texts in parallel batches, pipeline with async writer
    all_results: list[str] = []
    for b in range(n_batches):
        batch = all_texts[b * args.batch_size : (b + 1) * args.batch_size]
        print(f"Batch {b+1}/{n_batches}: {len(batch)} texts ...")
        results = await asyncio.to_thread(translate_batch, batch, tokenizer, sampler, config)
        all_results.extend(results)

    # Reassemble and enqueue for writing
    for problem, solution, start, n in meta:
        chunk_results = all_results[start : start + n]
        await write_queue.put({
            "id": problem_id(problem),
            "problem": problem,
            "problem_hi": chunk_results[0],
            "solution": solution,
            "solution_hi": "\n\n".join(chunk_results[1:]),
        })

    await write_queue.put(None)
    await writer_task

    print(f"\nDone — {len(examples)} examples saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
