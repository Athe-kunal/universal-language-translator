import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

from datasets import load_dataset
from jinja2 import Template
from openai import AsyncOpenAI
from tqdm import tqdm

CACHE_FILE = Path("translation_cache.jsonl")
LOG_FILE = Path("translation.log")
CONCURRENCY = 256
MODEL = "Qwen/Qwen2.5-32B-Instruct"

TRANSLATE_PROMPT = Template("""\
Translate the text inside <translate></translate> tags to Hindi. The surrounding <context></context> shows what comes before/after — do NOT translate or reproduce it. The boundary is strict: only return the Hindi translation of the content inside <translate></translate>, nothing else. Convert numbers and digits to Hindi (Devanagari) numerals. Keep LaTeX math expressions, variables, symbols (like \\sigma, \\frac, \\boxed, etc.) and any notation that has no Hindi equivalent exactly as-is.

<context>{{ before }}</context><translate>{{ chunk }}</translate><context>{{ after }}</context>\
""")


def problem_id(problem: str) -> str:
    return hashlib.md5(problem.encode()).hexdigest()


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def context_window(parts: list[str], idx: int, chars: int = 150) -> tuple[str, str]:
    before = " ".join(parts[:idx])[-chars:]
    after = " ".join(parts[idx + 1:])[:chars]
    return before, after


async def translate_chunk(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    chunk: str,
    before: str,
    after: str,
    log: logging.Logger,
) -> str:
    prompt = TRANSLATE_PROMPT.render(before=before, chunk=chunk, after=after)
    for attempt in range(3):
        async with sem:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
        content = (resp.choices[0].message.content or "").strip()
        if "<translate>" not in content and "</translate>" not in content:
            return content
        log.warning(f"Attempt {attempt + 1}: leaked <translate> tags, retrying")
    log.error("Leaked tags persisted after 3 attempts, returning raw")
    return content


async def translate(client: AsyncOpenAI, sem: asyncio.Semaphore, text: str, log: logging.Logger) -> str:
    parts = split_paragraphs(text)
    tasks = [
        translate_chunk(client, sem, chunk, *context_window(parts, i), log)
        for i, chunk in enumerate(parts)
    ]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)


def load_done() -> set[str]:
    if not CACHE_FILE.exists():
        return set()
    done = set()
    with open(CACHE_FILE) as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def worker(
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    cache_fh,
    write_lock: asyncio.Lock,
    pbar: tqdm,
    log: logging.Logger,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, ex = item
        pid = problem_id(ex["problem"])
        try:
            problem = ex["problem"]
            solution = strip_think(
                next(g for g, ok in zip(ex["generations"], ex["correctness_math_verify"]) if ok)
            )
            problem_hi, solution_hi = await asyncio.gather(
                translate(client, sem, problem, log),
                translate(client, sem, solution, log),
            )
            entry = {
                "id": pid,
                "problem": problem,
                "problem_hi": problem_hi,
                "solution": solution,
                "solution_hi": solution_hi,
            }
            async with write_lock:
                cache_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cache_fh.flush()
            log.info(f"idx={idx} id={pid} done — {pbar.n + 1}/{pbar.total}")
        except Exception as e:
            log.error(f"idx={idx} id={pid} failed: {e} — {pbar.n + 1}/{pbar.total}")
        finally:
            pbar.update(1)
            queue.task_done()


async def main(smoke: bool = False) -> None:
    cache_file = Path("smoke_translation_cache.jsonl") if smoke else CACHE_FILE
    log_file = Path("smoke_translation.log") if smoke else LOG_FILE

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    log = logging.getLogger("translate")

    done = load_done() if not smoke else set()
    log.info(f"Loaded {len(done)} already-translated entries from cache")

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    examples = [
        (i, ex) for i, ex in enumerate(ds)
        if any(ex["correctness_math_verify"]) and problem_id(ex["problem"]) not in done
    ]

    if smoke:
        examples = examples[:3]
        log.info("Smoke mode: running 3 examples")

    log.info(f"Queuing {len(examples)} examples ({len(done)} skipped)")

    concurrency = min(CONCURRENCY, len(examples))
    client = AsyncOpenAI(base_url="http://localhost:8069/v1", api_key="none")
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(64)

    pbar = tqdm(total=len(examples), desc="Translating", file=sys.stderr)

    with open(cache_file, "a", encoding="utf-8") as cache_fh:
        workers = [
            asyncio.create_task(worker(queue, client, sem, cache_fh, write_lock, pbar, log))
            for _ in range(concurrency)
        ]

        async def producer():
            for item in examples:
                await queue.put(item)
            for _ in range(concurrency):
                await queue.put(None)

        await asyncio.gather(producer(), *workers)

    pbar.close()

    if smoke:
        log.info(f"Smoke test complete — results in {cache_file}, logs in {log_file}")
        if cache_file.exists():
            with open(cache_file) as f:
                entries = [json.loads(l) for l in f]
            log.info(f"Wrote {len(entries)} entries")
            for e in entries:
                log.info(f"  problem[:80]: {e['problem'][:80]!r}")
                log.info(f"  problem_hi[:80]: {e['problem_hi'][:80]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Translate 3 examples to verify the pipeline")
    args = parser.parse_args()
    asyncio.run(main(smoke=args.smoke))
