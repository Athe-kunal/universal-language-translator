import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd
from datasets import load_dataset
from tqdm import tqdm

from data_gen.openai_client import AsyncChatClient

CACHE_FILE = Path("translation_chunked.jsonl")
LOG_FILE = Path("translation.log")
CONCURRENCY = 256
MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")

TRANSLATE_PROMPT = """\
Translate the following sentence to Hindi. Rules:
- Translate only the natural language text to Hindi.
- Keep all numbers and digits exactly as-is (do NOT convert to Devanagari numerals).
- Keep all LaTeX math expressions, variables, and symbols (like \\sigma, \\frac, \\boxed, etc.) exactly as-is.
- Keep punctuation, formatting, and structure intact.
Return only the Hindi translation, nothing else.\
"""


def problem_id(problem: str) -> str:
    return hashlib.md5(problem.encode()).hexdigest()


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

# Matches display math \[...\], inline math $...$, and bare \cmd{...}{...} outside dollar signs
_LATEX_RE = re.compile(
    r"(\\\[.*?\\\]|\$\$.*?\$\$|\$(?:[^$\\]|\\.)+?\$|\\[a-zA-Z]+(?:\{[^}]*\})+)",
    re.DOTALL,
)


def _mask_latex(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []

    def replacer(m: re.Match) -> str:
        tokens.append(m.group(0))
        return f"\x00LATEX{len(tokens) - 1}\x00"

    return _LATEX_RE.sub(replacer, text), tokens


def _restore_latex(text: str, tokens: list[str]) -> str:
    for i, tok in enumerate(tokens):
        text = text.replace(f"\x00LATEX{i}\x00", tok)
    return text


def split_paragraphs(text: str) -> list[str]:
    masked, tokens = _mask_latex(text)
    sentences = _SEGMENTER.segment(masked)
    return [_restore_latex(s.strip(), tokens) for s in sentences if s.strip()]


async def translate_chunk(client: AsyncChatClient, chunk: str) -> str:
    return await client.complete(
        messages=[
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0.0,
    )


async def translate(client: AsyncChatClient, text: str) -> list[dict]:
    parts = split_paragraphs(text)
    results = await asyncio.gather(*(translate_chunk(client, chunk) for chunk in parts))
    return [{"en": en, "hi": hi} for en, hi in zip(parts, results)]


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
    client: AsyncChatClient,
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
                translate(client, problem),
                translate(client, solution),
            )
            entry = {
                "id": pid,
                "problem": problem,
                "problem_chunks": problem_hi,
                "solution": solution,
                "solution_chunks": solution_hi,
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


async def main(num_examples: int | None = None) -> None:
    cache_file = CACHE_FILE
    log_file = LOG_FILE

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    log = logging.getLogger("translate")

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

    concurrency = min(CONCURRENCY, len(examples))
    client = AsyncChatClient(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8069/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "none"),
        model=MODEL,
        concurrency=64,
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 2)
    write_lock = asyncio.Lock()

    pbar = tqdm(total=len(examples), desc="Translating", file=sys.stderr)

    with open(cache_file, "a", encoding="utf-8") as cache_fh:
        workers = [
            asyncio.create_task(worker(queue, client, cache_fh, write_lock, pbar, log))
            for _ in range(concurrency)
        ]

        async def producer():
            for item in examples:
                await queue.put(item)
            for _ in range(concurrency):
                await queue.put(None)

        await asyncio.gather(producer(), *workers)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None, help="Cap the number of examples to translate (default: all)")
    args = parser.parse_args()
    asyncio.run(main(num_examples=args.num_examples))
