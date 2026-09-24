"""Chunked prompt tuning for the Jev translation judge. Writes nothing to disk.

Samples 100 steps (one random step from each of 100 random documents, fixed by --seed) and judges
them 10 at a time. Each invocation prints ONE chunk, so the prompt can be edited between chunks:

    python -m rl.tune_typesafe_prompt --chunk 0            # judge chunk 0 with the current prompt
    (read the output, note what Jev got wrong, edit _QUESTION in rl/typesafe_judge.py)
    python -m rl.tune_typesafe_prompt --chunk 1            # judge chunk 1 with the updated prompt
    python -m rl.tune_typesafe_prompt --chunk 0 --dry-run  # print the chunk with no API calls
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from rl.reward_components import language_switch_penalty, repetition_penalty
from rl.typesafe_judge import PROMPT_HASH, TypesafeJudge

DATA_PATH = "reasoning_hi_final_100k.jsonl"


def build_sample(n: int, seed: int) -> list[dict]:
    """One random step from each of n random documents, fixed by seed."""
    offsets, pos = [], 0
    with open(DATA_PATH, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    rng = random.Random(seed)
    sample = []
    with open(DATA_PATH, "rb") as f:
        for doc_idx in rng.sample(range(len(offsets)), n):
            f.seek(offsets[doc_idx])
            steps = json.loads(f.readline())["steps"]
            step_idx = rng.randrange(len(steps))
            sample.append({"doc": doc_idx, "step": step_idx, **steps[step_idx]})
    return sample


def auto_flags(en: str, hi: str) -> str:
    flags = []
    if hi.strip() == en.strip():
        flags.append("identical-to-EN")
    if repetition_penalty(hi) > 0.3:
        flags.append("repetition")
    if "�" in hi:
        flags.append("U+FFFD")
    if language_switch_penalty(hi) > 0.5:
        flags.append("mostly-Latin")
    return ", ".join(flags) or "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="print the chunk without calling the API")
    args = parser.parse_args()

    start = args.chunk * args.chunk_size
    chunk = build_sample(args.n, args.seed)[start : start + args.chunk_size]
    if not chunk:
        raise SystemExit(f"chunk {args.chunk} is out of range for n={args.n}")

    if args.dry_run:
        results = [None] * len(chunk)
    else:
        judge = TypesafeJudge()
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda s: judge.judge(s["en"], s["hi"]), chunk))

    print(f"# Chunk {args.chunk} (sample items {start}-{start + len(chunk) - 1}), prompt {PROMPT_HASH}\n")
    for i, (s, r) in enumerate(zip(chunk, results), 1):
        verdict = "Jev n/a" if r is None else (
            f"Jev {r.p_correct:.2f} [{' '.join(f'{q[:8]} {v:.2f}' for q, v in r.scores.items() if q != 'correct')}] "
            f"issue={r.issue}"
        )
        print(f"## {i}. doc {s['doc']} step {s['step']} - {verdict} - flags: {auto_flags(s['en'], s['hi'])}\n")
        print(f"EN:\n{s['en']}\n\nHI:\n{s['hi']}\n")


if __name__ == "__main__":
    main()
