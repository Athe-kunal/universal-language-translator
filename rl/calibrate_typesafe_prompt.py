"""Checks how well the Jev prompt separates good translations from known-bad ones. Writes nothing to disk.

Positives are real dataset pairs. Negatives are the same pairs corrupted in ways that match the failure modes
of the trained models (truncation, repetition, wrong pairing, prose left in English, altered digits, garbling).
Prints per-corruption scores for every question, whether the `issue` choice names the right
problem, AUROC, a suggested threshold, and a reliability table.

    python -m rl.calibrate_typesafe_prompt --n 40          # about 40 + 5 * 40 judge calls at most

Caveats: the positives are the dataset's own translations, which are imperfect, so read the lowest-scoring
positives it prints. The reliability table uses a synthetic class balance, so it shows ordering and separation,
not true calibration on the real distribution. That needs human labels on real model outputs.
"""

import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor

from rl.tune_typesafe_prompt import build_sample
from rl.typesafe_judge import PROMPT_HASH, TypesafeJudge

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


EXPECTED_ISSUE = {
    "original": "none",
    "truncated": "truncated",
    "repeated": "repeated",
    "prose_left_in_english": "untranslated",
    "garbled": "garbled",
    "wrong_pairing": "wrong_content",
    "altered_digits": "math_altered",
}


def _garble(hi: str, rng: random.Random) -> str:
    return "".join("�" if _DEVANAGARI.match(c) and rng.random() < 0.3 else c for c in hi)


def corruptions(step: dict, other: dict, rng: random.Random) -> dict[str, str]:
    en, hi = step["en"], step["hi"]
    out = {}
    if _DEVANAGARI.search(hi):
        out["truncated"] = hi[: max(1, int(len(hi) * 0.4))]
        out["repeated"] = (" ".join(hi.split()[:3]) + " ") * 6
        out["prose_left_in_english"] = en
        out["garbled"] = _garble(hi, rng)
    if other["hi"] != hi:
        out["wrong_pairing"] = other["hi"]
    if re.search(r"\d", hi):
        out["altered_digits"] = re.sub(r"\d", lambda m: str((int(m.group()) + 1) % 10), hi)
    return out


def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    steps = [s for s in build_sample(args.n * 2, args.seed) if len(s["en"]) > 40][: args.n]
    items = []  # (kind, en, hi)
    for i, s in enumerate(steps):
        items.append(("original", s["en"], s["hi"]))
        for kind, bad_hi in corruptions(s, steps[(i + 1) % len(steps)], rng).items():
            items.append((kind, s["en"], bad_hi))

    judge = TypesafeJudge()
    print(f"prompt {PROMPT_HASH}: judging {len(items)} pairs from {len(steps)} steps\n")
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda it: judge.judge(it[1], it[2]), items))
    scores = [r.p_correct for r in results]

    by_kind: dict[str, list[float]] = {}
    for (kind, _, _), p in zip(items, scores):
        by_kind.setdefault(kind, []).append(p)

    print(f"{'kind':24s} {'n':>4s} {'mean':>6s} {'>=0.5':>7s}")
    for kind, ps in by_kind.items():
        print(f"{kind:24s} {len(ps):4d} {sum(ps) / len(ps):6.2f} {sum(p >= 0.5 for p in ps) / len(ps):7.0%}")

    questions = list(results[0].scores)
    print(f"\nmean P(true) per question   ({'  '.join(q[:8] for q in questions)})  issue matches expected")
    for kind in by_kind:
        rows = [r for (k, _, _), r in zip(items, results) if k == kind]
        means = "  ".join(f"{sum(r.scores[q] for r in rows) / len(rows):8.2f}" for q in questions)
        hit = sum(r.issue == EXPECTED_ISSUE[kind] for r in rows) / len(rows)
        print(f"{kind:24s} {means}  {hit:6.0%}")

    pos = by_kind["original"]
    neg = [p for k, ps in by_kind.items() if k != "original" for p in ps]
    print(f"\nAUROC original vs corrupted: {auroc(pos, neg):.3f}")
    best = max((sum(p >= t for p in pos) / len(pos) + sum(p < t for p in neg) / len(neg) - 1, t)
               for t in [i / 20 for i in range(1, 20)])
    print(f"best threshold {best[1]:.2f} (balanced accuracy {(best[0] + 1) / 2:.2f}; current default is 0.50)")

    print("\nreliability (synthetic 1:5 class balance)")
    print(f"{'P(correct) bin':16s} {'n':>4s} {'actually original':>18s}")
    for lo in [i / 10 for i in range(10)]:
        in_bin = [k == "original" for (k, _, _), p in zip(items, scores) if lo <= p < lo + 0.1 + (p == 1.0) * 1e-9]
        if in_bin:
            print(f"{lo:.1f}-{lo + 0.1:.1f}          {len(in_bin):4d} {sum(in_bin) / len(in_bin):18.0%}")

    print("\nlowest-scoring originals (judge errors or dataset defects)")
    low = sorted(((p, it) for it, p in zip(items, scores) if it[0] == "original"), key=lambda x: x[0])[:5]
    for p, (_, en, hi) in low:
        print(f"  {p:.2f}  EN {en[:70]!r}\n        HI {hi[:70]!r}")


if __name__ == "__main__":
    main()
