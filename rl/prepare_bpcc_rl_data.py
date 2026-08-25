"""Builds a miles prompt/label JSONL from AI4Bharat's BPCC hin_Deva split.

Usage:
    uv run python -m rl.prepare_bpcc_rl_data
    uv run python -m rl.prepare_bpcc_rl_data --input_file bpcc_hin_deva.jsonl --eval_split 0.02
"""

import argparse
import json
import random
from pathlib import Path

from loguru import logger

TRANSLATION_INSTRUCTION = (
    "Translate the following English text to Hindi. Output only the Hindi "
    "translation, with no preamble, notes, or explanations.\n\nEnglish: {src}"
)

DEFAULT_INPUT_FILE = Path("bpcc_hin_deva.jsonl")
DEFAULT_TRAIN_OUTPUT = Path("bpcc_rl_train.jsonl")
DEFAULT_EVAL_OUTPUT = Path("bpcc_rl_eval.jsonl")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--train_output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval_output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--eval_split", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_records(input_file: Path) -> list[dict]:
    """Reads src/tgt JSONL rows into miles prompt/label records.

    Args:
        input_file: Path to a JSONL file with "src"/"tgt" fields per line.

    Returns:
        Records shaped {"prompt": ..., "label": ...}.
    """
    records = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            src, tgt = ex.get("src"), ex.get("tgt")
            if not src or not tgt:
                continue
            records.append({"prompt": TRANSLATION_INSTRUCTION.format(src=src), "label": tgt})
    return records


def main() -> None:
    args = parse_args()
    records = build_records(args.input_file)
    if not records:
        raise SystemExit(f"No usable src/tgt rows found in {args.input_file}")

    shuffled = records[:]
    random.Random(args.seed).shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * args.eval_split)) if args.eval_split > 0 else 0
    eval_records, train_records = shuffled[:n_eval], shuffled[n_eval:]

    for path, recs in ((args.train_output, train_records), (args.eval_output, eval_records)):
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        f"Wrote {len(train_records)} train / {len(eval_records)} eval examples "
        f"to {args.train_output} / {args.eval_output}"
    )


if __name__ == "__main__":
    main()
