"""Downloads Athekunal/english-hindi-reasoning-dataset from the Hugging Face Hub.

Each row is one document with a nested `steps` list of {"en", "hi", ...}
reasoning-step pairs (see the dataset card / data_gen/upload_hf.py, which
built it). Writes the `train` and `validation` splits to local JSONL files
in the same doc-per-line "chunked" shape train_translation.py already reads
via `dataset_format: chunked` (translation_chunked.jsonl's problem_chunks/
solution_chunks), just with a `steps` chunks_field instead - one line per
document, `{"id": ..., "steps": [{"en": ..., "hi": ...}, ...]}`. Steps with
a missing translation, or with fewer than `--min_token_count` English
tokens (per the dataset's own `token_count` field - too short to carry
useful translation signal, mostly stray fragments), are dropped.

Usage:
    uv run python data_gen/download_reasoning_hi.py
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
import os

DATASET_NAME = "Athekunal/english-hindi-reasoning-dataset"
DEFAULT_TRAIN_OUTPUT = Path("reasoning_hi_train.jsonl")
DEFAULT_VAL_OUTPUT = Path("reasoning_hi_val.jsonl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_reasoning_hi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val_output", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--num_examples", type=int, default=None,
                         help="Cap the number of documents per split (default: all).")
    parser.add_argument("--min_token_count", type=int, default=10,
                         help="Drop steps with fewer than this many English tokens (default: %(default)s).")
    return parser.parse_args()


def write_split(dataset, output_file: Path, min_token_count: int) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_steps = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in dataset:
            steps = [
                {"en": step["en"], "hi": step["hi"]}
                for step in doc["steps"]
                if not step.get("has_missing_translation")
                and step.get("en") and step.get("hi")
                and step.get("token_count", 0) >= min_token_count
            ]
            if not steps:
                continue
            f.write(json.dumps({"id": doc["doc_id"], "steps": steps}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_steps += len(steps)
    return n_docs, n_steps


def main() -> None:
    args = parse_args()
    load_dotenv()
    hf_token = os.environ.get("HF_API_KEY")

    for split_name, output_file in (("train", args.train_output), ("validation", args.val_output)):
        split = split_name if args.num_examples is None else f"{split_name}[:{args.num_examples}]"
        logger.info(f"Loading {DATASET_NAME} split={split}")
        dataset = load_dataset(DATASET_NAME, split=split, token=hf_token)
        n_docs, n_steps = write_split(dataset, output_file, args.min_token_count)
        logger.info(f"Wrote {n_docs} documents ({n_steps} steps) to {output_file}")


if __name__ == "__main__":
    main()
