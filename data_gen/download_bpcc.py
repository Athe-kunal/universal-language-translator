"""Downloads the English-Hindi split of AI4Bharat's BPCC dataset.

Fetches the `hin_Deva` split of a given BPCC config (default:
`bpcc-seed-latest`, the small human-curated seed corpus) from the Hugging
Face Hub and writes it to a JSONL file with `src` (English) and `tgt`
(Hindi) fields, ready to feed into `train_translation.py`.

Usage:
    uv run python data_gen/download_bpcc.py
    uv run python data_gen/download_bpcc.py --config massive
    uv run python data_gen/download_bpcc.py --num_examples 1000
"""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

DATASET_NAME = "ai4bharat/BPCC"
TARGET_LANG = "hin_Deva"
DEFAULT_CONFIG = "bpcc-seed-latest"
DEFAULT_OUTPUT_FILE = Path("bpcc_hin_deva.jsonl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_bpcc")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="BPCC config to pull the hin_Deva split from (default: %(default)s).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write the output JSONL file to (default: %(default)s).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Cap the number of examples downloaded (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    hf_token = os.environ.get("HF_API_KEY")
    if not hf_token:
        logger.warning("HF_API_KEY not set in environment or .env; proceeding without a token")

    split = TARGET_LANG if args.num_examples is None else f"{TARGET_LANG}[:{args.num_examples}]"
    logger.info(f"Loading {DATASET_NAME} config={args.config} split={split}")
    dataset = load_dataset(DATASET_NAME, args.config, split=split, token=hf_token)
    logger.info(f"Loaded {len(dataset)} examples")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            entry = {
                "src": example["src"],
                "tgt": example["tgt"],
                "src_lang": example["src_lang"],
                "tgt_lang": example["tgt_lang"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(dataset)} examples to {args.output_file}")


if __name__ == "__main__":
    main()
