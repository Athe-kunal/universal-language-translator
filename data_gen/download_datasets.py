"""Downloads a translation dataset from the Hugging Face Hub. Select which
with --dataset:

  bpcc: AI4Bharat's BPCC English-Hindi split, written as JSONL with `src`/
    `tgt` fields, ready to feed into train_translation.py.
  reasoning_hi: Athekunal/english-hindi-reasoning-dataset (train+validation),
    each row a document with a nested `steps` list of {"en","hi"} CoT
    reasoning-step pairs - see data_gen/upload_hf.py, which built it. Steps
    with a missing translation, or fewer than --min_token_count English
    tokens (per the dataset's own `token_count` field), are dropped.

Usage:
    uv run python -m data_gen.download_datasets --dataset bpcc
    uv run python -m data_gen.download_datasets --dataset bpcc --config massive --num_examples 1000
    uv run python -m data_gen.download_datasets --dataset reasoning_hi
"""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

BPCC_DATASET_NAME = "ai4bharat/BPCC"
BPCC_TARGET_LANG = "hin_Deva"
BPCC_DEFAULT_CONFIG = "bpcc-seed-latest"
BPCC_DEFAULT_OUTPUT_FILE = Path("bpcc_hin_deva.jsonl")

REASONING_HI_DATASET_NAME = "Athekunal/english-hindi-reasoning-dataset"
REASONING_HI_DEFAULT_TRAIN_OUTPUT = Path("reasoning_hi_train.jsonl")
REASONING_HI_DEFAULT_VAL_OUTPUT = Path("reasoning_hi_val.jsonl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_datasets")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", required=True, choices=["bpcc", "reasoning_hi"])
parser.add_argument("--num_examples", type=int, default=None, help="Cap examples/documents per split (default: all).")
parser.add_argument("--config", default=BPCC_DEFAULT_CONFIG, help="[bpcc] BPCC config to pull hin_Deva from.")
parser.add_argument("--output_file", type=Path, default=BPCC_DEFAULT_OUTPUT_FILE, help="[bpcc] Output JSONL path.")
parser.add_argument("--train_output", type=Path, default=REASONING_HI_DEFAULT_TRAIN_OUTPUT, help="[reasoning_hi]")
parser.add_argument("--val_output", type=Path, default=REASONING_HI_DEFAULT_VAL_OUTPUT, help="[reasoning_hi]")
parser.add_argument(
    "--min_token_count",
    type=int,
    default=10,
    help="[reasoning_hi] Drop steps with fewer than this many English tokens (default: %(default)s).",
)
args = parser.parse_args()

load_dotenv()
hf_token = os.environ.get("HF_API_KEY")
if not hf_token:
    logger.warning("HF_API_KEY not set in environment or .env; proceeding without a token")

if args.dataset == "bpcc":
    split = BPCC_TARGET_LANG if args.num_examples is None else f"{BPCC_TARGET_LANG}[:{args.num_examples}]"
    logger.info(f"Loading {BPCC_DATASET_NAME} config={args.config} split={split}")
    dataset = load_dataset(BPCC_DATASET_NAME, args.config, split=split, token=hf_token)
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

else:
    for split_name, output_file in (("train", args.train_output), ("validation", args.val_output)):
        split = split_name if args.num_examples is None else f"{split_name}[:{args.num_examples}]"
        logger.info(f"Loading {REASONING_HI_DATASET_NAME} split={split}")
        dataset = load_dataset(REASONING_HI_DATASET_NAME, split=split, token=hf_token)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        n_docs = n_steps = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in dataset:
                steps = [
                    {"en": step["en"], "hi": step["hi"]}
                    for step in doc["steps"]
                    if not step.get("has_missing_translation")
                    and step.get("en")
                    and step.get("hi")
                    and step.get("token_count", 0) >= args.min_token_count
                ]
                if not steps:
                    continue
                f.write(json.dumps({"id": doc["doc_id"], "steps": steps}, ensure_ascii=False) + "\n")
                n_docs += 1
                n_steps += len(steps)
        logger.info(f"Wrote {n_docs} documents ({n_steps} steps) to {output_file}")
