"""Dataset loader for GRPO RL training on the reasoning_hi a2d/mdlm checkpoint.

Loads Athekunal/english-hindi-reasoning-dataset directly from the Hugging
Face Hub (see `data_gen/download_datasets.py --dataset reasoning_hi`, which
applies the same filtering to build the local SFT JSONL) and flattens each document's `steps`
list into individual {"prompt", "hi"} rows for dllm's DiffuGRPOTrainer.

Prompt format matches train_translation.py's SFT chat template exactly - a
plain {"role": "user", "content": en} with no instruction wrapper and no
system prompt - so RL rollout starts from the same distribution the
checkpoint was fine-tuned on.
"""

import os
from functools import partial

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

DATASET_NAME = "Athekunal/english-hindi-reasoning-dataset"


def _flatten_steps(batch: dict, min_token_count: int) -> dict:
    """Flattens a batch of documents' `steps` lists into prompt/hi rows.

    Args:
        batch: A `{"steps": [[{"en", "hi", ...}, ...], ...]}` batch, one
            steps-list per document.
        min_token_count: Drop steps with fewer than this many English tokens
            (per the dataset's own `token_count` field), matching
            `data_gen/download_datasets.py --dataset reasoning_hi`'s default filtering.

    Returns:
        `{"prompt": [...], "hi": [...]}`, one entry per kept step.
    """
    prompts, his = [], []
    for steps in batch["steps"]:
        for step in steps:
            if step.get("has_missing_translation"):
                continue
            en, hi = step.get("en"), step.get("hi")
            if not en or not hi:
                continue
            if step.get("token_count", 0) < min_token_count:
                continue
            prompts.append([{"role": "user", "content": en}])
            his.append(hi)
    return {"prompt": prompts, "hi": his}


def get_reasoning_hi_dataset(split: str = "train", min_token_count: int = 10) -> Dataset:
    """Loads the reasoning_hi dataset from the HF Hub into a flat GRPO dataset.

    Args:
        split: "train" or "validation".
        min_token_count: Drop steps with fewer than this many English tokens
            (per the dataset's own `token_count` field), matching
            `data_gen/download_datasets.py --dataset reasoning_hi`'s default filtering.

    Returns:
        A Dataset with one row per step:
        {"prompt": [{"role": "user", "content": en}], "hi": hi}.
    """
    load_dotenv()
    hf_token = os.environ.get("HF_API_KEY")
    docs = load_dataset(DATASET_NAME, split=split, token=hf_token)
    return docs.map(
        partial(_flatten_steps, min_token_count=min_token_count),
        batched=True,
        remove_columns=docs.column_names,
    )
