"""
Train ModernBERT (masked diffusion) for Hindi<->English translation.

Run from the dllm/ subdirectory so accelerate configs are available:

    cd dllm
    accelerate launch \\
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        ../train_translation.py

Pass a custom config file:
    accelerate launch ... ../train_translation.py --config ../translation_config.yaml
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import yaml
import accelerate
import transformers
from datasets import Dataset, DatasetDict

import dllm
from loguru import logger


# ---------------------------------------------------------------------------
# Dataclasses — defaults come from the YAML; CLI flags override them.
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    jsonl_path: str = "translation_cache.jsonl"
    max_length: int = 512
    mask_prompt_loss: bool = True
    load_preprocessed_data: bool = False


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = ".models/modernbert-translation"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 1e-4
    group_by_length: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def yaml_to_argv(cfg: dict) -> list[str]:
    """Flatten YAML sections into HfArgumentParser-compatible argv strings."""
    argv = []
    for section in ("model", "data", "training"):
        for k, v in cfg.get(section, {}).items():
            if k == "tasks":  # handled separately
                continue
            if v is None:
                continue
            argv += [f"--{k}", str(v)]
    return argv


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_translation_dataset(jsonl_path: str, tasks: list[dict]) -> DatasetDict:
    """
    Each JSONL row produces one example per task.
    A task maps a source_field (English) to a target_field (Hindi).
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            for task in tasks:
                records.append({
                    "task": task["name"],
                    "messages": [
                        {"role": "user",      "content": ex[task["source_field"]]},
                        {"role": "assistant", "content": ex[task["target_field"]]},
                    ],
                })
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    # --- Parse --config before HfArgumentParser so YAML provides defaults ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, remaining = pre.parse_known_args()

    yaml_argv: list[str] = []
    tasks: list[dict] = [
        {"name": "question", "source_field": "problem",  "target_field": "problem_hi"},
        {"name": "solution", "source_field": "solution", "target_field": "solution_hi"},
    ]
    if known.config:
        cfg = load_yaml_config(known.config)
        yaml_argv = yaml_to_argv(cfg)
        tasks = cfg.get("data", {}).get("tasks", tasks)

    # YAML values act as defaults; CLI flags (remaining) override them.
    argv = yaml_argv + remaining

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=argv, look_for_args_file=False
    )

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # --- Model & tokenizer ---
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # --- Dataset ---
    with accelerate.PartialState().local_main_process_first():
        dataset = load_translation_dataset(data_args.jsonl_path, tasks)
        map_fn = partial(
            dllm.utils.default_sft_map_fn,
            tokenizer=tokenizer,
            mask_prompt_loss=data_args.mask_prompt_loss,
        )
        dataset = dataset.map(
            map_fn,
            num_proc=data_args.num_proc,
            desc="Tokenizing",
        )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info(
        f"Train examples: {len(dataset['train'])}  "
        f"Eval examples: {len(dataset['test'])}"
    )

    # --- Trainer ---
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=dllm.utils.NoAttentionMaskWrapper(
            transformers.DataCollatorForSeq2Seq(
                tokenizer,
                return_tensors="pt",
                padding=True,
                label_pad_token_id=tokenizer.pad_token_id,
            )
        ),
    )
    trainer.train()

    ckpt = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(ckpt)
    tokenizer.save_pretrained(ckpt)
    logger.info(f"Saved to {ckpt}")


if __name__ == "__main__":
    train()
