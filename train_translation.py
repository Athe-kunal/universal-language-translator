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
import re
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import yaml
import accelerate
import transformers
from datasets import Dataset, DatasetDict

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

import dllm
from loguru import logger

# Mirrors the split logic in data_gen/translate_ds.py exactly.
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
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


def split_en(text: str) -> list[str]:
    """Split English text into chunks using the same method used during translation."""
    masked, tokens = _mask_latex(text)
    sentences = _SEGMENTER.segment(masked)
    return [_restore_latex(s.strip(), tokens) for s in sentences if s.strip()]


def split_hi(text: str, n: int) -> list[str] | None:
    """Split Hindi text on \\n\\n into exactly n chunks (how it was assembled).
    Returns None if the count doesn't match, signalling fallback to whole-pair."""
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks if len(chunks) == n else None


class WandbArtifactCallback(transformers.TrainerCallback):
    """Upload model checkpoint as a W&B artifact after each epoch."""

    def __init__(self, project: str):
        self.project = project

    def on_epoch_end(self, args, state, control, **kwargs):
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed — skipping artifact upload.")
            return

        if not wandb.run:
            return

        epoch = int(state.epoch)
        ckpt_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")

        if model is not None:
            model.save_pretrained(ckpt_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(ckpt_dir)

        artifact = wandb.Artifact(
            name=f"modernbert-translation-epoch-{epoch}",
            type="model",
            description=f"ModernBERT translation checkpoint after epoch {epoch}",
            metadata={"epoch": epoch, "step": state.global_step},
        )
        artifact.add_dir(str(ckpt_dir))
        wandb.log_artifact(artifact)
        logger.info(f"Uploaded W&B artifact for epoch {epoch} from {ckpt_dir}")


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
    max_token_length: int = 8192  # drop examples whose tokenized length exceeds this
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
    report_to: str = "wandb"
    wandb_project: str = "universal-language-translator"


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
    Each JSONL row is split into sentence-level chunk pairs using the same
    split_paragraphs logic used during translation generation. English is split
    with pysbd (masking LaTeX first); Hindi is split on \\n\\n since that's
    how translate_ds.py assembled it. When counts match, each chunk pair becomes
    one training example. When they don't match, the whole pair is used as-is.
    """
    records = []
    whole_pair_fallbacks = 0

    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            for task in tasks:
                en_text = ex[task["source_field"]]
                hi_text = ex[task["target_field"]]

                en_chunks = split_en(en_text)
                hi_chunks = split_hi(hi_text, len(en_chunks))

                if hi_chunks is None:
                    # Counts don't align — use the whole pair as one example
                    whole_pair_fallbacks += 1
                    records.append({
                        "task": task["name"],
                        "messages": [
                            {"role": "user",      "content": en_text},
                            {"role": "assistant", "content": hi_text},
                        ],
                    })
                else:
                    for en_chunk, hi_chunk in zip(en_chunks, hi_chunks):
                        records.append({
                            "task": task["name"],
                            "messages": [
                                {"role": "user",      "content": en_chunk},
                                {"role": "assistant", "content": hi_chunk},
                            ],
                        })

    logger.info(
        f"Built {len(records)} examples from {jsonl_path} "
        f"({whole_pair_fallbacks} whole-pair fallbacks)"
    )
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

    if training_args.report_to in ("wandb", "all"):
        os.environ.setdefault("WANDB_PROJECT", training_args.wandb_project)

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

        # Drop examples longer than max_token_length tokens (they would be
        # truncated/cause indexing errors and skew the loss).
        max_tok = data_args.max_token_length
        before = {k: len(v) for k, v in dataset.items()}
        dataset = dataset.filter(
            lambda ex: len(ex["input_ids"]) <= max_tok,
            num_proc=data_args.num_proc,
            desc=f"Filtering > {max_tok} tokens",
        )
        for k, n_before in before.items():
            n_after = len(dataset[k])
            logger.info(
                f"{k}: dropped {n_before - n_after} / {n_before} examples "
                f"longer than {max_tok} tokens"
            )

        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info(
        f"Train examples: {len(dataset['train'])}  "
        f"Eval examples: {len(dataset['test'])}"
    )

    # --- Trainer ---
    callbacks = []
    if training_args.report_to in ("wandb", "all"):
        callbacks.append(WandbArtifactCallback(project=training_args.wandb_project))

    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        callbacks=callbacks,
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


if __name__ == "__main__":
    train()
