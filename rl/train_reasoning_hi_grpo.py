"""GRPO RL training for the reasoning_hi a2d/bd3lm checkpoint.

Continues RL post-training of a checkpoint already SFT'd by
train_translation.py on reasoning_hi_train.jsonl (see
data_gen/download_reasoning_hi.py), using rl.bd3lm_grpo_trainer's
BD3LMDiffuGRPOTrainer - a GRPO trainer whose log-prob computation matches
this checkpoint's block-diffusion (BD3LM) training objective, unlike dllm's
stock DiffuGRPOTrainer (which assumes MDLM).

Reward: rl/reasoning_hi_grpo_reward.py's embedding-similarity + penalty
score against the dataset's own Hindi reference (needs the reward server -
see `make rl-reward-server-up`).

Usage:
    uv run python -m rl.train_reasoning_hi_grpo --config configs/reasoning_hi_grpo_config.yaml
    uv run python -m rl.train_reasoning_hi_grpo --config configs/reasoning_hi_grpo_config.yaml \\
        training.learning_rate=1e-6 training.num_train_epochs=1
"""

import argparse
import os
from dataclasses import dataclass

from omegaconf import OmegaConf
from peft import LoraConfig
from trl import ModelConfig

import dllm
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from dllm.pipelines.rl import DiffuGRPOConfig

from rl.bd3lm_grpo_trainer import BD3LMDiffuGRPOTrainer
from rl.reasoning_hi_grpo_data import get_reasoning_hi_dataset
from rl.reasoning_hi_grpo_reward import reasoning_hi_reward_func
from rl.wandb_checkpoint_callback import WandbWeightOnlyCheckpointCallback

logger = dllm.utils.get_default_logger(__name__)

os.environ.setdefault("WANDB_PROJECT", "universal-language-translator")


@dataclass
class TrainingArguments(DiffuGRPOConfig):
    output_dir: str = ".models/qwen3-a2d-bd3lm-reasoning-hi/grpo"


def parse_config() -> tuple[ModelConfig, TrainingArguments]:
    """Loads --config (plus optional dotlist overrides) into typed configs.

    Args (CLI):
        --config: Path to a YAML file with top-level `model:` and
            `training:` sections.
        Remaining args: OmegaConf dotlist overrides, e.g.
            `training.learning_rate=1e-6`.

    Returns:
        (model_config, training_args), built from the YAML's `model:` and
        `training:` sections with any dotlist overrides applied on top.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.training, resolve=True))
    return model_config, training_args


def main() -> None:
    model_config, training_args = parse_config()

    train_set = get_reasoning_hi_dataset("train").shuffle(seed=training_args.seed)
    # Evaluation here is a periodic checkpointing signal, not a full metric -
    # GRPO generation over the full ~7.6k-row validation split every
    # eval_steps would dominate training time. A small fixed subsample keeps
    # each eval fast.
    eval_set = get_reasoning_hi_dataset("validation").shuffle(seed=training_args.seed).select(range(32))

    model_args = dllm.utils.ModelArguments(model_name_or_path=model_config.model_name_or_path)
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    model.config.use_cache = False

    # LoRA is NOT applied inside get_model - DiffuGRPOTrainer takes peft_config
    # separately so it can manage the reference model and adapter enable/disable
    # itself (see examples/rl/grpo/a2d/mdlm/train.py for the same convention).
    peft_config = None
    if model_config.lora_r and model_config.lora_r > 0:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
        )

    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    sampler_config = BD3LMSamplerConfig(
        steps=training_args.steps,
        max_new_tokens=training_args.max_completion_length,
        block_size=training_args.block_size,
        temperature=training_args.temperature or 0.0,
        cfg_scale=training_args.cfg_scale,
        remasking=training_args.remasking,
    )

    logger.info("Start GRPO training...")
    trainer = BD3LMDiffuGRPOTrainer(
        model=model,
        reward_funcs=[reasoning_hi_reward_func],
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        processing_class=tokenizer,
        peft_config=peft_config,
        sampler=sampler,
        sampler_config=sampler_config,
        callbacks=[WandbWeightOnlyCheckpointCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    main()
