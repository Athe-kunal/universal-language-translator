"""Miles GRPO launch script: Qwen3-0.6B on BPCC English-Hindi translation,
sglang rollout + FSDP2 actor, single node.

Run in the miles venv (`make rl-venv`), not this project's main one - miles
pins transformers==5.x, incompatible with dllm's transformers<5.0.

Usage:
    uv run python data_gen/download_bpcc.py
    uv run python -m rl.prepare_bpcc_rl_data
    PYTHONPATH=. .venv-miles/bin/python rl/run_qwen3_0_6b_bpcc_fsdp.py
"""

import os
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

HF_REPO = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen3-0.6B"
WANDB_GROUP = "qwen3-0.6b-fsdp-bpcc-translation"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_gpus_per_node: int = 4
    num_rollout: int = 200
    model_dir: str = "/root/models"
    train_data: str = "bpcc_rl_train.jsonl"
    eval_data: str = "bpcc_rl_eval.jsonl"
    wandb_project: str = "universal-language-translator-rl"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")


def execute(args: ScriptArgs):
    model_path = f"{args.model_dir}/{MODEL_NAME}"

    ckpt_args = f"--hf-checkpoint {model_path} --ref-load {model_path} "

    rollout_args = (
        f"--prompt-data {args.train_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--custom-rm-path rl.reward.custom_rm "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 10 "
        f"--eval-prompt-data bpcc {args.eval_data} "
        "--n-samples-per-eval-prompt 4 "
        "--eval-max-response-len 512 "
        "--eval-top-p 1 "
    )

    grpo_args = (
        "--use-kl-loss "
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {WANDB_GROUP} "
        if os.environ.get("WANDB_API_KEY")
        else ""
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-mem-fraction-static 0.75 "
        "--sglang-attention-backend fa3 "
        "--sglang-chunked-prefill-size 4096 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--update-weight-buffer-size 536870912 "
        "--gradient-checkpointing "
        "--attn-implementation flash_attention_3 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{eval_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{wandb_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
