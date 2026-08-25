#!/usr/bin/env bash
# Miles GRPO launch: Qwen3-0.6B on BPCC English-Hindi translation, sglang
# rollout + FSDP2 actor, single node. Mirrors miles's own
# scripts/run_qwen3_0_6b_fsdp.py (dapo-math-17k) with the reward model
# swapped for rl/reward.py's custom_rm and the prompt dataset swapped for
# BPCC. Invoke via `make rl-train-bpcc` (see Makefile), which puts the
# miles venv on PATH; run directly only if you've done that yourself.
#
# Prerequisites:
#   uv run python data_gen/download_bpcc.py
#   uv run python -m rl.prepare_bpcc_rl_data
#
# Required:
#   MILES_REPO         path to a `radixark/miles` checkout (has train.py at its root)
#
# Optional (defaults shown):
#   MODEL_DIR           /root/models
#   TRAIN_DATA          bpcc_rl_train.jsonl
#   EVAL_DATA           bpcc_rl_eval.jsonl
#   NUM_GPUS_PER_NODE   4
#   NUM_ROLLOUT         200
#   MASTER_ADDR         127.0.0.1
#   WANDB_PROJECT       universal-language-translator-rl
#   WANDB_API_KEY       unset -> wandb logging disabled
set -euo pipefail

: "${MILES_REPO:?set MILES_REPO to a radixark/miles checkout (see \`make rl-venv\`)}"
MODEL_DIR="${MODEL_DIR:-/root/models}"
TRAIN_DATA="${TRAIN_DATA:-bpcc_rl_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-bpcc_rl_eval.jsonl}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
WANDB_PROJECT="${WANDB_PROJECT:-universal-language-translator-rl}"
WANDB_GROUP="qwen3-0.6b-fsdp-bpcc-translation"

HF_REPO="Qwen/Qwen3-0.6B"
MODEL_PATH="$MODEL_DIR/Qwen3-0.6B"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$MODEL_DIR"
hf download "$HF_REPO" --local-dir "$MODEL_PATH"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP")
fi

TRAIN_ARGS=(
  --hf-checkpoint "$MODEL_PATH" --ref-load "$MODEL_PATH"
  --prompt-data "$TRAIN_DATA" --input-key prompt --label-key label
  --apply-chat-template --rollout-shuffle --balance-data
  --custom-rm-path rl.reward.custom_rm
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size 32 --n-samples-per-prompt 8
  --rollout-max-response-len 512 --rollout-temperature 1 --global-batch-size 256
  --eval-interval 10 --eval-prompt-data bpcc "$EVAL_DATA"
  --n-samples-per-eval-prompt 4 --eval-max-response-len 512 --eval-top-p 1
  --use-kl-loss --advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl
  --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.98
  "${WANDB_ARGS[@]}"
  --rollout-num-gpus-per-engine 1 --sglang-decode-log-interval 1000
  --sglang-mem-fraction-static 0.75 --sglang-attention-backend fa3
  --sglang-chunked-prefill-size 4096
  --train-backend fsdp --update-weight-buffer-size 536870912
  --gradient-checkpointing --attn-implementation flash_attention_3
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
  --use-dynamic-batch-size --max-tokens-per-gpu 9216
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --colocate --use-fault-tolerance
)

pkill -9 sglang || true
ray stop --force || true
pkill -9 -f "miles" || true
pkill -9 redis-server || true
sleep 3

export PYTHONUNBUFFERED=1
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats

RUNTIME_ENV_JSON="$(cat <<EOF
{"env_vars": {"PYTHONUNBUFFERED": "1", "MASTER_ADDR": "$MASTER_ADDR", "no_proxy": "127.0.0.1,$MASTER_ADDR", "PYTHONPATH": "$MILES_REPO:$PROJECT_ROOT:${PYTHONPATH:-}"}}
EOF
)"

no_proxy=127.0.0.1 ray job submit \
  --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 "$MILES_REPO/train.py" "${TRAIN_ARGS[@]}"
