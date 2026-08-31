#!/bin/bash
# Orchestrates: wait for the running reasoning_hi SFT job -> pick its final
# checkpoint -> point the GRPO config at it (NOT the stale pre-fix
# artifacts/...:v0 checkpoint, which was trained under the old
# max_length=512 truncation bug) -> launch GRPO RL training on the freed GPU.
set -uo pipefail
cd /home/recoverx/astarag/universal-language-translator
LOG_DIR=.orchestrator_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/orchestrator.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

SFT_PID=2944888
log "Watching SFT PID $SFT_PID (configs/qwen3_a2d_bd3lm_reasoning_hi_config.yaml, 50K-example subset, eval disabled, 1 epoch, per_device_train_batch_size=4/gradient_accumulation_steps=8; relaunched clean after a stale checkpoint-500 in output_dir caused a bad silent auto-resume)"
while kill -0 "$SFT_PID" 2>/dev/null; do
    sleep 60
done
log "SFT process $SFT_PID exited."

# Give the process group a moment to fully flush/save.
sleep 30

CKPT=$(ls -d .models/qwen3-a2d-bd3lm-reasoning-hi/checkpoint-* 2>/dev/null | sed -E 's#.*checkpoint-([0-9]+)$#\1 &#' | sort -n | awk '{print $2}' | tail -1)

if [ -z "$CKPT" ]; then
    log "ERROR: no checkpoint-* directory found under .models/qwen3-a2d-bd3lm-reasoning-hi/ - aborting, not launching RL."
    exit 1
fi
log "Selected final SFT checkpoint: $CKPT"

if [ ! -f "$CKPT/model.safetensors" ] && [ ! -f "$CKPT/model.safetensors.index.json" ]; then
    log "ERROR: $CKPT does not look like a complete checkpoint (no weights file) - aborting."
    exit 1
fi

cp configs/reasoning_hi_grpo_config.yaml "$LOG_DIR/reasoning_hi_grpo_config.yaml.bak"
python3 - "$CKPT" <<'PYEOF'
import sys
import re
path = "configs/reasoning_hi_grpo_config.yaml"
ckpt = sys.argv[1]
text = open(path).read()
new_text = re.sub(
    r'model_name_or_path:\s*".*"',
    f'model_name_or_path: "{ckpt}"',
    text,
    count=1,
)
assert new_text != text, "model_name_or_path line not found/replaced"
open(path, "w").write(new_text)
PYEOF
log "Updated configs/reasoning_hi_grpo_config.yaml model_name_or_path -> $CKPT"

launch_rl_2gpu() {
    log "Launching GRPO RL training on GPUs 0+1 (accelerate DDP, both freed by SFT completion)."
    CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch \
        --config_file dllm-src/scripts/accelerate_configs/ddp.yaml --num_processes 2 \
        -m rl.train_reasoning_hi_grpo --config configs/reasoning_hi_grpo_config.yaml \
        > "$LOG_DIR/rl_train.log" 2>&1 &
    echo $!
}

launch_rl_1gpu() {
    log "Falling back to single-GPU RL training on GPU 0 (config's own notes record 2-GPU DDP crashing before at other completion lengths)."
    CUDA_VISIBLE_DEVICES=0 uv run python -m rl.train_reasoning_hi_grpo \
        --config configs/reasoning_hi_grpo_config.yaml \
        > "$LOG_DIR/rl_train_1gpu_fallback.log" 2>&1 &
    echo $!
}

RL_PID=$(launch_rl_2gpu)
log "RL training (2-GPU) started, PID=$RL_PID, log=$LOG_DIR/rl_train.log"

# Watch the first few minutes for the known 2-GPU DDP failure signatures
# (OOM at step ~13, grad_norm blowup / unexplained kill at step ~140 seen
# previously at higher max_completion_length) before committing to it for
# the rest of the run.
EARLY_FAIL=0
for _ in $(seq 1 90); do
    if ! kill -0 "$RL_PID" 2>/dev/null; then
        EARLY_FAIL=1
        break
    fi
    if grep -qaE "Traceback|CUDA out of memory|NCCL error|RuntimeError" "$LOG_DIR/rl_train.log" 2>/dev/null; then
        EARLY_FAIL=1
        break
    fi
    sleep 10
done

if [ "$EARLY_FAIL" -eq 1 ]; then
    log "2-GPU RL run showed a failure signature (or exited) within the first ~15min - killing it and falling back to single GPU."
    kill "$RL_PID" 2>/dev/null
    wait "$RL_PID" 2>/dev/null
    RL_PID=$(launch_rl_1gpu)
    log "RL training (1-GPU fallback) started, PID=$RL_PID, log=$LOG_DIR/rl_train_1gpu_fallback.log"
fi

wait "$RL_PID"
RC=$?
log "RL training process (PID $RL_PID) exited with code $RC"

log "Orchestration complete. Compare rl_spot_check_results.jsonl / wandb run for reward-vs-SFT-baseline once eval checkpoints have logged (eval_steps=200)."
