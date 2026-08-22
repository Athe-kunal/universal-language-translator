#!/usr/bin/env bash
# Builds (if needed) and launches a vllm OpenAI-compatible server container.
# Invoke via `make vllm-up ...` (see Makefile) rather than running directly.
#
# Required:
#   GPUS           physical GPU ids, comma-separated, e.g. "0,1" or "2"
#
# Optional (defaults shown):
#   MODEL          Qwen/Qwen3.6-27B
#   TP             tensor-parallel-size; defaults to the GPU count in $GPUS
#   MAX_MODEL_LEN  262144
#   GPU_MEM_UTIL   unset -> vllm's own default (0.9)
#   PORT           18000
#   NAME           vllm-gpu<GPUS with commas turned into dashes>, e.g. vllm-gpu0-1
#   IMAGE          prediction-vllm:qwen3.6
#   EXTRA_ARGS     any other `vllm serve` flags, passed through verbatim,
#                  e.g. EXTRA_ARGS="--quantization awq --enable-prefix-caching"
#
# A container already running under the same NAME is replaced (docker rm -f) rather
# than left to collide on the port.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${GPUS:?set GPUS, e.g. GPUS=0,1 or GPUS=2}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
IFS=',' read -ra GPU_ARR <<<"$GPUS"
TP="${TP:-${#GPU_ARR[@]}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
PORT="${PORT:-18000}"
IMAGE="${IMAGE:-prediction-vllm:qwen3.6}"
NAME="${NAME:-vllm-gpu${GPUS//,/-}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Building $IMAGE ..." >&2
docker build -q -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.vllm" "$SCRIPT_DIR" >/dev/null

if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists — replacing it (docker rm -f)." >&2
  docker rm -f "$NAME" >/dev/null
fi

VLLM_ARGS=("$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN")
if [ -n "$GPU_MEM_UTIL" ]; then
  VLLM_ARGS+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi
if [ -n "$EXTRA_ARGS" ]; then
  # Intentional word-splitting: EXTRA_ARGS is a space-separated flag list.
  # shellcheck disable=SC2206
  VLLM_ARGS+=($EXTRA_ARGS)
fi

echo "Launching $NAME: gpus=$GPUS model=$MODEL tp=$TP max_model_len=$MAX_MODEL_LEN gpu_mem_util=${GPU_MEM_UTIL:-<default>} port=$PORT" >&2

# --gpus all (not an explicit device list) + -e CUDA_VISIBLE_DEVICES: see README.md —
# nvidia-container-toolkit remaps an explicit `--gpus '"device=..."'` list to local
# indices 0..N-1, which collides with CUDA_VISIBLE_DEVICES. `--gpus all` preserves
# physical GPU numbering so CUDA_VISIBLE_DEVICES is the single source of truth.
docker run -d \
  --name "$NAME" \
  --label vllm-server \
  --runtime nvidia --gpus all \
  --ipc host \
  --security-opt label=disable \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p "$PORT:8000" \
  "$IMAGE" "${VLLM_ARGS[@]}" >/dev/null

echo "$NAME started on port $PORT." >&2
echo "Tail logs:  docker logs -f $NAME  (or: make vllm-logs NAME=$NAME)" >&2
echo "Tear down:  make vllm-down NAME=$NAME" >&2
