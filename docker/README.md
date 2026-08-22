# Serving Qwen3.6 via vllm on this box (no host driver change)

## Quick start: `make vllm-up` / `make vllm-down`

Prefer these over the manual `docker build`/`docker run` below — same image, but you
only ever specify the knobs you care about (GPUs, model, tensor-parallel-size,
max-model-len, gpu-memory-utilization, ...); no image rebuild needed to change any of
them, since they're all passed at `docker run` time (see `vllm_up.sh`).

```bash
# Required: GPUS. Everything else has a sane default (see vllm_up.sh header).
make vllm-up GPUS=2 GPU_MEM_UTIL=0.95

# Two GPUs, custom model/context length, extra passthrough flags:
make vllm-up GPUS=0,1 MODEL=Qwen/Qwen3.6-27B MAX_MODEL_LEN=131072 \
  EXTRA_ARGS="--enable-prefix-caching"

# List running vllm servers / tail logs / tear one down:
make vllm-ps
make vllm-logs NAME=vllm-gpu2
make vllm-down NAME=vllm-gpu2      # or: make vllm-down VLLM_PORT=18001
```

`make vllm-up` picks a default container name (`vllm-gpu<GPUS with commas as dashes>`,
e.g. `vllm-gpu0-1`) and port (`18000`) if you don't set `NAME=`/`VLLM_PORT=` — it prints
both once the container is launched. Re-running `vllm-up` with the same `NAME` replaces
the existing container rather than erroring.

See `vllm_up.sh` / `vllm_down.sh` for the full list of knobs (`TP`, `IMAGE`, ...).

## Background

- Host GPUs: 4x A100 80GB, driver `550.163.01` (natively supports up to CUDA 12.4).
- `Qwen/Qwen3.6-27B` uses the `Qwen3_5ForConditionalGeneration` architecture, which
  requires `vllm>=0.17.0`, which requires `torch==2.10.0`, which has no `cu124` wheel —
  the minimum CUDA build is `cu126`.
- Rather than upgrading the host driver (this box has ~60+ active GPU processes from
  other users/projects — disruptive to touch), we use **NVIDIA CUDA Minor Version
  Compatibility**: on data-center GPUs (A100 qualifies), a newer CUDA 12.x userspace can
  run against an older CUDA 12.x driver via the `cuda-compat` package, as long as the
  driver clears the CUDA 12.0 floor (525.60.13) — 550.163.01 does. This does not extend
  to CUDA 13, which is why `pyproject.toml` pins `vllm==0.19.1`/`torch==2.10.0` rather
  than the latest `vllm==0.26.0`/`torch==2.11.0` (CUDA 13).

## Build

```bash
docker build -f prediction/llm_clinical/docker/Dockerfile.vllm \
  -t prediction-vllm:qwen3.6 \
  prediction/llm_clinical/docker
```

## Sanity check first (small, fast) before loading the 27B model

Confirms the compat trick actually works on this driver before committing GPU memory
to a 27B model:

```bash
docker run --rm --runtime=nvidia --gpus '"device=0"' \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  --entrypoint python3 \
  prediction-vllm:qwen3.6 \
  -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect something like `2.10.0+cu129 True NVIDIA A100 80GB PCIe`. If this fails with a
driver-version error, the compat package didn't shadow the driver's libcuda correctly —
stop here and report back rather than proceeding to the full model.

## Run the server (manual — `make vllm-up` above does this for you)

```bash
docker run -d --name qwen36-vllm --runtime=nvidia \
  --gpus all \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 18000:8000 \
  --ipc=host \
  prediction-vllm:qwen3.6
```

Use `--gpus all` here, not `--gpus '"device=0,3"'` — the nvidia-container-toolkit
*remaps* an explicit device list to local indices `0..N-1` inside the container, which
then collides with `CUDA_VISIBLE_DEVICES=0,3` baked into the image (index `3` doesn't
exist in a 2-GPU remapped view, collapsing it to 1 visible GPU and breaking
`--tensor-parallel-size 2`). With `--gpus all`, physical GPU numbering is preserved
inside the container, so `CUDA_VISIBLE_DEVICES` in `Dockerfile.vllm` is the single place
to change which physical GPUs get used — update it there and rebuild.

Host port `8000` was already taken by another process on this box (`uvicorn`, pid
346743) — mapped to `18000` instead. Check `ss -ltn` for what's free before picking a
port.

Before launching, check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
to avoid stepping on other users' jobs — GPUs 0 and 3 were idle when this was launched.

## Point the app at it

`ClinicalJourneyClient` (`prediction/llm_clinical/client.py`) takes any
`AsyncOpenAI`-compatible client, so point it at the local server instead of OpenAI:

```python
from openai import AsyncOpenAI
from prediction.llm_clinical.client import ClinicalJourneyClient

client = ClinicalJourneyClient(
    openai_client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
    model="Qwen/Qwen3.6-27B",
)
```

## Rollback

Nothing on the host changes — the driver, `nvidia-smi`, and every other process on
the box are untouched. `docker stop qwen36-vllm && docker rm qwen36-vllm` fully
reverts this.
