# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

English→Hindi translation dataset generation and masked diffusion language model (MDLM) training for translation. Uses the local `dllm` library (editable install under `dllm-src/`) built on ModernBERT.

## Commands

### Environment setup
```bash
uv sync --extra gpu   # GPU (installs torch + vllm)
uv sync --extra cpu   # CPU-only
```

### Training
```bash
make train                                       # 2-GPU DDP, default config
make train CONFIG=configs/translation_config.yaml GPUS=4
# Direct form:
accelerate launch --config_file dllm-src/scripts/accelerate_configs/ddp.yaml \
    --num_processes 2 train_translation.py --config configs/translation_config.yaml
```

### Inference
```bash
make translate
make translate MODEL=.models/my-checkpoint
```

### Dataset generation
```bash
uv run python data_gen/translate_ds.py                     # async real-time (local vLLM)
uv run python data_gen/translate_ds_batch.py               # OpenAI Batch API (50% cheaper)
uv run python data_gen/translate_ds.py --num_examples 100  # cap examples
uv run python data_gen/translate_ds_batch.py --dry_run     # preview without submitting
```

### Tests
```bash
uv run pytest
uv run pytest tests/test_foo.py   # single file
uv run pytest -m slow             # slow/integration tests
```

### RL (GRPO) translation training via Miles
```bash
make rl-venv                      # one-time: clones radixark/miles into .miles, creates .venv-miles, installs it
make dataset                      # -> bpcc_hin_deva.jsonl (if not already present)
make rl-dataset                   # -> bpcc_rl_train.jsonl / bpcc_rl_eval.jsonl
make rl-reward-server-up          # starts the reward-scoring HTTP server (main venv)
make rl-train-bpcc                # launches GRPO training
```
See "RL training (`rl/`)" below — this is a separate track from the MDLM SFT pipeline above (different backend, different venv, different base model).

## Architecture

### Data pipeline

```
data_gen/translate_ds.py        ──►  translation_chunked.jsonl  ──►  train_translation.py
data_gen/translate_ds_batch.py  ──►  (same output)
```

1. **translate_ds.py** — Async real-time pipeline (AsyncOpenAI, default endpoint `http://localhost:8069/v1`). Loads `open-r1/OpenR1-Math-220k`, filters for verified-correct solutions, splits text into sentence-level chunks (LaTeX masked via regex → pysbd segmentation → LaTeX restored), translates each chunk to Hindi, appends to `translation_chunked.jsonl`. Resume: MD5 hash of problem text as stable ID.

2. **translate_ds_batch.py** — Same chunking logic but submits OpenAI Batch API jobs sequentially: build JSONL → upload → create batch → poll until complete → parse → append to `translation_chunked.jsonl`. State tracked in `batch_jobs.jsonl`. Custom IDs encode `{pid}_{p|s}_{idx}` so chunks are reconstructed without a separate map.

3. **translation_chunked.jsonl** — One line per problem:
   ```json
   {"id": "<md5>", "problem": "...", "problem_chunks": [{"en": "...", "hi": "..."}], "solution": "...", "solution_chunks": [...]}
   ```
   Chunks are sentence-level `{"en", "hi"}` pairs with LaTeX preserved.

4. **train_translation.py** — Reads chunked JSONL, creates chat-format examples directly from each `{"en", "hi"}` pair (`user`=English, `assistant`=Hindi), tokenizes for MDLM training. Config via `configs/translation_config.yaml` or CLI flags.

### dllm library (`dllm-src/`)

Local editable package providing the masked diffusion infrastructure:
- `dllm.core.trainers` — `MDLMTrainer` (HF Trainer subclass), `MDLMConfig`
- `dllm.core.samplers` — `MDLMSampler`, `MDLMSamplerConfig`
- `dllm.core.schedulers` — Noise schedules
- `dllm.utils` — `get_model()`, `get_tokenizer()`, `default_sft_map_fn()`, `post_process_dataset()`, collators (`NoAttentionMaskWrapper`, `DataCollatorForSeq2Seq`)

### RL training (`rl/`)

GRPO fine-tuning of the plain autoregressive `Qwen/Qwen3-0.6B` (not the dllm a2d/MDLM checkpoints above) via [Miles](https://github.com/radixark/miles) — sglang for rollout generation, FSDP2 for the actor, on AI4Bharat's BPCC English→Hindi data.

- **`rl/prepare_bpcc_rl_data.py`** — Converts `bpcc_hin_deva.jsonl` (`src`/`tgt`) into miles's prompt/label JSONL format (`bpcc_rl_{train,eval}.jsonl`): each row's `prompt` wraps the English source in a translation instruction, `label` is the BPCC Hindi reference.
- **`rl/reward.py`** — `custom_rm(args, sample)`, wired in via miles's `--custom-rm-path` hook. Reward = jina-embeddings-v3 cosine similarity between the generated Hindi and the BPCC reference, discounted by two penalties from `rl/reward_components.py`: `repetition_penalty()` (degenerate/looping generation) and `language_switch_penalty()` (non-Devanagari, non-numeric tokens). The embedding similarity itself is scored over HTTP by `rl/reward_server.py` (`make rl-reward-server-up`), which runs in the main uv venv rather than the miles venv — jina-embeddings-v3's custom remote code needs this project's `transformers<5.0`; under miles's `transformers==5.x` its custom code hits a chain of compatibility bugs (tied-weights bookkeeping, cuBLAS context alloc, rotary-embedding cache).
- **`rl/run_qwen3_0_6b_bpcc_fsdp.sh`** — miles launch script: a plain bash equivalent of miles's own `scripts/run_qwen3_0_6b_fsdp.py` (`ray start --head` + `ray job submit -- python3 train.py <flags>`), single-node sglang + FSDP2, GRPO.

**Runs in its own venv** (`.venv-miles`, via `make rl-venv`, which also clones `radixark/miles` into `.miles`): miles pins `transformers==5.x`, which conflicts with this project's `transformers<5.0` (required by `dllm`/MDLM). `make rl-train-bpcc` puts `.venv-miles/bin` on `PATH` and passes `MILES_REPO=.miles` to the script. sglang and a matching torch/CUDA build aren't installed by `make rl-venv`; follow miles's own install docs for your hardware.

### Environment variables (`.env`)
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` — translation inference endpoint
- `WANDB_API_KEY`, `WANDB_PROJECT` — training metrics
