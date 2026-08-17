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

### Environment variables (`.env`)
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` — translation inference endpoint
- `WANDB_API_KEY`, `WANDB_PROJECT` — training metrics
