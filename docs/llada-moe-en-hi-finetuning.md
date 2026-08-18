# Fine-tuning LLaDA-MoE for En→Hi with dFactory

Companion to `mmbert-degeneration-fixes.md`. That document assumes the mmBERT
backbone. This one is the alternative path: drop the encoder entirely and
fine-tune a model that is already a diffusion LM.

Repo: <https://github.com/inclusionAI/dFactory> — "Easy and Efficient dLLM
Fine-Tuning", from the same org that publishes LLaDA-MoE. Built on VeOmni as a
submodule for distributed training. Docs: <https://www.inclusion-ai.org/dFactory/>

---

## Why this instead of mmBERT

| | mmBERT-base | LLaDA-MoE-7B-A1B-Instruct |
|---|---|---|
| Already a diffusion LM | no — needs adaptation stage | **yes** |
| Generative competence | understanding-only pretraining | ≈ Qwen2.5-3B-Instruct |
| Params to fine-tune | 307M | 1B active / 7B resident |
| Context | 8192 | 8192 |
| Termination behaviour | must be trained from scratch | already learned |

The decisive point: **fix 2 from the degeneration doc disappears.** No
BERT→diffusion adaptation stage, because the model was pretrained on the masked
diffusion objective. You are doing one job — teaching translation — not two.

Fix 1 (the EOS canvas) may also resolve itself, since an instruct-tuned diffusion
model has already learned to terminate. Verify rather than assume.

The cost is memory. 1B *active* does not mean 1B resident — all 7B parameters sit
on the GPUs regardless, since routing is per-token and any expert may be needed.
On 2×A100 that is the real constraint, not compute.

---

## Model choice — read this before starting

dFactory's shipped configs target the **LLaDA 2.0** family:

```
configs/model_configs/llada2_mini/     # 16B
configs/model_configs/llada2_flash/    # 100B
configs/sft/llada2_mini_bd_sft.yaml
configs/sft/llada2_mini_bd_with_dparallel_sft.yaml
```

There is **no shipped config for LLaDA-MoE-7B-A1B**, and — checked against the
actual dFactory source (`inclusionAI/dfactory`, commit `a385b14`) — it is
**not** the same family as `llada2_mini`/`llada2_flash` at the code level.
dFactory ships `LLaDA2MoeConfig` (`model_type="llada2_moe_veomni"`, fields like
`moe_intermediate_size`, `n_group`, `first_k_dense_replace`) with its own
`modeling_llada2_moe.py`. LLaDA-MoE-7B-A1B-Instruct is the older **LLaDA MoE**
architecture — `LLaDAMoEConfig` (`model_type="lladamoe"`, fields like
`expert_intermediate_size`, `shared_expert_intermediate_size`,
`router_num_group`) with different modeling code entirely. Pointing dFactory's
`llada2_mini` config or `scripts/moe_convertor.py` at an LLaDA-MoE-7B-A1B
checkpoint fails at load — the config schema and weight-key layout don't
match. "Same lineage" was wrong; it's "same publisher, different generation."

**This repo's `dllm-src` already vendors the correct architecture directly**
(`dllm-src/dllm/pipelines/llada/models/modeling_lladamoe.py` +
`configuration_lladamoe.py`), registered under `AutoModelForMaskedLM`
(`dllm-src/dllm/pipelines/llada/models/__init__.py`). That means
`train_translation.py` — the exact script that already trained mmBERT here —
can load and fine-tune LLaDA-MoE-7B-A1B-Instruct as-is, with no
dFactory/VeOmni dependency, no expert merge/split step, and no BERT→diffusion
adaptation stage. Use:

- `scripts/check_llada_tokenizer.py` — the Devanagari gate check below, as a
  runnable script.
- `configs/llada_moe_bpcc_translation_config.yaml` — training config wired to
  `bpcc_hin_deva.jsonl` via the existing flat-format loader.
- `make check-llada-tokenizer` then `make train-llada-moe` (FSDP2 by default —
  7B resident params don't fit plain DDP on 2×A100).

dFactory remains worth revisiting only if the LLaDA2.0-mini path (16B, option 2
below) becomes the priority, since that's the model family it actually ships
configs for.

Either way, **run the Devanagari tokenizer check first** — this gates everything
downstream:

```bash
uv run python scripts/check_llada_tokenizer.py
```

Roughly 8–12 tokens means real Devanagari subwords. Thirty-plus means byte
fallback — the SMDM trap again, and the whole plan needs rethinking. MoE routing
does not guarantee Hindi got attention during pretraining.

The rest of this section (dFactory's own pipeline: expert merge, dataset
format, `train.sh`) describes the **LLaDA2.0-mini path**, not the
LLaDA-MoE-7B-A1B path above. Keep it for option 2.

---

## Pipeline

### 1. Download and merge experts

Hub weights store experts separately; training wants them merged.

```bash
python ./scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --local_dir /path/to/separate_expert_model

python scripts/moe_convertor.py \
  --input-path /path/to/separate_expert_model \
  --output-path /path/to/save/merged_model \
  --mode merge
```

The merged directory is what the training config points at. Budget disk for two
full copies of the weights.

### 2. Build the dataset

dFactory expects a conversational format. `scripts/build_gsm8k_dataset.py` is the
worked example — read it and mirror the output shape.

Your BPCC and synthetic CoT pairs map onto it directly:

- user turn: the `<context>` / `<translate>` wrapped English chunk
- assistant turn: the Hindi translation

Keeping the same tag schema as the teacher pipeline means the student learns the
exact interface you already validated, and the passthrough behaviour (LaTeX,
code, numbers staying English) comes along as a property of the data rather than
something bolted on at decode time.

### 3. Train

```bash
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
  sh train.sh tasks/train_llada2_bd.py configs/sft/<your_config>.yaml
```

Two task entrypoints:

- `tasks/train_llada2_bd.py` — block diffusion. Start here.
- `tasks/train_llada2_bd_with_dparallel.py` — adds DPARALLEL. An optimisation,
  not a starting point.

On parallelism: the LLaDA2.0 paper describes combining data parallelism with
**expert parallelism** for exactly this reason — sharding experts across devices
rather than replicating the whole model. On 2×A100 that is likely mandatory, so
your current plain-DDP `GPU_IDS ?= 2,3` pattern will not carry over unchanged.

The paper also uses sequence packing (concatenating short sequences into longer
ones), which suits translation pairs well since most chunks are far shorter than
8192.

### 4. Convert back for inference

Training saves in merged format; inference wants separate experts. Split it back:

```bash
python scripts/moe_convertor.py \
  --input-path TRAIN_OUTPUT_DIR/checkpoints/global_step_XXX/hf_ckpt/ \
  --output-path /path/to/save/separate_expert_model \
  --mode split
```

Two gotchas the README calls out explicitly:

- `--input-path` is the `hf_ckpt/` subdirectory, **not** the training output root.
- After splitting, a **manual step** remains: copy the modeling file into the
  output directory. The conversion does not do this for you.

---

## What carries over from your existing repo

More than you'd expect. The MDLM objective is the same one dFactory trains
against, so the conceptual work transfers even where the code doesn't.

- **Data generation** — `data_gen/download_bpcc.py`, `data_gen/translate_ds.py`,
  `data_gen/translate_ds_batch.py` are unaffected. Reformat the output, keep the
  logic.
- **Validation** — `validate.py` and `validate_bpcc.py` port over. Drop the
  `estimate_max_new_tokens` heuristic and test whether the instruct model
  terminates on its own.
- **Streamlit app** — swap the inference backend, keep the interface.
- **`dllm-src/`** — the vendored dLLM package becomes redundant if you go all-in
  on dFactory. Worth keeping installed while you compare the two paths.

Note also that ML-GSAI's LLaDA repo states SFT differs from a standard
autoregressive trainer by only a few lines, with guidelines in `GUIDELINES.md`.

Update: for LLaDA-MoE-7B-A1B specifically, this "fallback" is now the primary
path, not a hedge — dFactory has no working path to that architecture at all
(see the corrected note in Model choice above), while `train_translation.py`
+ `dllm-src` already loads and trains it natively. You lose expert
parallelism (plain FSDP2 sharding instead), which is the real risk on 2×A100 —
watch for OOM and drop to LoRA (`lora: true` in
`configs/llada_moe_bpcc_translation_config.yaml`) if it bites.

---

## Suggested order

1. `make check-llada-tokenizer` (wraps `scripts/check_llada_tokenizer.py`) on
   LLaDA-MoE-7B-A1B-Instruct. **Gate on this.**
2. Zero-shot En→Hi on a handful of chunks before any training — establishes the
   baseline and confirms the tokenizer conclusion in practice. (`validate.py`
   against the base checkpoint, no training config needed yet.)
3. Overfit a few hundred BPCC pairs with
   `configs/llada_moe_bpcc_translation_config.yaml` (small `max_steps`, no eval)
   to prove the loop end-to-end.
4. Full BPCC warm-start via `make train-llada-moe`, then specialize on the
   synthetic CoT pairs (swap `data.jsonl_path`/`dataset_format` back to the
   chunked format for that stage).
5. Head-to-head against batched Gemma 3 12B under vLLM — quality *and* tokens/sec.

Step 2 is the cheap one that could save the most time. An instruct model with
real Devanagari coverage may already translate acceptably, which would reframe
the project from "train a translator" to "make an existing one fast and cheap."

---

## Open questions

- Does LLaDA-MoE's tokenizer handle Devanagari properly? Everything gates on this.
- Do 7B resident params fit alongside optimizer state on 2×A100, or is expert
  parallelism plus offloading required?
- Does the instruct model terminate cleanly on a fixed canvas, or does fix 1 from
  the degeneration doc still apply?
- Is LLaDA2.0-mini worth the memory pain for the quality gain?
- How do visual features route through an MoE trained only on text? Relevant for
  phase 3, not now.
