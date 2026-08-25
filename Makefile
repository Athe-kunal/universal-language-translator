CONFIG     ?= configs/bpcc_translation_config.yaml
GPUS       ?= 1
GPU_IDS    ?= 0
ACCEL_CFG  ?= dllm-src/scripts/accelerate_configs/ddp.yaml

# Downloads AI4Bharat's BPCC (English<->Hindi) hin_Deva split into
# bpcc_hin_deva.jsonl - the dataset every *-bpcc-* train target above/below
# reads via train_translation.py's --jsonl_path. Needs HF_API_KEY in .env for
# gated configs; the default "bpcc-seed-latest" config is small and ungated.
BPCC_CONFIG      ?= bpcc-seed-latest
BPCC_OUTPUT_FILE ?= bpcc_hin_deva.jsonl

dataset:
	uv run python data_gen/download_bpcc.py \
		--config "$(BPCC_CONFIG)" \
		--output_file "$(BPCC_OUTPUT_FILE)"

# Samples OpenThoughts3-1.2M (stratified by domain/source/difficulty) and
# facebook/natural_reasoning (uniform) into a combined TranslationDataset
# JSONL for the chunking + translation pipeline.
NUM_OPENTHOUGHTS3     ?= 50000
NUM_NATURAL_REASONING ?= 50000
SAMPLE_REASONING_SEED ?= 42
SAMPLE_REASONING_OUTPUT_FILE ?= sampled_reasoning.jsonl

sample-reasoning:
	uv run python data_gen/sample_reasoning.py \
		--num_openthoughts3 "$(NUM_OPENTHOUGHTS3)" \
		--num_natural_reasoning "$(NUM_NATURAL_REASONING)" \
		--seed "$(SAMPLE_REASONING_SEED)" \
		--output_file "$(SAMPLE_REASONING_OUTPUT_FILE)"

# Chunks + translates OpenThoughts3/natural_reasoning documents to Hindi via
# data_gen/translate_reasoning.py, against a vllm server started with vllm-up
# (or vllm-up-4b). Writes both the reconstructed per-document JSONL and a
# per-chunk JSONL updated incrementally — view either live in `streamlit run
# app.py`'s "Reasoning Translations" tab.
# Deliberately separate from NUM_OPENTHOUGHTS3/NUM_NATURAL_REASONING above
# (those default to 50000 for sample-reasoning) — translation is far more
# expensive per document than sampling, so this target defaults small.
TRANSLATE_NUM_OPENTHOUGHTS3   ?= 100
TRANSLATE_NUM_NATURAL_REASONING ?= 100
TRANSLATE_REASONING_BASE_URL ?= http://localhost:8077/v1
TRANSLATE_REASONING_MODEL    ?= Qwen/Qwen3-4B-Instruct-2507
TRANSLATE_REASONING_OUTPUT_FILE       ?= translated_reasoning.jsonl
TRANSLATE_REASONING_UNITS_OUTPUT_FILE ?= translated_reasoning_units.jsonl

translate-reasoning:
	uv run python data_gen/translate_reasoning.py \
		--num_openthoughts3 "$(TRANSLATE_NUM_OPENTHOUGHTS3)" \
		--num_natural_reasoning "$(TRANSLATE_NUM_NATURAL_REASONING)" \
		--base_url "$(TRANSLATE_REASONING_BASE_URL)" \
		--model "$(TRANSLATE_REASONING_MODEL)" \
		--output_file "$(TRANSLATE_REASONING_OUTPUT_FILE)" \
		--units_output_file "$(TRANSLATE_REASONING_UNITS_OUTPUT_FILE)"

export CUDA_HOME ?= /home/ubuntu/.local/fake-cuda

train:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(CONFIG)

MODEL ?= .models/modernbert-translation/checkpoint-final

translate:
	uv run python translate.py --model_name_or_path $(MODEL)

ADAPT_MODEL      ?= jhu-clsp/mmBERT-base
ADAPT_OUTPUT_DIR ?= .models/mmbert-diffusion-adapted

adapt-mmbert:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		dllm-src/examples/bert/pt.py \
		--model_name_or_path "$(ADAPT_MODEL)" \
		--dataset_args "wikitext[name:wikitext-103-v1]" \
		--text_field "text" \
		--insert_eos True \
		--max_length 512 \
		--num_train_epochs 1 \
		--learning_rate 1e-4 \
		--per_device_train_batch_size 16 \
		--per_device_eval_batch_size 16 \
		--output_dir "$(ADAPT_OUTPUT_DIR)"

# LLaDA-MoE-7B-A1B-Instruct fine-tune on BPCC. Already a diffusion LM, so no
# adaptation stage - but 7B resident params need FSDP2, not plain DDP.
LLADA_CONFIG    ?= configs/llada_moe_bpcc_translation_config.yaml
LLADA_ACCEL_CFG ?= dllm-src/scripts/accelerate_configs/fsdp2.yaml
LLADA_GPUS      ?= 2
LLADA_GPU_IDS   ?= $(GPU_IDS)

check-llada-tokenizer:
	uv run python scripts/check_llada_tokenizer.py

llada-moe-train-bpcc:
	$(if $(LLADA_GPU_IDS),CUDA_VISIBLE_DEVICES=$(LLADA_GPU_IDS)) uv run accelerate launch \
		--config_file $(LLADA_ACCEL_CFG) \
		--num_processes $(LLADA_GPUS) \
		train_translation.py --config $(LLADA_CONFIG)


A2D_MODEL      ?= Qwen/Qwen2.5-1.5B-Instruct
A2D_OUTPUT_DIR ?= .models/a2d/Qwen2.5-1.5B-Instruct

convert-a2d:
	uv run python dllm-src/dllm/pipelines/a2d/convert.py \
		--model-name-or-path "$(A2D_MODEL)" \
		--output-dir "$(A2D_OUTPUT_DIR)"

# convert-a2d only transplants AR weights into the non-causal A2D
# architecture - it has never learned to denoise from a fully-masked canvas.
# Warm it up with continual pretraining before BPCC SFT (same role as
# adapt-mmbert for the BERT path; same dataset/hyperparams for consistency).
# See dllm-src/examples/a2d/mdlm/pt.py and dllm-src/examples/a2d/README.md.
# Validated empirically: SFTing straight off convert-a2d's raw transplant
# (skipping this stage) produced garbled, ungrammatical Hindi even at higher
# sampling temperature - see a2d_qwen_bpcc_validation_results.txt.
A2D_WARMUP_OUTPUT_DIR ?= .models/a2d/Qwen2.5-1.5B-Instruct/mdlm/warmup

a2d-warmup:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		dllm-src/examples/a2d/mdlm/pt.py \
		--model_name_or_path "$(A2D_OUTPUT_DIR)" \
		--dataset_args "wikitext[name:wikitext-103-v1]" \
		--text_field "text" \
		--insert_eos True \
		--max_length 512 \
		--num_train_epochs 1 \
		--learning_rate 1e-4 \
		--per_device_train_batch_size 32 \
		--per_device_eval_batch_size 32 \
		--output_dir "$(A2D_WARMUP_OUTPUT_DIR)"

# SFT the warmed-up checkpoint (from a2d-warmup) on BPCC via
# train_translation.py - see configs/a2d_qwen_bpcc_translation_config.yaml
# for why this reuses the same trainer as the mmBERT/LLaDA-MoE BPCC configs.
A2D_TRAIN_CONFIG ?= configs/a2d_qwen_bpcc_translation_config.yaml

a2d-train-bpcc:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(A2D_TRAIN_CONFIG)

# Fast baseline: SFT dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 (already fully
# warmed - no convert-a2d/a2d-warmup needed) on BPCC directly from the Hub.
# See configs/qwen3_a2d_bpcc_translation_config.yaml for the tradeoffs vs.
# the from-scratch Qwen2.5-1.5B a2d-warmup path above.
QWEN3_A2D_TRAIN_CONFIG ?= configs/qwen3_a2d_bpcc_translation_config.yaml

qwen3-a2d-train-bpcc:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(QWEN3_A2D_TRAIN_CONFIG)

# BD3LM (block diffusion) counterpart to qwen3-a2d-train-bpcc, for a direct
# MDLM-vs-BD3LM comparison on the same base checkpoint family + BPCC data.
# train_translation.py picks BD3LMTrainer/BD3LMConfig automatically from this
# config's `training.trainer: bd3lm` (see configs/_base_bpcc_bd3lm.yaml).
QWEN3_A2D_BD3LM_TRAIN_CONFIG ?= configs/qwen3_a2d_bd3lm_bpcc_translation_config.yaml

qwen3-a2d-bd3lm-train-bpcc:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(QWEN3_A2D_BD3LM_TRAIN_CONFIG)

###
# vllm serving (local Qwen/etc. models — see prediction/llm_clinical/docker/README.md)
###
VLLM_DIR := docker
VLLM_MODEL ?= sarvamai/sarvam-translate
MAX_MODEL_LEN ?= 8192

.PHONY: vllm-up
vllm-up: # Build (if needed) and launch a vllm server. Required: GPUS="0,1". Optional: MODEL, TP, MAX_MODEL_LEN, GPU_MEM_UTIL, VLLM_PORT, NAME, EXTRA_ARGS (e.g. EXTRA_ARGS="--quantization awq").
	MODEL="$(VLLM_MODEL)" GPUS="$(GPUS)" TP="$(TP)" MAX_MODEL_LEN="$(MAX_MODEL_LEN)" \
	GPU_MEM_UTIL="$(GPU_MEM_UTIL)" PORT="$(VLLM_PORT)" NAME="$(NAME)" EXTRA_ARGS="$(EXTRA_ARGS)" \
	$(VLLM_DIR)/vllm_up.sh

# General-purpose instruct model (as opposed to VLLM_MODEL/sarvam-translate, a
# narrow dedicated MT model) — for comparing translation completeness, since a
# proper instruct model separates system-prompt instructions from user content
# instead of leaking/translating the instructions themselves. See
# data_gen/chunking.py for the chunker this feeds; pick a distinct NAME/VLLM_PORT
# so it can run alongside an existing vllm-up server.
VLLM_MODEL_4B ?= Qwen/Qwen3-4B-Instruct-2507

.PHONY: vllm-up-4b
vllm-up-4b: # Build (if needed) and launch a vllm server for the 4B instruct model. Required: GPUS="0,1". Optional: VLLM_MODEL_4B, TP, MAX_MODEL_LEN, GPU_MEM_UTIL, VLLM_PORT, NAME, EXTRA_ARGS.
	MODEL="$(VLLM_MODEL_4B)" GPUS="$(GPUS)" TP="$(TP)" MAX_MODEL_LEN="$(MAX_MODEL_LEN)" \
	GPU_MEM_UTIL="$(GPU_MEM_UTIL)" PORT="$(VLLM_PORT)" NAME="$(NAME)" EXTRA_ARGS="$(EXTRA_ARGS)" \
	$(VLLM_DIR)/vllm_up.sh

.PHONY: vllm-down
vllm-down: # Tear down a vllm server. NAME=<container> (as printed by vllm-up) or VLLM_PORT=<host port>.
	NAME="$(NAME)" PORT="$(VLLM_PORT)" $(VLLM_DIR)/vllm_down.sh

.PHONY: vllm-ps
vllm-ps: # List running vllm-server containers.
	docker ps --filter "label=vllm-server" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

.PHONY: vllm-logs
vllm-logs: # Tail logs for a vllm server. NAME=<container> (required).
	docker logs -f "$(NAME)"

###
# RL (GRPO) translation training via Miles (https://github.com/radixark/miles):
# sglang rollout + FSDP2 actor, reward from rl/reward.py (jina-embeddings-v3
# similarity to the BPCC reference, penalized for degenerate/repeated
# generation and non-numeric language switching - see rl/reward.py and
# rl/reward_components.py). Runs Qwen/Qwen3-0.6B (plain AR, not the
# dllm a2d/MDLM checkpoints the targets above train) in its OWN venv,
# separate from this project's main uv-managed one: miles pins
# transformers==5.x, which conflicts with dllm's transformers<5.0 - same
# isolation precedent as the COMET/MetricX venv in
# reward_metric_experiment.md.
###
MILES_VENV      ?= .venv-miles
MILES_RL_SCRIPT ?= rl/run_qwen3_0_6b_bpcc_fsdp.py

.PHONY: rl-venv
rl-venv: # One-time setup: creates $(MILES_VENV) and installs miles-rl + rl/requirements.txt into it. Still need sglang + a matching torch/CUDA build for your hardware - see https://github.com/radixark/miles for install instructions, this target doesn't pin those for you.
	python3 -m venv $(MILES_VENV)
	$(MILES_VENV)/bin/pip install -U pip
	$(MILES_VENV)/bin/pip install miles-rl
	$(MILES_VENV)/bin/pip install -r rl/requirements.txt

.PHONY: rl-dataset
rl-dataset: # Builds bpcc_rl_{train,eval}.jsonl from bpcc_hin_deva.jsonl (run `make dataset` first if that doesn't exist yet).
	uv run python -m rl.prepare_bpcc_rl_data

.PHONY: rl-train-bpcc
rl-train-bpcc: # Launches GRPO training (rl-venv and rl-dataset must have been run first).
	PYTHONPATH=. $(MILES_VENV)/bin/python $(MILES_RL_SCRIPT)

.PHONY: dataset sample-reasoning translate-reasoning train translate adapt-mmbert check-llada-tokenizer llada-moe-train-bpcc convert-llama-a2d a2d-warmup a2d-train-bpcc qwen3-a2d-train-bpcc qwen3-a2d-bd3lm-train-bpcc rl-venv rl-dataset rl-train-bpcc
