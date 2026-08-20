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
		--per_device_train_batch_size 16 \
		--per_device_eval_batch_size 16 \
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

.PHONY: dataset train translate adapt-mmbert check-llada-tokenizer llada-moe-train-bpcc convert-llama-a2d a2d-warmup a2d-train-bpcc
