CONFIG     ?= configs/bpcc_translation_config.yaml
GPUS       ?= 2
GPU_IDS    ?= 2,3
ACCEL_CFG  ?= dllm-src/scripts/accelerate_configs/ddp.yaml

train:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(CONFIG)

MODEL ?= .models/modernbert-translation/checkpoint-final

translate:
	uv run python translate.py --model_name_or_path $(MODEL)

# BERT -> diffusion continual-pretraining adaptation. Run this before
# fine-tuning a raw (non-chat) BERT-style checkpoint like jhu-clsp/mmBERT-base
# on a generation task — it teaches the model to denoise from a fully-masked
# canvas, which SFT alone does not. Mirrors the recipe that produced
# dllm-hub/ModernBERT-base-chat-v0.1 (see dllm-src/examples/bert/README.md).
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

# AR -> diffusion conversion for Llama-3.2-1B-Instruct via dllm's A2D
# pipeline (dllm-src/dllm/pipelines/a2d/convert.py). This only transplants
# the AR checkpoint's weights into the non-causal A2D architecture (single
# process, no accelerate launch needed) - the result is not yet a working
# diffusion model. It still needs continual-pretraining ("warmup") and SFT
# afterward, same as adapt-mmbert does for the BERT path.
#
# Llama 3.2's 1B/3B sizes are distilled/pruned from the 8B/70B rather than
# pretrained fresh, so treat this as a cheap smoke test of the A2D pipeline
# and the Hindi tokenizer, not a stand-in for converting the 8B checkpoint.
A2D_MODEL      ?= meta-llama/Llama-3.2-1B-Instruct
A2D_OUTPUT_DIR ?= .models/a2d/Llama-3.2-1B-Instruct

convert-llama-a2d:
	uv run python dllm-src/dllm/pipelines/a2d/convert.py \
		--model-name-or-path "$(A2D_MODEL)" \
		--output-dir "$(A2D_OUTPUT_DIR)"

.PHONY: train translate adapt-mmbert check-llada-tokenizer llada-moe-train-bpcc convert-llama-a2d
