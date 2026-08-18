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

# LLaDA-MoE-7B-A1B-Instruct fine-tune on BPCC (see
# docs/llada-moe-en-hi-finetuning.md). Already a diffusion LM, so no
# adaptation stage - but 7B resident params need FSDP2, not plain DDP.
LLADA_CONFIG    ?= configs/llada_moe_bpcc_translation_config.yaml
LLADA_ACCEL_CFG ?= dllm-src/scripts/accelerate_configs/fsdp2.yaml

check-llada-tokenizer:
	uv run python scripts/check_llada_tokenizer.py

train-llada-moe:
	$(if $(GPU_IDS),CUDA_VISIBLE_DEVICES=$(GPU_IDS)) uv run accelerate launch \
		--config_file $(LLADA_ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(LLADA_CONFIG)

.PHONY: train translate adapt-mmbert check-llada-tokenizer train-llada-moe
