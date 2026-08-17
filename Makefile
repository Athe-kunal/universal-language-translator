CONFIG     ?= bpcc_translation_config.yaml
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

.PHONY: train translate
