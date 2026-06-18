CONFIG     ?= translation_config.yaml
GPUS       ?= 1
ACCEL_CFG  ?= dllm/scripts/accelerate_configs/ddp.yaml

train:
	uv run accelerate launch \
		--config_file $(ACCEL_CFG) \
		--num_processes $(GPUS) \
		train_translation.py --config $(CONFIG)

.PHONY: train
