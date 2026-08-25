"""RL (GRPO) translation training support, run via the miles framework
(sglang rollout + FSDP2 actor - https://github.com/radixark/miles).

This package is imported from *inside miles's own process* (its
--custom-rm-path hook does `load_function("rl.reward.custom_rm")`), which
normally runs in a separate venv from this project's main one - see
rl/run_qwen3_0_6b_bpcc_fsdp.py for why. Keep imports in this package limited
to what that venv can actually provide (stdlib, sentence-transformers,
data_gen.embeddings) - don't import dllm or anything else that assumes the
main venv's dependency pins.
"""
