"""Composite RL reward for English-Hindi translation.

Wired into miles via --custom-rm-path (see rl/run_qwen3_0_6b_bpcc_fsdp.py).
"""

import os

from data_gen.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_similarity
from rl.reward_components import language_switch_penalty, repetition_penalty

REWARD_EMBEDDING_MODEL = os.environ.get("RL_REWARD_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
REWARD_EMBEDDING_DEVICE = os.environ.get("RL_REWARD_EMBEDDING_DEVICE", "cpu")


async def custom_rm(args, sample) -> float:
    """Returns the reward for one rollout sample.

    Args:
        args: miles's global run config (unused).
        sample: miles Sample with prompt, response, and label set.

    Returns:
        Embedding similarity between sample.response and sample.label,
        discounted by the repetition and language-switch penalties,
        clamped to [0, 1].
    """
    response = (sample.response or "").strip()
    reference = sample.label
    if not response or not reference:
        return 0.0

    similarity = max(
        0.0,
        await embedding_similarity(
            response, reference, REWARD_EMBEDDING_MODEL, REWARD_EMBEDDING_DEVICE
        ),
    )
    reward = (
        similarity
        * (1.0 - repetition_penalty(response))
        * (1.0 - language_switch_penalty(response))
    )
    return max(0.0, min(1.0, reward))
