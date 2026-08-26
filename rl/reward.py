"""Composite RL reward for English-Hindi translation.

Wired into miles via --custom-rm-path (see rl/run_qwen3_0_6b_bpcc_fsdp.sh).
Embedding similarity is scored by a separate process (rl/reward_server.py,
started with `make rl-reward-server-up`) over HTTP rather than loaded here -
see that module's docstring for why.
"""

import os

import httpx

from rl.reward_components import language_switch_penalty, repetition_penalty

REWARD_SERVER_URL = os.environ.get("RL_REWARD_SERVER_URL", "http://127.0.0.1:8090/similarity")
_client = httpx.AsyncClient(timeout=30.0)


async def _embedding_similarity(text_a: str, text_b: str) -> float:
    response = await _client.post(REWARD_SERVER_URL, json={"text_a": text_a, "text_b": text_b})
    response.raise_for_status()
    return response.json()["similarity"]


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

    similarity = max(0.0, await _embedding_similarity(response, reference))
    reward = (
        similarity
        * (1.0 - repetition_penalty(response))
        * (1.0 - language_switch_penalty(response))
    )
    return max(0.0, min(1.0, reward))
