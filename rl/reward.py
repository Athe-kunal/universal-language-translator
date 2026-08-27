"""Composite RL reward for English-Hindi translation.

Wired into miles via --custom-rm-path (see rl/run_qwen3_0_6b_bpcc_fsdp.sh).
Embedding similarity is scored by a separate process (rl/reward_server.py,
started with `make rl-reward-server-up`) over HTTP rather than loaded here -
see that module's docstring for why.
"""

import asyncio
import os

import httpx

from rl.reward_components import language_switch_penalty, repetition_penalty

REWARD_SERVER_URL = os.environ.get("RL_REWARD_SERVER_URL", "http://127.0.0.1:8090/similarity")
# rollout/eval batches can fan out to thousands of concurrent custom_rm calls
# (a full eval pass), while the server serializes actual GPU work to ~1 at a
# time - so requests queue server-side rather than failing. No client-side
# connection cap: with one at a time, a cap just moves the wait from "queued
# server-side" to "PoolTimeout waiting for a connection slot" at whatever the
# cap is. The long timeout gives a queued request room to wait its turn.
_client = httpx.AsyncClient(timeout=1200.0, limits=httpx.Limits(max_connections=None))
_MAX_RETRIES = 3


async def _embedding_similarity(text_a: str, text_b: str) -> float:
    # A single dropped connection out of a large eval fan-out shouldn't kill
    # a multi-hour training run - retry transient network errors before
    # letting one propagate up and crash the job.
    for attempt in range(_MAX_RETRIES):
        try:
            response = await _client.post(
                REWARD_SERVER_URL, json={"text_a": text_a, "text_b": text_b}
            )
            response.raise_for_status()
            return response.json()["similarity"]
        except httpx.TransportError:
            if attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2**attempt)


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
