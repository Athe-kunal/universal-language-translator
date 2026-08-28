"""GRPO reward adapter for reasoning_hi translation RL training.

Wraps rl/reward.py's embedding-similarity + penalty logic (already used by
the Miles/autoregressive RL track) in the batched reward-function signature
DiffuGRPOTrainer (dllm's TRL-based GRPO trainer) expects:
`(prompts, completions, **kwargs) -> list[float]`, as a coroutine - TRL
awaits async reward functions via asyncio.gather.
"""

import asyncio

from rl.reward import _embedding_similarity
from rl.reward_components import language_switch_penalty, repetition_penalty


async def _translation_reward(response: str, reference: str) -> float:
    """Scores one candidate translation against its reference.

    Args:
        response: The model's generated Hindi text.
        reference: The reference Hindi translation.

    Returns:
        Embedding similarity discounted by the repetition and
        language-switch penalties, clamped to [0, 1].
    """
    response = (response or "").strip()
    if not response or not reference:
        return 0.0
    similarity = max(0.0, await _embedding_similarity(response, reference))
    reward = (
        similarity
        * (1.0 - repetition_penalty(response))
        * (1.0 - language_switch_penalty(response))
    )
    return max(0.0, min(1.0, reward))


async def reasoning_hi_reward_func(prompts, completions, hi, **kwargs) -> list[float]:
    """DiffuGRPOTrainer reward function for the reasoning_hi translation task.

    Args:
        prompts: Batch of chat-format prompts (unused - reward only needs
            the generated response and its reference).
        completions: Batch of chat-format completions from the model.
        hi: Batch of Hindi reference translations (the "hi" dataset column
            from rl.reasoning_hi_grpo_data.get_reasoning_hi_dataset).
        **kwargs: Other dataset columns / trainer_state, unused.

    Returns:
        One reward in [0, 1] per sample.
    """
    responses = [completion[0]["content"] for completion in completions]
    return list(
        await asyncio.gather(
            *(_translation_reward(response, reference) for response, reference in zip(responses, hi))
        )
    )
