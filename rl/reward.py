"""Composite RL reward for English->Hindi translation, wired into miles via
--custom-rm-path (see rl/run_qwen3_0_6b_bpcc_fsdp.py).

miles's rollout loop awaits `custom_rm(args, sample)` once per generated
sample and uses the returned float as that sample's reward (see
docs/user-guide/customization.md and miles/rollout/rm_hub/__init__.py in
the miles repo). `sample.prompt` is the instruction miles sent the model,
`sample.response` is what it generated, and `sample.label` is whatever
rl/prepare_bpcc_rl_data.py put in the JSONL's "label" field - the BPCC
Hindi reference translation.

    reward = jina_similarity(response, reference)
             * (1 - repetition_penalty(response))
             * (1 - language_switch_penalty(response))

Both penalty terms are 0 for clean output, so a well-formed, on-topic,
Hindi-only translation's reward reduces to the embedding similarity alone;
degenerate/repeated or English-leaking output drags the reward toward 0
regardless of how similar it happens to look to the reference.
"""

import os

from data_gen.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_similarity
from rl.reward_components import language_switch_penalty, repetition_penalty

# jina-embeddings-v3 over LaBSE per reward_metric_experiment.md: LaBSE's
# 256-token tokenizer limit silently truncates most real translation-length
# text, jina-v3's 8192-token context doesn't, for a small non-systematic
# drop in discourse-marker catch rate. Overridable via env var so a run can
# swap models/devices without editing this file.
REWARD_EMBEDDING_MODEL = os.environ.get("RL_REWARD_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# CPU by default: the FSDP actor and sglang rollout engine are typically
# colocated on the same GPU(s) (miles's --colocate) already under memory
# pressure from a live RL run - don't add a third resident model to that
# contention unless the caller has GPU headroom to spare and sets this
# explicitly (e.g. RL_REWARD_EMBEDDING_DEVICE=cuda:0).
REWARD_EMBEDDING_DEVICE = os.environ.get("RL_REWARD_EMBEDDING_DEVICE", "cpu")


async def custom_rm(args, sample) -> float:
    """Per-sample reward hook - see docs/user-guide/customization.md in the
    miles repo for the `async def custom_rm(args, sample: Sample) -> float`
    contract this implements.
    """
    response = (sample.response or "").strip()
    reference = sample.label

    if not response or not reference:
        return 0.0

    similarity = await embedding_similarity(
        response, reference, REWARD_EMBEDDING_MODEL, REWARD_EMBEDDING_DEVICE
    )
    similarity = max(0.0, similarity)

    reward = (
        similarity
        * (1.0 - repetition_penalty(response))
        * (1.0 - language_switch_penalty(response))
    )
    return max(0.0, min(1.0, reward))
