"""Standalone reward-scoring HTTP server for RL translation training.

Runs jina-embeddings-v3 in the main project's venv (transformers<5.0, which
this model's custom remote code actually supports - see data_gen/embeddings.py),
separate from the miles GRPO training process (transformers==5.x, required
for FSDP2). rl/reward.py calls this over HTTP instead of loading the model
in-process: running jina under transformers 5.x hit a chain of compatibility
bugs in its custom code (tied-weights bookkeeping, cuBLAS context alloc,
rotary-embedding cache) that don't exist under the version it was built for.

Start with `make rl-reward-server-up` (see Makefile); stop with
`make rl-reward-server-down`.
"""

import asyncio
import os

from fastapi import FastAPI
from pydantic import BaseModel

from data_gen.embeddings import DEFAULT_EMBEDDING_MODEL, embedding_similarity

REWARD_EMBEDDING_MODEL = os.environ.get("RL_REWARD_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
REWARD_EMBEDDING_DEVICE = os.environ.get("RL_REWARD_EMBEDDING_DEVICE", "cpu")

# Concurrent requests each run their own forward pass on the one shared
# model with no limit otherwise - peak memory scales with request fan-in
# rather than the model's own footprint. Matters here since this can share
# a GPU with sglang's rollout engine (see RL_REWARD_EMBEDDING_DEVICE).
_CONCURRENCY = int(os.environ.get("RL_REWARD_EMBEDDING_CONCURRENCY", "2"))
_semaphore = asyncio.Semaphore(_CONCURRENCY)

app = FastAPI()


class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str


class SimilarityResponse(BaseModel):
    similarity: float


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(req: SimilarityRequest) -> SimilarityResponse:
    async with _semaphore:
        score = await embedding_similarity(
            req.text_a, req.text_b, REWARD_EMBEDDING_MODEL, REWARD_EMBEDDING_DEVICE
        )
    return SimilarityResponse(similarity=score)
