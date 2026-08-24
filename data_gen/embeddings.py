"""Shared cross-lingual embedding utility.

Lives in its own module (rather than `translate_reasoning.py`, where it
originated) so both `translate_reasoning.py` (embedding-similarity
validation during translation) and `segment_steps.py` (semantic-break step
boundaries, both post-hoc and pre-translation) can depend on it without a
circular import between the two.

Default model is `jinaai/jina-embeddings-v3`, not LaBSE — see
`reward_metric_experiment.md` for why: LaBSE's real tokenizer limit is 256
tokens, which silently truncated 67% of real translated steps (250-600+
tokens) in this pipeline; jina-embeddings-v3 has an 8192-token context and
truncated 0% of the same sample, for a small, benign (non-systematic) drop
in discourse-marker catch rate.
"""

import asyncio
import threading

DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
_EMBEDDING_TASK = "text-matching"

_embedding_model = None
_embedding_lock = threading.Lock()


def get_embedding_model(model_name: str, device: str):
    """Lazily loads and caches the cross-lingual embedding model, so
    importing this module doesn't pay the model-load cost unless it's
    actually needed (e.g. validation disabled, or no semantic-break usage).

    Guarded by a real (non-async) lock: callers typically run this inside
    `asyncio.to_thread`, so with concurrency > 1 many OS threads can reach
    this function on their first call simultaneously — without the lock,
    multiple threads racing to construct the same model onto the same
    device corrupts the load (observed as a "meta tensor" error).

    Args:
        model_name: SentenceTransformer model name.
        device: Device to load the model onto (e.g. "cpu", "cuda:0").

    Returns:
        A loaded `SentenceTransformer` instance.
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer

                _embedding_model = SentenceTransformer(
                    model_name, trust_remote_code=True, device=device
                )
    return _embedding_model


def _encode(model, texts: list[str]):
    """Encodes texts, using the "text-matching" task head for models (like
    jina-embeddings-v3) that support task-specific LoRA adapters; falls back
    to a plain encode for models (like LaBSE) that don't accept `task`.
    """
    try:
        return model.encode(texts, task=_EMBEDDING_TASK, normalize_embeddings=True)
    except TypeError:
        return model.encode(texts, normalize_embeddings=True)


async def embedding_similarity(text_a: str, text_b: str, model_name: str, device: str) -> float:
    """Computes cosine similarity between two texts' embeddings.

    Runs the (synchronous, GPU-or-CPU-bound) encode call in a thread so it
    doesn't block the asyncio event loop other tasks are using.

    Args:
        text_a: First text.
        text_b: Second text.
        model_name: SentenceTransformer model name.
        device: Device to run the model on.

    Returns:
        Cosine similarity in [-1, 1] (in practice, [0, 1]-ish for real text).
    """

    def _run() -> float:
        model = get_embedding_model(model_name, device)
        embeddings = _encode(model, [text_a, text_b])
        return float(embeddings[0] @ embeddings[1])

    return await asyncio.to_thread(_run)
