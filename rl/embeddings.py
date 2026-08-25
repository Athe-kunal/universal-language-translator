"""Standalone embedding-similarity helper for rl/reward.py.

Deliberately not imported from data_gen.embeddings: rl/ runs in a separate
venv from the rest of this project (see rl/run_qwen3_0_6b_bpcc_fsdp.sh), so
it keeps its own copy rather than depending on a package outside that venv.
"""

import asyncio
import threading

DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
_EMBEDDING_TASK = "text-matching"

_embedding_model = None
_embedding_lock = threading.Lock()


def get_embedding_model(model_name: str, device: str):
    """Lazily loads and caches the embedding model.

    Args:
        model_name: SentenceTransformer model name.
        device: Device to load the model onto (e.g. "cpu", "cuda:0").

    Returns:
        A loaded SentenceTransformer instance.
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer

                _embedding_model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return _embedding_model


def _encode(model, texts: list[str]):
    try:
        return model.encode(texts, task=_EMBEDDING_TASK, normalize_embeddings=True)
    except TypeError:
        return model.encode(texts, normalize_embeddings=True)


async def embedding_similarity(text_a: str, text_b: str, model_name: str, device: str) -> float:
    """Computes cosine similarity between two texts' embeddings.

    Args:
        text_a: First text.
        text_b: Second text.
        model_name: SentenceTransformer model name.
        device: Device to run the model on.

    Returns:
        Cosine similarity in [-1, 1].
    """

    def _run() -> float:
        model = get_embedding_model(model_name, device)
        embeddings = _encode(model, [text_a, text_b])
        return float(embeddings[0] @ embeddings[1])

    return await asyncio.to_thread(_run)
