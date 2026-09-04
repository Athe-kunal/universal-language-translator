"""Cross-lingual embedding client for `segment_steps.py`'s semantic-break
step boundaries. Two backends, picked per call via `backend`:

  "vllm":  calls a vLLM server's OpenAI-compatible /v1/embeddings endpoint
    (spin it up separately: `vllm serve jinaai/jina-embeddings-v3 --task
    embed --port 8081`). No in-process model load, so no thread-safety
    concerns - an HTTP call is inherently safe from multiple threads.
    vLLM's `embed` task L2-normalizes by default (its pooler config's
    `use_activation` defaults to true), so vectors come back ready to
    dot-product as-is - don't add client-side normalization back in unless
    the server's pooler config is overridden to disable it.

  "local": loads the model in-process via sentence-transformers onto a
    local GPU/CPU - for when spinning up a separate vLLM server isn't an
    option. Lazily loaded and cached; guarded by a lock since multiple
    threads can reach the first call simultaneously (without it, racing to
    construct the same model onto the same device corrupts the load -
    observed as a "meta tensor" error).

Default model is jinaai/jina-embeddings-v3, not LaBSE — see
`reward_metric_experiment.md` for why: LaBSE's real tokenizer limit is 256
tokens, which silently truncated 67% of real translated steps (250-600+
tokens) in this pipeline; jina-embeddings-v3 has an 8192-token context and
truncated 0% of the same sample, for a small, benign (non-systematic) drop
in discourse-marker catch rate.
"""

import threading

import numpy as np
from openai import OpenAI

DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_EMBEDDING_BACKEND = "vllm"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8081/v1"  # "vllm" backend
DEFAULT_EMBEDDING_DEVICE = "cpu"  # "local" backend, e.g. "cuda:0"
# Caps the actual per-call forward-pass batch size, regardless of how many
# texts a caller passes to embed() in one go - callers should batch calls
# across many documents for throughput (fewer Python/HTTP round trips), but
# without this cap a single oversized batch can exhaust GPU memory under
# "local"'s native (non-flash) attention, whose memory scales with
# batch_size * seq_len^2 (observed: a ~300-unit batch tried to allocate 64GiB).
DEFAULT_EMBED_BATCH_SIZE = 64
# jina-embeddings-v3's real context limit is 8192 tokens. chunking.py
# guarantees every non-table translatable unit respects its own max_tokens
# (see chunking.py's _hard_split), but tables are a deliberate, permanent
# exception (never split, by design - see test_oversized_table_is_not_split)
# and can still exceed jina's limit (observed: a real table hit 8195 tokens).
# Embeddings here are only a coarse semantic-similarity signal, not a
# precision task, so truncating a rare outlier is harmless - but get the
# cap right: dense pipe-table text tokenizes far closer to 1 char/token than
# prose's ~3-4 (a ~3 chars/token cap still overflowed on real table data),
# so this assumes the pessimistic 1:1 ratio - safe regardless of content
# density, at the cost of possibly truncating harder than strictly needed
# for non-table content (fine, since precision isn't the goal here).
_MAX_EMBED_CHARS = 6000

_clients: dict[str, OpenAI] = {}
_local_model = None
_local_model_lock = threading.Lock()


def _get_client(base_url: str) -> OpenAI:
    if base_url not in _clients:
        _clients[base_url] = OpenAI(base_url=base_url, api_key="none")
    return _clients[base_url]


def _get_local_model(model_name: str, device: str):
    """Lazily loads and caches the local sentence-transformers model."""
    global _local_model
    if _local_model is None:
        with _local_model_lock:
            if _local_model is None:
                from sentence_transformers import SentenceTransformer

                _local_model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return _local_model


def _embed_one_batch(texts: list[str], model: str, backend: str, base_url: str, device: str) -> np.ndarray:
    if backend == "vllm":
        resp = _get_client(base_url).embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
    if backend == "local":
        return _get_local_model(model, device).encode(texts, normalize_embeddings=True)
    raise ValueError(f"unknown embedding backend: {backend!r} (expected 'vllm' or 'local')")


def embed(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    device: str = DEFAULT_EMBEDDING_DEVICE,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Embeds a batch of texts, via a vLLM server or a local model.

    Callers should feel free to pass a large `texts` list spanning many
    documents in one call - this chunks internally to `batch_size` per
    actual forward pass, so batching embedding requests across many
    documents (fewer, larger Python-level calls instead of one tiny call
    per document) doesn't risk a single oversized forward pass.

    Args:
        texts: Texts to embed.
        model: Model name (vLLM server's model, or a sentence-transformers
            model name/path for "local").
        backend: "vllm" or "local".
        base_url: ["vllm" only] The vLLM server's OpenAI-compatible base URL.
        device: ["local" only] Device to load the model onto (e.g. "cpu", "cuda:0").
        batch_size: Max texts per actual backend call.

    Returns:
        (len(texts), dim) array of L2-normalized embeddings - callers get
        cosine similarity via a plain dot product.

    Raises:
        ValueError: If `backend` isn't "vllm" or "local".
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    texts = [t[:_MAX_EMBED_CHARS] for t in texts]
    batches = [
        _embed_one_batch(texts[i : i + batch_size], model, backend, base_url, device)
        for i in range(0, len(texts), batch_size)
    ]
    return np.concatenate(batches, axis=0)
