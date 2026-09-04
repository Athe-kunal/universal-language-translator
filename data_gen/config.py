"""Single source of control for the Fireworks translation pipeline - covers
both call modes:

  USE_BATCH_API = True:  Fireworks' Batch Inference API - its own Dataset +
    BatchInferenceJob resource model (not the OpenAI file+batch shape
    translate_ds_batch.py uses), submitted async and polled to completion.
    See prepare_fireworks_batch.py and
    https://docs.fireworks.ai/guides/batch-inference. Auth is a bearer
    token against FIREWORKS_ACCOUNT_ID-scoped URLs under FIREWORKS_BASE_URL.

  USE_BATCH_API = False: Fireworks' synchronous, OpenAI-compatible chat
    completions endpoint at FIREWORKS_ONLINE_BASE_URL - one request per
    call, no job/polling, for small runs or interactive use.

FIREWORKS_MODEL is shared by both modes. The Fireworks API key itself stays
out of this file - set FIREWORKS_API_KEY in .env (it's a secret, this file
is checked into git).
"""

# --- Mode toggle: flip this, nothing else, to switch how translation calls
# are made. ---
USE_BATCH_API = True

# Full model identifier, as Fireworks expects it in both a BatchInferenceJob's
# `model` field and an online chat-completion request's `model` field
# (e.g. "accounts/fireworks/models/llama-v3p1-8b-instruct").
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"

# Your Fireworks account id (the {ACCOUNT_ID} segment in every batch-API path).
FIREWORKS_ACCOUNT_ID = "fireworks"

# Batch Inference API host (account-scoped REST resources: datasets, jobs).
# Override only for a non-default region/deployment.
FIREWORKS_BASE_URL = "https://api.fireworks.ai"

# Synchronous, OpenAI-compatible chat completions endpoint (used when
# USE_BATCH_API is False).
FIREWORKS_ONLINE_BASE_URL = "https://api.fireworks.ai/inference/v1"

TARGET_LANGUAGE = "Hindi"

# Token budget for one translation unit's *protected* text (placeholders
# included - see chunking.py's Unit.token_count) before chunking.py's
# split fallback kicks in.
MIN_TOKENS = 60
MAX_TOKENS = 400

# Token budget for a regrouped "step" (several units merged for context) -
# see segment_steps.py.
MIN_STEP_TOKENS = 80
MAX_STEP_TOKENS = 600

SEMANTIC_PERCENTILE = 20.0
MIN_UNITS_FOR_SEMANTIC = 4
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"

# "vllm": call a separately-running vLLM embeddings server (spin it up with
#   `vllm serve jinaai/jina-embeddings-v3 --task embed --port 8081`).
# "local": load the model in-process onto EMBEDDING_DEVICE - for when
#   standing up a separate server isn't an option.
EMBEDDING_BACKEND = "vllm"
EMBEDDING_BASE_URL = "http://localhost:8081/v1"  # "vllm" backend
EMBEDDING_DEVICE = "cuda:0"  # "local" backend, e.g. "cpu" or "cuda:0"
# Max texts per actual embedding forward pass - embedding calls are batched
# across ALL documents in a run (one call per document is thousands of tiny
# calls dominated by overhead), but this still caps each real backend call
# so a huge combined batch can't exhaust GPU memory under "local"'s native
# (non-flash) attention (observed: an unbounded batch tried to allocate 64GiB).
EMBED_BATCH_SIZE = 64
