"""End-to-end EN -> HI reasoning-translation demo.

Picks a random document from reasoning_hi_train.jsonl (already chunked into
{"en", "hi"} steps by data_gen/download_datasets.py --dataset reasoning_hi), and
diffusion-decodes its steps through the BD3LM sampler (each step starts
fully masked), batching as many steps together per call as fit under
TOKEN_BUDGET (some docs have 100+ steps, so one batch per doc can OOM
regardless of per-step canvas size - see _chunk_steps). Runs on GPU 0.

Optionally animates the mask-resolving diffusion trajectory live per step,
Streamlit's counterpart to dllm.utils.TerminalVisualizer's Rich-console
animation used by translate.py's interactive CLI ("watch the masks
resolve") - Rich renders to a terminal, which doesn't work inside a
Streamlit page, so _animate_frame reuses TerminalVisualizer's own
tokenizer-decode/mask-count helpers (the framework-agnostic part) and drives
Streamlit placeholders with them instead of a Rich Live console.

Usage:
    streamlit run app.py
"""

import json
import os
import random
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import streamlit as st

import dllm
from translate import (
    BD3LMSamplerConfig,
    ScriptArguments,
    estimate_max_new_tokens,
    load_pipeline,
    translate_batch,
)

MODEL_PATH = ".models/qwen3-a2d-bd3lm-reasoning-hi-final10k-waitfix/checkpoint-933"
DATASET_PATH = "reasoning_hi_train.jsonl"
MAX_NEW_TOKENS_CAP = 2048
TOKEN_BUDGET = 4096


@st.cache_resource(show_spinner=f"Loading {MODEL_PATH} onto GPU 0…")
def _load():
    args = ScriptArguments(model_name_or_path=MODEL_PATH)
    _, tokenizer, sampler = load_pipeline(args, sampler_type="bd3lm")
    visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)
    return tokenizer, sampler, visualizer


def _animate_and_translate(
    chunk_texts: list[str],
    placeholders: list,
    tokenizer,
    sampler,
    visualizer: "dllm.utils.TerminalVisualizer",
    config: BD3LMSamplerConfig,
    fps: int = 12,
    max_redraws: int = 40,
) -> list[str]:
    """Runs one sampler.sample() call for a chunk of steps, animating each
    step's own placeholder frame-by-frame from the shared diffusion
    trajectory (mirrors TerminalVisualizer.visualize_one_history, but
    updates Streamlit placeholders instead of a Rich Live console), then
    returns the chunk's final trimmed translations.

    Args:
        chunk_texts: English source text for each step in this chunk.
        placeholders: One st.empty() per step, pre-created so the animation
            fills in already-visible layout rather than reflowing the page.
        tokenizer, sampler: From _load().
        visualizer: Supplies the tokenizer-decode/mask-count helpers this
            reuses - no Rich console involved.
        config: Sampler config for this chunk (shared max_new_tokens).
        fps: Animation frame rate.
        max_redraws: Caps how many of the (potentially hundreds of)
            diffusion steps actually trigger a Streamlit redraw, since
            redrawing every single step would make longer generations
            crawl - mirrors TerminalVisualizer's own every_n_steps knob.

    Returns:
        One trimmed Hindi translation string per step in chunk_texts.
    """
    messages = [[{"role": "user", "content": t}] for t in chunk_texts]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    outputs = sampler.sample(inputs, config, return_dict=True)

    history = outputs.histories
    every_n_steps = max(1, len(history) // max_redraws)
    sleep_s = 1.0 / max(1, fps)
    for step_idx, frame in enumerate(history, start=1):
        if every_n_steps > 1 and step_idx % every_n_steps != 0 and step_idx != len(history):
            continue
        for row, ph in enumerate(placeholders):
            text = visualizer._detok(frame[row], skip_special_tokens=False)
            masks_left = visualizer._count_masks(frame[row])
            ph.text(f"{text}\n\n[masks remaining: {masks_left} · step {step_idx}/{len(history)}]")
        time.sleep(sleep_s)

    sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    translations = [s.strip() for s in sequences]
    for ph, translation in zip(placeholders, translations):
        ph.text(translation or "<empty>")
    return translations


def _chunk_steps(texts: list[str], tokenizer) -> list[tuple[list[str], int]]:
    """Greedily groups step texts into sub-batches, each paired with the
    shared max_new_tokens its steps need, such that
    len(sub_batch) * max_new_tokens stays under TOKEN_BUDGET."""
    chunks = []
    current: list[str] = []
    current_max = 0
    for text in texts:
        step_max = estimate_max_new_tokens([text], tokenizer, max_tokens=MAX_NEW_TOKENS_CAP)
        candidate_max = max(current_max, step_max)
        if current and (len(current) + 1) * candidate_max > TOKEN_BUDGET:
            chunks.append((current, current_max))
            current, candidate_max = [], step_max
        current.append(text)
        current_max = candidate_max
    if current:
        chunks.append((current, current_max))
    return chunks


def _random_doc(path: str) -> dict:
    """Reads one random line from a JSONL file without loading it all into
    memory: seeks to a random byte offset, discards the (likely partial)
    line it lands in, and reads the next full line."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(random.randint(0, size - 1))
        f.readline()
        line = f.readline()
        if not line:
            f.seek(0)
            line = f.readline()
    return json.loads(line)


st.set_page_config(page_title="CoT Translator", layout="centered")
st.title("Chain-of-Thought Translator (EN → HI)")
st.caption(f"Model: `{MODEL_PATH}` · BD3LM block diffusion · GPU 0")

animate = st.sidebar.checkbox("Animate mask-resolving (watch the diffusion)", value=True)
fps = st.sidebar.slider("Animation fps", min_value=2, max_value=24, value=12, disabled=not animate)

if st.button("Pick random example", type="primary"):
    st.session_state["doc"] = _random_doc(DATASET_PATH)

doc = st.session_state.get("doc")

if doc:
    steps = doc["steps"]
    st.caption(f"Document `{doc['id'][:12]}` — {len(steps)} step(s)")

    tokenizer, sampler, visualizer = _load()
    chunks = _chunk_steps([s["en"] for s in steps], tokenizer)

    # Lay out every step's EN + reference up front, and one placeholder per
    # step for the HI translation, so the animation fills in already-visible
    # rows instead of the page reflowing as each chunk finishes.
    placeholders = []
    for i, step in enumerate(steps, 1):
        col1, col2 = st.columns(2)
        col1.caption(f"Step {i} — EN")
        col1.text(step["en"])
        col2.caption(f"Step {i} — HI (model)")
        ph = col2.empty()
        ph.text("<pending>")
        placeholders.append(ph)
        with st.expander("Reference HI (dataset)"):
            st.text(step["hi"])

    with st.spinner(f"Decoding {len(steps)} step(s) from masks in {len(chunks)} batch(es)…"):
        idx = 0
        for chunk_texts, chunk_max_new_tokens in chunks:
            chunk_placeholders = placeholders[idx : idx + len(chunk_texts)]
            config = BD3LMSamplerConfig(max_new_tokens=chunk_max_new_tokens, temperature=0.5)
            if animate:
                _animate_and_translate(
                    chunk_texts, chunk_placeholders, tokenizer, sampler, visualizer, config, fps=fps
                )
            else:
                translations = translate_batch(chunk_texts, tokenizer, sampler, config)
                for ph, translation in zip(chunk_placeholders, translations):
                    ph.text(translation or "<empty>")
            idx += len(chunk_texts)
