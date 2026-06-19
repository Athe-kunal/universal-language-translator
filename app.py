"""
Minimal Streamlit app for English → Hindi translation.

Usage:
    streamlit run app.py
"""

import warnings

import torch
import streamlit as st

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    from train_translation import split_en

from translate import ScriptArguments, SamplerConfig, load_pipeline, translate_one, translate_batch


@st.cache_resource(show_spinner="Loading model…")
def _load(model_path: str):
    args = ScriptArguments(model_name_or_path=model_path)
    _, tokenizer, sampler = load_pipeline(args)
    return tokenizer, sampler


@st.cache_data(show_spinner="Fetching example…")
def _fetch_example():
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    ex = next(iter(ds))
    return ex["problem"], ex["solution"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EN → HI Translator", layout="centered")
st.title("English → Hindi Translator")

# Sidebar
model_path = st.sidebar.text_input(
    "Checkpoint path",
    value=".models/modernbert-translation/checkpoint-final",
)
steps = st.sidebar.slider("Diffusion steps", 16, 256, 128, step=16)
max_new_tokens = st.sidebar.slider("Max new tokens", 32, 512, 128, step=32)
if torch.cuda.is_available():
    st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.warning("No GPU — running on CPU")

# Example loader
if st.button("Load example from OpenR1-Math-220k"):
    prob, sol = _fetch_example()
    st.session_state["question_en"] = prob
    st.session_state["solution_en"] = sol

question_en = st.text_area(
    "Question (English)", height=120,
    placeholder="Enter the English question…",
    value=st.session_state.get("question_en", ""),
)
solution_en = st.text_area(
    "Solution (English)", height=220,
    placeholder="Enter the English solution…",
    value=st.session_state.get("solution_en", ""),
)

if st.button("Translate", type="primary", disabled=not (question_en or solution_en)):
    try:
        tokenizer, sampler = _load(model_path)
    except Exception as e:
        st.error(f"Failed to load model from `{model_path}`:\n\n{e}")
        st.stop()

    config = SamplerConfig(steps=steps, max_new_tokens=max_new_tokens)

    q = question_en.strip()
    chunks = split_en(solution_en.strip()) if solution_en.strip() else []
    texts = ([q] if q else []) + chunks

    with st.spinner(f"Translating {len(texts)} text(s) in one batch…"):
        results = translate_batch(texts, tokenizer, sampler, config)

    idx = 0
    if q:
        col1, col2 = st.columns(2)
        col1.subheader("Question — EN")
        col1.write(q)
        col2.subheader("Question — HI")
        col2.write(results[idx] or "_<empty>_")
        idx += 1

    if chunks:
        st.subheader(f"Solution — {len(chunks)} chunk(s)")
        for i, (en, hi) in enumerate(zip(chunks, results[idx:]), 1):
            col1, col2 = st.columns(2)
            col1.caption(f"Chunk {i} — EN")
            col1.write(en)
            col2.caption(f"Chunk {i} — HI")
            col2.write(hi)
