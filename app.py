"""End-to-end EN -> HI reasoning-translation demo: dInfer translation + Typesafe (Jev) judging.

Picks a random held-out document (chunked into {"en", "hi"} steps), streams all its steps through the running
dInfer server (dinfer_server.py, default http://localhost:8092), which decodes steps of similar length in parallel
as one batch and returns each batch as soon as it is done, and judges every step as its translation arrives with Typesafe's calibrated decisions (overall P(correct) plus per-aspect scores).

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv-dinfer/bin/python dinfer_server.py --model <checkpoint-fused>   # once
    streamlit run app.py
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

from rl.typesafe_judge import TypesafeJudge

BACKENDS = {  # name -> dInfer server URL (see dinfer_server.py --model_type)
    "Qwen3-0.6B MDLM (dInfer)": os.environ.get("DINFER_QWEN_URL", "http://localhost:8093"),
    "LLaDA-MoE-7B-A1B (dInfer)": os.environ.get("DINFER_SERVER_URL", "http://localhost:8092"),
}
DATASET_PATH = "reasoning_hi_final_100k_val_waitfix.jsonl"


@st.cache_resource
def _judge() -> TypesafeJudge:
    return TypesafeJudge()


@st.cache_resource
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(8)


def _translate_stream(url: str, texts: list[str]):
    """Yields (index, translation, seconds) as the server finishes each canvas-size group; steps in a group are
    decoded in parallel, and short steps arrive first."""
    with requests.post(f"{url}/translate_stream", json={"texts": texts}, stream=True, timeout=1200) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                yield d["idx"], d["hi"], d["seconds"]


def _random_doc(path: str) -> dict:
    """Reads one random line without loading the whole file: seek to a random byte offset, skip the
    (likely partial) line, read the next full one."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(random.randint(0, size - 1))
        f.readline()
        line = f.readline()
        if not line:
            f.seek(0)
            line = f.readline()
    return json.loads(line)


def _show_judgement(box, res, ref_res=None):
    verdict = "✅" if res.correct else "❌"
    text = f"{verdict} **P(correct) {res.p_correct:.2f}**"
    if ref_res is not None:
        text += f" · reference {ref_res.p_correct:.2f}"
    s = res.scores
    text += f" · complete {s.get('complete', 0):.2f} · prose {s.get('prose_translated', 0):.2f}"
    text += f" · math {s.get('math_preserved', 0):.2f} · fluent {s.get('fluent', 0):.2f} · issue: {res.issue}"
    box.markdown(text)


st.set_page_config(page_title="CoT Translator", layout="wide")
st.title("Chain-of-Thought Translator (EN → HI)")
st.caption("dInfer translation · judged by Typesafe Jev")

with st.sidebar:
    backend = st.radio("Model", list(BACKENDS))
    st.caption(f"server `{BACKENDS[backend]}`")
    judge_ref = st.checkbox("Also judge the dataset reference", value=False)
    max_steps = st.number_input("Max steps to translate", 1, 100, 12)

if st.button("Pick random example", type="primary"):
    st.session_state["doc"] = _random_doc(DATASET_PATH)
    st.session_state["run"] = True

doc = st.session_state.get("doc")
if doc:
    steps = doc["steps"][: int(max_steps)]
    st.caption(f"Document `{doc['id'][:12]}` — showing {len(steps)} of {len(doc['steps'])} step(s)")

    # Layout first so each step fills in live: translation when decoded, judgement when Typesafe answers.
    slots = []
    for i, step in enumerate(steps, 1):
        col1, col2 = st.columns(2)
        col1.caption(f"Step {i} — EN")
        col1.code(step["en"], language=None, wrap_lines=True)  # raw text: st.write would mangle code as markdown
        col2.caption(f"Step {i} — HI (dInfer)")
        slots.append((col2.empty(), col2.empty(), col2.expander("Reference HI (dataset)")))
        slots[-1][2].code(step["hi"], language=None, wrap_lines=True)

    judge, pool = _judge(), _pool()
    for hi_box, _, _ in slots:
        hi_box.info("decoding…")
    futures = [None] * len(steps)
    results = {}

    def show_done(wait=False):
        """Renders every judgement whose Typesafe call has finished (all of them when wait=True)."""
        for j, jobs in enumerate(futures):
            if jobs is not None and j not in results and (wait or all(f.done() for f in jobs)):
                results[j] = jobs[0].result()
                _show_judgement(slots[j][1], results[j], jobs[1].result() if len(jobs) > 1 else None)

    for i, hi, secs in _translate_stream(BACKENDS[backend], [step["en"] for step in steps]):
        hi_box, judge_box, _ = slots[i]
        hi_box.code(hi or "<empty>", language=None, wrap_lines=True)
        judge_box.caption(f"decoded in {secs}s · judging…")
        jobs = [pool.submit(judge.judge, steps[i]["en"], hi)]
        if judge_ref:
            jobs.append(pool.submit(judge.judge, steps[i]["en"], steps[i]["hi"]))
        futures[i] = jobs
        show_done()
    show_done(wait=True)

    st.divider()
    scored = list(results.values())
    st.metric("Mean P(correct)", f"{sum(r.p_correct for r in scored) / len(scored):.2f}",
              f"{sum(r.correct for r in scored)}/{len(scored)} steps judged correct")
