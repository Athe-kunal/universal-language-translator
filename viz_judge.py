"""Browse the 100K EN->HI reasoning dataset with a calibrated correctness verdict per step.

Shows each document's English / Hindi steps side by side, and the TypeSafe Jev
calibrated decision (P(translation correct)) for each step. Verdicts are cached in
typesafe_judgements.jsonl (keyed by the step pair plus the prompt version) so nothing is billed twice.

Usage:
    TYPESAFE_API_KEY in .env, then:  streamlit run viz_judge.py
"""

import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from rl.typesafe_judge import PROMPT_HASH, TypesafeJudge

DATA_PATH = "reasoning_hi_final_100k.jsonl"
CACHE_PATH = "typesafe_judgements.jsonl"
MAX_WORKERS = 8


@st.cache_resource(show_spinner="Indexing dataset…")
def _offsets(path: str) -> list[int]:
    offsets, pos = [], 0
    with open(path, "rb") as f:
        for line in f:
            offsets.append(pos)
            pos += len(line)
    return offsets


def _load_doc(path: str, offset: int) -> dict:
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def _key(en: str, hi: str) -> str:
    return hashlib.sha1(f"{en}\x00{hi}\x00{PROMPT_HASH}".encode()).hexdigest()


@st.cache_resource
def _verdicts() -> dict[str, dict]:
    out = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out[row["key"]] = row
    return out


def _judge_steps(judge: TypesafeJudge, steps: list[dict]) -> None:
    verdicts = _verdicts()
    todo = [s for s in steps if _key(s["en"], s["hi"]) not in verdicts]

    def run(step):
        try:
            r = judge.judge(step["en"], step["hi"])
            return _key(step["en"], step["hi"]), {"p_correct": r.p_correct, "correct": r.correct,
                                                  "scores": r.scores, "issue": r.issue}
        except Exception as e:  # noqa: BLE001 - surface API errors in the UI, keep going
            return _key(step["en"], step["hi"]), {"error": str(e)}

    with ThreadPoolExecutor(MAX_WORKERS) as pool, open(CACHE_PATH, "a", encoding="utf-8") as f:
        for key, res in pool.map(run, todo):
            if "error" in res:
                st.warning(f"Judge error: {res['error']}")
                continue
            row = {"key": key, **res}
            verdicts[key] = row
            f.write(json.dumps(row) + "\n")


def _badge(v: dict | None) -> str:
    if v is None:
        return "not judged yet"
    mark = "CORRECT" if v["correct"] else "INCORRECT"
    return f"{mark} — P(correct) = {v['p_correct']:.2f}"


st.set_page_config(page_title="Translation judge", layout="wide")
st.title("EN → HI translation: calibrated correctness")

offsets = _offsets(DATA_PATH)
verdicts = _verdicts()

if "idx" not in st.session_state:
    st.session_state["idx"] = random.randrange(len(offsets))

col_a, col_b, col_c = st.sidebar.columns([2, 1, 1])
idx = col_a.number_input("Document", 0, len(offsets) - 1, st.session_state["idx"])
if col_b.button("Rand"):
    idx = random.randrange(len(offsets))
    st.session_state["idx"] = idx
    st.rerun()
st.session_state["idx"] = int(idx)

judged = list(verdicts.values())
if judged:
    st.sidebar.metric("Steps judged", len(judged))
    st.sidebar.metric("Judged correct", f"{sum(v['correct'] for v in judged) / len(judged):.0%}")

doc = _load_doc(DATA_PATH, offsets[int(idx)])
steps = doc["steps"]
st.caption(f"Document {int(idx)} · id `{doc['id'][:12]}` · {len(steps)} steps")

judge = None
if os.environ.get("TYPESAFE_API_KEY") or os.path.exists(".env"):
    try:
        judge = TypesafeJudge()
    except KeyError:
        judge = None
if judge is None:
    st.info("Set TYPESAFE_API_KEY in .env to get verdicts; browsing works without it.")
elif st.button("Judge all steps in this document", type="primary"):
    with st.spinner(f"Judging {len(steps)} steps…"):
        _judge_steps(judge, steps)
    st.rerun()

for i, step in enumerate(steps, 1):
    v = verdicts.get(_key(step["en"], step["hi"]))
    with st.container(border=True):
        head, action = st.columns([5, 1])
        head.markdown(f"**Step {i}**")
        if v is not None:
            (st.success if v["correct"] else st.error)(_badge(v))
            if "scores" in v:
                detail = "  ·  ".join(f"{q} {p:.2f}" for q, p in v["scores"].items() if q != "correct")
                st.caption(f"{detail}  ·  main issue: {v['issue']}")
        elif judge is not None and action.button("Judge", key=f"judge_{i}"):
            _judge_steps(judge, [step])
            st.rerun()
        left, right = st.columns(2)
        left.caption("English")
        left.text(step["en"])
        right.caption("Hindi")
        right.text(step["hi"])
