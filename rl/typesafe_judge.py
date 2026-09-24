"""Minimal OpenAI-SDK client for TypeSafe's Jev calibrated-decision model.

Jev is a "System One" model: it returns typed, calibrated decisions, not text,
so it is served at POST {base}/v1/systemone rather than /chat/completions.
The OpenAI SDK still works for it - same Bearer auth, custom path via
`client.post(...)` - which is what this module uses.

One request carries several questions. `correct` is the overall verdict, the other yes/no
questions and the `issue` choice say what is wrong when it is not.

Env (.env is loaded if python-dotenv is installed):
    TYPESAFE_API_KEY   required (TYPESAFE_AI is also accepted)
    TYPESAFE_BASE_URL  optional, default https://api.typesafe.ai
                       (OpenRouter: https://openrouter.ai/api)
    TYPESAFE_MODEL     optional, default jev-latest
"""

import hashlib
import json
import os
from dataclasses import dataclass, field

from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"

QUESTIONS = {
    "correct": {
        "type": "noul",
        "instructions": (
            "Is the Hindi a faithful, complete and fluent translation of the English? "
            "Math equations, formulas, numbers, tables, code and variable names must stay exactly as in the English, "
            "so a verbatim copy of such content is correct and must not be penalized. "
            "Only the surrounding English prose should be translated into Hindi. "
            "Consider missing content, mistranslation, garbling, repetition, prose left in English "
            "and math or code that was altered or transliterated."
        ),
        "criteria": {
            "true": "Prose is faithfully translated into Hindi, and math, numbers and code are kept exactly unchanged",
            "false": "Prose is missing, wrong, garbled, repeated or left in English, or math or code was altered",
        },
    },
    "complete": {
        "type": "noul",
        "instructions": "Is every part of the English content present in the Hindi, with nothing dropped or cut off?",
        "criteria": {
            "true": "All content is present and the Hindi ends where the English ends",
            "false": "Part of the content is missing or the Hindi is cut off",
        },
    },
    "prose_translated": {
        "type": "noul",
        "instructions": (
            "Is all of the English prose translated into Hindi, apart from math, code "
            "and technical terms that may stay in English?"
        ),
        "criteria": {
            "true": "All prose is written in Hindi in Devanagari script",
            "false": "Some prose is left in English or written in romanized Hindi",
        },
    },
    "math_preserved": {
        "type": "noul",
        "instructions": "Are all equations, numbers, code and variable names in the Hindi identical to the English?",
        "criteria": {
            "true": "Every equation, number and piece of code is unchanged",
            "false": "Some math, number or code was changed, transliterated or dropped",
        },
    },
    "fluent": {
        "type": "noul",
        "instructions": "Is the Hindi free of repetition and garbled text, and does it read naturally?",
        "criteria": {
            "true": "Natural Hindi with no repeated loops and no corrupted characters",
            "false": "Contains repeated phrases, corrupted characters or unreadable text",
        },
    },
    "issue": {
        "type": "choice",
        "instructions": "What is the main problem with the Hindi translation, if any?",
        "criteria": {
            "none": "No real problem",
            "truncated": "Content is cut off or missing",
            "repeated": "Phrases repeat in a loop",
            "untranslated": "Prose is left in English or in romanized Hindi",
            "math_altered": "Math, numbers or code were changed",
            "garbled": "Corrupted or unreadable text",
            "wrong_content": "The Hindi does not match the English",
        },
    },
}


def prompt_hash(questions: dict) -> str:
    """Short stable id of the question set, so cached verdicts can be tied to the prompt that made them."""
    return hashlib.sha1(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:8]


PROMPT_HASH = prompt_hash(QUESTIONS)


@dataclass
class JudgeResult:
    p_correct: float
    correct: bool
    model: str
    scores: dict[str, float] = field(default_factory=dict)  # P(true) for every yes/no question
    issue: str = "none"  # main problem chosen by the `issue` question


class TypesafeJudge:
    """Judges a Hindi translation with calibrated probabilities, overall and per aspect."""

    def __init__(self, api_key=None, base_url=None, model=None, threshold=0.5, http_client=None):
        base = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.client = OpenAI(
            api_key=api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ["TYPESAFE_AI"],
            base_url=f"{base}/v1",
            http_client=http_client,
        )
        self.model = model or os.environ.get("TYPESAFE_MODEL") or DEFAULT_MODEL
        self.threshold = threshold

    def judge(self, en: str, hi: str) -> JudgeResult:
        body = {
            "model": self.model,
            "state": f"English source:\n{en}\n\nHindi translation:\n{hi}",
            "questions": QUESTIONS,
        }
        resp = self.client.post("/systemone", body=body, cast_to=object)
        answers = resp["answers"]
        scores = {
            name: float(answers[name].get("noul", answers[name].get("value")))
            for name, q in QUESTIONS.items()
            if q["type"] == "noul"
        }
        p = scores["correct"]
        return JudgeResult(
            p_correct=p,
            correct=p >= self.threshold,
            model=resp.get("model", self.model),
            scores=scores,
            issue=answers["issue"].get("choice", "none"),
        )
