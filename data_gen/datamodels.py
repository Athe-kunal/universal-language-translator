from dataclasses import dataclass
from typing import Any, Literal



@dataclass
class TranslationDataset:
    id: str  # stable hash for dedup/resume, e.g. md5(question)
    question: str
    reference_answer: str | None
    cot_answer: str
    metadata: dict[str, Any] #useful for provenance
    source: Literal["openthoughts3", "natural-reasoning", "opencodereasoning"]