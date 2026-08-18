"""Gate check: does LLaDA-MoE-7B-A1B-Instruct's tokenizer handle Devanagari as
real subwords, or fall back to byte-level encoding?

The whole LLaDA-MoE fine-tuning plan hinges on this. Roughly 8-12 tokens for
the sentence below means real Devanagari subwords. Thirty-plus means byte
fallback (the SMDM trap) and the plan needs rethinking before any training
runs.

Usage:
    uv run python scripts/check_llada_tokenizer.py
    uv run python scripts/check_llada_tokenizer.py --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct
"""

import argparse

from transformers import AutoTokenizer

DEFAULT_MODEL = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

# A short, unremarkable Hindi sentence ("This is a test sentence.") used
# purely to count subword fragments, not to judge translation quality.
TEST_SENTENCES = [
    "यह एक परीक्षण वाक्य है।",
    "मुझे हिंदी में अनुवाद करना पसंद है।",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Tokenizer: {args.model}")
    print(f"Vocab size: {tok.vocab_size}\n")

    verdicts = []
    for sentence in TEST_SENTENCES:
        ids = tok.encode(sentence, add_special_tokens=False)
        tokens = tok.convert_ids_to_tokens(ids)
        n_chars = len(sentence)
        ratio = len(ids) / n_chars
        verdicts.append(len(ids))
        print(f"Sentence: {sentence}")
        print(f"  chars={n_chars}  tokens={len(ids)}  chars/token={n_chars / len(ids):.2f}")
        print(f"  token pieces: {tokens}\n")

    avg = sum(verdicts) / len(verdicts)
    print("-" * 60)
    if avg <= 15:
        print(f"PASS  avg {avg:.1f} tokens/sentence — looks like real Devanagari subwords.")
    elif avg <= 25:
        print(f"BORDERLINE  avg {avg:.1f} tokens/sentence — inspect the token pieces above by hand.")
    else:
        print(
            f"FAIL  avg {avg:.1f} tokens/sentence — looks like byte-level fallback. "
            "The LLaDA-MoE plan needs rethinking."
        )


if __name__ == "__main__":
    main()
