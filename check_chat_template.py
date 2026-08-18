"""
Verify a checkpoint's chat template round-trips identically between training
(dllm.utils.default_sft_map_fn) and inference (translate.py::translate_batch),
and that mask/eos/pad token ids are distinct and sane — the MDLM sampler
depends on all three being different tokens.

Usage:
    uv run python check_chat_template.py jhu-clsp/mmBERT-base
    uv run python check_chat_template.py .models/mmbert-diffusion-adapted/checkpoint-final
"""

import argparse
import sys

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)

    ok = True

    print(f"=== {args.model_name_or_path} ===")

    # --- chat template presence ---
    if tok.chat_template is None:
        print("FAIL: no chat_template — apply_chat_template will fall back to a "
              "transformers default, which will not match what the model was "
              "trained to expect.")
        ok = False
    else:
        print("OK: chat_template is set.")
        print("--- template ---")
        print(tok.chat_template)
        print("----------------")

    # --- training vs inference formatting parity ---
    messages_full = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "जाँच"},
    ]

    try:
        # Mirrors translate.py::translate_batch (inference-side prompt).
        inference_ids = tok.apply_chat_template(
            [messages_full[0]], add_generation_prompt=True, tokenize=True
        )
        # Mirrors dllm.utils.default_sft_map_fn's prompt_tokens (training-side
        # prompt used to compute prompt_len / mask the prompt out of the loss).
        training_prompt_ids = tok.apply_chat_template(
            messages_full[:-1], add_generation_prompt=True, tokenize=True
        )
        if inference_ids == training_prompt_ids:
            print("OK: inference prompt formatting matches the training-side "
                  "prompt formatting (same tokens up to the generation prompt).")
        else:
            print("FAIL: inference prompt tokens differ from the training-side "
                  "prompt tokens — training and inference are feeding the model "
                  "different formats for the same logical input.")
            print(f"  inference:       {inference_ids}")
            print(f"  training prompt: {training_prompt_ids}")
            ok = False

        # Mirrors default_sft_map_fn's full prompt+response encoding.
        full_ids = tok.apply_chat_template(
            messages_full, tokenize=True, add_generation_prompt=False
        )
        if full_ids[: len(training_prompt_ids)] == training_prompt_ids:
            print("OK: full prompt+response encoding starts with the same "
                  "prompt tokens used for prompt_len masking.")
        else:
            print("FAIL: full encoding does not start with the training prompt "
                  "tokens — default_sft_map_fn's prompt-length masking "
                  "(labels[:len(prompt_tokens)] = -100) will mask the wrong span.")
            ok = False
    except Exception as e:
        print(f"FAIL: apply_chat_template raised: {e!r}")
        ok = False

    # --- special token sanity ---
    mask_id, eos_id, pad_id = tok.mask_token_id, tok.eos_token_id, tok.pad_token_id
    print(f"mask_token_id={mask_id}  eos_token_id={eos_id}  pad_token_id={pad_id}")

    if mask_id is None:
        print("FAIL: mask_token_id is None — MDLMSampler needs a real [MASK] token.")
        ok = False
    if eos_id is None:
        print("FAIL: eos_token_id is None — the sampler's canvas background and "
              "training's EOS-padding both need a real EOS token.")
        ok = False
    if mask_id is not None and mask_id == eos_id:
        print("FAIL: mask_token_id == eos_token_id — the sampler can't tell "
              "'still masked' from 'terminated'.")
        ok = False
    if pad_id is not None and pad_id == mask_id:
        print("FAIL: pad_token_id == mask_token_id — padded positions would "
              "look like unresolved masks.")
        ok = False
    if ok and mask_id is not None and eos_id is not None and mask_id != eos_id:
        print("OK: mask/eos are distinct.")

    print()
    print("PASS" if ok else "FAIL — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
