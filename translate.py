"""
Interactive English → Hindi translator using a trained masked diffusion model.

Usage:
    uv run python translate.py --model_name_or_path .models/modernbert-translation/checkpoint-final
    uv run python translate.py --model_name_or_path .models/modernbert-translation/checkpoint-final --steps 64
"""

import argparse
from dataclasses import dataclass

import transformers
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

import dllm

console = Console()


@dataclass
class ScriptArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/modernbert-translation/checkpoint-final"


@dataclass
class MDLMSamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 512
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


@dataclass
class BD3LMSamplerConfig(dllm.core.samplers.BD3LMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 512
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    right_shift_logits: bool = False


# Kept as an alias: existing callers (e.g. validate_bpcc.py) importing
# `SamplerConfig` get the MDLM variant, same as before this file grew BD3LM
# support.
SamplerConfig = MDLMSamplerConfig


def load_pipeline(model_args: ScriptArguments, sampler_type: str = "mdlm"):
    """A checkpoint trained with BD3LMTrainer (block-causal attention +
    per-block noising, see train_translation.py's --trainer bd3lm) must be
    sampled with BD3LMSampler, not MDLMSampler - the two use different
    attention-mask/position-id construction during generation
    (dllm-src/dllm/core/samplers/{mdlm,bd3lm}.py), so using the wrong one
    produces malformed output even though the model loads and runs fine."""
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    sampler_cls = (
        dllm.core.samplers.BD3LMSampler
        if sampler_type == "bd3lm"
        else dllm.core.samplers.MDLMSampler
    )
    sampler = sampler_cls(model=model, tokenizer=tokenizer)
    return model, tokenizer, sampler


def estimate_max_new_tokens(
    texts: list[str],
    tokenizer,
    factor: float = 2.5,
    min_tokens: int = 16,
    max_tokens: int = 96,
) -> int:
    """
    Sizes the fixed diffusion canvas to roughly fit the expected translation
    length. Sizing max_new_tokens much larger than the real translation
    leaves trailing positions with no signal to denoise, which this MDLM
    checkpoint fills with repetitive garbage instead of padding cleanly.
    """
    longest_src = max(len(tokenizer(t)["input_ids"]) for t in texts)
    return max(min_tokens, min(max_tokens, round(longest_src * factor)))


def translate_one(text: str, tokenizer, sampler, config) -> str:
    return translate_batch([text], tokenizer, sampler, config)[0]


def translate_batch(texts: list[str], tokenizer, sampler, config) -> list[str]:
    messages = [[{"role": "user", "content": t}] for t in texts]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    outputs = sampler.sample(inputs, config, return_dict=True)
    sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    return [s.strip() for s in sequences]


def main():
    # Pre-parse --sampler before HfArgumentParser so we know whether to build
    # an MDLMSamplerConfig or BD3LMSamplerConfig (mirrors train_translation.py's
    # --trainer pre-parse, for the same reason: the two sampler configs are
    # different dataclasses, so the choice has to happen before the parser
    # for the *actual* config class is constructed).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sampler", default="mdlm", choices=["mdlm", "bd3lm"])
    known, remaining = pre.parse_known_args()

    sampler_config_cls = (
        BD3LMSamplerConfig if known.sampler == "bd3lm" else MDLMSamplerConfig
    )
    parser = transformers.HfArgumentParser((ScriptArguments, sampler_config_cls))
    script_args, sampler_config = parser.parse_args_into_dataclasses(
        args=remaining, look_for_args_file=False
    )

    console.print(Rule("[bold magenta]Universal Language Translator[/]"))
    console.print(f"[dim]Loading model from [cyan]{script_args.model_name_or_path}[/cyan]...[/dim]")

    _, tokenizer, sampler = load_pipeline(script_args, sampler_type=known.sampler)
    visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

    console.print("[green]Model ready.[/green] Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n")

    while True:
        try:
            text = Prompt.ask("[bold cyan]English[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not text or text.lower() in ("exit", "quit"):
            break

        messages = [[{"role": "user", "content": text}]]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        console.print()
        console.print(Rule("[dim]Translating — watch the masks resolve[/dim]"))

        outputs = sampler.sample(inputs, sampler_config, return_dict=True)
        sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

        # Play back the demasking animation
        visualizer.visualize(outputs.histories, rich=True)

        translation = sequences[0].strip() if sequences[0].strip() else "<empty>"
        console.print()
        console.print(Panel(translation, title="[bold green]Hindi[/bold green]", border_style="green"))
        console.print()


if __name__ == "__main__":
    main()
