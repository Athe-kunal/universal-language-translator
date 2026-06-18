"""
Interactive English → Hindi translator using a trained masked diffusion model.

Usage:
    uv run python translate.py --model_name_or_path .models/modernbert-translation/checkpoint-final
    uv run python translate.py --model_name_or_path .models/modernbert-translation/checkpoint-final --steps 64
"""

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
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 512
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()

    console.print(Rule("[bold magenta]Universal Language Translator[/]"))
    console.print(f"[dim]Loading model from [cyan]{script_args.model_name_or_path}[/cyan]...[/dim]")

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
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
