"""Uploads weight-only checkpoints to W&B at each eval interval, without
keeping a local copy - for hosts with limited local disk.

Unlike train_translation.py's WandbArtifactCallback (which fires on
on_epoch_end and leaves the saved directory on disk), this fires on
on_evaluate (so its cadence follows eval_steps, meaningful for a fixed
max_steps GRPO run rather than an epoch boundary) and deletes its temporary
save directory immediately after the artifact upload completes.
"""

import shutil
import tempfile
from pathlib import Path

import transformers

logger = transformers.utils.logging.get_logger(__name__)


class WandbWeightOnlyCheckpointCallback(transformers.TrainerCallback):
    """Uploads a weights-only checkpoint to W&B on every evaluation."""

    def on_evaluate(self, args, state, control, **kwargs):
        import wandb

        if not wandb.run:
            logger.warning("wandb not initialized - skipping checkpoint upload.")
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None:
            return

        with tempfile.TemporaryDirectory(prefix="wandb-ckpt-") as tmp_dir:
            model.save_pretrained(tmp_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)

            artifact = wandb.Artifact(
                name=f"{Path(args.output_dir).name}-checkpoint",
                type="model",
                metadata={"step": state.global_step, "epoch": state.epoch},
            )
            artifact.add_dir(tmp_dir)
            wandb.log_artifact(artifact)

        logger.info(f"Uploaded weight-only checkpoint at step {state.global_step} to W&B.")
