"""GRPO trainer for BD3LM (block diffusion) checkpoints.

dllm's DiffuGRPOTrainer hardcodes MDLM's forward process in
_get_per_token_logps: a single bidirectional forward pass over one noised
sequence. BD3LM checkpoints (dllm.core.trainers.bd3lm.BD3LMTrainer) are
trained differently - each forward pass concatenates [noised_x_t; clean_x_0]
along the sequence dimension and applies a specialized block-causal
attention mask, only reading logits from the first half. Reusing
DiffuGRPOTrainer unmodified against a BD3LM checkpoint would compute GRPO's
policy-ratio and KL terms under a forward process the model wasn't trained
under.

This subclass overrides _get_per_token_logps to mirror BD3LMTrainer's actual
forward pass, so the RL objective matches the checkpoint's SFT objective.
"""

import torch
import torch.nn.functional as F

from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask
from dllm.pipelines.rl import DiffuGRPOTrainer


class BD3LMDiffuGRPOTrainer(DiffuGRPOTrainer):
    """DiffuGRPOTrainer variant using BD3LM's block-diffusion forward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = self.args.block_size
        # DiffuGRPOTrainer._generate_and_score_completions reads
        # self.max_prompt_length, an attribute older TRL versions set from
        # GRPOConfig.max_prompt_length. trl>=0.29 dropped that field (and the
        # attribute) entirely, so the base class never sets it - restore the
        # old "no truncation" default so that code path doesn't AttributeError.
        self.max_prompt_length = None
        # Same rename as above: DiffuGRPOTrainer._generate_and_score_completions
        # writes to self._textual_logs (prompt/completion/rewards/advantages
        # deques), the old TRL name for what trl>=0.29 now calls self._logs.
        # Both refer to the same dict/keys, so alias rather than duplicate.
        self._textual_logs = self._logs

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        seed=None,
    ) -> torch.Tensor:
        """Computes per-token log-probs under BD3LM's block-diffusion forward process.

        Mirrors dllm.core.trainers.bd3lm.BD3LMTrainer.compute_loss's forward
        pass: [noised_x_t; clean_x_0] concatenated along the sequence
        dimension, one forward pass under the block-causal attention mask,
        loss read from the first half's logits. The noising itself (which
        tokens get masked) reuses the base class's _forward_process
        unchanged - that recipe (mask completion fully, prompt with
        p_mask_prompt) is GRPO's variance-reduction convention, independent
        of the base model's architecture.

        Does not support classifier-free guidance (sampler_config.cfg_scale
        > 0) - raises if set, rather than silently computing the wrong loss.
        """
        if self.sampler_config.cfg_scale > 0.0:
            raise NotImplementedError(
                "BD3LMDiffuGRPOTrainer does not support cfg_scale > 0."
            )
        if seed is None:
            seed = getattr(self, "_current_mask_seed", None)
        batch_size = batch_size or input_ids.size(0)
        mask_id = self.processing_class.mask_token_id
        seq_len = input_ids.size(1)
        prompt_length = seq_len - logits_to_keep

        prompt_index = torch.arange(seq_len, device=input_ids.device) < prompt_length

        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            batch = input_ids[i : i + batch_size]
            b, l = batch.shape

            noised_input_ids = self._forward_process(
                batch, prompt_index, mask_id, seed=seed
            )

            # [x_t; x_0] concatenation + block-diffusion attention mask,
            # exactly as BD3LMTrainer.compute_loss does at SFT time.
            concat_input_ids = torch.cat([noised_input_ids, batch], dim=1)
            base_pos = torch.arange(l, device=batch.device).unsqueeze(0).expand(b, l)
            concat_position_ids = torch.cat([base_pos, base_pos], dim=1)
            block_attention_mask = (
                _create_bd3lm_attention_mask(
                    b=None,
                    h=None,
                    q_idx=torch.arange(l * 2, device=batch.device)[:, None],
                    kv_idx=torch.arange(l * 2, device=batch.device)[None, :],
                    block_size=self.block_size,
                    n=l,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(b, 1, 2 * l, 2 * l)
            )

            logits = model(
                input_ids=concat_input_ids,
                attention_mask=block_attention_mask,
                position_ids=concat_position_ids,
            ).logits[:, :l]

            completion_logits = logits[:, -logits_to_keep:, :]
            completion_targets = batch[:, -logits_to_keep:]
            loss = F.cross_entropy(
                completion_logits.reshape(-1, completion_logits.size(-1)),
                completion_targets.reshape(-1),
                reduction="none",
            )
            all_logps.append(-loss.view(b, logits_to_keep).to(torch.float32))

        per_token_logps = torch.cat(all_logps, dim=0)
        completion_mask = attention_mask[:, -logits_to_keep:]
        per_token_logps = per_token_logps * completion_mask
        return per_token_logps
