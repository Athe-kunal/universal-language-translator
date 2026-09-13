"""Step 5: a scheduled-quota decoder for BD3LM-trained Qwen3-a2d models
running inside dInfer, replacing dInfer's stock ThresholdParallelDecoder.

Root cause this fixes (see bd3lm_iteration.py's Step 4 diagnosis): dInfer's
ThresholdParallelDecoder commits *every* masked position whose confidence
clears a threshold in a single step - for our BD3LM checkpoint this often
resolves an entire 32-token block in one or two forward calls, including a
stray high-confidence EOS prediction that gets locked in before the block
ever has a chance at iterative refinement. The proven `dllm` BD3LM sampler
never does this: it runs a small FIXED number of steps per block
(`get_num_transfer_tokens` - dInfer already ships this exact utility,
originally written for LLaDA's own uniform linear schedule), committing
only a few of the currently-most-confident masked positions each step
regardless of their absolute confidence value. That gives already-committed
neighboring tokens a chance to change the model's mind about a
premature-looking EOS on a later step, before it gets frozen in.

This decoder reproduces that exact schedule-driven behavior via dInfer's
public ParallelDecoder interface (block_init + decode), so it plugs into
BD3LMBlockWiseDiffusionLLM the same way ThresholdParallelDecoder did -
dInfer's own package is still untouched.
"""

import math

import torch
import torch.nn.functional as F

from dinfer.decoding.parallel_strategy import ParallelDecoder, add_gumbel_noise
from dinfer.decoding.utils import get_num_transfer_tokens


class ScheduledQuotaDecoder(ParallelDecoder):
    def __init__(self, temperature, steps_per_block, mask_id, eos_id, remasking="low_confidence"):
        super().__init__(temperature, remasking, mask_id)
        if remasking != "low_confidence":
            raise NotImplementedError(remasking)
        self.steps_per_block = steps_per_block
        self.eos_id = eos_id
        self._schedule = None
        self._step = 0

    def block_init(self, block_x, block_id):
        mask_index = block_x == self.mask_id
        self._schedule = get_num_transfer_tokens(mask_index, self.steps_per_block)  # [B, steps]
        self._step = 0

    def decode(self, logits, block_start, block_end, x):
        curr_x = x[:, block_start:block_end]
        mask_index = curr_x == self.mask_id

        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
        else:
            logits_with_noise = logits
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]

        x0 = torch.where(mask_index, x0, curr_x)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

        step = min(self._step, self._schedule.shape[1] - 1)
        transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
        for b in range(mask_index.shape[0]):
            k = int(self._schedule[b, step].item())
            valid_count = int((confidence[b] > -float("inf")).sum().item())
            k = min(k, valid_count)
            if k <= 0:
                continue
            _, idx = torch.topk(confidence[b], k=k)
            transfer_index[b, idx] = True

        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        self._step += 1
