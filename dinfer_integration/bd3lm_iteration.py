"""Step 4: block-causal DiffusionIteration for BD3LM-trained Qwen3-a2d models
running inside dInfer's decode loop.

dInfer's own `BaseDiffusionIteration` (used by `BlockWiseDiffusionLLM` when
`cache_factory=None`) calls `model(x.data)` with no attention_mask - fine for
MDLM (our model defaults to full bidirectional attention when
attention_mask is None), but wrong for BD3LM, which was trained with a
block-causal mask: a query at physical position q may attend to any key at
position k with block_id(k) <= block_id(q), where block_id(pos) = pos //
block_size - i.e. causal *between* blocks, fully bidirectional *within* a
block. See dllm-src/dllm/core/samplers/bd3lm.py's `_prepare_for_sampling`
for the reference implementation this is adapted from.

Unlike that reference (which grows the sequence block-by-block and has to
recompute the mask incrementally), dInfer pre-allocates the whole canvas
(prompt + gen_length, all-mask_id) up front via `TokenArray`, so the mask
depends only on physical position, not on what's been decoded yet - it can
be built once per `generate()` call and reused for every diffusion step.

Batch note: this assumes batch_size=1 (no padding). Batched rows with
different real prompt lengths would need the mask itself to be row-specific
(dInfer's canvas right-pads shorter prompts with mask_id, which is
indistinguishable from generation content, so block alignment would be
wrong for a batch of mixed prompt lengths) - out of scope for this step.
"""

import torch

from dinfer.decoding.generate_uniform import BaseDiffusionIteration, BlockWiseDiffusionLLM


def build_block_causal_mask(total_len: int, block_size: int, device) -> torch.Tensor:
    """[1, 1, T, T] bool mask: True where key may be attended to by query."""
    pos = torch.arange(total_len, device=device)
    block_ids = torch.div(pos, block_size, rounding_mode="floor")
    bid_q = block_ids.view(1, 1, total_len, 1)
    bid_k = block_ids.view(1, 1, 1, total_len)
    return bid_k <= bid_q


class BD3LMDiffusionIteration(BaseDiffusionIteration):
    """Same as BaseDiffusionIteration, but passes a block-causal attention_mask
    (built once, cached) instead of leaving attention_mask=None."""

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size
        self._cached_mask = None
        self._cached_len = None

    def _mask_for(self, total_len: int, device) -> torch.Tensor:
        if self._cached_len != total_len:
            self._cached_mask = build_block_causal_mask(total_len, self.block_size, device)
            self._cached_len = total_len
        return self._cached_mask

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        assert kv_cache is None, "BD3LMDiffusionIteration only supports cache_factory=None for now (Step 4)"
        mask = self._mask_for(x.data.shape[1], x.data.device)
        logits = model(x.data, attention_mask=mask).logits[:, block_loc.start:block_loc.end]
        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return None, logits


class BD3LMBlockWiseDiffusionLLM(BlockWiseDiffusionLLM):
    """BlockWiseDiffusionLLM wired up with BD3LMDiffusionIteration instead of
    the base (mask-agnostic) iteration. cache_factory must be None (Step 4
    scope); KV-caching for BD3LM is a later step."""

    def __init__(self, model, decoder, iterator_factory, block_size: int, early_stop=True, maximum_unroll=4, expected_tpf=8):
        super().__init__(model, decoder, iterator_factory, early_stop=early_stop, cache_factory=None, maximum_unroll=maximum_unroll, expected_tpf=expected_tpf)
        self.diff_iteration = BD3LMDiffusionIteration(block_size)
        from dinfer.decoding.generate_uniform import BlockRunner
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)
