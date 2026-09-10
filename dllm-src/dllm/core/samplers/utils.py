import torch

from dllm.core.schedulers import BaseAlphaScheduler


def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    scheduler: BaseAlphaScheduler,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute the number of tokens to unmask at each diffusion step.

    For each sample, determines how many masked tokens should be revealed
    per step based on the reverse diffusion schedule.

    Args:
        mask_index: Boolean tensor [B, L] indicating masked positions.
        steps: Number of diffusion steps.
        scheduler: Alpha scheduler defining the masking schedule.
        stochastic: If True, sample from a binomial distribution (probabilistic);
            if False, use deterministic rounding of the expected number of tokens.

    Returns:
        Integer tensor [B, steps] with number of tokens to unmask per step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    )
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps - 1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = (
                    torch.distributions.Binomial(n, reverse_transfer_prob)
                    .sample()
                    .to(torch.int64)
                )
            num_transfer_tokens[i, j] = torch.minimum(
                num_transfer_tokens[i, j], mask_num[i, 0]
            )
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0:
                break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


def apply_repetition_penalty(
    logits: torch.Tensor,
    x: torch.Tensor,
    penalty: float,
    exclude_ids: set[int],
) -> torch.Tensor:
    """
    Standard repetition penalty (Keskar et al. 2019 / HF's
    RepetitionPenaltyLogitsProcessor), adapted for a diffusion sampler where
    "already generated" means "already committed anywhere in the current
    sequence x" rather than "earlier in a strictly left-to-right history":
    for every token id seen in x (excluding pad/mask), divide its logit by
    `penalty` if positive or multiply by `penalty` if negative - discourages
    (without forbidding) picking that token again, penalizing the
    short-phrase repetition loops diffusion decoding can fall into on some
    inputs (e.g. reasoning steps containing "Wait").

    Args:
        logits: [B, L, V] logits for the block currently being denoised.
        x: [B, T] full sequence so far (prompt + already-committed tokens).
        penalty: >1.0 discourages repeats; 1.0 is a no-op.
        exclude_ids: Token ids (pad, mask, ...) never penalized.

    Returns:
        Penalized logits, same shape as `logits`.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    exclude = torch.as_tensor(list(exclude_ids), device=x.device)
    for b in range(x.size(0)):
        seen = torch.unique(x[b])
        seen = seen[~torch.isin(seen, exclude)]
        if seen.numel() == 0:
            continue
        vals = logits[b, :, seen]
        logits[b, :, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
