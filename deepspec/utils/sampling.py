from __future__ import annotations

import torch


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs

    filtered_logits = logits.float() / temperature
    vocab_size = int(filtered_logits.shape[-1])
    if top_k is not None:
        top_k = int(top_k)
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}.")
        if 0 < top_k < vocab_size:
            top_indices = torch.topk(filtered_logits, top_k, dim=-1).indices
            remove = torch.ones_like(filtered_logits, dtype=torch.bool)
            remove.scatter_(-1, top_indices, False)
            filtered_logits = filtered_logits.masked_fill(remove, -torch.inf)

    if top_p is not None:
        top_p = float(top_p)
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                filtered_logits,
                descending=True,
                dim=-1,
            )
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            filtered_logits = torch.empty_like(sorted_logits).scatter(
                -1,
                sorted_indices,
                sorted_logits,
            )

    return torch.softmax(filtered_logits, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, vocab_size = probs.shape
    flat = probs.reshape(-1, vocab_size)
    return torch.multinomial(flat, num_samples=1).reshape(bsz, seq_len)


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    return sample_from_probs(logits_to_probs(logits, temperature))


def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass == 0.0):
        residual = torch.where(residual_mass == 0.0, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass
    return sample_from_probs(residual.unsqueeze(1)).squeeze(1)


__all__ = [
    "gather_token_probs",
    "logits_to_probs",
    "sample_from_probs",
    "sample_residual",
    "sample_tokens",
]
