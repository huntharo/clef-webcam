"""Faster gated delta rule for Apple Silicon.

Qwen3.5's linear-attention layers fall back to a reference PyTorch implementation when the CUDA-only
flash-linear-attention kernels are unavailable. On MPS, that fallback spends most of its time in
torch.linalg.solve_triangular. This version computes the same result, but inverts each chunk's unit
lower-triangular system by recursive block inversion (log2(chunk_size) levels of batched matmuls),
which Apple GPUs run quickly.
"""

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _unit_lower_inverse(strict_lower: torch.Tensor) -> torch.Tensor:
    """Inverse of (I + strict_lower) for a batch of square matrices whose size is a power of two.

    Recursive 2x2 block inversion, vectorized per level: inv([[A, 0], [C, D]]) = [[A', 0], [-D' C A', D']].
    It takes log2(size) levels of batched matmuls and is as stable as forward substitution.
    """
    size = strict_lower.shape[-1]
    batch = strict_lower.shape[:-2]
    blocks = torch.ones(*batch, size, 1, 1, dtype=strict_lower.dtype, device=strict_lower.device)
    width = 1
    while width < size:
        count = size // (2 * width)
        tiles = strict_lower.reshape(*batch, count, 2 * width, count, 2 * width)
        lower_left = torch.diagonal(tiles, dim1=-4, dim2=-2).movedim(-1, -3)[..., width:, :width]
        upper, lower = blocks[..., 0::2, :, :], blocks[..., 1::2, :, :]
        corner = -(lower @ lower_left @ upper)
        top = torch.cat([upper, torch.zeros_like(upper)], dim=-1)
        bottom = torch.cat([corner, lower], dim=-1)
        blocks = torch.cat([top, bottom], dim=-2)
        width *= 2
    return blocks[..., 0, :, :]


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in replacement for transformers' torch_chunk_gated_delta_rule (same arguments and outputs)."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query = query * k_head_dim**-0.5

    pad = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    )
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    num_chunks = decay.shape[2]

    cum_decay = decay.cumsum(dim=-1)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    pairwise_decay = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    inverse = _unit_lower_inverse(ut_system.tril(-1))
    new_values = inverse @ v_beta
    k_cumdecay = inverse @ (k_beta * cum_decay.exp().unsqueeze(-1))

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    state = (
        torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    outputs = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        outputs.append(query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = torch.stack(outputs, dim=2).reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def patch_qwen3_5() -> None:
    """Route Qwen3.5 linear attention through the faster implementation above."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
