# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental sparse/window attention helpers for Cosmos3 GEN tokens."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class Cosmos3SparseWindowConfig:
    enabled: bool = False
    temporal_radius: int = 4
    spatial_radius: int = 4
    query_block_size: int = 256
    min_latent_frames: int = 16

    @classmethod
    def from_value(cls, value: bool | Mapping[str, Any] | None) -> Cosmos3SparseWindowConfig:
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls(enabled=value)
        valid_fields = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in valid_fields})


@dataclass(frozen=True)
class Cosmos3SparseWindowMetadata:
    video_shape: tuple[int, int, int]
    patch_grid: tuple[int, int]
    local_token_start: int
    local_token_count: int
    context_token_start: int
    halo_tokens: int
    rank: int
    world_size: int
    process_group: dist.ProcessGroup | None
    query_ranges: tuple[tuple[int, int], ...]
    key_indices: torch.Tensor
    key_lens: torch.Tensor
    key_lens_cpu: tuple[int, ...]


def _block_key_indices(
    *,
    q_start: int,
    q_end: int,
    frame_size: int,
    hp: int,
    wp: int,
    total_tokens: int,
    context_start: int,
    context_end: int,
    temporal_radius: int,
    spatial_radius: int,
) -> list[int]:
    q_first = q_start
    q_last = q_end - 1
    t0 = q_first // frame_size
    t1 = q_last // frame_size

    h0 = hp
    h1 = 0
    w0 = wp
    w1 = 0
    for q_idx in range(q_start, q_end):
        rem = q_idx % frame_size
        h = rem // wp
        w = rem % wp
        h0 = min(h0, h)
        h1 = max(h1, h)
        w0 = min(w0, w)
        w1 = max(w1, w)

    kt0 = max(0, t0 - temporal_radius)
    kt1 = min((total_tokens // frame_size) - 1, t1 + temporal_radius)
    kh0 = max(0, h0 - spatial_radius)
    kh1 = min(hp - 1, h1 + spatial_radius)
    kw0 = max(0, w0 - spatial_radius)
    kw1 = min(wp - 1, w1 + spatial_radius)

    indices: list[int] = []
    for kt in range(kt0, kt1 + 1):
        frame_base = kt * frame_size
        for kh in range(kh0, kh1 + 1):
            row_base = frame_base + kh * wp
            for kw in range(kw0, kw1 + 1):
                key_idx = row_base + kw
                if context_start <= key_idx < context_end:
                    indices.append(key_idx - context_start)
    return indices


def build_sparse_window_metadata(
    *,
    config: Cosmos3SparseWindowConfig,
    video_shape: tuple[int, int, int],
    patch_grid: tuple[int, int],
    local_token_start: int,
    local_token_count: int,
    rank: int = 0,
    world_size: int = 1,
    process_group: dist.ProcessGroup | None = None,
    device: torch.device,
) -> Cosmos3SparseWindowMetadata | None:
    if not config.enabled:
        return None

    t, _, _ = video_shape
    hp, wp = patch_grid
    frame_size = hp * wp
    total_tokens = t * frame_size
    if t < config.min_latent_frames:
        return None
    if local_token_start % frame_size != 0:
        return None
    if local_token_count % frame_size != 0:
        return None

    halo_tokens = 0
    if world_size > 1:
        halo_tokens = min(config.temporal_radius * frame_size, local_token_count)
    context_start = max(0, local_token_start - halo_tokens)
    context_end = min(total_tokens, local_token_start + local_token_count + halo_tokens)

    query_ranges: list[tuple[int, int]] = []
    block_indices: list[list[int]] = []
    for local_q_start in range(0, local_token_count, config.query_block_size):
        local_q_end = min(local_q_start + config.query_block_size, local_token_count)
        global_q_start = local_token_start + local_q_start
        global_q_end = local_token_start + local_q_end
        indices = _block_key_indices(
            q_start=global_q_start,
            q_end=global_q_end,
            frame_size=frame_size,
            hp=hp,
            wp=wp,
            total_tokens=total_tokens,
            context_start=context_start,
            context_end=context_end,
            temporal_radius=config.temporal_radius,
            spatial_radius=config.spatial_radius,
        )
        query_ranges.append((local_q_start, local_q_end))
        block_indices.append(indices)

    max_keys = max(len(indices) for indices in block_indices)
    key_indices = torch.empty((len(block_indices), max_keys), dtype=torch.long, device=device)
    key_lens = torch.empty((len(block_indices),), dtype=torch.int32, device=device)
    key_lens_cpu: list[int] = []
    for block_idx, indices in enumerate(block_indices):
        lens = len(indices)
        key_lens_cpu.append(lens)
        key_lens[block_idx] = lens
        key_indices[block_idx, :lens] = torch.tensor(indices, dtype=torch.long, device=device)
        if lens < max_keys:
            key_indices[block_idx, lens:] = indices[-1]

    return Cosmos3SparseWindowMetadata(
        video_shape=video_shape,
        patch_grid=patch_grid,
        local_token_start=local_token_start,
        local_token_count=local_token_count,
        context_token_start=context_start,
        halo_tokens=halo_tokens,
        rank=rank,
        world_size=world_size,
        process_group=process_group,
        query_ranges=tuple(query_ranges),
        key_indices=key_indices,
        key_lens=key_lens,
        key_lens_cpu=tuple(key_lens_cpu),
    )


def _all_gather_edges(
    tensor: torch.Tensor,
    *,
    edge_tokens: int,
    rank: int,
    world_size: int,
    process_group: dist.ProcessGroup,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    left_edge = tensor[:, :edge_tokens].contiguous()
    right_edge = tensor[:, -edge_tokens:].contiguous()
    left_edges = [torch.empty_like(left_edge) for _ in range(world_size)]
    right_edges = [torch.empty_like(right_edge) for _ in range(world_size)]
    dist.all_gather(left_edges, left_edge, group=process_group)
    dist.all_gather(right_edges, right_edge, group=process_group)

    left_halo = right_edges[rank - 1] if rank > 0 else None
    right_halo = left_edges[rank + 1] if rank + 1 < world_size else None
    return left_halo, right_halo


def _build_context(
    tensor: torch.Tensor,
    metadata: Cosmos3SparseWindowMetadata,
) -> torch.Tensor:
    if metadata.world_size == 1:
        return tensor
    assert metadata.process_group is not None
    left_halo, right_halo = _all_gather_edges(
        tensor,
        edge_tokens=metadata.halo_tokens,
        rank=metadata.rank,
        world_size=metadata.world_size,
        process_group=metadata.process_group,
    )
    parts = []
    if left_halo is not None:
        parts.append(left_halo)
    parts.append(tensor)
    if right_halo is not None:
        parts.append(right_halo)
    return torch.cat(parts, dim=1)


def _sparse_window_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    joint_key: torch.Tensor,
    joint_value: torch.Tensor,
    metadata: Cosmos3SparseWindowMetadata,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    key_context = _build_context(key, metadata)
    value_context = _build_context(value, metadata)

    output = torch.empty_like(query)
    enable_gqa = query.shape[2] != key.shape[2]
    prefix_k = joint_key.transpose(1, 2)
    prefix_v = joint_value.transpose(1, 2)

    for block_idx, (q_start, q_end) in enumerate(metadata.query_ranges):
        key_len = metadata.key_lens_cpu[block_idx]
        indices = metadata.key_indices[block_idx, :key_len]
        block_k = key_context.index_select(1, indices).transpose(1, 2)
        block_v = value_context.index_select(1, indices).transpose(1, 2)
        attn_k = torch.cat([prefix_k, block_k], dim=2)
        attn_v = torch.cat([prefix_v, block_v], dim=2)
        block_q = query[:, q_start:q_end].transpose(1, 2)
        block_out = F.scaled_dot_product_attention(
            block_q,
            attn_k,
            attn_v,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
            enable_gqa=enable_gqa,
        )
        output[:, q_start:q_end] = block_out.transpose(1, 2)

    return output


def sparse_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    joint_key: torch.Tensor,
    joint_value: torch.Tensor,
    metadata: Cosmos3SparseWindowMetadata,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    return _sparse_window_attention_reference(
        query,
        key,
        value,
        joint_key,
        joint_value,
        metadata,
        softmax_scale=softmax_scale,
    )
