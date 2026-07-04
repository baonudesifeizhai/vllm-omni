# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
#  from https://github.com/feifeibear/long-context-attention/blob/main/yunchang/comm/all_to_all.py
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from vllm_omni.diffusion.distributed.symm_mem_ulysses_ops import (
    load_symm_mem_ulysses_ops,
)
from vllm_omni.platforms import current_omni_platform

__all__ = [
    "all_to_all_4D",
    "all_to_all_4D_qkv",
    "all_to_all_5D",
    "SeqAllToAll4D",
    "SeqAllToAll5D",
    "RingComm",
]

try:
    from torch.distributed import _symmetric_memory as torch_symm_mem
except ImportError:
    torch_symm_mem = None

_USE_SYMM_MEM_ALL2ALL_ENV = "VLLM_OMNI_USE_SYMM_MEM_ALL2ALL"
_USE_SYMM_MEM_PACKED_QKV_ALL2ALL_ENV = "VLLM_OMNI_USE_SYMM_MEM_PACKED_QKV_ALL2ALL"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _use_symm_mem_all2all() -> bool:
    value = os.environ.get(_USE_SYMM_MEM_ALL2ALL_ENV, "")
    return value.strip().lower() in _TRUE_ENV_VALUES


def _use_symm_mem_packed_qkv_all2all() -> bool:
    value = os.environ.get(_USE_SYMM_MEM_PACKED_QKV_ALL2ALL_ENV, "")
    return _use_symm_mem_all2all() and value.strip().lower() in _TRUE_ENV_VALUES


def _get_group_name(group: dist.ProcessGroup | None):
    pg = group if group is not None else dist.group.WORLD
    return getattr(pg, "group_name", None)


def _can_use_symm_mem_all2all(
    input: Tensor,
    group: dist.ProcessGroup | None,
    world_size: int,
) -> bool:
    return (
        _use_symm_mem_all2all()
        and world_size > 1
        and input.is_cuda
        and torch_symm_mem is not None
        and hasattr(torch_symm_mem, "_pipelined_produce_and_all2all")
        and _get_group_name(group) is not None
    )


@torch.compiler.disable
def _symm_mem_opaque_all_to_all_4d_qkv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup | None,
    use_sync: bool,
    seq_world_size: int,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if (
        not _use_symm_mem_packed_qkv_all2all()
        or not _can_use_symm_mem_all2all(query, group, seq_world_size)
        or group is None
        or not (key.is_cuda and value.is_cuda)
        or not (query.device == key.device == value.device)
        or not (query.dtype == key.dtype == value.dtype)
        or scatter_idx != 2
        or gather_idx != 1
        or query.dim() != 4
        or key.dim() != 4
        or value.dim() != 4
    ):
        return None

    bs, shard_seqlen, q_heads, head_size = query.shape
    if key.shape[:2] != (bs, shard_seqlen) or value.shape[:2] != (bs, shard_seqlen):
        return None
    if key.shape[3] != head_size or value.shape[3] != head_size:
        return None
    if q_heads % seq_world_size != 0 or key.shape[2] % seq_world_size != 0 or value.shape[2] % seq_world_size != 0:
        return None

    load_symm_mem_ulysses_ops()
    output = torch.ops.vllm_omni.symm_mem_ulysses_exchange_qkv(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        group.boxed(),
    )
    if use_sync:
        current_omni_platform.synchronize()
    return output


@torch.compiler.disable
def _symm_mem_all_to_all_4d(
    input: Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup | None,
    use_sync: bool,
    seq_world_size: int,
) -> Tensor | None:
    if not _can_use_symm_mem_all2all(input, group, seq_world_size):
        return None

    group_name = _get_group_name(group)
    assert group_name is not None
    assert torch_symm_mem is not None

    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = input.shape
        if hc % seq_world_size != 0:
            return None
        shard_hc = hc // seq_world_size
        seqlen = shard_seqlen * seq_world_size
        output_t = input.new_empty((seq_world_size, shard_seqlen, bs, shard_hc, hs))

        def chunk_producer(dst_rank: int, buf: Tensor) -> None:
            head_start = dst_rank * shard_hc
            head_end = head_start + shard_hc
            buf.copy_(input[:, :, head_start:head_end, :].transpose(0, 1))

        torch_symm_mem._pipelined_produce_and_all2all(
            chunk_producer,
            output_t,
            group_name,
            out_chunk_dim=0,
        )
        if use_sync:
            current_omni_platform.synchronize()
        return output_t.reshape(seqlen, bs, shard_hc, hs).transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

    if scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = input.shape
        if seqlen % seq_world_size != 0:
            return None
        shard_seqlen = seqlen // seq_world_size
        hc = shard_hc * seq_world_size
        output_t = input.new_empty((seq_world_size, shard_hc, shard_seqlen, bs, hs))

        def chunk_producer(dst_rank: int, buf: Tensor) -> None:
            seq_start = dst_rank * shard_seqlen
            seq_end = seq_start + shard_seqlen
            buf.copy_(input[:, seq_start:seq_end, :, :].permute(2, 1, 0, 3))

        torch_symm_mem._pipelined_produce_and_all2all(
            chunk_producer,
            output_t,
            group_name,
            out_chunk_dim=0,
        )
        if use_sync:
            current_omni_platform.synchronize()
        return output_t.reshape(hc, shard_seqlen, bs, hs).transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

    return None


@torch.compiler.disable
def _symm_mem_all_to_all_4d_qkv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scatter_idx: int,
    gather_idx: int,
    group: dist.ProcessGroup | None,
    use_sync: bool,
    seq_world_size: int,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if (
        not _use_symm_mem_packed_qkv_all2all()
        or not _can_use_symm_mem_all2all(query, group, seq_world_size)
        or not (key.is_cuda and value.is_cuda)
        or not (query.device == key.device == value.device)
        or not (query.dtype == key.dtype == value.dtype)
        or scatter_idx != 2
        or gather_idx != 1
        or query.dim() != 4
        or key.dim() != 4
        or value.dim() != 4
    ):
        return None

    bs, shard_seqlen, q_heads, head_size = query.shape
    if key.shape[:2] != (bs, shard_seqlen) or value.shape[:2] != (bs, shard_seqlen):
        return None
    if key.shape[3] != head_size or value.shape[3] != head_size:
        return None

    k_heads = int(key.shape[2])
    v_heads = int(value.shape[2])
    if q_heads % seq_world_size != 0 or k_heads % seq_world_size != 0 or v_heads % seq_world_size != 0:
        return None

    q_shard_heads = q_heads // seq_world_size
    k_shard_heads = k_heads // seq_world_size
    v_shard_heads = v_heads // seq_world_size
    packed_shard_heads = q_shard_heads + k_shard_heads + v_shard_heads
    seqlen = shard_seqlen * seq_world_size
    output_t = query.new_empty((seq_world_size, shard_seqlen, bs, packed_shard_heads, head_size))

    group_name = _get_group_name(group)
    assert group_name is not None
    assert torch_symm_mem is not None

    def chunk_producer(dst_rank: int, buf: Tensor) -> None:
        if buf.dim() == 5:
            assert buf.shape[0] == 1
            buf = buf[0]
        q_start = dst_rank * q_shard_heads
        q_end = q_start + q_shard_heads
        k_start = dst_rank * k_shard_heads
        k_end = k_start + k_shard_heads
        v_start = dst_rank * v_shard_heads
        v_end = v_start + v_shard_heads

        offset = 0
        next_offset = offset + q_shard_heads
        buf[:, :, offset:next_offset, :].copy_(query[:, :, q_start:q_end, :].transpose(0, 1))
        offset = next_offset
        next_offset = offset + k_shard_heads
        buf[:, :, offset:next_offset, :].copy_(key[:, :, k_start:k_end, :].transpose(0, 1))
        offset = next_offset
        next_offset = offset + v_shard_heads
        buf[:, :, offset:next_offset, :].copy_(value[:, :, v_start:v_end, :].transpose(0, 1))

    torch_symm_mem._pipelined_produce_and_all2all(
        chunk_producer,
        output_t,
        group_name,
        out_chunk_dim=0,
    )
    if use_sync:
        current_omni_platform.synchronize()

    packed = output_t.reshape(seqlen, bs, packed_shard_heads, head_size).transpose(0, 1)
    q_out, k_out, v_out = packed.split(
        [q_shard_heads, k_shard_heads, v_shard_heads],
        dim=2,
    )
    return q_out.contiguous(), k_out.contiguous(), v_out.contiguous()


def all_to_all_4D_qkv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Packed strict-Ulysses Q/K/V pre-attention all-to-all.

    Returns None when the env-gated symmetric-memory packed path is unavailable,
    so callers can fall back to the regular per-tensor all-to-all path.
    """
    seq_world_size = dist.get_world_size(group)
    opaque_qkv = _symm_mem_opaque_all_to_all_4d_qkv(
        query,
        key,
        value,
        scatter_idx,
        gather_idx,
        group,
        use_sync,
        seq_world_size,
    )
    if opaque_qkv is not None:
        return opaque_qkv

    return _symm_mem_all_to_all_4d_qkv(
        query,
        key,
        value,
        scatter_idx,
        gather_idx,
        group,
        use_sync,
        seq_world_size,
    )


def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)
    symm_mem_output = _symm_mem_all_to_all_4d(
        input,
        scatter_idx,
        gather_idx,
        group,
        use_sync,
        seq_world_size,
    )
    if symm_mem_output is not None:
        return symm_mem_output

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)->
        #  (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        return all_to_all_4D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


def all_to_all_5D(
    input: torch.tensor, scatter_idx: int = 3, gather_idx: int = 1, group=None, use_sync: bool = False
) -> torch.tensor:
    """
    all-to-all for QKV
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group (torch.distributed.ProcessGroup): torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, 3, hc, hs)
    """
    assert input.dim() == 5, f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 3 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, 3, hc, hs) output: (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape

        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, 3, hc, hs) -reshape-> (bs, seq_len/P, 3, P, hc/P, hs) -transpose(0,3)->
        #  (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs).transpose(0, 3).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, 3, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, 3, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)

        # (seq_len, 3, bs, hc/P, hs) -trans-> (bs, seq_len, 3, hc/P, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()

        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
    elif scatter_idx == 1 and gather_idx == 3:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, 3, hc/P, hs) -reshape-> (bs, P, seq_len/P, 3, hc/P, hs) -transpose(0, 4)->
        # (hc/P, P, seqlen/P, 3, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                current_omni_platform.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)

        # (hc, seqlen/N, bs, hs) -transpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 3).contiguous()

        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        use_sync: bool = False,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return all_to_all_5D(input, scatter_idx, gather_idx, group=group, use_sync=use_sync)


class RingComm:
    """Ring communication utility for Ring Attention P2P communication."""

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: torch.Tensor | None = None) -> torch.Tensor:
        # Ensure to_send is contiguous for P2P
        if not to_send.is_contiguous():
            to_send = to_send.contiguous()

        if recv_tensor is None:
            # Create a contiguous buffer for receiving
            res = torch.empty_like(to_send, memory_format=torch.contiguous_format)
            # print(f"send_recv: empty_like {to_send.shape}")
        else:
            res = recv_tensor
            if not res.is_contiguous():
                res = res.contiguous()

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
