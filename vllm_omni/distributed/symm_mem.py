# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = [
    "all_to_all_single",
    "all_to_all_vdev",
    "all_to_all_vdev_2d",
    "all_to_all_vdev_2d_offset",
    "empty",
    "get_backend",
    "get_mem_pool",
    "get_workspace",
    "group_name",
    "is_nvshmem_available",
    "multimem_all_gather_out",
    "multimem_all_reduce_",
    "one_shot_all_reduce",
    "one_shot_all_reduce_out",
    "pipelined_produce_and_all2all",
    "rendezvous",
    "set_backend",
    "two_shot_all_reduce_",
    "use_mem_pool",
    "use_symm_mem_all2all",
    "use_symm_mem_allreduce",
    "use_symm_mem_kernel",
]

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_USE_SYMM_MEM_ENV = "VLLM_OMNI_USE_SYMM_MEM"
_USE_SYMM_MEM_ALL2ALL_ENV = "VLLM_OMNI_USE_SYMM_MEM_ALL2ALL"
_USE_SYMM_MEM_ALLREDUCE_ENV = "VLLM_OMNI_USE_SYMM_MEM_ALLREDUCE"
_USE_SYMM_MEM_KERNEL_ENV = "VLLM_OMNI_USE_SYMM_MEM_KERNEL"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_ENV_VALUES


def use_symm_mem_all2all() -> bool:
    return _env_flag(_USE_SYMM_MEM_ENV) and _env_flag(_USE_SYMM_MEM_ALL2ALL_ENV, default=True)


def use_symm_mem_allreduce() -> bool:
    return _env_flag(_USE_SYMM_MEM_ENV) and _env_flag(_USE_SYMM_MEM_ALLREDUCE_ENV, default=True)


def use_symm_mem_kernel() -> bool:
    return use_symm_mem_all2all() and _env_flag(_USE_SYMM_MEM_KERNEL_ENV)


def _symm_mem_module() -> Any:
    import torch.distributed._symmetric_memory as symm_mem

    return symm_mem


def group_name(group: Any | None) -> str:
    if group is None:
        import torch.distributed as dist

        group = dist.group.WORLD
    return group.group_name


def empty(
    *size: int | tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return _symm_mem_module().empty(*size, dtype=dtype, device=device)


def rendezvous(tensor: torch.Tensor, group: Any | None) -> Any:
    return _symm_mem_module().rendezvous(tensor, group_name(group))


def is_nvshmem_available() -> bool:
    return _symm_mem_module().is_nvshmem_available()


def set_backend(name: str) -> None:
    _symm_mem_module().set_backend(name)


def get_backend(device: torch.device | str) -> str | None:
    return _symm_mem_module().get_backend(device)


def get_mem_pool(device: torch.device | str) -> Any:
    return _symm_mem_module().get_mem_pool(device)


def get_workspace(group: Any | None, min_size: int) -> Any:
    from torch.distributed import _symmetric_memory

    return _symmetric_memory.get_symm_mem_workspace(group_name(group), min_size)


@contextmanager
def use_mem_pool(device: torch.device | str):
    import torch

    with torch.cuda.use_mem_pool(get_mem_pool(device)):
        yield


def one_shot_all_reduce(input: torch.Tensor, group: Any | None, *, reduce_op: str = "sum") -> torch.Tensor:
    import torch

    return torch.ops.symm_mem.one_shot_all_reduce(input, reduce_op, group_name(group))


def one_shot_all_reduce_out(
    input: torch.Tensor,
    output: torch.Tensor,
    group: Any | None,
    *,
    reduce_op: str = "sum",
) -> torch.Tensor:
    import torch

    return torch.ops.symm_mem.one_shot_all_reduce_out(input, reduce_op, group_name(group), out=output)


def multimem_all_reduce_(input: torch.Tensor, group: Any | None, *, reduce_op: str = "sum") -> torch.Tensor:
    import torch

    return torch.ops.symm_mem.multimem_all_reduce_(input, reduce_op, group_name(group))


def multimem_all_gather_out(input: torch.Tensor, output: torch.Tensor, group: Any | None) -> torch.Tensor:
    import torch

    return torch.ops.symm_mem.multimem_all_gather_out(input, group_name(group), out=output)


def two_shot_all_reduce_(input: torch.Tensor, group: Any | None, *, reduce_op: str = "sum") -> torch.Tensor:
    import torch

    return torch.ops.symm_mem.two_shot_all_reduce_(input, reduce_op, group_name(group))


def all_to_all_vdev(
    input: torch.Tensor,
    output: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group: Any | None,
) -> None:
    import torch

    torch.ops.symm_mem.all_to_all_vdev(input, output, in_splits, out_splits_offsets, group_name(group))


def all_to_all_vdev_2d(
    input: torch.Tensor,
    output: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group: Any | None,
    *,
    major_align: int | None = None,
) -> None:
    import torch

    if major_align is None:
        torch.ops.symm_mem.all_to_all_vdev_2d(input, output, in_splits, out_splits_offsets, group_name(group))
    else:
        torch.ops.symm_mem.all_to_all_vdev_2d(
            input,
            output,
            in_splits,
            out_splits_offsets,
            group_name(group),
            major_align,
        )


def all_to_all_vdev_2d_offset(
    input: torch.Tensor,
    output: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group: Any | None,
) -> None:
    import torch

    torch.ops.symm_mem.all_to_all_vdev_2d_offset(
        input,
        output,
        in_splits_offsets,
        out_splits_offsets,
        group_name(group),
    )


def pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, torch.Tensor], None],
    output: torch.Tensor,
    group: Any | None,
    *,
    out_chunk_dim: int = 0,
) -> None:
    from torch.distributed import _symmetric_memory

    name = group_name(group)
    if out_chunk_dim == 0:
        _symmetric_memory._pipelined_produce_and_all2all(chunk_producer, output, name)
    else:
        _symmetric_memory._pipelined_produce_and_all2all(
            chunk_producer,
            output,
            name,
            out_chunk_dim=out_chunk_dim,
        )


def all_to_all_single(
    input: torch.Tensor,
    output: torch.Tensor,
    group: Any | None,
    *,
    out_chunk_dim: int = 0,
) -> None:
    def chunk_producer(dst_rank: int, buf: torch.Tensor) -> None:
        buf.copy_(input[dst_rank : dst_rank + 1])

    pipelined_produce_and_all2all(chunk_producer, output, group, out_chunk_dim=out_chunk_dim)
