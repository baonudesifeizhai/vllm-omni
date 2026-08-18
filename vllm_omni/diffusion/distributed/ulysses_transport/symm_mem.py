# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch Symmetric Memory + Copy Engine Ulysses transport.

The payload path follows the central Fast-Ulysses idea: rendezvous a peer-
addressable symmetric window, then express the sequence/head relayout as
pitched ``cudaMemcpy2DAsync`` copies directly into each destination window.
The payload therefore runs on GPU Copy Engines rather than in an SM-resident
copy kernel.

This first implementation deliberately supports the uniform strict Ulysses
layout.  UAA keeps using the NCCL transport until uneven split plans are added.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm_omni.platforms import current_omni_platform

UlyssesBufferSlot = Literal["q", "k", "v", "o"]

_BUILD_LOCK = threading.Lock()
_KERNEL_LOADED = False


@torch.compiler.disable
def _ensure_kernel_loaded() -> None:
    """JIT-build and load the Fast-Ulysses operators once per process."""
    global _KERNEL_LOADED
    if _KERNEL_LOADED:
        return

    with _BUILD_LOCK:
        if _KERNEL_LOADED:
            return

        source = Path(__file__).with_name("csrc") / "symm_mem_ce.cu"
        if not source.is_file():
            raise RuntimeError(f"Fast Ulysses CUDA source is missing from the installed vLLM-Omni package: {source}")

        from torch.utils.cpp_extension import load

        load(
            name="vllm_omni_symm_mem_ulysses",
            sources=[str(source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            is_python_module=False,
            verbose=False,
        )
        _KERNEL_LOADED = True


@dataclass
class _SymmetricWindow:
    tensor: torch.Tensor
    handle: object
    capacity: int


class SymmetricMemoryUlyssesTransport:
    """Uniform single-node Ulysses all-to-all over symmetric memory.

    Windows are high-water-mark allocations keyed by Q/K/V/O role and dtype.
    Distinct roles are required because Q, K, and V remain live together until
    attention executes.  Replaced windows remain owned so a previously
    captured CUDA Graph never observes an unmapped address.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        *,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
    ) -> None:
        if scatter_idx != 2 or gather_idx != 1:
            raise ValueError(
                "The symm_mem Ulysses transport currently requires "
                f"scatter_idx=2 and gather_idx=1, got {scatter_idx} and {gather_idx}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The symm_mem Ulysses transport requires CUDA.")

        _ensure_kernel_loaded()

        self._process_group = process_group
        self._group_name = process_group.group_name
        self._world_size = dist.get_world_size(process_group)
        self._use_sync = use_sync
        self._windows: dict[tuple[UlyssesBufferSlot, torch.dtype, torch.device], _SymmetricWindow] = {}
        self._retired_windows: list[_SymmetricWindow] = []
        self._window_lock = threading.Lock()

        if self._world_size > 8:
            raise ValueError(
                "The symm_mem Copy-Engine transport is single-node and currently "
                f"supports at most 8 ranks, got {self._world_size}."
            )

        backend = symm_mem.get_backend(torch.device("cuda", torch.accelerator.current_device_index()))
        if backend not in (None, "CUDA"):
            raise RuntimeError(
                "The symm_mem Copy-Engine transport requires PyTorch's CUDA "
                f"symmetric-memory backend, but the process selected {backend!r}."
            )

        # Required by torch 2.11 and harmless on newer releases.  It binds the
        # process-group store used by rendezvous without changing the backend.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            symm_mem.enable_symm_mem_for_group(self._group_name)

    def _window(
        self,
        *,
        slot: UlyssesBufferSlot,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        numel = 1
        for dim in shape:
            numel *= dim
        key = (slot, dtype, device)

        window = self._windows.get(key)
        if window is None or window.capacity < numel:
            window = self._grow_window(key=key, numel=numel, dtype=dtype, device=device)

        return window.tensor.narrow(0, 0, numel).view(shape)

    @torch.compiler.disable
    def _grow_window(
        self,
        *,
        key: tuple[UlyssesBufferSlot, torch.dtype, torch.device],
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _SymmetricWindow:
        """Collectively grow a window outside Dynamo's captured region."""
        with self._window_lock:
            window = self._windows.get(key)
            if window is None or window.capacity < numel:
                tensor = symm_mem.empty(numel, dtype=dtype, device=device)
                handle = symm_mem.rendezvous(tensor, self._process_group)
                torch.ops.vllm_omni_symm_mem.init_ulysses_window_(tensor, self._group_name)
                replacement = _SymmetricWindow(tensor=tensor, handle=handle, capacity=numel)
                if window is not None:
                    self._retired_windows.append(window)
                self._windows[key] = replacement
                window = replacement
        return window

    def scatter_heads(self, x: torch.Tensor, *, slot: UlyssesBufferSlot) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Ulysses scatter expects a 4-D tensor, got shape {tuple(x.shape)}.")
        if not x.is_cuda:
            raise ValueError("The symm_mem Ulysses transport only accepts CUDA tensors.")
        if not x.is_contiguous():
            x = x.contiguous()

        batch, seq_local, heads, head_dim = x.shape
        if heads <= 0:
            raise ValueError("The symm_mem Ulysses transport requires at least one head.")
        if heads % self._world_size == 0:
            heads_local = heads // self._world_size
        elif self._world_size % heads == 0:
            # MQA, or GQA with fewer KV heads than Ulysses ranks: each KV
            # head is copied to the consecutive ranks whose Q-head shard uses
            # it. The CE plan performs the replication directly into peers.
            heads_local = 1
        else:
            raise ValueError(
                "The symm_mem Ulysses transport requires nested head/rank "
                "partitions: heads must divide world_size or world_size must "
                f"divide heads, got heads={heads}, world_size={self._world_size}."
            )
        output_shape = (
            batch,
            seq_local * self._world_size,
            heads_local,
            head_dim,
        )
        output = self._window(slot=slot, shape=output_shape, dtype=x.dtype, device=x.device)
        torch.ops.vllm_omni_symm_mem.ce_ulysses_a2a(x, output, 0, self._group_name)
        if self._use_sync:
            current_omni_platform.synchronize()
        return output

    def scatter_kv(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("The symm_mem fused K/V exchange requires 4-D tensors.")
        if key.shape != value.shape:
            raise ValueError(
                "The symm_mem fused K/V exchange requires matching K/V shapes, "
                f"got key={tuple(key.shape)} and value={tuple(value.shape)}."
            )
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("The symm_mem fused K/V exchange requires matching K/V dtype and device.")
        if not key.is_cuda:
            raise ValueError("The symm_mem Ulysses transport only accepts CUDA tensors.")
        if not key.is_contiguous():
            key = key.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()

        batch, seq_local, heads, head_dim = key.shape
        if heads % self._world_size == 0:
            heads_local = heads // self._world_size
        elif self._world_size % heads == 0:
            heads_local = 1
        else:
            raise ValueError(
                "The symm_mem fused K/V exchange requires nested head/rank "
                f"partitions, got heads={heads}, world_size={self._world_size}."
            )
        output_shape = (batch, seq_local * self._world_size, heads_local, head_dim)
        key_out = self._window(slot="k", shape=output_shape, dtype=key.dtype, device=key.device)
        value_out = self._window(slot="v", shape=output_shape, dtype=value.dtype, device=value.device)
        torch.ops.vllm_omni_symm_mem.ce_ulysses_scatter_kv_(
            key,
            value,
            key_out,
            value_out,
            self._group_name,
        )
        if self._use_sync:
            current_omni_platform.synchronize()
        return key_out, value_out

    def gather_heads(self, x: torch.Tensor, *, slot: UlyssesBufferSlot = "o") -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Ulysses gather expects a 4-D tensor, got shape {tuple(x.shape)}.")
        if not x.is_cuda:
            raise ValueError("The symm_mem Ulysses transport only accepts CUDA tensors.")
        if not x.is_contiguous():
            x = x.contiguous()

        batch, seq_global, heads_local, head_dim = x.shape
        if seq_global % self._world_size != 0:
            raise ValueError(
                "The strict symm_mem Ulysses transport requires the global sequence "
                f"length to divide world_size: seq_global={seq_global}, "
                f"world_size={self._world_size}."
            )
        output_shape = (
            batch,
            seq_global // self._world_size,
            heads_local * self._world_size,
            head_dim,
        )
        output = self._window(slot=slot, shape=output_shape, dtype=x.dtype, device=x.device)
        torch.ops.vllm_omni_symm_mem.ce_ulysses_a2a(x, output, 1, self._group_name)
        if self._use_sync:
            current_omni_platform.synchronize()
        return output
