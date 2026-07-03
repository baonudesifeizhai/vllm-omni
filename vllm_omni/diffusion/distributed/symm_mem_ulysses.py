# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.symm_mem_ulysses_ops import (
    load_symm_mem_ulysses_ops,
)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
USE_SYMM_MEM_ALL2ALL_ENV = "VLLM_OMNI_USE_SYMM_MEM_ALL2ALL"
USE_SYMM_MEM_ASYNC_ULYSSES_ENV = "VLLM_OMNI_USE_SYMM_MEM_ASYNC_ULYSSES"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def is_symm_mem_async_ulysses_enabled() -> bool:
    return _env_flag(USE_SYMM_MEM_ALL2ALL_ENV) and _env_flag(USE_SYMM_MEM_ASYNC_ULYSSES_ENV)


def _group_name(group: dist.ProcessGroup) -> str | None:
    return getattr(group, "group_name", None)


@dataclass(slots=True)
class _PreparedAllToAll:
    name: str
    recv: torch.Tensor
    send_handle: Any


class SymmMemUlyssesTransport:
    """Fused CUDA symmetric-memory transport for strict Ulysses input A2A."""

    _side_streams: dict[int, torch.cuda.Stream] = {}

    @staticmethod
    def load_ops() -> None:
        load_symm_mem_ulysses_ops()

    def __init__(self, group: dist.ProcessGroup, device: torch.device) -> None:
        self.load_ops()
        self.group = group
        self.group_boxed = group.boxed()
        self.group_name = group.group_name
        self.world_size = dist.get_world_size(group)
        self.device = device
        self._pending: list[_PreparedAllToAll] = []

        device_index = device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        if device_index not in self._side_streams:
            self._side_streams[device_index] = torch.cuda.Stream(device=device)
        self.side_stream = self._side_streams[device_index]

    @staticmethod
    def is_available(group: dist.ProcessGroup | None = None) -> bool:
        return (
            is_symm_mem_async_ulysses_enabled()
            and torch.cuda.is_available()
            and group is not None
            and _group_name(group) is not None
            and dist.get_world_size(group) > 1
        )

    @torch.compiler.disable
    def prepare(self, tensor: torch.Tensor, *, name: str) -> _PreparedAllToAll:
        with torch.profiler.record_function("symm_mem_ulysses_prepare"):
            recv, send_handle = torch.ops.vllm_omni.symm_mem_ulysses_prepare(
                tensor,
                self.group_boxed,
            )
        return _PreparedAllToAll(name=name, recv=recv, send_handle=send_handle)

    @torch.compiler.disable
    def push(self, work: _PreparedAllToAll) -> None:
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.side_stream):
            self.side_stream.wait_event(ready)
            with torch.profiler.record_function("symm_mem_ulysses_push"):
                torch.ops.vllm_omni.symm_mem_ulysses_push(
                    work.send_handle,
                    self.group_boxed,
                )
        self._pending.append(work)

    @torch.compiler.disable
    def join(self) -> tuple[torch.Tensor, ...]:
        with torch.cuda.stream(self.side_stream):
            with torch.profiler.record_function("symm_mem_ulysses_join"):
                for _ in self._pending:
                    torch.ops.vllm_omni.symm_mem_ulysses_barrier(self.group_boxed)
            done = torch.cuda.Event()
            done.record(self.side_stream)
        torch.cuda.current_stream(self.device).wait_event(done)

        outputs = tuple(work.recv for work in self._pending)
        self._pending.clear()
        return outputs

    @torch.compiler.disable
    def post_unscatter_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.profiler.record_function("symm_mem_ulysses_post_unscatter"):
            return torch.ops.vllm_omni.symm_mem_ulysses_post_unscatter_qkv(
                query,
                key,
                value,
            )


_TRANSPORTS: dict[tuple[str, int], SymmMemUlyssesTransport] = {}


def get_symm_mem_ulysses_transport(
    group: dist.ProcessGroup,
    device: torch.device,
) -> SymmMemUlyssesTransport:
    group_name = group.group_name
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key = (group_name, device_index)
    transport = _TRANSPORTS.get(key)
    if transport is None:
        transport = SymmMemUlyssesTransport(group, device)
        _TRANSPORTS[key] = transport
    return transport
