# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.ulysses_transport.base import (
    UlyssesBufferSlot,
    UlyssesTransport,
)


class NcclUlyssesTransport(UlyssesTransport):
    """The existing permute + ``all_to_all_single`` implementation."""

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        *,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
    ) -> None:
        self._process_group = process_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._use_sync = use_sync

    @property
    def backend(self) -> str:
        return "nccl"

    def scatter_heads(self, x: torch.Tensor, *, slot: UlyssesBufferSlot) -> torch.Tensor:
        del slot
        return SeqAllToAll4D.apply(
            self._process_group,
            x,
            self._scatter_idx,
            self._gather_idx,
            self._use_sync,
        )

    def gather_heads(self, x: torch.Tensor, *, slot: UlyssesBufferSlot = "o") -> torch.Tensor:
        del slot
        return SeqAllToAll4D.apply(
            self._process_group,
            x,
            self._gather_idx,
            self._scatter_idx,
            self._use_sync,
        )
