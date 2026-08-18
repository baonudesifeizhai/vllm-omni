# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch

UlyssesBufferSlot = Literal["q", "k", "v", "o"]


class UlyssesTransport(ABC):
    """Transport boundary for Ulysses sequence/head resharding.

    Attention compute backends only consume the tensors returned here.  Keeping
    communication behind this interface lets NCCL, symmetric-memory Copy
    Engine, and future transports share the same attention integration.
    """

    @property
    @abstractmethod
    def backend(self) -> str:
        """Effective backend name."""

    @abstractmethod
    def scatter_heads(self, x: torch.Tensor, *, slot: UlyssesBufferSlot) -> torch.Tensor:
        """[B, S_local, H, D] -> [B, S_global, H_local, D]."""

    @abstractmethod
    def gather_heads(self, x: torch.Tensor, *, slot: UlyssesBufferSlot = "o") -> torch.Tensor:
        """[B, S_global, H_local, D] -> [B, S_local, H, D]."""

    def scatter_kv(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange K/V together when the backend has a fused implementation."""
        return self.scatter_heads(key, slot="k"), self.scatter_heads(value, slot="v")
