# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch.distributed as dist

from vllm_omni.diffusion.distributed.ulysses_transport.base import UlyssesTransport
from vllm_omni.diffusion.distributed.ulysses_transport.nccl import NcclUlyssesTransport


def build_ulysses_transport(
    fast_ulysses: bool,
    process_group: dist.ProcessGroup,
    *,
    scatter_idx: int,
    gather_idx: int,
    use_sync: bool,
) -> UlyssesTransport:
    """Install the optional fast path on top of ordinary Ulysses SP.

    This mirrors vLLM AsyncTP's relationship with sequence parallelism: the
    boolean enables an optimization, while the underlying parallel algorithm
    and its default communication path stay unchanged.
    """
    if fast_ulysses:
        world_size = dist.get_world_size(process_group)
        if world_size <= 1:
            raise ValueError("fast_ulysses requires an Ulysses process group larger than 1")
        from vllm_omni.diffusion.distributed.ulysses_transport.symm_mem import (
            SymmetricMemoryUlyssesTransport,
        )

        return SymmetricMemoryUlyssesTransport(
            process_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )
    return NcclUlyssesTransport(
        process_group,
        scatter_idx=scatter_idx,
        gather_idx=gather_idx,
        use_sync=use_sync,
    )
