# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_symm_mem_ulysses_ops() -> None:
    source_root = Path(__file__).resolve().parents[2] / "csrc"
    load(
        name="vllm_omni_symm_mem_ulysses",
        sources=[
            str(source_root / "symm_mem_ulysses.cpp"),
        ],
        extra_cflags=["-O3"],
        with_cuda=True,
        is_python_module=False,
        verbose=os.environ.get("VLLM_OMNI_EXT_VERBOSE", "0") == "1",
    )
