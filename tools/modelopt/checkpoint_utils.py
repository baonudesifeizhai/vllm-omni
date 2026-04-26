#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local helpers for assembling Diffusers-style checkpoint directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

WEIGHT_CANDIDATES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.pt",
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
)
WEIGHT_INDEX_CANDIDATES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class CheckpointAssemblyError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightSpec:
    kind: str  # "single" | "sharded"
    single_file: Path | None = None
    index_file: Path | None = None
    shard_files: tuple[Path, ...] = ()


def load_shard_files_from_index(index_file: Path, role: str = "weight") -> tuple[Path, ...]:
    try:
        payload = json.loads(index_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointAssemblyError(f"Failed to parse {role} index file {index_file}: {exc}") from exc

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointAssemblyError(f"Invalid or empty weight_map in {role} index file: {index_file}")

    shard_names = sorted({str(v) for v in weight_map.values()})
    shard_files = tuple(index_file.parent / shard_name for shard_name in shard_names)
    missing = [str(path) for path in shard_files if not path.is_file()]
    if missing:
        raise CheckpointAssemblyError(
            f"{role} index {index_file} references missing shard file(s): " + ", ".join(missing)
        )
    return shard_files


def resolve_weight_spec(path: Path, role: str = "weight") -> WeightSpec:
    if path.is_file():
        return WeightSpec(kind="single", single_file=path)

    if not path.is_dir():
        raise CheckpointAssemblyError(f"{role} path does not exist: {path}")

    for candidate_name in WEIGHT_CANDIDATES:
        candidate = path / candidate_name
        if candidate.is_file():
            return WeightSpec(kind="single", single_file=candidate)

    for index_name in WEIGHT_INDEX_CANDIDATES:
        index_file = path / index_name
        if index_file.is_file():
            return WeightSpec(
                kind="sharded",
                index_file=index_file,
                shard_files=load_shard_files_from_index(index_file, role=role),
            )

    raise CheckpointAssemblyError(
        f"Could not resolve {role} under {path}. Expected one of "
        f"{', '.join(WEIGHT_CANDIDATES + WEIGHT_INDEX_CANDIDATES)}."
    )


def canonical_weight_name(weight_file: Path) -> str:
    suffix = weight_file.suffix.lower()
    if suffix == ".safetensors":
        return "diffusion_pytorch_model.safetensors"
    if suffix == ".bin":
        return "diffusion_pytorch_model.bin"
    if suffix == ".pt":
        return "diffusion_pytorch_model.pt"
    return weight_file.name


def materialize_weight(weight: WeightSpec, dst_dir: Path, role: str = "weight") -> tuple[Path, ...]:
    if weight.kind == "single":
        assert weight.single_file is not None
        dst = dst_dir / canonical_weight_name(weight.single_file)
        shutil.copy2(weight.single_file, dst)
        return (dst,)

    if weight.kind == "sharded":
        assert weight.index_file is not None
        copied: list[Path] = []
        index_dst = dst_dir / weight.index_file.name
        shutil.copy2(weight.index_file, index_dst)
        copied.append(index_dst)
        for shard_file in weight.shard_files:
            shard_dst = dst_dir / shard_file.name
            shutil.copy2(shard_file, shard_dst)
            copied.append(shard_dst)
        return tuple(copied)

    raise CheckpointAssemblyError(f"Unknown {role} kind: {weight.kind}")


def copy_or_link_dir(src: Path, dst: Path, asset_mode: str) -> None:
    if asset_mode == "copy":
        shutil.copytree(src, dst)
        return
    if asset_mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
        return
    raise CheckpointAssemblyError(f"Unknown asset mode: {asset_mode}")


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise CheckpointAssemblyError(
                f"Output directory already exists: {output_dir}. Use overwrite to remove and recreate it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
