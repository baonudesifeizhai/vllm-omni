# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact pre-load identity for final-layout BF16 diffusion artifacts."""

from __future__ import annotations

import hashlib
import inspect
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import regex as re
import torch
from torch import nn

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    ComponentIdentity,
    ProducerIdentity,
    RuntimeWeightLayout,
    WeightArtifactIdentity,
    WeightRepresentation,
    WeightSourceIdentity,
)
from vllm_omni.host_weight_runtime.identity import canonical_json

from .producers.final_layout_bf16 import (
    FINAL_LAYOUT_BF16_MANIFEST_SCHEMA,
    FINAL_LAYOUT_BF16_PRODUCER_ID,
    FINAL_LAYOUT_BF16_REPRESENTATION,
    FINAL_LAYOUT_BF16_RESTORER_SCHEMA,
    FINAL_LAYOUT_BF16_VERSION,
    FinalLayoutBF16Producer,
)
from .restorer import FinalLayoutBF16Restorer
from .tensor_layout import (
    FINAL_LAYOUT_BF16_MODEL_CONTRACT,
    FinalLayoutContractError,
    collect_final_layout_targets,
    tensor_contract_sha256,
    validate_final_layout_model_contract,
)

_HASH_CHUNK_BYTES = 8 * 1024**2
_IMMUTABLE_REVISION_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class FinalLayoutBF16Request:
    """Loader-selected semantics for one supported final-layout representation.

    DP coordinates and transfer policy are intentionally absent because they
    do not change replicated final-layout bytes.
    """

    model_id: str
    runtime_dtype: torch.dtype = torch.bfloat16
    load_format: str = "default"
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    sequence_parallel_size: int = 1
    sequence_parallel_backend: str = "none"
    loader_metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)
    sequence_parallel_metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)
    layout_metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)
    adaptation: AdaptationIdentity = field(default_factory=AdaptationIdentity)
    pipeline_parallel_size: int = 1
    cfg_parallel_size: int = 1
    use_hsdp: bool = False
    enable_expert_parallel: bool = False
    quantization: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("final-layout BF16 identity requires a canonical model ID or path")
        if self.runtime_dtype is not torch.bfloat16:
            raise ValueError(f"final-layout BF16 artifacts require torch.bfloat16, got {self.runtime_dtype}")
        if self.load_format != "default":
            raise ValueError(f"final-layout BF16 artifacts require load_format='default', got {self.load_format!r}")

        tp_size = _require_positive_int("tensor_parallel_size", self.tensor_parallel_size)
        if isinstance(self.tensor_parallel_rank, bool) or not isinstance(self.tensor_parallel_rank, int):
            raise ValueError("tensor_parallel_rank must be an integer")
        if not 0 <= self.tensor_parallel_rank < tp_size:
            raise ValueError("tensor_parallel_rank must be within tensor_parallel_size")
        sp_size = _require_positive_int("sequence_parallel_size", self.sequence_parallel_size)
        if not isinstance(self.sequence_parallel_backend, str) or not self.sequence_parallel_backend:
            raise ValueError("sequence_parallel_backend must not be empty")
        if sp_size == 1 and self.sequence_parallel_backend != "none":
            raise ValueError("sequence_parallel_backend must be 'none' when sequence_parallel_size is one")
        if sp_size > 1 and self.sequence_parallel_backend == "none":
            raise ValueError("sequence_parallel_backend must describe SP semantics when sequence_parallel_size > 1")

        for name, metadata in (
            ("loader_metadata", self.loader_metadata),
            ("sequence_parallel_metadata", self.sequence_parallel_metadata),
            ("layout_metadata", self.layout_metadata),
        ):
            if not isinstance(metadata, CanonicalJson):
                raise ValueError(f"{name} must use CanonicalJson")
        if not isinstance(self.adaptation, AdaptationIdentity):
            raise ValueError("adaptation must use AdaptationIdentity")
        if self.adaptation.kind != "base" or self.adaptation.fingerprint is not None:
            raise ValueError("merged or static LoRA requires a representation-specific producer")

        if _require_positive_int("pipeline_parallel_size", self.pipeline_parallel_size) != 1:
            raise ValueError("pipeline-parallel ownership is not supported by the final-layout BF16 producer")
        if _require_positive_int("cfg_parallel_size", self.cfg_parallel_size) != 1:
            raise ValueError("CFG-parallel ownership is not supported by the final-layout BF16 producer")
        if not isinstance(self.use_hsdp, bool) or not isinstance(self.enable_expert_parallel, bool):
            raise ValueError("HSDP and expert-parallel flags must be booleans")
        if self.use_hsdp:
            raise ValueError("HSDP/DTensor layouts are not supported by the final-layout BF16 producer")
        if self.enable_expert_parallel:
            raise ValueError("expert-parallel ownership is not supported by the final-layout BF16 producer")
        if self.quantization is not None:
            raise ValueError("quantized layouts require a representation-specific producer")


@dataclass(frozen=True)
class PreparedWeightSource:
    """Canonical source files selected by the ordinary loader."""

    model_or_path: str
    subfolder: str | None
    requested_revision: str | None
    prefix: str
    resolved_root: Path
    weight_files: tuple[Path, ...]
    use_safetensors: bool
    checkpoint_adapter_type: type | None = None

    def __post_init__(self) -> None:
        if not self.model_or_path or not isinstance(self.prefix, str):
            raise ValueError("prepared source model and prefix must be valid strings")
        if not isinstance(self.resolved_root, Path):
            raise ValueError("prepared source root must use pathlib.Path")
        if not isinstance(self.weight_files, tuple) or not self.weight_files:
            raise ValueError("prepared source must contain an immutable tuple of weight files")
        if any(not isinstance(path, Path) for path in self.weight_files):
            raise ValueError("prepared source weight files must use pathlib.Path")
        if not isinstance(self.use_safetensors, bool):
            raise ValueError("prepared source safetensors mode must be a boolean")


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    relative_name: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    symlink_target: str | None
    content_id: str

    def semantic_dict(self) -> dict[str, object]:
        return {
            "relative_name": self.relative_name,
            "size": self.size,
            "content_id": self.content_id,
        }

    def unchanged(self) -> bool:
        try:
            current = self.path.stat()
            symlink_target = os.readlink(self.path) if self.path.is_symlink() else None
        except OSError:
            return False
        return (
            current.st_size == self.size
            and current.st_dev == self.device
            and current.st_ino == self.inode
            and current.st_mtime_ns == self.mtime_ns
            and current.st_ctime_ns == self.ctime_ns
            and symlink_target == self.symlink_target
        )


@dataclass(frozen=True)
class _SourceSnapshot:
    semantic: CanonicalJson
    files: tuple[_FileSnapshot, ...]

    def unchanged(self) -> bool:
        return all(file.unchanged() for file in self.files)


@dataclass(frozen=True)
class FinalLayoutBF16IdentityContext:
    """Exact identity plus one-load source and tensor ownership guards."""

    identity: WeightArtifactIdentity
    tensor_names: frozenset[str]
    dit_names: tuple[str, ...]
    source_snapshots: tuple[_SourceSnapshot, ...]
    pipeline_type: type[nn.Module]
    dit_types: tuple[type[nn.Module], ...]

    def sources_unchanged(self) -> bool:
        return all(snapshot.unchanged() for snapshot in self.source_snapshots)

    def ensure_sources_unchanged(self) -> None:
        if not self.sources_unchanged():
            raise FinalLayoutContractError(
                "source_changed",
                "canonical source changed after final-layout identity resolution",
            )


def _type_identity(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _blob_content_id(path: Path) -> str | None:
    if not path.is_symlink():
        return None
    try:
        target = os.readlink(path)
    except OSError:
        return None
    blob_name = Path(target).name
    if _IMMUTABLE_REVISION_RE.fullmatch(blob_name) is None:
        return None
    return f"immutable-blob:{blob_name.lower()}"


def _snapshot_revision(path: Path) -> str | None:
    parts = path.absolute().parts
    for index, part in enumerate(parts[:-1]):
        if part != "snapshots":
            continue
        candidate = parts[index + 1]
        if _IMMUTABLE_REVISION_RE.fullmatch(candidate) is not None:
            return candidate.lower()
    return None


def _logical_model_id(value: str) -> str:
    candidate = Path(value).expanduser()
    try:
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        pass
    return value


def _snapshot_source(source: PreparedWeightSource) -> tuple[_SourceSnapshot, str]:
    root = source.resolved_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"canonical source root is not a directory: {root}")
    immutable_revision = _snapshot_revision(root)
    if immutable_revision is None and source.requested_revision is not None:
        requested = source.requested_revision.strip()
        if _IMMUTABLE_REVISION_RE.fullmatch(requested) is not None:
            immutable_revision = requested.lower()

    files: list[_FileSnapshot] = []
    for path in sorted(source.weight_files, key=lambda item: str(item)):
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.absolute()
        try:
            relative_name = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"weight file {candidate} is outside its canonical source root {root}") from exc
        try:
            current = candidate.stat()
        except OSError as exc:
            raise ValueError(f"cannot stat canonical weight file {candidate}") from exc
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"canonical weight source is not a regular file: {candidate}")
        symlink_target = os.readlink(candidate) if candidate.is_symlink() else None
        content_id = _blob_content_id(candidate)
        if content_id is None:
            content_id = f"sha256:{_sha256_file(candidate)}"
        files.append(
            _FileSnapshot(
                path=candidate,
                relative_name=relative_name,
                size=current.st_size,
                device=current.st_dev,
                inode=current.st_ino,
                mtime_ns=current.st_mtime_ns,
                ctime_ns=current.st_ctime_ns,
                symlink_target=symlink_target,
                content_id=content_id,
            )
        )

    file_semantics = [file.semantic_dict() for file in files]
    file_fingerprint = hashlib.sha256(canonical_json(file_semantics)).hexdigest()
    revision = immutable_revision or f"content-{file_fingerprint}"
    semantic = CanonicalJson.from_value(
        {
            "checkpoint_adapter": (
                _type_identity(source.checkpoint_adapter_type) if source.checkpoint_adapter_type is not None else None
            ),
            "files": file_semantics,
            "model_or_path": _logical_model_id(source.model_or_path),
            "prefix": source.prefix,
            "resolved_revision": revision,
            "subfolder": source.subfolder,
            "use_safetensors": source.use_safetensors,
        }
    )
    return _SourceSnapshot(semantic=semantic, files=tuple(files)), revision


def _implementation_fingerprint(objects: Sequence[object]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for value in objects:
        identity = (
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        )
        if identity in seen:
            continue
        seen.add(identity)
        try:
            source = inspect.getsource(cast(Any, value))
        except (OSError, TypeError) as exc:
            raise FinalLayoutContractError(
                "implementation_fingerprint_unavailable",
                f"cannot fingerprint final-layout implementation {identity}",
            ) from exc
        digest.update(identity.encode("utf-8"))
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def build_final_layout_bf16_identity(
    pipeline: nn.Module,
    *,
    dit_modules: Sequence[tuple[str, nn.Module]],
    loader_type: type,
    prepared_sources: Sequence[PreparedWeightSource],
    request: FinalLayoutBF16Request,
) -> FinalLayoutBF16IdentityContext:
    """Build an exact pre-load identity or reject an unsupported layout."""
    if not isinstance(request, FinalLayoutBF16Request):
        raise TypeError("final-layout BF16 identity requires FinalLayoutBF16Request")
    validate_final_layout_model_contract(dit_modules)
    structural_targets = collect_final_layout_targets(
        pipeline,
        dit_modules,
        require_materialized=False,
    )
    target_names = frozenset(target.name for target in structural_targets)
    relevant_sources = tuple(
        source for source in prepared_sources if any(name.startswith(source.prefix) for name in target_names)
    )
    if not relevant_sources:
        raise ValueError("no canonical weight source covers the discovered DiT tensors")

    snapshots_and_revisions = tuple(_snapshot_source(source) for source in relevant_sources)
    source_snapshots = tuple(item[0] for item in snapshots_and_revisions)
    source_revisions = tuple(item[1] for item in snapshots_and_revisions)
    source_semantics = [snapshot.semantic.to_value() for snapshot in source_snapshots]
    source_fingerprint = hashlib.sha256(canonical_json(source_semantics)).hexdigest()
    unique_revisions = tuple(dict.fromkeys(source_revisions))
    aggregate_revision = (
        unique_revisions[0]
        if len(unique_revisions) == 1
        else f"aggregate-{hashlib.sha256(canonical_json(unique_revisions)).hexdigest()}"
    )

    dit_names = tuple(name for name, _ in dit_modules)
    implementation_objects: list[object] = [
        build_final_layout_bf16_identity,
        FinalLayoutBF16Producer,
        FinalLayoutBF16Restorer,
        collect_final_layout_targets,
        tensor_contract_sha256,
        validate_final_layout_model_contract,
        loader_type,
        type(pipeline),
    ]
    implementation_objects.extend(type(module) for _, module in dit_modules)
    implementation_objects.extend(
        source.checkpoint_adapter_type for source in relevant_sources if source.checkpoint_adapter_type is not None
    )

    identity = WeightArtifactIdentity(
        schema_version=1,
        source=WeightSourceIdentity(
            model_id=_logical_model_id(request.model_id),
            revision=aggregate_revision,
            fingerprint=source_fingerprint,
            metadata=CanonicalJson.from_value({"sources": source_semantics}),
        ),
        component=ComponentIdentity(
            name="diffusion-dit",
            ownership="complete-final-layout-tensors",
            metadata=CanonicalJson.from_value({"component_names": list(dit_names)}),
        ),
        representation=WeightRepresentation(
            name=FINAL_LAYOUT_BF16_REPRESENTATION,
            dtype=str(torch.bfloat16),
            metadata=CanonicalJson.from_value(
                {
                    "load_format": request.load_format,
                    "loader": request.loader_metadata.to_value(),
                    "mixed_precision_policy": "bf16-with-preserved-fp32",
                }
            ),
        ),
        layout=RuntimeWeightLayout(
            name="diffusion-final-module-layout-v1",
            tensor_parallel_size=request.tensor_parallel_size,
            tensor_parallel_rank=request.tensor_parallel_rank,
            sequence_parallel_size=request.sequence_parallel_size,
            sequence_parallel_backend=request.sequence_parallel_backend,
            metadata=CanonicalJson.from_value(
                {
                    "consumer_layout": request.layout_metadata.to_value(),
                    "model_contract": FINAL_LAYOUT_BF16_MODEL_CONTRACT,
                    "sequence_parallel": request.sequence_parallel_metadata.to_value(),
                    "tensor_contract_sha256": tensor_contract_sha256(structural_targets),
                }
            ),
        ),
        adaptation=request.adaptation,
        producer=ProducerIdentity(
            producer_id=FINAL_LAYOUT_BF16_PRODUCER_ID,
            version=FINAL_LAYOUT_BF16_VERSION,
            implementation_fingerprint=_implementation_fingerprint(implementation_objects),
            manifest_schema=FINAL_LAYOUT_BF16_MANIFEST_SCHEMA,
            restorer_schema=FINAL_LAYOUT_BF16_RESTORER_SCHEMA,
        ),
    )
    return FinalLayoutBF16IdentityContext(
        identity=identity,
        tensor_names=target_names,
        dit_names=dit_names,
        source_snapshots=source_snapshots,
        pipeline_type=type(pipeline),
        dit_types=tuple(type(module) for _, module in dit_modules),
    )


__all__ = [
    "FinalLayoutBF16IdentityContext",
    "FinalLayoutBF16Request",
    "PreparedWeightSource",
    "build_final_layout_bf16_identity",
]
