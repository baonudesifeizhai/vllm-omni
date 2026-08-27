# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host Weight Runtime integration for diffusion model loaders.

This mixin keeps HWR identity, restore, publication, and startup-recovery
policy out of the ordinary diffusers loader.  The host loader supplies the
canonical source-preparation and weight-loading hooks; this module owns only
the optional final-layout transaction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    TensorBinding,
    build_checkpoint_mmap_plan,
    has_online_quantization,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.diffusion.model_loader.host_weights.identity_adapter import FinalLayoutIdentityContext
    from vllm_omni.diffusion.model_loader.host_weights.source_identity import (
        NodeSourceDigestCache,
        PreparedWeightSource,
    )
    from vllm_omni.host_weight_runtime import RuntimeMode


class _WeightSource(Protocol):
    model_or_path: str
    subfolder: str | None
    revision: str | None
    prefix: str
    fall_back_to_pt: bool
    allow_patterns_overrides: list[str] | None


class _AllGatherCoordinator(Protocol):
    world_size: int
    rank_in_group: int
    ranks: Sequence[int]
    cpu_group: Any
    device_group: Any


class _HWRCommitError(RuntimeError):
    """A committed warm restore made the current model disposable."""


class HWRLoaderMixin:
    """Optional final-layout HWR behavior shared by diffusion loaders."""

    if TYPE_CHECKING:
        od_config: Any
        parallel_config: Any
        quant_config: Any
        host_weight_plan: HostWeightPlan | None
        _hwr_state: dict[str, object] | None
        _last_load_request: dict[str, object] | None
        _force_canonical_load: bool

        def _prepare_weights(
            self,
            model_name_or_path: Path | str,
            subfolder: str | None,
            revision: str | None,
            fall_back_to_pt: bool,
            allow_patterns_overrides: list[str] | None,
        ) -> tuple[Path | str, list[str], bool]: ...

        def _get_checkpoint_adapter(
            self,
            model: nn.Module,
            source: Any,
            use_safetensors: bool,
        ) -> Any: ...

        def load_model(self, **kwargs: object) -> nn.Module: ...

    @staticmethod
    def _identity_value(value: object) -> object:
        """Convert config objects into deterministic identity metadata."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dict):
            return {str(key): HWRLoaderMixin._identity_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [HWRLoaderMixin._identity_value(item) for item in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return HWRLoaderMixin._identity_value(to_dict())
            except TypeError:
                pass
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return HWRLoaderMixin._identity_value(dataclasses.asdict(cast(Any, value)))
        return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"

    @staticmethod
    def _identity_fingerprint(value: object) -> str:
        from vllm_omni.host_weight_runtime import CanonicalJson

        return hashlib.sha256(CanonicalJson.from_value(value).encoded).hexdigest()

    @staticmethod
    def _snapshot_final_layout_tensors(model: nn.Module, names: Iterable[str]) -> dict[str, tuple[int, str]]:
        snapshot: dict[str, tuple[int, str]] = {}
        for name in names:
            parent_path, _, leaf_name = name.rpartition(".")
            parent = model.get_submodule(parent_path)
            tensor = parent._parameters.get(leaf_name)
            if tensor is None:
                tensor = parent._buffers.get(leaf_name)
            if tensor is None or tensor.is_meta:
                raise RuntimeError(f"cannot snapshot missing or meta final-layout tensor {name!r}")
            value = tensor.detach()
            if value.device.type != "cpu":
                value = value.cpu()
            value = value.contiguous()
            digest = hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()
            snapshot[name] = (tensor.untyped_storage().data_ptr(), digest)
        return snapshot

    @classmethod
    def _assert_final_layout_tensors_unchanged(
        cls,
        model: nn.Module,
        snapshot: dict[str, tuple[int, str]],
    ) -> None:
        current = cls._snapshot_final_layout_tensors(model, snapshot)
        changed = [name for name in snapshot if current[name] != snapshot[name]]
        if changed:
            raise RuntimeError(
                "shared warm finalization changed restored final-layout tensor bytes or backing pointers: "
                f"{changed[:5]}"
            )

    def _hwr_eligibility_mode(
        self,
        model: nn.Module,
        modules: object,
        *,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
    ) -> RuntimeMode | None:
        """Return the enabled HWR mode only after all zero-interaction gates."""
        from vllm_omni.host_weight_runtime import RuntimeMode

        raw_mode = getattr(self.od_config, "host_weight_runtime_mode", "disabled")
        try:
            mode = RuntimeMode(raw_mode)
        except ValueError as exc:
            raise ValueError("host_weight_runtime_mode must be disabled, preferred, or required") from exc

        # These gates intentionally precede HWR imports, source preparation,
        # identity construction, and store creation.
        if mode is RuntimeMode.DISABLED or not dist_offload:
            return None

        parallel = self.parallel_config
        reasons: list[str] = []
        if load_format != "default":
            reasons.append("load_format must be 'default'")
        if bool(getattr(parallel, "use_hsdp", False)):
            reasons.append("HSDP layouts are not supported by the final-layout BF16 consumer")
        if self.quant_config is not None:
            reasons.append("quantized layouts require a representation-specific HWR producer")
        if getattr(self.od_config, "lora_path", None):
            reasons.append("adapted weights are not reusable base-model artifacts")

        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        if not dit_modules:
            reasons.append("no DiT modules were discovered")
        elif any(
            getattr(dit, "host_weight_restore_contract", None) is None
            or not callable(getattr(dit, "validate_restored_host_weights", None))
            for _, dit in dit_modules
        ):
            reasons.append("every owned DiT must declare the final-layout restore contract")

        if reasons:
            message = "; ".join(reasons)
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        return mode

    def _prepare_hwr_sources(
        self,
        model: nn.Module,
        modules: object,
        sources: Sequence[_WeightSource],
    ) -> tuple[PreparedWeightSource, ...]:
        from vllm_omni.diffusion.model_loader.host_weights import (
            ImplementationIdentity,
            PreparedWeightSource,
            WeightSourceKind,
        )

        dit_prefixes = tuple(f"{name}." for name in getattr(modules, "dit_names", ()))
        selected_sources = tuple(
            source
            for source in sources
            if not source.prefix or any(prefix.startswith(source.prefix) for prefix in dit_prefixes)
        )
        if not selected_sources:
            raise ValueError("final-layout HWR requires canonical weight sources covering every DiT")

        prepared: list[PreparedWeightSource] = []
        for source in selected_sources:
            resolved_root, weight_files, use_safetensors = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
            adapter = self._get_checkpoint_adapter(model, source, use_safetensors)
            adapter_identity = None
            if adapter is not None:
                adapter_name = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
                adapter_identity = ImplementationIdentity(
                    implementation_id=adapter_name,
                    version="1",
                    fingerprint=self._identity_fingerprint({"adapter": adapter_name}),
                )
            source_kind = (
                WeightSourceKind.LOCAL_PATH
                if os.path.isdir(os.fspath(source.model_or_path))
                else WeightSourceKind.HUGGING_FACE_HUB
            )
            prepared.append(
                PreparedWeightSource(
                    model_or_path=os.fspath(source.model_or_path),
                    subfolder=source.subfolder,
                    requested_revision=source.revision,
                    prefix=source.prefix,
                    resolved_root=Path(os.fspath(resolved_root)),
                    weight_files=tuple(Path(os.fspath(path)) for path in weight_files),
                    use_safetensors=use_safetensors,
                    checkpoint_adapter=adapter_identity,
                    source_kind=source_kind,
                )
            )
        return tuple(prepared)

    def _resolve_hwr_allgather_coordinator(self, use_allgather: bool) -> _AllGatherCoordinator | None:
        """Resolve the same DP-or-SP cohort used later by DLO AllGather."""
        parallel = self.parallel_config
        dp_size = int(getattr(parallel, "data_parallel_size", 1) or 1)
        sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
        expected_size = dp_size if dp_size > 1 else sp_size
        if not use_allgather or expected_size <= 1:
            return None
        if not torch.distributed.is_initialized():
            raise RuntimeError("BF16 HWR AllGather resolution requires initialized distributed groups")

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_world_size,
            get_dp_group,
            get_sp_group,
        )

        coordinator = get_dp_group() if get_data_parallel_world_size() > 1 else get_sp_group()
        return cast(_AllGatherCoordinator, coordinator)

    @staticmethod
    def _coordinate_hwr_group(
        coordinator: _AllGatherCoordinator | None,
        phase: str,
        local_record: dict[str, object],
        *,
        require_identity: bool = True,
    ) -> list[dict[str, object]]:
        """Gather one bounded startup record and validate cohort membership."""
        if coordinator is None:
            return [
                {
                    **local_record,
                    "phase": phase,
                    "group_size": 1,
                    "group_rank": 0,
                    "group_ranks": [0],
                }
            ]

        world_size = int(getattr(coordinator, "world_size"))
        group_rank = int(getattr(coordinator, "rank_in_group"))
        group_ranks = tuple(int(rank) for rank in getattr(coordinator, "ranks"))
        record = {
            **local_record,
            "phase": phase,
            "group_size": world_size,
            "group_rank": group_rank,
            "group_ranks": list(group_ranks),
        }
        gathered: list[object] = [None] * world_size
        group = getattr(coordinator, "cpu_group", None) or getattr(coordinator, "device_group")
        torch.distributed.all_gather_object(gathered, record, group=group)
        records = [candidate for candidate in gathered if isinstance(candidate, dict)]
        if len(records) != world_size:
            raise RuntimeError(f"BF16 HWR AllGather {phase} consensus returned malformed rank records")
        if (
            any(candidate.get("phase") != phase for candidate in records)
            or any(candidate.get("group_size") != world_size for candidate in records)
            or any(candidate.get("group_ranks") != list(group_ranks) for candidate in records)
            or {candidate.get("group_rank") for candidate in records} != set(range(world_size))
        ):
            raise RuntimeError(f"BF16 HWR AllGather process-group resolution differs during {phase}")
        if require_identity:
            identities = {candidate.get("identity_digest") for candidate in records}
            if len(identities) != 1 or None in identities:
                raise RuntimeError(f"BF16 HWR AllGather weight identity differs during {phase}")
        return records

    def _build_hwr_context(
        self,
        model: nn.Module,
        modules: object,
        *,
        load_format: str,
        sources: Sequence[_WeightSource],
        source_digest_cache: NodeSourceDigestCache | None = None,
    ) -> FinalLayoutIdentityContext:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        from vllm_omni.diffusion.model_loader.host_weights import (
            FINAL_LAYOUT_BF16_POLICY,
            FinalLayoutLoaderIdentity,
            FinalLayoutParallelIdentity,
            FinalLayoutRequest,
            ImplementationIdentity,
            build_final_layout_identity,
        )

        parallel = self.parallel_config
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
        try:
            tp_rank = int(get_tensor_model_parallel_rank()) if tp_size > 1 else 0
        except Exception:
            tp_rank = int(getattr(parallel, "tensor_parallel_rank", 0))
        sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
        loader_config = {
            "dtype": str(self.od_config.dtype),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_config": self._identity_value(getattr(self.od_config, "tf_model_config", None)),
            "load_format": load_format,
            "quantization": self._identity_value(self.quant_config),
        }
        contracts = [
            self._identity_value(getattr(dit, "host_weight_restore_contract")) for dit in getattr(modules, "dits", ())
        ]
        loader_identity = FinalLayoutLoaderIdentity(
            implementation=ImplementationIdentity(
                implementation_id="vllm-omni.diffusion.diffusers-loader",
                version="final-layout-v1",
                fingerprint=self._identity_fingerprint(
                    {
                        "loader": "diffusers-loader-final-layout-v1",
                        "pipeline": f"{type(model).__module__}.{type(model).__qualname__}",
                    }
                ),
            ),
            model_config_fingerprint=self._identity_fingerprint(loader_config),
            weight_transform_fingerprint=self._identity_fingerprint(
                {
                    "contracts": contracts,
                    "transform": "diffusion-final-layout-loader-transforms-v1",
                }
            ),
        )
        semantic_parallel = FinalLayoutParallelIdentity(
            tensor_parallel_size=tp_size,
            tensor_parallel_rank=tp_rank,
            sequence_parallel_size=sp_size,
            ulysses_degree=int(getattr(parallel, "ulysses_degree", 1)),
            ring_degree=int(getattr(parallel, "ring_degree", 1)),
            allgather_degree=int(getattr(parallel, "allgather_degree", 1)),
            ulysses_mode=str(getattr(parallel, "ulysses_mode", "strict")),
            pipeline_parallel_size=int(getattr(parallel, "pipeline_parallel_size", 1)),
            cfg_parallel_size=int(getattr(parallel, "cfg_parallel_size", 1)),
            use_hsdp=bool(getattr(parallel, "use_hsdp", False)),
            enable_expert_parallel=bool(getattr(parallel, "enable_expert_parallel", False)),
        )
        model_id = str(getattr(self.od_config, "model", "") or "")
        if not model_id:
            raise ValueError("final-layout HWR requires a canonical model identifier")
        request = FinalLayoutRequest(
            model_id=model_id,
            loader=loader_identity,
            parallel=semantic_parallel,
            load_format=load_format,
        )
        prepared_sources = self._prepare_hwr_sources(model, modules, sources)
        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        return build_final_layout_identity(
            model,
            dit_modules=dit_modules,
            prepared_sources=prepared_sources,
            request=request,
            policy=FINAL_LAYOUT_BF16_POLICY,
            source_digest_cache=source_digest_cache,
        )

    def _derivation_plan_digest(self, plan: HostWeightPlan) -> str:
        bindings = []
        for runtime_name, binding in sorted(plan.bindings.items()):
            transform = binding.transform
            bindings.append(
                {
                    "runtime_name": runtime_name,
                    "checkpoint_key": binding.checkpoint_key,
                    "file_path": str(Path(binding.file_path).resolve()),
                    "transform": (
                        None
                        if transform is None
                        else (
                            f"{getattr(transform, '__module__', type(transform).__module__)}."
                            f"{getattr(transform, '__qualname__', type(transform).__qualname__)}"
                        )
                    ),
                }
            )
        return self._identity_fingerprint(
            {
                "backing_kind": plan.backing_kind,
                "bindings": bindings,
                "planned_source_prefixes": sorted(plan.planned_source_prefixes),
            }
        )

    def _resolve_hwr_allgather_derivation(
        self,
        model: nn.Module,
        modules: object,
        *,
        coordinator: _AllGatherCoordinator,
        mode: RuntimeMode,
        load_format: str,
        sources: Sequence[_WeightSource],
    ) -> dict[str, object] | None:
        """Resolve BF16 AllGather as deferred derivation from a checkpoint.

        Native BF16 checkpoints already contain the representation consumed by
        DLO. HWR authorizes the pre-artifact derivation lifetime while the
        loader owns model-specific bindings and distributed consensus.
        """
        from vllm_omni.diffusion.model_loader.host_weights import NodeSourceDigestCache
        from vllm_omni.host_weight_runtime import (
            HostWeightDerivationLease,
            HostWeightLeaseCarrier,
            HostWeightRuntime,
            HostWeightRuntimeConfig,
            ProductionPolicy,
            ResolutionOutcome,
            RuntimeMode,
            SourceDerivationSpec,
            StorageDomainPolicy,
        )
        from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

        context = None
        plan = None
        plan_digest = None
        runtime = None
        fallback_reason = None
        preflight_error: Exception | None = None
        try:
            parallel = self.parallel_config
            dp_size = int(getattr(parallel, "data_parallel_size", 1) or 1)
            sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
            expected_size = dp_size if dp_size > 1 else sp_size
            if int(getattr(coordinator, "world_size")) != expected_size:
                raise RuntimeError(
                    "BF16 HWR AllGather group size differs from the configured DLO topology: "
                    f"resolved={coordinator.world_size}, expected={expected_size}"
                )

            expected_prefixes = frozenset(f"{name}." for name in getattr(modules, "dit_names", ()))
            available_prefixes = frozenset(getattr(source, "prefix", "") for source in sources)
            if not expected_prefixes <= available_prefixes:
                fallback_reason = "source derivation requires one dedicated source prefix per owned DiT"
            else:
                overlapping_prefixes = frozenset(
                    prefix
                    for prefix in available_prefixes
                    if any(dit_prefix.startswith(prefix) for dit_prefix in expected_prefixes)
                )
                if not overlapping_prefixes <= expected_prefixes:
                    fallback_reason = "source derivation rejects mixed DiT/component checkpoint sources"

            if fallback_reason is None:
                root_value = getattr(self.od_config, "host_weight_runtime_root", None)
                if not isinstance(root_value, str) or not root_value.strip():
                    raise ValueError("enabled Host Weight Runtime requires host_weight_runtime_root")
                root = Path(root_value).expanduser()
                root.mkdir(parents=True, exist_ok=True)
                runtime = HostWeightRuntime.from_config(
                    HostWeightRuntimeConfig(
                        mode=mode,
                        domain=StorageDomainPolicy(root=root, storage_class=detect_storage_class(root)),
                        production=ProductionPolicy(
                            allow_local_build=False,
                            allow_post_load_publish=False,
                        ),
                    )
                )
                context = self._build_hwr_context(
                    model,
                    modules,
                    load_format=load_format,
                    sources=sources,
                    source_digest_cache=NodeSourceDigestCache(
                        root,
                        timeout_seconds=runtime.config.wait.coordination_timeout_seconds,
                    ),
                )
                dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
                result = build_checkpoint_mmap_plan(
                    model,
                    dit_modules=dit_modules,
                    sources=sources,
                    model_path=str(getattr(self.od_config, "model", "")) or None,
                    tensor_parallel_size=int(getattr(parallel, "tensor_parallel_size", 1)),
                    use_hsdp=bool(getattr(parallel, "use_hsdp", False)),
                    online_quantization=has_online_quantization(model),
                )
                if result.plan is None:
                    fallback_reason = result.fallback_reason or "checkpoint mmap planning failed"
                else:
                    context.ensure_sources_unchanged()
                    plan = dataclasses.replace(
                        result.plan,
                        backing_kind="host_weight_runtime_derivation",
                    )
                    plan_digest = self._derivation_plan_digest(plan)
        except Exception as exc:
            preflight_error = exc

        records = self._coordinate_hwr_group(
            coordinator,
            "derivation_plan",
            {
                "identity_digest": context.identity.key if context is not None else None,
                "content_digest": context.identity.source.fingerprint if context is not None else None,
                "plan_digest": plan_digest,
                "status": "error" if preflight_error is not None else "fallback" if fallback_reason else "ready",
                "error_type": type(preflight_error).__name__ if preflight_error is not None else None,
            },
            require_identity=False,
        )
        failed_ranks = [record.get("group_rank") for record in records if record.get("status") == "error"]
        if failed_ranks:
            raise RuntimeError(
                f"BF16 HWR AllGather derivation-plan preflight failed on group ranks {failed_ranks}"
            ) from preflight_error

        fallback_ranks = [record.get("group_rank") for record in records if record.get("status") == "fallback"]
        if fallback_ranks:
            if len(fallback_ranks) != len(records):
                raise RuntimeError(
                    f"BF16 HWR AllGather derivation-plan eligibility differs across ranks: fallback={fallback_ranks}"
                )
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(
                    "required Host Weight Runtime derivation plan is ineligible: "
                    f"{fallback_reason or 'peer derivation-plan fallback'}"
                )
            logger.info(
                "BF16 HWR derivation plan is ineligible on group ranks %s; using canonical DLO: %s",
                fallback_ranks,
                fallback_reason or "peer derivation-plan fallback",
            )
            return None

        for field, label in (
            ("identity_digest", "source identity"),
            ("content_digest", "source content"),
            ("plan_digest", "checkpoint binding plan"),
        ):
            values = {record.get(field) for record in records}
            if len(values) != 1 or None in values:
                raise RuntimeError(f"BF16 HWR AllGather {label} differs across ranks")

        assert context is not None
        assert plan is not None
        assert plan_digest is not None
        assert runtime is not None
        resolution = runtime.resolve_derivation(
            SourceDerivationSpec(
                identity=context.identity,
                source_content_digest=context.identity.source.fingerprint,
                plan_digest=plan_digest,
                validate_source=context.ensure_sources_unchanged,
            )
        )
        derivation_lease = resolution.lease if isinstance(resolution.lease, HostWeightDerivationLease) else None
        try:
            resolution_records = self._coordinate_hwr_group(
                coordinator,
                "derivation_resolution",
                {
                    "identity_digest": context.identity.key,
                    "content_digest": context.identity.source.fingerprint,
                    "plan_digest": plan_digest,
                    "status": (
                        "ready"
                        if resolution.report.outcome is ResolutionOutcome.SOURCE_DERIVATION
                        else "fallback"
                        if resolution.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK
                        else "error"
                    ),
                    "outcome": resolution.report.outcome.value,
                },
            )
        except Exception:
            if derivation_lease is not None:
                derivation_lease.close()
            raise
        failed_ranks = [record.get("group_rank") for record in resolution_records if record.get("status") == "error"]
        fallback_ranks = [
            record.get("group_rank") for record in resolution_records if record.get("status") == "fallback"
        ]
        if failed_ranks or fallback_ranks:
            if derivation_lease is not None:
                derivation_lease.close()
            if fallback_ranks and len(fallback_ranks) == len(resolution_records):
                logger.info("BF16 HWR derivation fell back on all group ranks: %s", fallback_ranks)
                return None
            affected_ranks = failed_ranks or fallback_ranks
            raise RuntimeError(
                f"BF16 HWR AllGather derivation resolution differs or failed on group ranks {affected_ranks}"
            )
        outcomes = {record.get("outcome") for record in resolution_records}
        if outcomes != {ResolutionOutcome.SOURCE_DERIVATION.value} or derivation_lease is None:
            if derivation_lease is not None:
                derivation_lease.close()
            raise RuntimeError("BF16 HWR AllGather derivation resolution returned no derivation lease")
        plan = dataclasses.replace(plan, lease_carrier=HostWeightLeaseCarrier(derivation_lease))
        logger.info(
            "Resolved BF16 HWR checkpoint derivation: outcome=%s, identity=%s, tensors=%d",
            resolution.report.outcome.value,
            context.identity.key,
            len(plan.bindings),
        )
        return {
            "mode": mode,
            "context": context,
            "runtime": runtime,
            "outcome": resolution.report.outcome,
            "plan": plan,
        }

    def _resolve_hwr(
        self,
        model: nn.Module,
        modules: object,
        *,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
        sources: Sequence[_WeightSource],
    ) -> dict[str, object] | None:
        """Resolve an eligible final-layout BF16 HWR transaction."""
        from vllm_omni.diffusion.model_loader.host_weights import (
            FinalLayoutTensorRestorer,
            NodeSourceDigestCache,
        )
        from vllm_omni.host_weight_runtime import (
            HostWeightLease,
            HostWeightLeaseCarrier,
            HostWeightRuntime,
            HostWeightRuntimeConfig,
            ProductionPolicy,
            ResolutionOutcome,
            RuntimeMode,
            StorageDomainPolicy,
        )
        from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

        try:
            requested_mode = RuntimeMode(getattr(self.od_config, "host_weight_runtime_mode", "disabled"))
        except ValueError as exc:
            raise ValueError("host_weight_runtime_mode must be disabled, preferred, or required") from exc
        if requested_mode is RuntimeMode.DISABLED or not dist_offload:
            self._hwr_eligibility_mode(
                model,
                modules,
                dist_offload=dist_offload,
                use_allgather=use_allgather,
                load_format=load_format,
            )
            return None

        coordinator = self._resolve_hwr_allgather_coordinator(use_allgather)
        mode = None
        eligibility_error: Exception | None = None
        try:
            mode = self._hwr_eligibility_mode(
                model,
                modules,
                dist_offload=dist_offload,
                use_allgather=use_allgather,
                load_format=load_format,
            )
        except Exception as exc:
            eligibility_error = exc
        eligibility_records = self._coordinate_hwr_group(
            coordinator,
            "eligibility",
            {
                "identity_digest": "bf16-final-layout-eligibility-v1",
                "status": "error" if eligibility_error is not None else "ineligible" if mode is None else "eligible",
                "error_type": type(eligibility_error).__name__ if eligibility_error is not None else None,
            },
            require_identity=False,
        )
        resolution_scope = "BF16 HWR AllGather" if coordinator is not None else "Host Weight Runtime"
        failed_ranks = [record.get("group_rank") for record in eligibility_records if record.get("status") == "error"]
        if failed_ranks:
            if coordinator is None and eligibility_error is not None:
                raise eligibility_error
            raise RuntimeError(
                f"{resolution_scope} eligibility failed on group ranks {failed_ranks}"
            ) from eligibility_error
        ineligible_ranks = [
            record.get("group_rank") for record in eligibility_records if record.get("status") == "ineligible"
        ]
        if ineligible_ranks:
            if len(ineligible_ranks) == len(eligibility_records):
                return None
            raise RuntimeError(f"BF16 HWR AllGather eligibility differs across ranks: ineligible={ineligible_ranks}")
        assert isinstance(mode, RuntimeMode)
        if coordinator is not None:
            return self._resolve_hwr_allgather_derivation(
                model,
                modules,
                coordinator=coordinator,
                mode=mode,
                load_format=load_format,
                sources=sources,
            )

        expected_prefixes = frozenset(f"{name}." for name in getattr(modules, "dit_names", ()))
        available_prefixes = frozenset(getattr(source, "prefix", "") for source in sources)
        if not expected_prefixes <= available_prefixes:
            message = "final-layout HWR requires one dedicated source prefix per owned DiT"
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        overlapping_prefixes = frozenset(
            prefix
            for prefix in available_prefixes
            if any(dit_prefix.startswith(prefix) for dit_prefix in expected_prefixes)
        )
        if not overlapping_prefixes <= expected_prefixes:
            message = "final-layout HWR requires dedicated DiT sources and rejects mixed component sources"
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        root_value = getattr(self.od_config, "host_weight_runtime_root", None)
        if not isinstance(root_value, str) or not root_value.strip():
            raise ValueError("enabled Host Weight Runtime requires host_weight_runtime_root")
        root = Path(root_value).expanduser()
        runtime = HostWeightRuntime.from_config(
            HostWeightRuntimeConfig(
                mode=mode,
                domain=StorageDomainPolicy(root=root, storage_class=detect_storage_class(root)),
                production=ProductionPolicy(
                    allow_local_build=False,
                    allow_post_load_publish=True,
                ),
            )
        )
        source_digest_cache = NodeSourceDigestCache(
            root,
            timeout_seconds=runtime.config.wait.coordination_timeout_seconds,
        )
        context = self._build_hwr_context(
            model,
            modules,
            load_format=load_format,
            sources=sources,
            source_digest_cache=source_digest_cache,
        )
        resolution = runtime.resolve(context.identity)
        state: dict[str, object] = {
            "mode": mode,
            "context": context,
            "runtime": runtime,
            "outcome": resolution.report.outcome,
        }
        if resolution.report.outcome is ResolutionOutcome.FAILED:
            failure = next(
                (attempt.failure for attempt in reversed(resolution.report.attempts) if attempt.failure is not None),
                None,
            )
            detail = failure.message if failure is not None else "resolution failed without a typed detail"
            raise RuntimeError(f"Host Weight Runtime resolution failed: {detail}")
        if resolution.report.outcome is not ResolutionOutcome.LOCAL_HIT:
            return state

        lease = resolution.lease
        if not isinstance(lease, HostWeightLease):
            raise RuntimeError("Host Weight Runtime returned LOCAL_HIT without an artifact lease")
        restorer = FinalLayoutTensorRestorer(context)
        try:
            restore_plan = restorer.plan_restore(model, lease)
        except Exception:
            lease.close()
            if mode is RuntimeMode.REQUIRED:
                raise
            logger.warning("HWR warm restore planning failed; falling back to canonical loading", exc_info=True)
            state["outcome"] = ResolutionOutcome.CANONICAL_FALLBACK
            return state

        try:
            restore_plan.commit()
        except Exception as exc:
            lease.close()
            raise _HWRCommitError(
                "Host Weight Runtime restore commit failed; the partially restored model must be discarded"
            ) from exc

        try:
            carrier = HostWeightLeaseCarrier(lease)
            warm_snapshot = self._snapshot_final_layout_tensors(model, context.tensor_names)
        except Exception as exc:
            lease.close()
            raise _HWRCommitError(
                "Host Weight Runtime committed restore could not establish its startup ownership boundary"
            ) from exc
        source_metadata = context.identity.source.metadata.to_value()
        target_bindings_value = source_metadata.get("target_bindings") if isinstance(source_metadata, dict) else None
        target_bindings = target_bindings_value if isinstance(target_bindings_value, list) else []
        planned_prefix_values: set[str] = set()
        for binding in target_bindings:
            if isinstance(binding, dict):
                source_prefix = binding.get("source_prefix")
                if isinstance(source_prefix, str):
                    planned_prefix_values.add(source_prefix)
        planned_prefixes = frozenset(planned_prefix_values)
        if not planned_prefixes:
            lease.close()
            raise _HWRCommitError("Host Weight Runtime identity did not record owned canonical source prefixes")
        state["plan"] = HostWeightPlan(
            backing_kind="host_weight_runtime",
            bindings={name: TensorBinding(name, "") for name in context.tensor_names},
            planned_source_prefixes=planned_prefixes,
            lease_carrier=carrier,
        )
        state["warm_snapshot"] = warm_snapshot
        return state

    def load_fresh_canonical_model(self) -> nn.Module:
        """Reload a disposable startup model without HWR or checkpoint mmap."""
        if self._last_load_request is None:
            raise RuntimeError("cannot construct a fresh canonical model before the initial load")
        request = dict(self._last_load_request)
        self._force_canonical_load = True
        try:
            return self.load_model(**request)
        finally:
            self._force_canonical_load = False

    def _publish_hwr_after_load(self, model: nn.Module, modules: object, state: dict[str, object] | None) -> None:
        from vllm_omni.diffusion.model_loader.host_weights import FinalLayoutBF16Producer
        from vllm_omni.host_weight_runtime import HostWeightRuntime, PostLoadPublicationOutcome, ResolutionOutcome

        if state is None or state.get("outcome") is not ResolutionOutcome.CANONICAL_FALLBACK:
            return
        context = cast("FinalLayoutIdentityContext", state["context"])
        runtime = cast(HostWeightRuntime, state["runtime"])
        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        try:
            producer = FinalLayoutBF16Producer(context, model, dit_modules)
            report = runtime.publish_after_load(context.identity, producer=producer)
        except Exception:
            logger.warning("Host Weight Runtime post-load publication failed", exc_info=True)
            return
        if report.outcome is PostLoadPublicationOutcome.FAILED:
            logger.warning("Host Weight Runtime post-load publication failed: %s", report.failure)

    def _attach_offload_startup_state(self, model: nn.Module) -> None:
        """Publish generic offload startup state on the loaded pipeline."""
        if self.host_weight_plan is None:
            return
        from vllm_omni.diffusion.offloader.startup import OffloadStartupState, attach_offload_startup_state

        hwr_state = self._hwr_state
        allow_retry = False
        if hwr_state is not None and isinstance(hwr_state.get("plan"), HostWeightPlan):
            from vllm_omni.host_weight_runtime import RuntimeMode

            allow_retry = hwr_state.get("mode") is RuntimeMode.PREFERRED
        attach_offload_startup_state(
            model,
            OffloadStartupState(
                host_weight_plan=self.host_weight_plan,
                fresh_model_loader=self.load_fresh_canonical_model if allow_retry else None,
                allow_fresh_retry=allow_retry,
            ),
        )
