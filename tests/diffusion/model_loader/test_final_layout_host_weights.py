# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for final-layout BF16 Host Weight Runtime artifacts."""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.model_loader.host_weights import (
    FinalLayoutBF16IdentityContext,
    FinalLayoutBF16Producer,
    FinalLayoutBF16Request,
    FinalLayoutBF16Restorer,
    PreparedWeightSource,
    build_final_layout_bf16_identity,
)
from vllm_omni.diffusion.model_loader.host_weights.tensor_layout import (
    FINAL_LAYOUT_BF16_MODEL_CONTRACT,
    FinalLayoutContractError,
    collect_final_layout_targets,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    CoordinationScope,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    LookupPhase,
    PostLoadPublicationOutcome,
    ProductionPolicy,
    ProductionSourceMode,
    ResolutionOutcome,
    RuntimeMode,
    StorageDomainPolicy,
    StorageScope,
    WaitPolicy,
)
from vllm_omni.host_weight_runtime.filesystem import FilesystemHostWeightStore, detect_storage_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TinyDiT(nn.Module):
    host_weight_restore_contract = FINAL_LAYOUT_BF16_MODEL_CONTRACT

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        target = torch.device(device)
        self.restore_validations = 0
        self.proj = nn.Linear(3, 2, dtype=torch.bfloat16, device=target)
        self.fp32_gain = nn.Parameter(torch.empty(1, dtype=torch.float32, device=target))
        self.register_buffer("scale", torch.empty(1, dtype=torch.float32, device=target))
        self.register_buffer(
            "derived",
            torch.tensor([7.0], dtype=torch.float32),
            persistent=False,
        )

    def validate_restored_host_weights(self) -> None:
        self.restore_validations += 1
        assert not self.proj.weight.is_meta
        assert self.proj.weight.dtype is torch.bfloat16
        assert self.fp32_gain.dtype is torch.float32
        assert self.scale.dtype is torch.float32


class _TinyPipeline(nn.Module):
    def __init__(self, *, dit_device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.transformer = _TinyDiT(device=dit_device)
        self.text_encoder = nn.Linear(2, 2, dtype=torch.bfloat16)
        self.vae = nn.Module()
        self.vae.register_buffer("gain", torch.tensor([9.0], dtype=torch.float32))


class _AlternateTinyPipeline(_TinyPipeline):
    pass


class _TestLoader:
    pass


def _prepared_source(
    tmp_path: Path,
    *,
    content: bytes = b"canonical-source-for-identity",
    requested_revision: str | None = None,
    resolved_revision: str | None = None,
    directory: str = "canonical",
) -> PreparedWeightSource:
    source_root = tmp_path / directory
    if resolved_revision is not None:
        source_root = source_root / "snapshots" / resolved_revision
    source_root.mkdir(parents=True, exist_ok=True)
    weight_file = source_root / "model.safetensors"
    if not weight_file.exists():
        weight_file.write_bytes(content)
    elif weight_file.read_bytes() != content:
        weight_file.write_bytes(content)
    return PreparedWeightSource(
        model_or_path="test-org/tiny-diffusion",
        subfolder=None,
        requested_revision=requested_revision,
        prefix="transformer.",
        resolved_root=source_root,
        weight_files=(weight_file,),
        use_safetensors=True,
    )


def _request(**changes: object) -> FinalLayoutBF16Request:
    request = FinalLayoutBF16Request(
        model_id="test-org/tiny-diffusion",
        loader_metadata=CanonicalJson.from_value({"attention": "eager"}),
    )
    return dataclasses.replace(request, **changes)


def _identity(
    model: _TinyPipeline,
    source: PreparedWeightSource,
    *,
    request: FinalLayoutBF16Request | None = None,
) -> FinalLayoutBF16IdentityContext:
    return build_final_layout_bf16_identity(
        model,
        dit_modules=(("transformer", model.transformer),),
        loader_type=_TestLoader,
        prepared_sources=(source,),
        request=request or _request(),
    )


def _runtime(root: Path) -> HostWeightRuntime:
    return HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=StorageDomainPolicy(
                root=root,
                scope=StorageScope.NODE,
                domain_id="node",
                storage_class=detect_storage_class(root.parent),
            ),
            production=ProductionPolicy(
                allow_local_build=False,
                allow_post_load_publish=True,
            ),
            wait=WaitPolicy(coordination_timeout_seconds=5.0),
        )
    )


def _dit_modules(model: _TinyPipeline) -> tuple[tuple[str, nn.Module], ...]:
    return (("transformer", model.transformer),)


def _fill_final_weights(model: _TinyPipeline) -> None:
    with torch.no_grad():
        model.transformer.proj.weight.copy_(torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3))
        model.transformer.proj.bias.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))
        model.transformer.fp32_gain.copy_(torch.tensor([6.5]))
        model.transformer.scale.copy_(torch.tensor([2.5]))
        model.text_encoder.weight.fill_(11)
        model.text_encoder.bias.fill_(12)


def _pointer_snapshot(model: nn.Module) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        pointer = 0 if tensor.is_meta else tensor.untyped_storage().data_ptr()
        snapshot[name] = (id(tensor), pointer, tensor.device.type)
    return snapshot


def test_preload_identity_is_stable_for_cpu_and_meta_skeletons(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    cpu_model = _TinyPipeline()
    meta_model = _TinyPipeline(dit_device="meta")

    cpu = _identity(cpu_model, source)
    meta = _identity(meta_model, source)

    assert cpu.identity == meta.identity
    assert cpu.tensor_names == {
        "transformer.fp32_gain",
        "transformer.proj.bias",
        "transformer.proj.weight",
        "transformer.scale",
    }
    assert "transformer.derived" not in cpu.tensor_names
    assert all("text_encoder" not in name and "vae" not in name for name in cpu.tensor_names)

    request_fields = {field.name for field in dataclasses.fields(FinalLayoutBF16Request)}
    assert not any("data_parallel" in name for name in request_fields)
    identity_bytes = cpu.identity.canonical_bytes
    assert b"data_parallel" not in identity_bytes
    assert b"allgather" not in identity_bytes
    assert b"registration" not in identity_bytes


def test_identity_uses_resolved_revision_and_exact_semantics(tmp_path: Path) -> None:
    model = _TinyPipeline()
    commit = "0123456789abcdef0123456789abcdef01234567"
    source = _prepared_source(tmp_path, requested_revision="main", resolved_revision=commit)
    by_alias = _identity(model, source)
    by_commit = _identity(model, dataclasses.replace(source, requested_revision=commit))

    assert by_alias.identity == by_commit.identity

    base = _request()
    variants = (
        dataclasses.replace(base, loader_metadata=CanonicalJson.from_value({"attention": "cudnn"})),
        dataclasses.replace(base, tensor_parallel_size=2, tensor_parallel_rank=0),
        dataclasses.replace(base, tensor_parallel_size=2, tensor_parallel_rank=1),
        dataclasses.replace(
            base,
            sequence_parallel_size=2,
            sequence_parallel_backend="ulysses",
            sequence_parallel_metadata=CanonicalJson.from_value({"ulysses_degree": 2, "mode": "strict"}),
        ),
        dataclasses.replace(base, layout_metadata=CanonicalJson.from_value({"packing": "v2"})),
    )
    base_identity = _identity(model, source, request=base).identity
    assert all(_identity(model, source, request=variant).identity != base_identity for variant in variants)

    changed_source = _prepared_source(
        tmp_path,
        directory="changed",
        content=b"different-canonical-source",
    )
    assert _identity(model, changed_source).identity != base_identity

    changed_abi = dataclasses.replace(
        base_identity,
        producer=dataclasses.replace(base_identity.producer, restorer_schema="future-restorer-v2"),
    )
    assert changed_abi.key != base_identity.key


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"runtime_dtype": torch.float16}, "require torch.bfloat16"),
        ({"load_format": "diffusers"}, "load_format='default'"),
        ({"pipeline_parallel_size": 2}, "pipeline-parallel"),
        ({"cfg_parallel_size": 2}, "CFG-parallel"),
        ({"use_hsdp": True}, "HSDP"),
        ({"enable_expert_parallel": True}, "expert-parallel"),
        ({"quantization": "fp8"}, "quantized layouts"),
        (
            {"adaptation": AdaptationIdentity(kind="merged-lora", fingerprint="adapter-sha256")},
            "LoRA",
        ),
        ({"sequence_parallel_size": 2}, "describe SP semantics"),
    ],
)
def test_request_fails_closed_for_unsupported_semantics(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**changes)


def test_tensor_ownership_is_complete_mixed_precision_and_alias_free(tmp_path: Path) -> None:
    model = _TinyPipeline()
    records = collect_final_layout_targets(model, _dit_modules(model), require_materialized=True)
    by_name = {record.name: record for record in records}

    assert set(by_name) == {
        "transformer.fp32_gain",
        "transformer.proj.bias",
        "transformer.proj.weight",
        "transformer.scale",
    }
    assert by_name["transformer.proj.weight"].tensor.dtype is torch.bfloat16
    assert by_name["transformer.fp32_gain"].tensor.dtype is torch.float32
    assert by_name["transformer.scale"].role == "persistent_buffer"

    model.transformer.register_parameter("alias", model.transformer.proj.weight)
    with pytest.raises(FinalLayoutContractError, match="aliases tensor object"):
        _identity(model, _prepared_source(tmp_path))


def test_identity_requires_explicit_model_restore_contract(tmp_path: Path) -> None:
    model = _TinyPipeline()
    model.transformer.host_weight_restore_contract = "unsupported-contract"

    with pytest.raises(FinalLayoutContractError, match="does not declare"):
        _identity(model, _prepared_source(tmp_path))


def test_publication_is_warm_only_and_restore_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _prepared_source(tmp_path)
    cold_model = _TinyPipeline()
    _fill_final_weights(cold_model)
    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "store")
    producer = FinalLayoutBF16Producer(context, cold_model, _dit_modules(cold_model), max_shard_bytes=16)
    assert producer.spec.lookup_phase is LookupPhase.POST_LOAD_ONLY
    assert producer.spec.source_mode is ProductionSourceMode.FINALIZED_MODEL
    assert producer.spec.coordination_scope is CoordinationScope.SINGLE_PROCESS

    miss = runtime.resolve(context.identity)
    assert miss.report.outcome is ResolutionOutcome.CANONICAL_FALLBACK
    cold_pointers = _pointer_snapshot(cold_model)
    cold_values = {name: tensor.detach().clone() for name, tensor in cold_model.named_parameters()}

    publication = runtime.publish_after_load(context.identity, producer=producer)

    assert publication.outcome is PostLoadPublicationOutcome.PUBLISHED
    assert publication.failure is None
    assert _pointer_snapshot(cold_model) == cold_pointers
    assert all(torch.equal(dict(cold_model.named_parameters())[name], value) for name, value in cold_values.items())
    assert cold_model.transformer.restore_validations == 1

    def unexpected_produce(_writer: object) -> object:
        raise AssertionError("already-present publication must not invoke the producer")

    monkeypatch.setattr(producer, "produce", unexpected_produce)
    already_present = runtime.publish_after_load(context.identity, producer=producer)
    assert already_present.outcome is PostLoadPublicationOutcome.ALREADY_PRESENT

    warm_model = _TinyPipeline(dit_device="meta")
    warm_context = _identity(warm_model, source)
    assert warm_context.identity == context.identity
    hit = runtime.resolve(warm_context.identity)
    assert hit.report.outcome is ResolutionOutcome.LOCAL_HIT
    assert hit.lease is not None

    before_plan = _pointer_snapshot(warm_model)
    text_weight = warm_model.text_encoder.weight
    vae_gain = warm_model.vae.gain
    plan = FinalLayoutBF16Restorer(warm_context).plan_restore(warm_model, hit.lease)
    assert _pointer_snapshot(warm_model) == before_plan
    assert warm_model.transformer.restore_validations == 0

    plan.commit()
    with pytest.raises(RuntimeError, match="already committed"):
        plan.commit()

    expected = {
        "transformer.proj.weight": torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3),
        "transformer.proj.bias": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "transformer.fp32_gain": torch.tensor([6.5]),
        "transformer.scale": torch.tensor([2.5]),
    }
    restored = dict(warm_model.named_parameters()) | dict(warm_model.named_buffers())
    assert all(torch.equal(restored[name], value) for name, value in expected.items())
    assert all(not restored[name].is_meta for name in warm_context.tensor_names)
    assert warm_model.transformer.restore_validations == 1
    assert warm_model.text_encoder.weight is text_weight
    assert warm_model.vae.gain is vae_gain
    assert warm_model.transformer.derived.item() == 7.0
    for name in warm_context.tensor_names:
        assert restored[name].untyped_storage().data_ptr() == hit.lease.tensors[name].untyped_storage().data_ptr()

    del restored, warm_model, plan
    gc.collect()
    hit.lease.close()
    assert isinstance(runtime.store, FilesystemHostWeightStore)
    assert runtime.store.cleanup(context.identity) is None


def test_independent_publications_are_content_deterministic(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    model = _TinyPipeline()
    _fill_final_weights(model)
    context = _identity(model, source)
    first_runtime = _runtime(tmp_path / "first-store")
    second_runtime = _runtime(tmp_path / "second-store")

    first = first_runtime.publish_after_load(
        context.identity,
        producer=FinalLayoutBF16Producer(context, model, _dit_modules(model), max_shard_bytes=16),
    )
    second = second_runtime.publish_after_load(
        context.identity,
        producer=FinalLayoutBF16Producer(context, model, _dit_modules(model), max_shard_bytes=1024),
    )
    assert first.outcome is PostLoadPublicationOutcome.PUBLISHED
    assert second.outcome is PostLoadPublicationOutcome.PUBLISHED

    first_hit = first_runtime.resolve(context.identity)
    second_hit = second_runtime.resolve(context.identity)
    assert first_hit.lease is not None and second_hit.lease is not None
    assert first_hit.lease.manifest.artifact_content_sha256 == second_hit.lease.manifest.artifact_content_sha256
    assert first_hit.lease.manifest.format_metadata == second_hit.lease.manifest.format_metadata
    assert {entry.name: entry.sha256 for entry in first_hit.lease.manifest.tensors} == {
        entry.name: entry.sha256 for entry in second_hit.lease.manifest.tensors
    }
    first_hit.lease.close()
    second_hit.lease.close()


def test_source_replacement_prevents_publication(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    model = _TinyPipeline()
    _fill_final_weights(model)
    context = _identity(model, source)
    runtime = _runtime(tmp_path / "store")
    source.weight_files[0].write_bytes(b"replacement-source")

    publication = runtime.publish_after_load(
        context.identity,
        producer=FinalLayoutBF16Producer(context, model, _dit_modules(model)),
    )

    assert publication.outcome is PostLoadPublicationOutcome.FAILED
    assert publication.failure is not None
    assert not context.sources_unchanged()
    assert runtime.resolve(context.identity).report.outcome is ResolutionOutcome.CANONICAL_FALLBACK


def test_source_replacement_between_restore_plan_and_commit_causes_no_mutation(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    cold_model = _TinyPipeline()
    _fill_final_weights(cold_model)
    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "store")
    assert (
        runtime.publish_after_load(
            context.identity,
            producer=FinalLayoutBF16Producer(context, cold_model, _dit_modules(cold_model)),
        ).outcome
        is PostLoadPublicationOutcome.PUBLISHED
    )
    hit = runtime.resolve(context.identity)
    assert hit.lease is not None

    warm_model = _TinyPipeline()
    warm_context = _identity(warm_model, source)
    plan = FinalLayoutBF16Restorer(warm_context).plan_restore(warm_model, hit.lease)
    before = _pointer_snapshot(warm_model)
    source.weight_files[0].write_bytes(b"source-changed-before-commit")

    with pytest.raises(FinalLayoutContractError, match="canonical source changed"):
        plan.commit()
    assert _pointer_snapshot(warm_model) == before
    del plan
    gc.collect()
    hit.lease.close()


def test_restore_rejects_wrong_identity_and_coverage_without_mutation(tmp_path: Path) -> None:
    source = _prepared_source(tmp_path)
    cold_model = _TinyPipeline()
    _fill_final_weights(cold_model)
    context = _identity(cold_model, source)
    runtime = _runtime(tmp_path / "store")
    assert (
        runtime.publish_after_load(
            context.identity,
            producer=FinalLayoutBF16Producer(context, cold_model, _dit_modules(cold_model)),
        ).outcome
        is PostLoadPublicationOutcome.PUBLISHED
    )
    hit = runtime.resolve(context.identity)
    assert hit.lease is not None

    alternate = _AlternateTinyPipeline()
    _fill_final_weights(alternate)
    with pytest.raises(ValueError, match="producer model implementation differs"):
        FinalLayoutBF16Producer(context, alternate, _dit_modules(alternate))
    with pytest.raises(ValueError, match="restore model implementation differs"):
        FinalLayoutBF16Restorer(context).plan_restore(alternate, hit.lease)

    model = _TinyPipeline()
    wrong_context = _identity(
        model,
        source,
        request=_request(tensor_parallel_size=2, tensor_parallel_rank=1),
    )
    with pytest.raises(ValueError, match="semantic identity differs"):
        FinalLayoutBF16Restorer(wrong_context).plan_restore(model, hit.lease)

    exact_context = _identity(model, source)
    before = _pointer_snapshot(model)
    model.transformer.register_buffer("new_persistent_state", torch.tensor([1.0]))
    with pytest.raises(FinalLayoutContractError, match="tensor contract differs"):
        FinalLayoutBF16Restorer(exact_context).plan_restore(model, hit.lease)
    after = _pointer_snapshot(model)
    assert all(after[name] == value for name, value in before.items())
    hit.lease.close()


def test_minimax_h3_declares_the_final_layout_restore_contract() -> None:
    assert MiniMaxH3DiTModel.host_weight_restore_contract == FINAL_LAYOUT_BF16_MODEL_CONTRACT
    assert callable(MiniMaxH3DiTModel.validate_restored_host_weights)
    assert "data_parallel" not in FinalLayoutBF16Request.__dataclass_fields__
