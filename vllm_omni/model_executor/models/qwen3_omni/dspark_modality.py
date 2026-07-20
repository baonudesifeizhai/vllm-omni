"""vLLM compatibility for modality-routed Qwen3-Omni DSpark checkpoints."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import maybe_prefix

_MODALITY_NAMES: Final = ("text", "image", "audio", "video")


class _ModalityResidualHead(nn.Module):
    """TP-compatible equivalent of Speculators' ``ModalityLogitHead``."""

    def __init__(
        self,
        hidden_size: int,
        draft_vocab_size: int,
        rank: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.down_proj = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "down_proj"),
            return_bias=False,
        )
        self.up_proj = ParallelLMHead(
            draft_vocab_size,
            rank,
            params_dtype=dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "up_proj"),
        )


class _ModalityRouter(nn.Module):
    """TP-replicated equivalent of Speculators' ``ModalityRouter``."""

    def __init__(self, hidden_size: int, dtype: torch.dtype, prefix: str) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            hidden_size,
            len(_MODALITY_NAMES),
            bias=True,
            params_dtype=dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "proj"),
            return_bias=False,
        )


def _install_config_patch() -> None:
    from vllm.transformers_utils.configs.speculators import algos

    current = algos.SUPPORTED_SPECULATORS_TYPES["dspark"]
    if getattr(current, "_omni_modality_heads_patched", False):
        return

    def update_dspark_with_modalities(config_dict: dict, pre_trained_config: dict) -> None:
        current(config_dict=config_dict, pre_trained_config=pre_trained_config)
        # Match the actual training layout instead of forcing the DFlash-style
        # bonus-anchor layout for every Speculators checkpoint.
        pre_trained_config["dspark_bonus_anchor"] = not bool(config_dict.get("sample_from_anchor", True))
        for key in ("modality_head_rank", "modality_token_ids"):
            if config_dict.get(key) is not None:
                pre_trained_config[key] = config_dict[key]

    update_dspark_with_modalities._omni_modality_heads_patched = True
    algos.SUPPORTED_SPECULATORS_TYPES["dspark"] = update_dspark_with_modalities


def _finish_logits(logits_processor, local_logits: torch.Tensor):
    """Finish a locally sharded LM projection with one TP gather."""
    logits = logits_processor._gather_logits(local_logits)  # noqa: SLF001
    if logits is None:
        return None
    logits = logits[..., : logits_processor.org_vocab_size]
    if logits_processor.soft_cap is not None:
        logits = torch.tanh(logits / logits_processor.soft_cap)
        logits = logits * logits_processor.soft_cap
    if logits_processor.scale != 1.0:
        logits *= logits_processor.scale
    return logits


def _install_model_patch() -> None:
    from vllm.model_executor.models import qwen3_dspark

    model_cls = qwen3_dspark.Qwen3DSparkModel
    causal_lm_cls = qwen3_dspark.Qwen3DSparkForCausalLM
    original_init = model_cls.__init__
    if getattr(original_init, "_omni_modality_heads_patched", False):
        return

    def patched_model_init(
        self,
        *,
        vllm_config,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        original_init(
            self,
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        rank = int(getattr(self.config, "modality_head_rank", 0) or 0)
        self.modality_heads = nn.ModuleDict()
        self.modality_router = None
        if rank <= 0:
            return

        hidden_size = self.config.hidden_size
        draft_vocab_size = getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        dtype = vllm_config.model_config.dtype
        self.modality_heads = nn.ModuleDict(
            {
                name: _ModalityResidualHead(
                    hidden_size,
                    draft_vocab_size,
                    rank,
                    dtype,
                    maybe_prefix(prefix, f"modality_heads.{name}"),
                )
                for name in _MODALITY_NAMES
            }
        )
        self.modality_router = _ModalityRouter(
            hidden_size,
            dtype,
            maybe_prefix(prefix, "modality_router"),
        )

    original_compute_draft_logits = causal_lm_cls.compute_draft_logits

    def patched_compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.model.modality_heads:
            return original_compute_draft_logits(self, hidden_states)

        block_size = int(self.config.block_size)
        if hidden_states.shape[0] % block_size != 0:
            raise ValueError(
                "Modality-routed DSpark expects complete draft blocks, got "
                f"{hidden_states.shape[0]} hidden states for block_size={block_size}."
            )
        hidden_blocks = hidden_states.view(-1, block_size, hidden_states.shape[-1])
        router_logits = self.model.modality_router.proj(hidden_blocks[:, 0])
        route_ids = router_logits.argmax(dim=-1).repeat_interleave(block_size)

        # Combine all local-vocabulary projections first, then communicate once.
        # The four fixed branches are CUDA-graph safe; masks select one residual
        # per request without a data-dependent Python branch.
        processor = self.logits_processor
        local_logits = processor._apply_head(  # noqa: SLF001
            self.lm_head, hidden_states, None
        )
        for modality_id, name in enumerate(_MODALITY_NAMES):
            head = self.model.modality_heads[name]
            low_rank = F.silu(head.down_proj(hidden_states))
            residual = processor._apply_head(  # noqa: SLF001
                head.up_proj, low_rank, None
            )
            route_mask = (route_ids == modality_id).to(residual.dtype).unsqueeze(-1)
            local_logits = local_logits + residual * route_mask
        return _finish_logits(processor, local_logits)

    patched_model_init._omni_modality_heads_patched = True
    patched_compute_draft_logits._omni_modality_heads_patched = True
    model_cls.__init__ = patched_model_init
    causal_lm_cls.compute_draft_logits = patched_compute_draft_logits


def install_qwen3_omni_dspark_modality_patch() -> None:
    """Install config, loader, and inference support for the four draft heads."""
    _install_config_patch()
    _install_model_patch()
