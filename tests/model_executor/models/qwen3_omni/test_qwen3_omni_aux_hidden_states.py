# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3-Omni Thinker auxiliary hidden states."""

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeThinker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aux_layers: tuple[int, ...] = ()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_layers = layers

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return (2, 24, 45)


def _make_thinker_stage() -> Qwen3OmniMoeForConditionalGeneration:
    model = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "thinker"
    model.thinker = _FakeThinker()
    return model


def test_thinker_preserves_aux_hidden_states_around_omni_output() -> None:
    model = _make_thinker_stage()
    final_hidden = torch.randn(2, 3, 8)
    captured = {"hidden_states": {"layers": {0: torch.randn(6, 8)}}}
    aux_hidden_states = [torch.randn(6, 8) for _ in range(6)]

    output = model.make_omni_output(((final_hidden, captured), aux_hidden_states))

    assert isinstance(output, tuple)
    omni_output, returned_aux = output
    assert isinstance(omni_output, OmniOutput)
    assert torch.equal(
        omni_output.text_hidden_states,
        final_hidden.reshape(-1, final_hidden.shape[-1]),
    )
    assert omni_output.multimodal_outputs is captured
    assert returned_aux is aux_hidden_states


def test_thinker_delegates_eagle3_aux_layer_configuration() -> None:
    model = _make_thinker_stage()

    assert supports_eagle3(model)
    model.set_aux_hidden_state_layers((5, 14, 24, 33, 43, 48))

    assert model.thinker.aux_layers == (5, 14, 24, 33, 43, 48)
    assert model.get_eagle3_default_aux_hidden_state_layers() == (2, 24, 45)


def test_non_thinker_stage_rejects_aux_hidden_states() -> None:
    model = _make_thinker_stage()
    model.model_stage = "talker"

    with pytest.raises(RuntimeError, match="only available on the Thinker"):
        model.set_aux_hidden_state_layers((5, 14, 24, 33, 43, 48))
