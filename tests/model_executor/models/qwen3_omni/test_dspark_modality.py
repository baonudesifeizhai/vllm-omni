import torch

from vllm_omni.model_executor.models.qwen3_omni.dspark_modality import (
    _finish_logits,
    install_qwen3_omni_dspark_modality_patch,
)


def test_dspark_config_propagates_modality_heads_and_anchor_layout():
    from vllm.transformers_utils.configs.speculators.algos import (
        SUPPORTED_SPECULATORS_TYPES,
    )

    install_qwen3_omni_dspark_modality_patch()
    config = {
        "aux_hidden_state_layer_ids": [2, 13, 24, 35, 46],
        "sample_from_anchor": True,
        "draft_vocab_size": 32000,
        "mask_token_id": 151643,
        "markov_rank": 256,
        "block_size": 7,
        "modality_head_rank": 128,
        "modality_token_ids": {
            "image": [151655],
            "audio": [151675],
            "video": [151656],
        },
    }
    converted = {
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000,
            "mrope_section": [24, 20, 20],
            "interleaved": True,
        }
    }
    SUPPORTED_SPECULATORS_TYPES["dspark"](config, converted)

    assert converted["dspark_bonus_anchor"] is False
    assert converted["dflash_config"] == {
        "mask_token_id": config["mask_token_id"],
        "target_layer_ids": [1, 12, 23, 34, 45],
    }
    assert converted["modality_head_rank"] == 128
    assert converted["modality_token_ids"] == config["modality_token_ids"]
    assert converted["rope_parameters"] == {
        "rope_type": "default",
        "rope_theta": 1_000_000,
    }


def test_dspark_config_preserves_bonus_anchor_mode_when_requested():
    from vllm.transformers_utils.configs.speculators.algos import (
        SUPPORTED_SPECULATORS_TYPES,
    )

    install_qwen3_omni_dspark_modality_patch()
    config = {
        "aux_hidden_state_layer_ids": [2, 13, 24],
        "sample_from_anchor": False,
        "draft_vocab_size": 128,
        "mask_token_id": 127,
        "markov_rank": 8,
        "block_size": 4,
    }
    converted = {}
    SUPPORTED_SPECULATORS_TYPES["dspark"](config, converted)
    assert converted["dspark_bonus_anchor"] is True
    assert converted["dflash_config"]["mask_token_id"] == 127


def test_finish_logits_gathers_once_then_applies_vocab_and_scale():
    class FakeProcessor:
        org_vocab_size = 3
        soft_cap = None
        scale = 2.0

        def __init__(self):
            self.gathers = 0

        def _gather_logits(self, logits):
            self.gathers += 1
            return logits

    processor = FakeProcessor()
    local_logits = torch.tensor([[1.0, 2.0, 3.0, 99.0]])
    logits = _finish_logits(processor, local_logits)
    assert processor.gathers == 1
    assert torch.equal(logits, torch.tensor([[2.0, 4.0, 6.0]]))
