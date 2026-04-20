# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online smoke test for a public ModelOpt FP8 image checkpoint.

This validates that a native ModelOpt FP8 diffusers checkpoint can be served
through Omni, accepts an Images API request, and returns a decodable image.
"""

import base64
from io import BytesIO
from pathlib import Path

import pytest
import requests
from PIL import Image

from tests.conftest import OmniServer, OmniServerParams, assert_image_valid
from tests.utils import hardware_marks

MODEL = "feizhai123/flux2-dev-modelopt-fp8"
STAGE_CONFIG = str(Path(__file__).parent.parent / "stage_configs" / "flux2_dev_dit_2gpu_fp8.yaml")
PROMPT = (
    "An art deco locomotive crossing a high bridge above a misty canyon at sunrise, cinematic light, highly detailed."
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, watermark"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 4
TRUE_CFG_SCALE = 4.0
SEED = 42


def _post_image_request(server: OmniServer) -> Image.Image:
    response = requests.post(
        f"http://{server.host}:{server.port}/v1/images/generations",
        headers={"Authorization": "Bearer EMPTY"},
        json={
            "model": server.model,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "size": f"{WIDTH}x{HEIGHT}",
            "response_format": "b64_json",
            "n": 1,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "seed": SEED,
        },
        timeout=900,
    )
    response.raise_for_status()
    payload = response.json()

    assert "data" in payload and len(payload["data"]) == 1
    encoded_image = payload["data"][0]["b64_json"]
    image = Image.open(BytesIO(base64.b64decode(encoded_image)))
    image.load()
    return image.convert("RGB")


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=STAGE_CONFIG,
                init_timeout=900,
                stage_init_timeout=900,
            ),
            id="flux2_dev_modelopt_fp8_2gpu",
            marks=hardware_marks(res={"cuda": "H100"}, num_cards=2),
        )
    ],
    indirect=True,
)
def test_modelopt_fp8_images_api_returns_valid_image(omni_server: OmniServer) -> None:
    image = _post_image_request(omni_server)
    assert_image_valid(image, width=WIDTH, height=HEIGHT)
