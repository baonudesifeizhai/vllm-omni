#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample ModelOpt FP8 conversion script for a single diffusion model.

This file is intentionally not a one-size-fits-all converter.
Different diffusion models need different quantization/export flows, so the
practical pattern here is "one model, one conversion method".

This sample only covers the qwen-image-2512 path:

* load the Diffusers transformer checkpoint,
* quantize selected Linear weights to FP8 per channel,
* persist weight_scale tensors, and
* write the matching ModelOpt-style quantization_config.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

FP8_E4M3_MAX = 448.0

DEFAULT_TARGETS = [
    "ff.net.0.proj",
    "ff.net.2",
    "img_attn.proj",
    "img_mlp.fc1",
    "img_mlp.fc2",
    "img_mod.lin",
    "txt_attn.proj",
    "txt_mlp.fc1",
    "txt_mlp.fc2",
    "txt_mod.lin",
    "to_add_out",
    "to_k",
    "to_out.0",
    "to_q",
    "to_v",
]
PROFILES: dict[str, list[str]] = {
    "all-linear": [],
    "attention-only": [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "to_add_out",
        "img_attn.proj",
        "txt_attn.proj",
    ],
    "default": DEFAULT_TARGETS,
}
IGNORE_NAME_REWRITES = (
    (".to_q.", ".to_qkv."),
    (".to_k.", ".to_qkv."),
    (".to_v.", ".to_qkv."),
    (".add_k_proj.", ".add_qkv_proj."),
    (".add_v_proj.", ".add_qkv_proj."),
    (".ff.net.0.proj.", ".ff.net.0."),
)


class ConvertError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Input qwen-image-2512 model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output converted model directory.")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype used for loading the source model.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    parser.add_argument("--max-shard-size", default="10GB", help="Shard size passed to save_pretrained().")
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(PROFILES)),
        default="default",
        help="Named target set for qwen-image-2512 Linear layers.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Additional Linear-name suffix to quantize. May be repeated.",
    )
    return parser.parse_args()


def parse_torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = aliases.get(name.lower(), name.lower())
    dtype = getattr(torch, normalized, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ConvertError(f"Unsupported torch dtype: {name}")
    return dtype


def require_existing_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise ConvertError(f"{description} does not exist or is not a directory: {path}")


def require_existing_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ConvertError(f"{description} does not exist: {path}")


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ConvertError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def copy_model_tree(model_dir: Path, output_dir: Path) -> None:
    shutil.copytree(model_dir, output_dir, symlinks=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_weight_dtypes(weights_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for file_path in sorted(weights_path.glob("*.safetensors")):
        with safe_open(file_path, framework="pt") as handle:
            for key in handle.keys():
                counts[str(handle.get_tensor(key).dtype)] += 1
    return dict(sorted(counts.items()))


def check_has_fp8_weights(weights_path: Path) -> bool:
    dtype_summary = summarize_weight_dtypes(weights_path)
    return any("float8" in dtype_name for dtype_name in dtype_summary)


def ignored_names_for_profile(profile: str) -> list[str]:
    targets = PROFILES[profile]
    if not targets:
        return []
    rewritten: set[str] = set()
    for target in targets:
        name = f".{target}."
        for src, dst in IGNORE_NAME_REWRITES:
            name = name.replace(src, dst)
        rewritten.add(name.strip("."))
    return sorted(rewritten)


def build_quantization_config(profile: str) -> dict[str, Any]:
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 8,
                    "strategy": "token",
                    "type": "float",
                },
                "weights": {
                    "dynamic": False,
                    "num_bits": 8,
                    "strategy": "channel",
                    "type": "float",
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ignored_names_for_profile(profile),
        "producer": {
            "name": "modelopt",
            "version": "manual-fp8-per-channel-per-token",
        },
        "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
        "quant_method": "modelopt",
        "source_profile": profile,
    }


def should_quantize(name: str, targets: list[str]) -> bool:
    if not targets:
        return True
    return any(name.endswith(target) for target in targets)


def quantize_weight_per_channel(module: torch.nn.Linear) -> None:
    weight = module.weight.detach().to(torch.float32)
    amax = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_E4M3_MAX
    quantized = (weight / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    module.weight = torch.nn.Parameter(quantized, requires_grad=False)
    module.register_parameter(
        "weight_scale",
        torch.nn.Parameter(scale.squeeze(1).contiguous(), requires_grad=False),
    )


def validate_inputs(model_dir: Path) -> None:
    require_existing_dir(model_dir, "input model directory")
    require_existing_file(model_dir / "model_index.json", "diffusers model_index.json")
    require_existing_dir(model_dir / "transformer", "transformer directory")
    require_existing_file(model_dir / "transformer" / "config.json", "transformer config")


def run_conversion(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    from diffusers import QwenImageTransformer2DModel

    transformer_dir = args.model / "transformer"
    model = QwenImageTransformer2DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )

    targets = list(PROFILES[args.profile]) + list(args.target)
    quantized_names: list[str] = []
    with torch.inference_mode():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and should_quantize(name, targets):
                quantize_weight_per_channel(module)
                quantized_names.append(name)

    if not quantized_names:
        raise ConvertError("No Linear modules matched the requested target set.")

    copy_model_tree(args.model, args.output)
    out_transformer_dir = args.output / "transformer"
    if out_transformer_dir.exists():
        shutil.rmtree(out_transformer_dir)

    model.save_pretrained(
        out_transformer_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    config_path = out_transformer_dir / "config.json"
    config = load_json(config_path)
    config["quantization_config"] = build_quantization_config(args.profile)
    write_json(config_path, config)

    if not check_has_fp8_weights(out_transformer_dir):
        raise ConvertError(f"Expected FP8 tensors under {out_transformer_dir}, but none were found.")

    return {
        "model": str(args.model),
        "output": str(args.output),
        "profile": args.profile,
        "quantized_linear_modules": len(quantized_names),
        "dtype_summary": summarize_weight_dtypes(out_transformer_dir),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    dtype = parse_torch_dtype(args.dtype)
    validate_inputs(args.model)
    ensure_clean_output(args.output, args.overwrite)
    summary = run_conversion(args, dtype)
    print_summary(summary)


if __name__ == "__main__":
    main()
