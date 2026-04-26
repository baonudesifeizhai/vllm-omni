#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble ModelOpt diffusion checkpoints with a small, config-driven pipeline.

This tool is intentionally separate from runtime/loader support. It only helps
with offline conversion tasks such as:

1. running an external ModelOpt quantization command,
2. assembling a Diffusers/vLLM-Omni-style model directory, and
3. verifying the resulting layout.

The config schema is step-based so different models can describe their own
conversion flow without needing one script per model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_utils import (  # noqa: E402
    CheckpointAssemblyError,
    copy_or_link_dir,
    ensure_clean_output,
    materialize_weight,
    resolve_weight_spec,
)


class ConvertError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a config-driven ModelOpt diffusion conversion pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config describing the conversion steps.",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a top-level config variable. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print the plan without touching the filesystem.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow assemble steps to remove and recreate existing output directories.",
    )
    return parser.parse_args()


def parse_cli_vars(raw_items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ConvertError(f"Invalid --var {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConvertError(f"Invalid --var {item!r}; key cannot be empty.")
        parsed[key] = value
    return parsed


def load_config(config_path: Path, cli_vars: dict[str, str]) -> dict[str, Any]:
    if not config_path.is_file():
        raise ConvertError(f"Config file does not exist: {config_path}")

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(raw, dict):
        raise ConvertError("Top-level YAML config must be a mapping.")

    cfg_vars = raw.get("vars", {})
    if cfg_vars is None:
        cfg_vars = {}
    if not isinstance(cfg_vars, dict):
        raise ConvertError("Config field 'vars' must be a mapping if provided.")
    merged_vars = {str(k): v for k, v in cfg_vars.items()}
    merged_vars.update(cli_vars)
    raw["vars"] = resolve_vars(merged_vars)
    return render_value(raw, raw["vars"])


def render_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(variables)
        except KeyError as exc:
            missing = exc.args[0]
            raise ConvertError(f"Missing variable {{{missing}}} while rendering config.") from exc
    if isinstance(value, list):
        return [render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {k: render_value(v, variables) for k, v in value.items()}
    return value


def resolve_vars(variables: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(variables)
    for _ in range(max(len(resolved), 1)):
        next_resolved = render_value(resolved, resolved)
        if next_resolved == resolved:
            return next_resolved
        resolved = next_resolved
    return resolved


def deep_merge(base: Any, updates: Any) -> Any:
    if isinstance(base, dict) and isinstance(updates, dict):
        merged = dict(base)
        for key, value in updates.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return updates


def import_object(import_path: str) -> Any:
    module_name, sep, attr_path = import_path.partition(":")
    if not sep:
        module_name, sep, attr_path = import_path.rpartition(".")
    if not sep or not module_name or not attr_path:
        raise ConvertError(f"Invalid import path {import_path!r}. Use 'pkg.module:attr' or 'pkg.module.attr'.")

    module = importlib.import_module(module_name)
    obj: Any = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def parse_torch_dtype(dtype_name: str | None) -> Any | None:
    if dtype_name is None:
        return None

    import torch

    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = aliases.get(dtype_name.lower(), dtype_name.lower())
    dtype = getattr(torch, normalized, None)
    if dtype is None:
        raise ConvertError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


def modelopt_recipe_step(step: dict[str, Any]) -> None:
    recipe_name = step.get("recipe")
    output_path = step.get("output_path")
    if not recipe_name or not output_path:
        raise ConvertError(f"ModelOpt recipe step {step.get('name', '<unnamed>')} needs 'recipe' and 'output_path'.")

    from modelopt.recipe import load_recipe

    recipe = load_recipe(recipe_name)
    payload = recipe.model_dump(mode="json")
    overrides = step.get("overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ConvertError(
                f"ModelOpt recipe step {step.get('name', '<unnamed>')} field 'overrides' must be a mapping."
            )
        payload = deep_merge(payload, overrides)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True)
    output_file.write_text(yaml_text, encoding="utf-8")


def modelopt_export_hf_step(step: dict[str, Any]) -> None:
    model_path = step.get("model_path")
    export_dir = step.get("export_dir")
    if not model_path or not export_dir:
        raise ConvertError(f"ModelOpt export step {step.get('name', '<unnamed>')} needs 'model_path' and 'export_dir'.")

    loader_path = step.get("loader", "diffusers:DiffusionPipeline")
    loader_obj = import_object(loader_path)
    if not hasattr(loader_obj, "from_pretrained"):
        raise ConvertError(f"Loader {loader_path!r} must resolve to an object exposing from_pretrained().")

    load_kwargs = step.get("load_kwargs", {})
    if not isinstance(load_kwargs, dict):
        raise ConvertError(
            f"ModelOpt export step {step.get('name', '<unnamed>')} field 'load_kwargs' must be a mapping."
        )

    torch_dtype = parse_torch_dtype(step.get("torch_dtype"))
    if torch_dtype is not None and "torch_dtype" not in load_kwargs:
        load_kwargs = {**load_kwargs, "torch_dtype": torch_dtype}

    model = loader_obj.from_pretrained(model_path, **load_kwargs)

    export_kwargs = step.get("export_kwargs", {})
    if not isinstance(export_kwargs, dict):
        raise ConvertError(
            f"ModelOpt export step {step.get('name', '<unnamed>')} field 'export_kwargs' must be a mapping."
        )

    from modelopt.torch.export.unified_export_hf import export_hf_checkpoint

    export_hf_checkpoint(
        model=model,
        dtype=torch_dtype,
        export_dir=Path(export_dir),
        components=step.get("components"),
        **export_kwargs,
    )


def run_step(step: dict[str, Any]) -> None:
    command = step.get("command")
    if not isinstance(command, list) or not command:
        raise ConvertError(f"Run step {step.get('name', '<unnamed>')} needs a non-empty command list.")

    cwd = step.get("cwd")
    env = None
    if "env" in step:
        if not isinstance(step["env"], dict):
            raise ConvertError(f"Run step {step.get('name', '<unnamed>')} field 'env' must be a mapping.")
        env = {**os.environ, **{str(k): str(v) for k, v in step["env"].items()}}

    subprocess.run(
        [str(arg) for arg in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def assemble_step(step: dict[str, Any], overwrite: bool) -> None:
    skeleton_dir = Path(step["skeleton_dir"])
    output_dir = Path(step["output_dir"])
    asset_mode = step.get("asset_mode", "copy")

    if not skeleton_dir.is_dir():
        raise ConvertError(f"Assemble step skeleton_dir is not a directory: {skeleton_dir}")

    try:
        ensure_clean_output(output_dir, overwrite=overwrite)
    except CheckpointAssemblyError as exc:
        raise ConvertError(str(exc)) from exc

    copy_spec = step.get("copy", {})
    if not isinstance(copy_spec, dict):
        raise ConvertError(f"Assemble step {step.get('name', '<unnamed>')} field 'copy' must be a mapping.")

    for file_name in copy_spec.get("root_files", []):
        src_file = skeleton_dir / file_name
        if not src_file.is_file():
            raise ConvertError(f"Missing root file in skeleton: {src_file}")
        dst_file = output_dir / file_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    for dir_name in copy_spec.get("dirs", []):
        src_dir = skeleton_dir / dir_name
        if not src_dir.is_dir():
            raise ConvertError(f"Missing asset directory in skeleton: {src_dir}")
        copy_or_link_dir(src_dir, output_dir / dir_name, asset_mode)

    for component in step.get("components", []):
        if not isinstance(component, dict):
            raise ConvertError("Each assemble component must be a mapping.")
        target_subdir = component["target_subdir"]
        component_dir = output_dir / target_subdir
        component_dir.mkdir(parents=True, exist_ok=True)

        config_src = Path(component["config_src"])
        if not config_src.is_file():
            raise ConvertError(f"Missing component config source: {config_src}")
        config_dst = component_dir / "config.json"
        if "config_updates" not in component:
            shutil.copy2(config_src, config_dst)
        else:
            base_config = json.loads(config_src.read_text(encoding="utf-8"))
            merged = deep_merge(base_config, component["config_updates"])
            config_dst.write_text(json.dumps(merged, indent=2, sort_keys=False) + "\n", encoding="utf-8")

        weights_src = Path(component["weights_src"])
        try:
            weight_spec = resolve_weight_spec(weights_src, role=target_subdir)
            materialize_weight(weight_spec, component_dir, role=target_subdir)
        except CheckpointAssemblyError as exc:
            raise ConvertError(str(exc)) from exc


def verify_exists(path: Path) -> None:
    if not path.exists():
        raise ConvertError(f"Verification failed: expected path to exist: {path}")


def verify_step(step: dict[str, Any]) -> None:
    for check in step.get("checks", []):
        if not isinstance(check, dict):
            raise ConvertError("Each verify check must be a mapping.")
        if "exists" in check:
            verify_exists(Path(check["exists"]))
            continue
        if "any_exists" in check:
            candidates = [Path(item) for item in check["any_exists"]]
            if not any(path.exists() for path in candidates):
                joined = ", ".join(str(path) for path in candidates)
                raise ConvertError(f"Verification failed: expected at least one path to exist: {joined}")
            continue
        raise ConvertError(f"Unknown verify check: {check}")


def execute_steps(config: dict[str, Any], overwrite: bool, dry_run: bool) -> None:
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ConvertError("Config must include a non-empty 'steps' list.")

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ConvertError(f"Step #{index} must be a mapping.")
        step_type = step.get("type")
        step_name = step.get("name", f"step_{index}")
        if dry_run:
            print(f"[dry-run] {index}. {step_name} ({step_type})")
            if step_type == "run":
                print("  command:", " ".join(str(arg) for arg in step.get("command", [])))
            elif step_type == "modelopt_recipe":
                print(f"  recipe: {step.get('recipe')}")
                print(f"  output: {step.get('output_path')}")
            elif step_type == "modelopt_export_hf":
                print(f"  model:  {step.get('model_path')}")
                print(f"  export: {step.get('export_dir')}")
            elif step_type == "assemble":
                print(f"  skeleton: {step.get('skeleton_dir')}")
                print(f"  output:   {step.get('output_dir')}")
            elif step_type == "verify":
                print(f"  checks:   {len(step.get('checks', []))}")
            else:
                raise ConvertError(f"Unsupported step type: {step_type}")
            continue

        print(f"[{index}/{len(steps)}] {step_name} ({step_type})")
        if step_type == "run":
            run_step(step)
        elif step_type == "modelopt_recipe":
            modelopt_recipe_step(step)
        elif step_type == "modelopt_export_hf":
            modelopt_export_hf_step(step)
        elif step_type == "assemble":
            assemble_step(step, overwrite=overwrite)
        elif step_type == "verify":
            verify_step(step)
        else:
            raise ConvertError(f"Unsupported step type: {step_type}")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config, parse_cli_vars(args.var))
        execute_steps(config, overwrite=args.overwrite, dry_run=args.dry_run)
    except ConvertError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: step command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
