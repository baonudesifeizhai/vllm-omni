#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a ModelOpt diffusion checkpoint for vLLM-Omni.

The script intentionally keeps validation model-agnostic:

1. check the assembled Diffusers-style directory layout,
2. parse component quantization_config through vLLM-Omni's quantization
   factory,
3. verify the checkpoint adapter will be selected for ModelOpt FP8 weights,
4. optionally run the standard diffusion serving benchmark against an already
   running vLLM-Omni server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from checkpoint_utils import CheckpointAssemblyError, WeightSpec, resolve_weight_spec  # noqa: E402


class ValidationError(RuntimeError):
    pass


@dataclass
class CheckResult:
    name: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    model: Path
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, **details: Any) -> None:
        self.checks.append(CheckResult(name=name, details=details))

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": str(self.model),
                "checks": [{"name": item.name, **item.details} for item in self.checks],
            },
            indent=2,
            sort_keys=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ModelOpt diffusion checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Assembled vLLM-Omni/Diffusers model directory.")
    parser.add_argument(
        "--component",
        action="append",
        default=None,
        help="Component subdir to validate. May be passed multiple times. Defaults to transformer.",
    )
    parser.add_argument("--expected-method", default="modelopt", help="Expected quantization config get_name().")
    parser.add_argument(
        "--expected-algo",
        action="append",
        default=None,
        help="Allowed quant_algo value. May be passed multiple times.",
    )
    parser.add_argument("--stage-config", type=Path, default=None, help="Optional serving stage config path to check.")
    parser.add_argument(
        "--allow-component-only",
        action="store_true",
        help="Do not require root model_index.json/config.json. Useful for validating a raw component directory.",
    )
    parser.add_argument(
        "--skip-omni-config",
        action="store_true",
        help="Skip OmniDiffusionConfig construction check.",
    )
    parser.add_argument("--report-json", type=Path, default=None, help="Optional path to write a JSON report.")

    benchmark = parser.add_argument_group("online benchmark")
    benchmark.add_argument("--run-benchmark", action="store_true", help="Run benchmark against an existing server.")
    benchmark.add_argument(
        "--benchmark-script",
        type=Path,
        default=REPO_ROOT / "benchmarks/diffusion/diffusion_benchmark_serving.py",
    )
    benchmark.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable for benchmark.")
    benchmark.add_argument("--backend", default="vllm-omni")
    benchmark.add_argument("--host", default="127.0.0.1")
    benchmark.add_argument("--port", type=int, default=8000)
    benchmark.add_argument("--dataset", default="random")
    benchmark.add_argument("--task", default="t2i")
    benchmark.add_argument("--num-prompts", type=int, default=1)
    benchmark.add_argument("--request-rate", default="inf")
    benchmark.add_argument("--warmup-requests", type=int, default=0)
    benchmark.add_argument("--width", type=int, default=1024)
    benchmark.add_argument("--height", type=int, default=1024)
    benchmark.add_argument("--num-inference-steps", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--max-concurrency", type=int, default=1)
    benchmark.add_argument("--enable-negative-prompt", action="store_true")
    benchmark.add_argument("--output-file", type=Path, default=None)
    benchmark.add_argument(
        "--print-benchmark-command",
        action="store_true",
        help="Print the benchmark command without executing it.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Failed to read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"Expected JSON object in {path}")
    return payload


def get_quant_algo(quant_config: dict[str, Any]) -> str | None:
    nested = quant_config.get("quantization")
    if isinstance(nested, dict) and nested.get("quant_algo") is not None:
        return str(nested["quant_algo"])
    if quant_config.get("quant_algo") is not None:
        return str(quant_config["quant_algo"])
    return None


def weight_kind(weight_spec: WeightSpec) -> str:
    if weight_spec.kind == "single":
        assert weight_spec.single_file is not None
        return weight_spec.single_file.name
    assert weight_spec.index_file is not None
    return f"{weight_spec.index_file.name} ({len(weight_spec.shard_files)} shards)"


def is_safetensors_weight(weight_spec: WeightSpec) -> bool:
    if weight_spec.kind == "single":
        assert weight_spec.single_file is not None
        return weight_spec.single_file.suffix == ".safetensors"
    assert weight_spec.index_file is not None
    return weight_spec.index_file.name.endswith(".safetensors.index.json")


def validate_component(
    model_dir: Path,
    component: str,
    expected_method: str | None,
    expected_algos: set[str] | None,
    report: ValidationReport,
) -> None:
    from vllm_omni.diffusion.data import TransformerConfig
    from vllm_omni.diffusion.model_loader.checkpoint_adapters.modelopt_fp8 import ModelOptFp8CheckpointAdapter
    from vllm_omni.quantization import build_quant_config

    component_dir = model_dir / component
    if not component_dir.is_dir():
        raise ValidationError(f"Missing component directory: {component_dir}")
    config_path = component_dir / "config.json"
    if not config_path.is_file():
        raise ValidationError(f"Missing component config: {config_path}")

    config_payload = load_json(config_path)
    quant_payload = config_payload.get("quantization_config")
    if not isinstance(quant_payload, dict):
        raise ValidationError(f"{config_path} is missing dict quantization_config")

    quant_config = build_quant_config(quant_payload)
    if quant_config is None:
        raise ValidationError(f"{config_path} quantization_config resolved to None")

    method = quant_config.get_name()
    if expected_method is not None and method != expected_method:
        raise ValidationError(f"{config_path} resolved method {method!r}, expected {expected_method!r}")

    quant_algo = get_quant_algo(quant_payload)
    if expected_algos is not None and quant_algo not in expected_algos:
        raise ValidationError(f"{config_path} has quant_algo {quant_algo!r}, expected one of {sorted(expected_algos)}")

    try:
        weight_spec = resolve_weight_spec(component_dir, role=component)
    except CheckpointAssemblyError as exc:
        raise ValidationError(str(exc)) from exc

    TransformerConfig.from_dict(config_payload)
    source = SimpleNamespace(subfolder=component, prefix=f"{component}.")
    adapter_selected = ModelOptFp8CheckpointAdapter.is_compatible(
        source=source,
        quant_config=quant_config,
        use_safetensors=is_safetensors_weight(weight_spec),
    )
    if method == "modelopt" and component == "transformer" and not adapter_selected:
        raise ValidationError("ModelOpt FP8 checkpoint adapter was not selected for transformer safetensors weights")

    report.add(
        "component",
        component=component,
        method=method,
        quant_algo=quant_algo,
        weights=weight_kind(weight_spec),
        adapter_selected=adapter_selected,
    )


def validate_root_layout(model_dir: Path, allow_component_only: bool, report: ValidationReport) -> None:
    if not model_dir.is_dir():
        raise ValidationError(f"Model directory does not exist: {model_dir}")

    if not allow_component_only:
        root_config_exists = (model_dir / "model_index.json").is_file() or (model_dir / "config.json").is_file()
        if not root_config_exists:
            raise ValidationError(f"{model_dir} must contain model_index.json or config.json")
    report.add(
        "root_layout",
        model_index=(model_dir / "model_index.json").is_file(),
        config=(model_dir / "config.json").is_file(),
    )


def validate_stage_config(stage_config: Path | None, report: ValidationReport) -> None:
    if stage_config is None:
        return
    if not stage_config.is_file():
        raise ValidationError(f"Stage config does not exist: {stage_config}")
    report.add("stage_config", path=str(stage_config))


def validate_omni_config(model_dir: Path, skip: bool, report: ValidationReport) -> None:
    if skip:
        return
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    od_config = OmniDiffusionConfig(model=str(model_dir))
    quant_config = getattr(od_config, "quantization_config", None)
    report.add(
        "omni_config",
        model_class_name=getattr(od_config, "model_class_name", None),
        quantization=(
            quant_config.get_name() if quant_config is not None and hasattr(quant_config, "get_name") else None
        ),
    )


def build_benchmark_command(args: argparse.Namespace) -> list[str]:
    output_file = args.output_file
    if output_file is None:
        output_file = Path("outputs") / f"{args.model.name}_modelopt_validation.json"

    command = [
        str(args.python),
        str(args.benchmark_script),
        "--backend",
        args.backend,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(args.model),
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--warmup-requests",
        str(args.warmup_requests),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--seed",
        str(args.seed),
        "--max-concurrency",
        str(args.max_concurrency),
        "--output-file",
        str(output_file),
    ]
    if args.enable_negative_prompt:
        command.append("--enable-negative-prompt")
    return command


def maybe_run_benchmark(args: argparse.Namespace, report: ValidationReport) -> None:
    if not args.run_benchmark and not args.print_benchmark_command:
        return
    if not args.benchmark_script.is_file():
        raise ValidationError(f"Benchmark script does not exist: {args.benchmark_script}")

    command = build_benchmark_command(args)
    report.add("benchmark_command", command=command)
    if args.print_benchmark_command:
        print(" ".join(command))
    if not args.run_benchmark:
        return

    output_file = Path(command[-1])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    report.add("benchmark", output_file=str(output_file))


def run_validation(args: argparse.Namespace) -> ValidationReport:
    components = args.component or ["transformer"]
    expected_algos = set(args.expected_algo) if args.expected_algo else None
    report = ValidationReport(model=args.model)

    validate_root_layout(args.model, args.allow_component_only, report)
    validate_stage_config(args.stage_config, report)
    for component in components:
        validate_component(args.model, component, args.expected_method, expected_algos, report)
    validate_omni_config(args.model, args.skip_omni_config, report)
    maybe_run_benchmark(args, report)
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run_validation(args)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: benchmark failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(report.to_json() + "\n", encoding="utf-8")
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
