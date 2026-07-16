# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert iWorld-Bench Memory trajectories into LingBot-World actions.

iWorld-Bench stores each trajectory as an 81-row, 19-column TXT file:

    timestamp, fx, fy, cx, cy, distortion[2], flattened OpenCV W2C[3, 4]

The Memory trajectories use normalized intrinsics. LingBot-World expects
pixel-space ``intrinsics.npy`` and OpenCV camera-to-world ``poses.npy``.

Example:
    python examples/offline_inference/diffusion/convert_iworldbench_lingbot_actions.py \
        --input-dir /path/to/iWorld-Bench/camera_trajectories/inference_txt \
        --output-dir /path/to/iWorld-Bench-LingBot/actions
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_EXPECTED_COLUMNS = 19
_REFERENCE_WIDTH = 832
_REFERENCE_HEIGHT = 480


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing iWorld-Bench memory_*.txt trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination root; each trajectory is written to its own directory.",
    )
    parser.add_argument("--width", type=int, default=_REFERENCE_WIDTH)
    parser.add_argument("--height", type=int, default=_REFERENCE_HEIGHT)
    parser.add_argument(
        "--pattern",
        default="memory_*.txt",
        help="Input filename glob. Defaults to the eight Memory trajectories.",
    )
    return parser.parse_args(argv)


def _validate_rotations(rotations: np.ndarray, source: Path) -> None:
    identity = np.eye(3, dtype=np.float64)
    orthogonality_error = np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - identity))
    determinants = np.linalg.det(rotations)
    if orthogonality_error > 1e-5 or not np.allclose(determinants, 1.0, atol=1e-5):
        raise ValueError(f"{source} does not contain rigid camera rotations.")


def convert_trajectory(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
) -> dict[str, object]:
    values = np.loadtxt(source, dtype=np.float64, ndmin=2)
    if values.ndim != 2 or values.shape[1] != _EXPECTED_COLUMNS:
        raise ValueError(f"{source} must have shape [frames, {_EXPECTED_COLUMNS}], got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite values.")

    world_to_camera = values[:, 7:].reshape(-1, 3, 4)
    rotations = world_to_camera[:, :3, :3]
    translations = world_to_camera[:, :3, 3]
    _validate_rotations(rotations, source)

    # iWorld Memory matrices are OpenCV W2C transforms. Rigid inversion gives
    # the OpenCV C2W convention consumed by the official LingBot implementation.
    inverse_rotations = rotations.transpose(0, 2, 1)
    inverse_translations = -np.einsum("fij,fj->fi", inverse_rotations, translations)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], values.shape[0], axis=0)
    poses[:, :3, :3] = inverse_rotations.astype(np.float32)
    poses[:, :3, 3] = inverse_translations.astype(np.float32)

    normalized_intrinsics = values[:, 1:5]
    intrinsics = normalized_intrinsics.astype(np.float32)
    intrinsics[:, (0, 2)] *= width
    intrinsics[:, (1, 3)] *= height
    if np.any(intrinsics[:, :2] <= 0):
        raise ValueError(f"{source} contains non-positive focal lengths.")

    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "poses.npy", poses, allow_pickle=False)
    np.save(destination / "intrinsics.npy", intrinsics, allow_pickle=False)

    loop_pose_error = float(np.max(np.abs(poses[0] - poses[-1])))
    return {
        "source": str(source.resolve()),
        "action_dir": str(destination.resolve()),
        "frames": int(values.shape[0]),
        "width": width,
        "height": height,
        "coordinate_system": "OpenCV camera-to-world",
        "intrinsics_order": ["fx", "fy", "cx", "cy"],
        "first_intrinsics": intrinsics[0].tolist(),
        "loop_pose_error": loop_pose_error,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError("--input-dir must point to an existing directory.")

    sources = sorted(input_dir.glob(args.pattern))
    if not sources:
        raise ValueError(f"No trajectories matching {args.pattern!r} found in {input_dir}.")

    manifest = [
        convert_trajectory(
            source,
            output_dir / source.stem,
            width=args.width,
            height=args.height,
        )
        for source in sources
    ]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for item in manifest:
        print(
            f"{Path(str(item['action_dir'])).name}: frames={item['frames']} "
            f"intrinsics={item['first_intrinsics']} loop_error={item['loop_pose_error']:.3g}"
        )
    print(f"Wrote {len(manifest)} LingBot action directories to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
