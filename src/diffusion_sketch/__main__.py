"""CLI entry point: python -m diffusion_sketch [--config PATH] [overrides...]

Examples:
    python -m diffusion_sketch
    python -m diffusion_sketch --config configs/custom.yaml
    python -m diffusion_sketch training.epochs=50 training.batch_size=8
"""

import argparse
import sys

from diffusion_sketch.config import load_config
from diffusion_sketch.training import run_training


def _parse_overrides(args: list[str]) -> dict:
    overrides = {}
    for arg in args:
        if "=" not in arg:
            print(f"Warning: ignoring malformed override '{arg}' (expected key=value)")
            continue
        key, val = arg.split("=", 1)
        for cast in (int, float):
            try:
                val = cast(val)
                break
            except ValueError:
                continue
        if val == "true":
            val = True
        elif val == "false":
            val = False
        overrides[key] = val
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Diffusion-Sketch: Conditional DDPM for sketch colorization",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args, remaining = parser.parse_known_args()
    overrides = _parse_overrides(remaining)

    cfg = load_config(path=args.config, overrides=overrides or None)
    run_training(cfg)


if __name__ == "__main__":
    main()
