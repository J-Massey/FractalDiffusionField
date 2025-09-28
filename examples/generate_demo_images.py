"""Rebuild the README demo images with consistent parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

from fractal_diffusion_field import LognormalMultifractalParams, generate_field

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")


def _case(name: str, **overrides) -> tuple[str, LognormalMultifractalParams]:
    params = LognormalMultifractalParams(
        nx=256,
        ny=256,
        steps=600,
        seed=overrides.pop("seed"),
        save_path=name,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return name, params


def build_demo_images(output_dir: Path) -> Iterable[Path]:
    """Generate the PNGs showcased in the README and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _case("demo_baseline.png", seed=42),
        _case("demo_sigmah035.png", seed=7, sigma_h=0.35),
        _case("demo_Lint01.png", seed=2024, L_int=0.10),
    ]

    saved_paths = []
    for filename, params in cases:
        params.save_path = str(output_dir / filename)
        generate_field(params, save=True, show=False)
        saved_paths.append(Path(params.save_path))
    return saved_paths


def main() -> int:
    here = Path(__file__).resolve().parent
    output_dir = (here / "../docs/assets").resolve()
    paths = build_demo_images(output_dir)
    print("Generated:")
    for path in paths:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
