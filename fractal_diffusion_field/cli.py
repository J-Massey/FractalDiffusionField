"""Command line interface for the fractal-diffusion-field package."""
from __future__ import annotations

import argparse
import json
from typing import Iterable, Optional

from .params import LognormalMultifractalParams
from .sampling import sample_field_lognormal_diffusion
from .visualization import save_field_image

DEFAULT_PARAMS = LognormalMultifractalParams()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractal-diffusion-field",
        description="Generate multifractal fields from analytic lognormal statistics.",
    )
    parser.add_argument("--nx", type=int, default=DEFAULT_PARAMS.nx)
    parser.add_argument("--ny", type=int, default=DEFAULT_PARAMS.ny)
    parser.add_argument("--box-size", type=float, default=DEFAULT_PARAMS.box_size)
    parser.add_argument("--h0", type=float, default=DEFAULT_PARAMS.h0)
    parser.add_argument("--sigma-h", type=float, default=DEFAULT_PARAMS.sigma_h)
    parser.add_argument("--L-int", type=float, default=DEFAULT_PARAMS.L_int)
    parser.add_argument("--sigma-u", type=float, default=DEFAULT_PARAMS.sigma_u)
    parser.add_argument("--steps", type=int, default=DEFAULT_PARAMS.steps)
    parser.add_argument("--beta-min", type=float, default=DEFAULT_PARAMS.beta_min)
    parser.add_argument("--beta-max", type=float, default=DEFAULT_PARAMS.beta_max)
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAMS.seed or 0, help="Random seed (set -1 for nondeterministic RNG)")
    parser.add_argument("--eig-floor", type=float, default=DEFAULT_PARAMS.eig_floor)
    parser.add_argument("--output", type=str, default=DEFAULT_PARAMS.save_path, help="Output image path")
    parser.add_argument("--no-save", action="store_true", help="Skip writing image to disk")
    parser.add_argument("--show", action="store_true", help="Display the sampled field interactively")
    parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON")
    return parser


def _namespace_to_params(ns: argparse.Namespace) -> LognormalMultifractalParams:
    seed: Optional[int]
    if ns.seed == -1:
        seed = None
    else:
        seed = int(ns.seed)
    return LognormalMultifractalParams(
        nx=ns.nx,
        ny=ns.ny,
        box_size=ns.box_size,
        h0=ns.h0,
        sigma_h=ns.sigma_h,
        L_int=ns.L_int,
        sigma_u=ns.sigma_u,
        steps=ns.steps,
        beta_min=ns.beta_min,
        beta_max=ns.beta_max,
        seed=seed,
        eig_floor=ns.eig_floor,
        save_path=None if ns.no_save else ns.output,
        show=ns.show,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    params = _namespace_to_params(ns)

    field, diagnostics = sample_field_lognormal_diffusion(params)
    saved_path = None
    if params.save_path or params.show:
        saved_path = save_field_image(field, params.save_path, show=params.show)

    if ns.json:
        payload = diagnostics.copy()
        payload["saved_path"] = str(saved_path) if saved_path else None
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"gamma = {diagnostics['gamma']:.5f}")
        eig_min, eig_max = diagnostics["covariance_fft_eigs_minmax"]
        print(f"covariance FFT eigenvalues (min, max) = ({eig_min:.3e}, {eig_max:.3e})")
        print(f"alpha(T=1) = {diagnostics['alpha_T']:.6f}")
        if saved_path:
            print(f"saved image -> {saved_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
