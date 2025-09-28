"""Unit tests for the sampling API."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fractal_diffusion_field import (
    LognormalMultifractalParams,
    generate_field,
    sample_field_lognormal_diffusion,
)


def _test_params(**overrides) -> LognormalMultifractalParams:
    base = LognormalMultifractalParams(
        nx=32,
        ny=32,
        steps=50,
        seed=1234,
        save_path=None,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_sample_field_basic_shape_and_diagnostics():
    params = _test_params()
    field, diagnostics = sample_field_lognormal_diffusion(params)

    assert field.shape == (params.ny, params.nx)
    eig_min, eig_max = diagnostics["covariance_fft_eigs_minmax"]
    assert eig_min > 0.0
    assert eig_max >= eig_min
    assert "gamma" in diagnostics
    assert isinstance(diagnostics["gamma"], float)


def test_generate_field_skip_save(tmp_path: Path):
    params = _test_params(save_path=str(tmp_path / "field.png"))
    field, diagnostics = generate_field(params, save=False)

    assert field.shape == (params.ny, params.nx)
    assert not (tmp_path / "field.png").exists()
    assert diagnostics["params"]["nx"] == params.nx
