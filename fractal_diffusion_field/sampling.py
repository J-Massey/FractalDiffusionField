"""Core sampling routines for multifractal diffusion fields."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np

from .params import LognormalMultifractalParams


Array = np.ndarray


def _periodic_min_distance_coords(n: int, box: float) -> Array:
    """Coordinates of minimal periodic distance from the origin along one axis."""
    dx = box / n
    x = np.arange(n) * dx
    return np.minimum(x, box - x)


def _covariance_grid_from_eq46(
    nx: int,
    ny: int,
    box: float,
    L_int: float,
    h0: float,
    sigma_h: float,
    sigma_u: float,
    gamma: float,
    eig_floor: float,
) -> Tuple[Array, Array]:
    """Return the lognormal covariance grid and its FFT eigenvalues (Eq. 4.6)."""
    dxs = _periodic_min_distance_coords(nx, box)
    dys = _periodic_min_distance_coords(ny, box)
    X, Y = np.meshgrid(dxs, dys, indexing="xy")
    r = np.hypot(X, Y)

    C = np.empty_like(r)
    mask = r > 0
    rr = r[mask]

    z = np.log1p((L_int / rr) ** 2)
    expo = -h0 - gamma * np.sqrt(2.0 * (sigma_h ** 2) * z)
    term = np.power(1.0 + (L_int / rr) ** 2, expo)
    C[mask] = (sigma_u ** 2) - (sigma_u ** 2) * term
    C[~mask] = sigma_u ** 2

    Lam = np.real(np.fft.fft2(C))
    Lam = np.maximum(Lam, eig_floor * np.max(Lam))
    return C, Lam


def _alpha_of_t(t: float, beta_min: float, beta_max: float) -> float:
    """Variance-preserving SDE coefficient alpha(t)."""
    b0 = beta_min
    b1 = beta_max - beta_min
    integral = b0 * t + 0.5 * b1 * t * t
    return float(np.exp(-0.5 * integral))


def _beta_of_t(t: float, beta_min: float, beta_max: float) -> float:
    """Linear beta(t) schedule."""
    return float(beta_min + t * (beta_max - beta_min))


def _apply_precision_t_spatial(x: Array, Lam: Array, alpha: float) -> Array:
    """Apply the time-dependent precision matrix using FFT diagonalization."""
    xhat = np.fft.fft2(x)
    denom = (alpha * alpha) * Lam + (1.0 - alpha * alpha)
    yhat = xhat / denom
    return np.fft.ifft2(yhat).real


def sample_field_lognormal_diffusion(
    params: LognormalMultifractalParams,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Dict[str, object]]:
    """Draw a multifractal field from the analytic lognormal model.

    Parameters
    ----------
    params:
        Structured configuration for the sampler.
    rng:
        Optional NumPy random generator. If omitted, ``np.random.default_rng``
        is constructed from ``params.seed``.

    Returns
    -------
    field, diagnostics
        ``field`` is a ``(ny, nx)`` array sampled from the target covariance.
        ``diagnostics`` contains metadata about the draw and sampler state.
    """
    p = params
    if rng is None:
        rng = np.random.default_rng(p.seed)

    gamma = float(rng.normal())

    _, Lam = _covariance_grid_from_eq46(
        nx=p.nx,
        ny=p.ny,
        box=p.box_size,
        L_int=p.L_int,
        h0=p.h0,
        sigma_h=p.sigma_h,
        sigma_u=p.sigma_u,
        gamma=gamma,
        eig_floor=p.eig_floor,
    )

    x = rng.standard_normal((p.ny, p.nx), dtype=np.float64)

    T = 1.0
    N = int(p.steps)
    dt = T / N if N > 0 else 1.0
    t_grid = np.linspace(T, 0.0, N + 1)

    for n in range(N, 0, -1):
        t = t_grid[n]
        beta_t = _beta_of_t(t, p.beta_min, p.beta_max)
        alpha_t = _alpha_of_t(t, p.beta_min, p.beta_max)
        Sigma_inv_x = _apply_precision_t_spatial(x, Lam, alpha_t)
        drift = beta_t * (0.5 * x - Sigma_inv_x)
        x = x + drift * dt

    diagnostics: Dict[str, object] = {
        "gamma": gamma,
        "params": asdict(p),
        "covariance_fft_eigs_minmax": (
            float(np.min(Lam)),
            float(np.max(Lam)),
        ),
        "alpha_T": _alpha_of_t(1.0, p.beta_min, p.beta_max),
    }
    return x, diagnostics


__all__ = [
    "LognormalMultifractalParams",
    "sample_field_lognormal_diffusion",
]
