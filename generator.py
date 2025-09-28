#!/usr/bin/env python3
"""
Generate 2‑D multifractal random fields with a training‑free diffusion sampler
whose score is derived from the statistical model in:

  Warnecke, Bentkamp, Apolinario, Wilczek, Johnson
  "An ensemble of Gaussian fields with multifractal statistics for turbulence"
  arXiv:2509.19622

We implement the paper's lognormal example (Sec. 4). The structure function
and covariance used here are exactly Eqs. (4.5) and (4.6), with S = 2(C(0)-C(l))
from Eq. (2.16).

References to equations are for convenience:
  - Characteristic functional of Gaussian mixture: Eq. (2.13)
  - Increment structure function from covariance: Eq. (2.16)
  - Lognormal conditional structure function: Eq. (4.5)
  - Lognormal conditional covariance: Eq. (4.6)

This file requires only NumPy and Matplotlib.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Tuple, Optional


# -----------------------------
# Model & sampler configuration
# -----------------------------

@dataclass
class LognormalMultifractalParams:
    # Statistical parameters (see Eq. 4.5–4.6 in the paper)
    h0: float = 1.0 / 3.0       # central Hölder exponent (Kolmogorov 1941 baseline)
    sigma_h: float = 0.20       # width of singularity spectrum (tunes intermittency)
    L_int: float = 0.25         # integral scale "L" (in domain-length units)
    sigma_u: float = 1.0        # RMS of the large-scale velocity (sets overall variance)

    # Domain/grid
    nx: int = 128
    ny: int = 128
    box_size: float = 1.0       # physical box side length; grid spacing = box_size/nx etc.

    # Diffusion ODE (variance-preserving SDE probability-flow ODE)
    steps: int = 500
    beta_min: float = 0.1
    beta_max: float = 20.0
    seed: Optional[int] = 0     # random seed (set None for nondeterministic)

    # Small floor to avoid division by zero / negative FFT eigenvalues
    eig_floor: float = 1e-12

    # Output / visualization
    save_path: Optional[str] = "mf_field.png"
    show: bool = False


# -----------------------------
# Utilities
# -----------------------------

def _periodic_min_distance_coords(n: int, box: float) -> np.ndarray:
    """
    Coordinates of minimal periodic distance from origin along one axis.
    Returns shape (n,), values in [0, box/2] mirrored across the periodic boundary.
    """
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
    eig_floor: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the 2-D covariance grid C_e(r; gamma) from Eq. (4.6):
        C_e(ell; gamma) = sigma_u^2
                          - sigma_u^2 * [1 + (L/ell)^2]^(-h0 - gamma * sqrt(2*sigma_h^2 * log(1 + (L/ell)^2)))
    with the convention C_e(0; gamma) = sigma_u^2.
    Distances are computed on the periodic torus of size 'box'.
    Returns:
      C : covariance grid, shape (ny, nx)
      Lam : eigenvalues of the circulant covariance operator = real(FFT2(C))
    """
    # Minimal periodic distances along x and y from the origin
    dxs = _periodic_min_distance_coords(nx, box)
    dys = _periodic_min_distance_coords(ny, box)
    X, Y = np.meshgrid(dxs, dys, indexing="xy")
    r = np.hypot(X, Y)

    C = np.empty_like(r)
    C.fill(np.nan)

    # Handle r > 0 via Eq. (4.6)
    mask = r > 0
    rr = r[mask]
    # robust log: log(1 + (L/ell)^2)
    z = np.log1p((L_int / rr) ** 2)
    expo = -h0 - gamma * np.sqrt(2.0 * (sigma_h ** 2) * z)
    term = np.power(1.0 + (L_int / rr) ** 2, expo)
    C[mask] = (sigma_u ** 2) - (sigma_u ** 2) * term

    # By definition C(0) = sigma_u^2 (limit as ell -> 0)
    C[~mask] = sigma_u ** 2

    # Real FFT eigenvalues (diagonalization of circular convolution)
    Lam = np.real(np.fft.fft2(C))
    # Clip tiny negative values from discretization/roundoff
    Lam = np.maximum(Lam, eig_floor * np.max(Lam))

    return C, Lam


def _alpha_of_t(t: float, beta_min: float, beta_max: float) -> float:
    """
    For a VP SDE with time-varying beta(t) = beta_min + t*(beta_max - beta_min),
    alpha(t) = exp(-0.5 * ∫_0^t beta(s) ds) = exp(-0.5*(beta_min*t + 0.5*(beta_max-beta_min)*t^2)).
    """
    b0 = beta_min
    b1 = beta_max - beta_min
    integral = b0 * t + 0.5 * b1 * t * t
    return np.exp(-0.5 * integral)


def _beta_of_t(t: float, beta_min: float, beta_max: float) -> float:
    """Linear schedule beta(t)."""
    return beta_min + t * (beta_max - beta_min)


def _apply_precision_t_spatial(x: np.ndarray, Lam: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply Σ_t(γ)^{-1} to x using diagonalization in Fourier space.
    For the VP SDE, Σ_t = α^2 Σ + (1 - α^2) I, where diag(Σ) in Fourier = Lam.
    So in Fourier: (Σ_t)^{-1}(k) = 1 / (α^2 * Lam(k) + (1 - α^2)).
    """
    xhat = np.fft.fft2(x)
    denom = (alpha * alpha) * Lam + (1.0 - alpha * alpha)
    yhat = xhat / denom
    y = np.fft.ifft2(yhat).real
    return y


# -----------------------------
# Main sampler (probability‑flow ODE)
# -----------------------------

def sample_field_lognormal_diffusion(params: LognormalMultifractalParams) -> Tuple[np.ndarray, dict]:
    """
    Draw one field sample by:
      1) Sample γ ~ N(0,1) (global latent of the Gaussian mixture)
      2) Build covariance C_e(.; γ) from Eq. (4.6) and its FFT eigenvalues Lam
      3) Integrate the probability‑flow ODE (reverse‑time deterministic sampler)
         dx/dt = β(t) * [ 0.5 * x + s_t(x) ],   with s_t(x) = ∇_x log p_t(x|γ) = - Σ_t^{-1} x
         (VP SDE; see Song et al. 2021 for derivation of probability‑flow ODE)
      4) Return x(t=0), which is exactly distributed as N(0, Σ_γ) for the chosen γ.

    No training, no learned network; the score is analytic from the paper’s statistics.
    """
    p = params
    rng = np.random.default_rng(p.seed)

    # 1) mixture latent
    gamma = rng.normal()

    # 2) covariance grid and its FFT eigenvalues (Σ diagonal in Fourier)
    C_grid, Lam = _covariance_grid_from_eq46(
        nx=p.nx, ny=p.ny, box=p.box_size,
        L_int=p.L_int, h0=p.h0, sigma_h=p.sigma_h,
        sigma_u=p.sigma_u, gamma=gamma, eig_floor=p.eig_floor
    )

    # 3) probability-flow ODE integration from t=1 -> t=0 (deterministic Euler)
    # Initialize x(t=1) ~ N(0, I)
    x = rng.standard_normal((p.ny, p.nx), dtype=np.float64)

    # Time grid (uniform)
    T = 1.0
    N = int(p.steps)
    dt = T / N
    t_grid = np.linspace(T, 0.0, N + 1)  # includes both endpoints; we will use t_n at each step

    for n in range(N, 0, -1):  # n: N..1, stepping t_n -> t_{n-1}
        t = t_grid[n]
        beta_t = _beta_of_t(t, p.beta_min, p.beta_max)
        alpha_t = _alpha_of_t(t, p.beta_min, p.beta_max)

        # Analytic score for Gaussian at time t: s_t(x) = - Σ_t^{-1} x
        Sigma_inv_x = _apply_precision_t_spatial(x, Lam, alpha_t)

        # Probability‑flow ODE: x' = β(t) * (0.5 * x + s_t(x))
        drift = beta_t * (0.5 * x - Sigma_inv_x)  # note s_t(x) = -Sigma_inv_x
        x = x + drift * dt  # Euler step

    # x is the t=0 sample ~ N(0, Σ_γ) (for this γ)
    info = {
        "gamma": gamma,
        "params": asdict(p),
        "covariance_fft_eigs_minmax": (float(np.min(Lam)), float(np.max(Lam))),
        "alpha_T": _alpha_of_t(1.0, p.beta_min, p.beta_max),
    }
    return x, info


# -----------------------------
# Visualization helper
# -----------------------------

def _plot_and_save(field: np.ndarray, info: dict, path: Optional[str], show: bool):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
    ax.set_title("Multifractal field via diffusion (lognormal model)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("u(x,y)")
    if path:
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        print(f"[saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def main():
    p = LognormalMultifractalParams()
    print("Sampling with parameters:", p)

    field, info = sample_field_lognormal_diffusion(p)
    print("gamma =", info["gamma"])
    print("covariance FFT eigenvalues (min, max) =", info["covariance_fft_eigs_minmax"])
    print("alpha(T=1) =", info["alpha_T"])

    # Normalize the field to unit RMS *if* you want (commented by default).
    # field = field / np.std(field)

    _plot_and_save(field, info, p.save_path, p.show)


if __name__ == "__main__":
    main()
