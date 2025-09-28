"""Configuration objects for fractal diffusion field sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class LognormalMultifractalParams:
    """Parameters for the lognormal multifractal sampler.

    The defaults reproduce the lognormal turbulence model from
    Warnecke et al. (2024), "An ensemble of Gaussian fields with multifractal
    statistics for turbulence" (arXiv:2509.19622, Sec. 4).
    """

    # Statistical parameters (Eqs. 4.5-4.6)
    h0: float = 1.0 / 3.0
    sigma_h: float = 0.20
    L_int: float = 0.25
    sigma_u: float = 1.0

    # Domain/grid resolution
    nx: int = 128
    ny: int = 128
    box_size: float = 1.0

    # Diffusion probability-flow ODE schedule
    steps: int = 500
    beta_min: float = 0.1
    beta_max: float = 20.0
    seed: Optional[int] = 0

    # Numerical stabilizers
    eig_floor: float = 1e-12

    # Output options
    save_path: Optional[str] = "mf_field.png"
    show: bool = False

    def as_dict(self) -> dict:
        """Return a JSON-serializable parameter dictionary."""
        return {
            "h0": self.h0,
            "sigma_h": self.sigma_h,
            "L_int": self.L_int,
            "sigma_u": self.sigma_u,
            "nx": self.nx,
            "ny": self.ny,
            "box_size": self.box_size,
            "steps": self.steps,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "seed": self.seed,
            "eig_floor": self.eig_floor,
            "save_path": self.save_path,
            "show": self.show,
        }
