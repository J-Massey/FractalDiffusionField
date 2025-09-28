"""Visualization helpers for fractal diffusion fields."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .sampling import Array


def save_field_image(
    field: Array,
    path: Optional[str],
    *,
    show: bool = False,
    cmap: str = "viridis",
) -> Optional[Path]:
    """Save the sampled field to disk and optionally display it interactively."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1], interpolation="nearest", cmap=cmap)
    ax.set_title("Multifractal field via diffusion (lognormal model)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("u(x, y)")

    saved_path: Optional[Path] = None
    if path:
        saved_path = Path(path).expanduser().resolve()
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(saved_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return saved_path


__all__ = ["save_field_image"]
