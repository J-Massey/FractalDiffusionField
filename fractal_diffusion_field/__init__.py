"""Generate multifractal diffusion fields from analytic statistics."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .params import LognormalMultifractalParams
from .sampling import sample_field_lognormal_diffusion
from .visualization import save_field_image

Array = np.ndarray


def generate_field(
    params: Optional[LognormalMultifractalParams] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    save: bool = True,
    show: Optional[bool] = None,
) -> Tuple[Array, Dict[str, object]]:
    """Sample a field and optionally save the visualization.

    Parameters
    ----------
    params:
        Sampler configuration. If omitted, ``LognormalMultifractalParams()`` is used.
    rng:
        Optional random number generator.
    save:
        Persist the field image to ``params.save_path`` when True.
    show:
        Override ``params.show`` when provided.
    """
    if params is None:
        params = LognormalMultifractalParams()

    field, diagnostics = sample_field_lognormal_diffusion(params, rng=rng)

    effective_show = params.show if show is None else show
    if save and params.save_path:
        save_field_image(field, params.save_path, show=effective_show)
    elif effective_show:
        save_field_image(field, None, show=True)

    return field, diagnostics


__all__ = [
    "Array",
    "LognormalMultifractalParams",
    "generate_field",
    "sample_field_lognormal_diffusion",
    "save_field_image",
]
