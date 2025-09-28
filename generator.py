#!/usr/bin/env python3
"""Compatibility shim that invokes the fractal-diffusion-field CLI."""
from fractal_diffusion_field.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
