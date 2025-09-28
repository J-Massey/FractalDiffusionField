"""CLI smoke tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_json_output(tmp_path: Path):
    output_path = tmp_path / "field.png"
    cmd = [
        sys.executable,
        "-m",
        "fractal_diffusion_field.cli",
        "--nx",
        "8",
        "--ny",
        "8",
        "--steps",
        "10",
        "--no-save",
        "--output",
        str(output_path),
        "--json",
    ]

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    project_root = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(project_root), existing]))

    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(completed.stdout)
    assert payload["saved_path"] is None
    eig_min, eig_max = payload["covariance_fft_eigs_minmax"]
    assert eig_min > 0.0
    assert eig_max >= eig_min
    assert payload["params"]["nx"] == 8
