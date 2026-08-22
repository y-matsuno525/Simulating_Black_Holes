"""Check whether the Fig. 4 doubler peak survives removal of the right boundary."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "paper_audit" / "boundary_control.json"
TIMES = np.asarray([0.0, 3.0, 3.5, 4.0])
KD = 2.0 * np.arccos(1.0 / 1.2)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def prepared(overrides: dict) -> dict:
    from config import DEFAULT_CONFIG, deep_merge, prepare_config

    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_config(deep_merge(DEFAULT_CONFIG, overrides))


def diagnostics(profiles: np.ndarray) -> dict:
    n_sites = profiles.shape[1]
    k = 2.0 * np.pi * np.fft.rfftfreq(n_sites, d=1.0)
    spectra = np.abs(np.fft.rfft(profiles - profiles.mean(axis=1, keepdims=True), axis=1))
    low = k <= 0.35
    doubler = np.abs(k - KD) <= 0.18
    weights = np.abs(profiles)
    edge_fraction = (
        weights[:, :10].sum(axis=1) + weights[:, -10:].sum(axis=1)
    ) / weights.sum(axis=1)
    low_peak = spectra[:, low].max(axis=1)
    doubler_peak = spectra[:, doubler].max(axis=1)
    return {
        "low_peak": low_peak.tolist(),
        "doubler_peak": doubler_peak.tolist(),
        "doubler_to_low_ratio": (doubler_peak / low_peak).tolist(),
        "edge_fraction_10_sites": edge_fraction.tolist(),
    }


def main() -> None:
    from paper_figures.exact_profiles import compute_h_plus_profiles
    from paper_figures.reproduce_panel import PANEL_SPECS

    base = dict(PANEL_SPECS["fig3a"].overrides)
    extended = dict(base)
    # Keep epsilon, x0, j0, and sigma fixed while doubling the physical box.
    extended.update(
        {
            "L": 600,
            "l": 4.0 * np.pi,
            "beta_center_fraction": 1.0 / 3.0,
            "j0_fraction": 60.0 / 600.0,
            "sigma_fraction": 15.0 / 600.0,
            "run_name": "FIG4_boundary_control_L600",
        }
    )

    report = {"times": TIMES.tolist(), "kd": float(KD), "systems": {}}
    for label, overrides in (("paper_L300", base), ("extended_L600", extended)):
        print(f"[boundary-control] calculating {label}", flush=True)
        config = prepared(overrides)
        profiles = compute_h_plus_profiles(config, TIMES)
        report["systems"][label] = {
            "L": int(config["L"]),
            "ell": float(config["l"]),
            "epsilon": float(config["epsilon"]),
            "x0": float(config["beta_center_fraction"] * config["l"]),
            "j0": int(config["j0"]),
            "sigma": float(config["sigma"]),
            **diagnostics(profiles),
        }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[boundary-control] wrote {REPORT}")
    for label, result in report["systems"].items():
        print(label)
        print("  doubler/low:", ", ".join(f"{v:.4f}" for v in result["doubler_to_low_ratio"]))
        print("  edge fraction:", ", ".join(f"{v:.4f}" for v in result["edge_fraction_10_sites"]))


if __name__ == "__main__":
    main()
