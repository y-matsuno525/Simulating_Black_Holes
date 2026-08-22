"""Independently verify cached paper profiles from the current BdG model.

The production simulation constructs an L x L operator matrix for every local
observable.  This audit instead evaluates the one-quasiparticle excess of the
same fermion bilinears directly from the time-dependent mode amplitudes.  It is
therefore both memory-light and an independent check of the cached CSV data.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO / "paper_reproduction" / "runs"
REPORT_PATH = REPO / "paper_audit" / "profile_audit.json"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


RUN_OBSERVABLES = {
    "FIG2a": ("H_p",),
    "FIG2b": ("H_m",),
    "FIG2c": ("H_p",),
    "FIG2d": ("H_m",),
    "FIG3a": ("H_p",),
    "FIG3b": ("H_m",),
    "FIG3c": ("H_p",),
    "FIG3d": ("H_m",),
    "FIG5_p0": ("H_m",),
    "FIG5_p1": ("H_m",),
    "FIG6a": ("H_p",),
    "FIG7": ("H_p", "H_m", "H_pm"),
}


def load_config(run_name: str) -> dict:
    path = RUN_ROOT / run_name / "config.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_cached_profile(run_name: str, observable: str) -> tuple[np.ndarray, np.ndarray]:
    path = RUN_ROOT / run_name / f"{observable}_val.csv"
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=complex)
    return np.real(raw[:, 0]), np.real(raw[:, 1:])


def direct_profiles(config: dict, times: np.ndarray) -> dict[str, np.ndarray]:
    """Return vacuum-subtracted local densities without local operator matrices."""
    from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS
    import simulation

    with contextlib.redirect_stdout(io.StringIO()):
        simulation.configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)

    params = simulation.PARAMS
    L = int(params.L)
    epsilon = float(params.epsilon)
    h_bdg = simulation.build_bdg_matrix(params)
    eigenvalues, eigenvectors = simulation.diagonalize_bdg_matrix(h_bdg, L)
    eigenvectors = simulation.enforce_particle_hole_symmetry(eigenvectors, L)
    psi0, _ = simulation.build_initial_state(
        L,
        eigenvectors,
        int(config["j0"]),
        float(config["sigma"]),
        bool(config["PBC"]),
        int(config["initial_direction_sign"]),
    )

    amplitudes = (
        np.exp(-1j * times[:, None] * eigenvalues[None, :L])
        * psi0[:, 0][None, :]
    )
    particle = amplitudes @ eigenvectors[:L, :L].T
    hole_bra = amplitudes.conj() @ eigenvectors[:L, L:].T

    # Excess contractions on bond b=(b,b+1).  The last column is the absent
    # OBC wrap bond and remains zero.
    pair = np.zeros((len(times), L), dtype=complex)
    hopping = np.zeros_like(pair)
    pair[:, : L - 1] = (
        hole_bra[:, 1:] * particle[:, :-1]
        - hole_bra[:, :-1] * particle[:, 1:]
    )
    hopping[:, : L - 1] = (
        particle[:, 1:].conj() * particle[:, :-1]
        - hole_bra[:, :-1] * hole_bra[:, 1:].conj()
    )
    if params.PBC:
        pair[:, L - 1] = (
            hole_bra[:, 0] * particle[:, L - 1]
            - hole_bra[:, L - 1] * particle[:, 0]
        )
        hopping[:, L - 1] = (
            particle[:, 0].conj() * particle[:, L - 1]
            - hole_bra[:, L - 1] * hole_bra[:, 0].conj()
        )

    core_plus = -1j * pair - hopping + hopping.conj() - 1j * pair.conj()
    core_minus = 1j * pair - hopping + hopping.conj() + 1j * pair.conj()
    cross = -1j * hopping - 1j * hopping.conj()

    beta_right = np.asarray(
        [simulation.beta_for_params(b + 0.5, params) for b in range(L)],
        dtype=float,
    )
    beta_left = np.roll(beta_right, 1)
    plus = -1j / (8 * epsilon**2) * (
        (1 + beta_left[None, :]) * np.roll(core_plus, 1, axis=1)
        + (1 + beta_right[None, :]) * core_plus
    )
    minus = 1j / (8 * epsilon**2) * (
        (1 - beta_left[None, :]) * np.roll(core_minus, 1, axis=1)
        + (1 - beta_right[None, :]) * core_minus
    )

    delta_number = np.abs(particle) ** 2 - np.abs(hole_bra) ** 2
    interaction = (
        -1j * params.p / (4 * epsilon**2)
        * (np.roll(cross, 1, axis=1) + cross)
        + (params.p - epsilon * params.m) / epsilon**2 * delta_number
    )
    return {
        "H_p": np.real_if_close(plus, tol=1000).real,
        "H_m": np.real_if_close(minus, tol=1000).real,
        "H_pm": np.real_if_close(interaction, tol=1000).real,
    }


def compare(run_name: str, observables: tuple[str, ...]) -> dict:
    config = load_config(run_name)
    first_times, _ = load_cached_profile(run_name, observables[0])
    calculated = direct_profiles(config, first_times)
    results = {}
    for observable in observables:
        times, cached = load_cached_profile(run_name, observable)
        if not np.array_equal(times, first_times):
            raise AssertionError(f"time grid differs within {run_name}")
        fresh = calculated[observable]
        difference = fresh - cached
        scale = max(float(np.max(np.abs(cached))), np.finfo(float).tiny)
        results[observable] = {
            "shape": list(cached.shape),
            "first_time": float(times[0]),
            "last_time": float(times[-1]),
            "max_abs_error": float(np.max(np.abs(difference))),
            "max_relative_to_peak": float(np.max(np.abs(difference)) / scale),
            "rms_error": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        }
    del calculated
    gc.collect()
    return {
        "effective_settings": {
            "L": int(config["L"]),
            "ell": float(config["l"]),
            "p": float(config["p"]),
            "m": float(config["m"]),
            "beta_profile": config["beta_profile"],
            "beta_sign": config["beta_sign"],
            "beta_amplitude": float(config["beta_amplitude"]),
            "beta_width": float(config["beta_width"]),
            "beta_center_fraction": float(config["beta_center_fraction"]),
            "centered_beta_amplitude": float(config["centered_beta_amplitude"]),
            "centered_beta_width": float(config["centered_beta_width"]),
            "centered_beta_center_fraction": float(config["centered_beta_center_fraction"]),
            "j0": int(config["j0"]),
            "sigma": float(config["sigma"]),
            "chirality": config["chirality"],
            "initial_direction": config["initial_direction"],
            "dt": float(config["dt"]),
        },
        "observables": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", choices=sorted(RUN_OBSERVABLES))
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    selected = args.runs or list(RUN_OBSERVABLES)
    report = {}
    for run_name in selected:
        print(f"[profile-audit] {run_name}", flush=True)
        report[run_name] = compare(run_name, RUN_OBSERVABLES[run_name])
        for observable, result in report[run_name]["observables"].items():
            print(
                f"  {observable}: max_abs={result['max_abs_error']:.3e}, "
                f"relative={result['max_relative_to_peak']:.3e}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[profile-audit] wrote {args.output}")


if __name__ == "__main__":
    main()
