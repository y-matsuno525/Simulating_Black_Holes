"""Memory-light evaluation of branch profiles at explicitly requested times."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np


def compute_h_plus_profiles(config: dict, times: np.ndarray) -> np.ndarray:
    """Return vacuum-subtracted H_+ profiles at the supplied physical times."""
    from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS
    import simulation

    sample_times = np.asarray(times, dtype=float)
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
        np.exp(-1j * sample_times[:, None] * eigenvalues[None, :L])
        * psi0[:, 0][None, :]
    )
    particle = amplitudes @ eigenvectors[:L, :L].T
    hole_bra = amplitudes.conj() @ eigenvectors[:L, L:].T

    pair = np.zeros((len(sample_times), L), dtype=complex)
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

    core = -1j * pair - hopping + hopping.conj() - 1j * pair.conj()
    beta_right = np.asarray(
        [simulation.beta_for_params(b + 0.5, params) for b in range(L)],
        dtype=float,
    )
    beta_left = np.roll(beta_right, 1)
    profiles = -1j / (8 * epsilon**2) * (
        (1 + beta_left[None, :]) * np.roll(core, 1, axis=1)
        + (1 + beta_right[None, :]) * core
    )
    return np.real_if_close(profiles, tol=1000).real


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def save_profile_provenance(
    output_dir: Path,
    config: dict,
    times: np.ndarray,
    profiles: np.ndarray,
) -> None:
    """Save exact snapshots and their prepared configuration together."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([np.asarray(times, dtype=float), profiles])
    header = ",".join(["time"] + [f"j{site}" for site in range(profiles.shape[1])])
    np.savetxt(output_dir / "H_p_val.csv", data, delimiter=",", header=header, comments="")
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(config), handle, indent=2)
