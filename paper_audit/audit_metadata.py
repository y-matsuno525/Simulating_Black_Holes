"""Audit manuscript settings, trajectories, conservation, fits, and FFT times."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


REPO = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO / "paper_reproduction" / "runs"
REPORT_PATH = REPO / "paper_audit" / "metadata_audit.json"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import beta_from_config  # noqa: E402
from paper_audit.audit_profiles import direct_profiles, load_config  # noqa: E402
from paper_figures.make_doubler_fft import doubler_k  # noqa: E402
from paper_figures.measure_fig6b_scaling import (  # noqa: E402
    CURVATURE_THRESHOLD,
    WH_A,
    WH_B,
    WH_X0,
    beta_second_derivative,
    central_curvature_boundary,
    load_results as load_fig6b_results,
)


DYNAMICS_RUNS = (
    "FIG2a", "FIG2b", "FIG2c", "FIG2d",
    "FIG3a", "FIG3b", "FIG3c", "FIG3d",
    "FIG6a", "FIG7",
)


def effective_beta_parameters(config: dict) -> tuple[float, float, float, float]:
    sign = 1.0 if config["beta_sign"] == "plus" else -1.0
    if config["beta_profile"] == "pos_horizon":
        amplitude = sign * float(config["beta_amplitude"])
        return (
            amplitude,
            3.0 / float(config["beta_width"]),
            amplitude,
            float(config["beta_center_fraction"]) * float(config["l"]),
        )
    if config["beta_profile"] == "centered_horizon":
        amplitude = sign * float(config["centered_beta_amplitude"])
        return (
            amplitude,
            float(config["centered_beta_width"]),
            amplitude,
            float(config["centered_beta_center_fraction"]) * float(config["l"]),
        )
    raise ValueError(config["beta_profile"])


def solve_horizon(config: dict, target: float) -> tuple[float, float, float] | None:
    A, B, C, x0 = effective_beta_parameters(config)
    argument = (target - C) / A
    if not -1.0 < argument < 1.0:
        return None
    x_h = x0 + np.arctanh(argument) / B
    j_h = x_h / float(config["epsilon"])
    derivative = A * B / np.cosh(B * (x_h - x0)) ** 2
    return float(x_h), float(j_h), float(abs(derivative))


def analytic_horizon(config: dict) -> tuple[float, float, float]:
    metric_target = 1.0 if config["beta_sign"] == "plus" else -1.0
    result = solve_horizon(config, metric_target)
    if result is None:
        raise AssertionError("configured black/white-hole profile has no metric horizon")
    return result


def load_profile(run_name: str, observable: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(
        RUN_ROOT / run_name / f"{observable}_val.csv",
        delimiter=",",
        skiprows=1,
        dtype=complex,
    )
    return np.real(raw[:, 0]), np.real(raw[:, 1:])


def geodesic_check(run_name: str, config: dict) -> dict:
    stored = np.loadtxt(RUN_ROOT / run_name / "geodesic.dat", delimiter=",")
    x_stored, times = stored[:, 0], stored[:, 1]

    def velocity(_time, x):
        j = x[0] / float(config["epsilon"])
        return [float(beta_from_config(j, config) + config["initial_direction_sign"])]

    reference = solve_ivp(
        velocity,
        (float(times[0]), float(times[-1])),
        [float(config["j0"]) * float(config["epsilon"])],
        dense_output=True,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.01,
    ).sol(times)[0]
    error = reference - x_stored
    return {
        "first_time": float(times[0]),
        "last_time": float(times[-1]),
        "initial_j": float(x_stored[0] / config["epsilon"]),
        "last_x": float(x_stored[-1]),
        "max_path_error_x": float(np.max(np.abs(error))),
        "max_path_error_sites": float(np.max(np.abs(error)) / config["epsilon"]),
    }


def conservation_and_boundary_check(run_name: str, config: dict) -> dict:
    arrays = {}
    times = None
    for observable in ("H_p", "H_m", "H_pm"):
        obs_times, arrays[observable] = load_profile(run_name, observable)
        if times is None:
            times = obs_times
        elif not np.array_equal(times, obs_times):
            raise AssertionError(f"time grids differ for {run_name}")

    integrated = float(config["epsilon"]) * sum(arrays.values()).sum(axis=1)
    energy_scale = max(float(np.max(np.abs(integrated))), np.finfo(float).tiny)
    branch = "H_p" if config["chirality"] == "chi_plus" else "H_m"
    weights = np.abs(arrays[branch])
    edge_fraction = (
        weights[:, :10].sum(axis=1) + weights[:, -10:].sum(axis=1)
    ) / weights.sum(axis=1)

    boundary_times = {}
    for threshold in (0.01, 0.05, 0.10):
        indices = np.flatnonzero(edge_fraction > threshold)
        boundary_times[f"first_time_edge_fraction_gt_{threshold:g}"] = (
            float(times[indices[0]]) if len(indices) else None
        )
    return {
        "integrated_energy_mean": float(np.mean(integrated)),
        "relative_peak_to_peak_energy_drift": float(
            (integrated.max() - integrated.min()) / energy_scale
        ),
        "selected_branch": branch,
        "final_edge_fraction_10_sites_each_side": float(edge_fraction[-1]),
        **boundary_times,
    }


def run_checks() -> dict:
    report = {}
    for run_name in DYNAMICS_RUNS:
        config = load_config(run_name)
        x_h, j_h, kappa = analytic_horizon(config)
        stored_horizon = np.atleast_2d(
            np.loadtxt(RUN_ROOT / run_name / "horizon_positions.txt", skiprows=1)
        )[0]
        report[run_name] = {
            "effective_beta_A_B_C_x0": list(effective_beta_parameters(config)),
            "analytic_horizon": {"x_h": x_h, "j_h": j_h, "kappa": kappa},
            "selected_branch_horizon": solve_horizon(
                config,
                -1.0 if config["chirality"] == "chi_plus" else 1.0,
            ),
            "stored_horizon": {"j_h": float(stored_horizon[0]), "x_h": float(stored_horizon[1])},
            "horizon_error_sites": float(stored_horizon[0] - j_h),
            "geodesic": geodesic_check(run_name, config),
            "conservation_and_boundary": conservation_and_boundary_check(run_name, config),
        }
    return report


def fit_surface_gravity() -> dict:
    def model(t, amplitude, rate):
        return amplitude * (np.exp(rate * t) - 1.0)

    report = {}
    for suffix in ("", "_pre_density_fix"):
        version = "current_density" if not suffix else "pre_density_fix"
        report[version] = {}
        for p in (0, 1):
            path = REPO / "paper_data" / f"H_m_sigmas_p{p}{suffix}.txt"
            data = np.loadtxt(path)
            times = data[:, 0]
            physical_width = data[:, 1] * (2 * np.pi / 500)
            parameters, _ = curve_fit(
                model,
                times,
                physical_width,
                p0=[0.01, 1.0],
                maxfev=10000,
            )
            residual = physical_width - model(times, *parameters)
            r_squared = 1.0 - np.sum(residual**2) / np.sum(
                (physical_width - physical_width.mean()) ** 2
            )
            report[version][f"p{p}"] = {
                "amplitude": float(parameters[0]),
                "kappa_fit": float(parameters[1]),
                "relative_error_percent": float(abs(parameters[1] - 1.0) * 100),
                "r_squared": float(r_squared),
            }
    return report


def fft_time_check() -> dict:
    config = load_config("FIG3a")
    requested_times = np.asarray([0.0, 3.0, 3.5, 4.0])
    exact_profiles = direct_profiles(config, requested_times)["H_p"]

    def normalized_fft(profiles):
        spectra = np.asarray(
            [np.abs(np.fft.rfft(row - row.mean())) for row in profiles]
        )
        return spectra / spectra.max()

    exact = normalized_fft(exact_profiles)
    cached_times, cached = load_profile("FIG3a", "H_p")
    dt = float(config["dt"])
    legacy_indices = [
        min(int(round(target / dt)), len(cached_times) - 1)
        for target in requested_times
    ]
    legacy_times = cached_times[legacy_indices]
    legacy = normalized_fft(cached[legacy_indices])
    k = 2 * np.pi * np.fft.rfftfreq(config["L"])
    kd = doubler_k()
    kd_index = int(np.argmin(np.abs(k - kd)))
    return {
        "requested_times": requested_times.tolist(),
        "legacy_selected_times": legacy_times.tolist(),
        "k_d_analytic": float(kd),
        "k_d_fft_grid": float(k[kd_index]),
        "max_spectrum_difference_by_time": [
            float(value) for value in np.max(np.abs(exact - legacy), axis=1)
        ],
        "exact_normalized_amplitude_at_kd": [
            float(value) for value in exact[:, kd_index]
        ],
        "legacy_normalized_amplitude_at_kd": [
            float(value) for value in legacy[:, kd_index]
        ],
    }


def fig6b_check() -> dict:
    Ls, log_l, t_lat = load_fig6b_results()
    slope, intercept = np.polyfit(log_l, t_lat, 1)
    prediction = slope * log_l + intercept
    r_squared = 1.0 - np.sum((t_lat - prediction) ** 2) / np.sum(
        (t_lat - t_lat.mean()) ** 2
    )
    boundary = central_curvature_boundary()
    x_h = WH_X0 + np.arctanh(2 / 3) / WH_B
    return {
        "L_values": Ls.astype(int).tolist(),
        "T_lat": t_lat.tolist(),
        "linear_fit_slope": float(slope),
        "linear_fit_intercept": float(intercept),
        "linear_fit_r_squared": float(r_squared),
        "curvature_threshold": CURVATURE_THRESHOLD,
        "t_in_spatial_boundary": float(boundary),
        "beta_inflection_x0": float(WH_X0),
        "white_hole_horizon_x": float(x_h),
        "abs_beta_second_at_boundary": float(abs(beta_second_derivative(np.asarray(boundary)))),
        "abs_beta_second_at_horizon": float(abs(beta_second_derivative(np.asarray(x_h)))),
    }


def main() -> None:
    report = {
        "dynamics": run_checks(),
        "surface_gravity": fit_surface_gravity(),
        "fig4_fft_times": fft_time_check(),
        "fig6b": fig6b_check(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[metadata-audit] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
