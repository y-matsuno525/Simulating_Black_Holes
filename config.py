import json
from pathlib import Path
from datetime import datetime

import numpy as np

# CONFIG では短い別名も使えるようにしておき、計算本体では canonical な名前だけを扱う。
# 例: "pos" と "pos_horizon" は同じ beta profile として扱われる。
BETA_PROFILE_ALIASES = {
    "pos": "pos_horizon",
    "pos_horizon": "pos_horizon",
    "center": "centered_horizon",
    "centered_horizon": "centered_horizon",
    "flat": "flat",
}

# chirality と beta_sign だけをユーザーに見せ、古い lr/ur 名は内部からも外す。
CHIRALITY_ALIASES = {
    "chi+": "chi_plus",
    "chi_plus": "chi_plus",
    "plus": "chi_plus",
    "+": "chi_plus",
    "chi-": "chi_minus",
    "chi_minus": "chi_minus",
    "minus": "chi_minus",
    "-": "chi_minus",
}

BETA_SIGN_ALIASES = {
    "+": "plus",
    "plus": "plus",
    "positive": "plus",
    "beta_plus": "plus",
    "-": "minus",
    "minus": "minus",
    "negative": "minus",
    "beta_minus": "minus",
}

SCENARIO_ALIASES = {
    "BH_chi_plus": "BH_chi_plus",
    "bh_chi_plus": "BH_chi_plus",
    "bh_chi+": "BH_chi_plus",
    "BH_chi+": "BH_chi_plus",
    "WH_chi_plus": "WH_chi_plus",
    "wh_chi_plus": "WH_chi_plus",
    "wh_chi+": "WH_chi_plus",
    "WH_chi+": "WH_chi_plus",
    "BH_chi_minus": "BH_chi_minus",
    "bh_chi_minus": "BH_chi_minus",
    "bh_chi-": "BH_chi_minus",
    "BH_chi-": "BH_chi_minus",
}

SCENARIO_CONFIGS = {
    "BH_chi_plus": {
        "chirality": "chi_plus",
        "beta_sign": "minus",
    },
    "WH_chi_plus": {
        "chirality": "chi_plus",
        "beta_sign": "plus",
    },
    "BH_chi_minus": {
        "chirality": "chi_minus",
        "beta_sign": "plus",
    },
}

DEFAULT_CONFIG = {
    # --- lattice / physical parameters ---
    "L": 100,
    "l": 2*np.pi,
    "p": 1,
    "m": 0.0001,
    "scenario": "WH_chi_plus",
    "chirality": "chi_plus",
    "beta_sign": "plus",
    "t_i": 0,
    "t_f": 5,
    "dt_scale": 0.01,
    "PBC": False,

    # --- beta profile parameters ---
    "beta_profile": "pos",
    "surface_gravity_beta": False,
    "beta_width": 1,
    "beta_amplitude": 0.6,
    "beta_center_fraction": 2/3,
    "centered_beta_width": 1,
    "centered_beta_amplitude": 1,
    "centered_beta_center_fraction": 1/2,

    # --- initial wave-packet parameters ---
    "j0_fraction": 0.25,
    "sigma_fraction": 0.05,
    "initial_direction": "right",

    # --- optional diagnostics kept from the older scripts ---
    "mode_function_count": 10,
    "geodesic_points": 500,
    "surface_gravity_output_path": "figures/surface_gravity_fit.png",
    "stagnation_position": 5.4789375878605995,
    "fft_observables": ["H_p"],
    "fft_remove_spatial_mean": True,
    "output_base_dir": "outputs",
    "run_name": None,

    # Heavy output switches.  These are deliberately separate from the numerical
    # parameters so exploratory runs can skip slow plots without changing physics.
    "outputs": {
        "show_beta_profile": True,
        "heatmaps": True,
        "gifs": True,
        "mode_functions": True,
        "geodesic": True,
        "surface_gravity_fit": True,
        "fft": True,
    },
}

DENSITY_PLOT_CONFIGS = {
    "H_p": {
        "output_path": "figures/H_p.png",
        "colorbar_label": r"$H_j^+$",
        "geodesic_time_scale": "t_f",
        "geodesic_color": "cyan",
        "geodesic_linestyle": "--",
        "geodesic_linewidth": 1,
    },
    "H_m": {
        "output_path": "figures/H_m.png",
        "colorbar_label": r"$H_j^-$",
        "geodesic_time_scale": 1.21,
        "geodesic_color": "cyan",
        "geodesic_linestyle": "--",
    },
    "c_dag_c": {
        "output_path": "figures/c_dag_c.png",
        "colorbar_label": r"$c_j^\dagger c_j$",
        "geodesic_time_scale": 20.0,
        "geodesic_color": "cyan",
        "geodesic_linestyle": "--",
    },
    "H_pm": {
        "output_path": "figures/H_pm.png",
        "colorbar_label": r"$H_j^{+-}$",
        "geodesic_time_scale": 20.0,
        "geodesic_color": "cyan",
        "geodesic_linestyle": "--",
    },
}

ANIMATION_CONFIGS = {
    "H_p": {
        "gif_path": "figures/H_p.gif",
        "cmap_line": "blue",
    },
    "H_m": {
        "gif_path": "figures/H_m.gif",
        "cmap_line": "orange",
    },
    "H_pm": {
        "gif_path": "figures/H_pm.gif",
        "cmap_line": "green",
    },
}


def deep_merge(base, overrides):
    """Return base recursively updated with override values."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path="config.json"):
    """Load user-facing config from JSON and derive numerical constants."""
    config_path = Path(path)
    if config_path.exists():
        overrides = json.loads(config_path.read_text())
        config = deep_merge(DEFAULT_CONFIG, overrides)
    else:
        config = dict(DEFAULT_CONFIG)
    return prepare_config(config)


def prepare_config(config):
    """Validate config values and derive constants used by the numerical pipeline."""
    prepared = dict(config)

    if prepared.get("surface_gravity_beta", False):
        prepared["scenario"] = "BH_chi_minus"
        prepared["j0_fraction"] = 0.45
        prepared["sigma_fraction"] = 0.003

    if prepared["scenario"] is not None:
        if prepared["scenario"] not in SCENARIO_ALIASES:
            raise ValueError("scenario must be 'BH_chi_plus', 'WH_chi_plus', 'BH_chi_minus', or null")
        prepared["scenario"] = SCENARIO_ALIASES[prepared["scenario"]]
        prepared.update(SCENARIO_CONFIGS[prepared["scenario"]])

    if prepared["chirality"] not in CHIRALITY_ALIASES:
        raise ValueError("chirality must be 'chi_plus' or 'chi_minus'")
    prepared["chirality"] = CHIRALITY_ALIASES[prepared["chirality"]]

    if prepared["beta_sign"] not in BETA_SIGN_ALIASES:
        raise ValueError("beta_sign must be 'plus' or 'minus'")
    prepared["beta_sign"] = BETA_SIGN_ALIASES[prepared["beta_sign"]]
    prepared["beta_sign_value"] = 1 if prepared["beta_sign"] == "plus" else -1

    if prepared["beta_profile"] not in BETA_PROFILE_ALIASES:
        raise ValueError("beta_profile must be 'pos', 'center', or 'flat'")
    prepared["beta_profile"] = BETA_PROFILE_ALIASES[prepared["beta_profile"]]

    direction_sign_by_name = {
        "right": 1,
        "left": -1,
    }
    if prepared["initial_direction"] not in direction_sign_by_name:
        raise ValueError("initial_direction must be 'right' or 'left'")
    # chirality が解決済みであれば chi_plus → 右向き / chi_minus → 左向きを優先する。
    # manual モード (scenario=None) では initial_direction をそのまま使う。
    if prepared["scenario"] is not None:
        prepared["initial_direction_sign"] = 1 if prepared["chirality"] == "chi_plus" else -1
    else:
        prepared["initial_direction_sign"] = direction_sign_by_name[prepared["initial_direction"]]

    L = prepared["L"]
    prepared["epsilon"] = prepared["l"] / L
    prepared["dt"] = prepared["dt_scale"]*(300/L)
    prepared["times"] = np.arange(prepared["t_i"] + prepared["dt"], prepared["t_f"], prepared["dt"])
    prepared["sigma"] = prepared["sigma_fraction"]*L
    prepared["j0"] = int(prepared["j0_fraction"]*L)
    if not prepared["run_name"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prepared["scenario"] is None:
            run_label = f"manual_{prepared['chirality']}_beta_{prepared['beta_sign']}"
        else:
            run_label = prepared["scenario"]
        prepared["run_name"] = f"{timestamp}_{run_label}_L{L}"
    prepared["output_dir"] = str(Path(prepared["output_base_dir"]) / prepared["run_name"])
    return prepared
