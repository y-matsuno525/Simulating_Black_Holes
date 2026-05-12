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

# horizon_case は物理的な呼び方、pos は既存コードで使ってきた幾何配置の短縮名。
# lr: left-to-right 側の chi_+ / white-hole case, ur: upper/right 側の chi_- / black-hole case。
HORIZON_CASE_ALIASES = {
    "chi_plus_wh": "lr",
    "p_wh": "lr",
    "lr": "lr",
    "chi_minus_bh": "ur",
    "m_bh": "ur",
    "ur": "ur",
}

DEFAULT_CONFIG = {
    # --- lattice / physical parameters ---
    "L": 100,
    "l": 2*np.pi,
    "p": 1,
    "m": 0.0001,
    "horizon_case": "chi_plus_wh",
    "t_i": 0,
    "t_f": 5,
    "dt_scale": 0.01,
    "PBC": False,

    # --- beta profile parameters ---
    "beta_profile": "pos",
    "beta_width": 1,
    "beta_amplitude": 0.6,
    "beta_center_fraction": 2/3,
    "centered_beta_width": 1,
    "centered_beta_amplitude": 1,
    "centered_beta_center_fraction": 1/2,

    # --- initial wave-packet parameters ---
    "j0_by_case": {
        "chi_plus_wh": 25,
        "chi_minus_bh": 25,
    },
    "sigma_fraction": 0.05,
    "initial_direction": "right",

    # --- optional diagnostics kept from the older scripts ---
    "mode_function_count": 10,
    "geodesic_x_start_fraction": 0.75,
    "geodesic_x_end_offset": 0.4,
    "geodesic_t_start": 0.0,
    "geodesic_points": 500,
    "surface_gravity_output_path": "figures/surface_gravity_fit.png",
    "stagnation_position": 5.4789375878605995,
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
    },
}

DENSITY_PLOT_CONFIGS = {
    "H_p": {
        "output_path": "figures/H_p.png",
        "geodesic_time_scale": "t_f",
        "geodesic_color": "red",
        "geodesic_linestyle": "--",
        "geodesic_linewidth": 1,
    },
    "H_m": {
        "output_path": "figures/H_m.png",
        "geodesic_time_scale": 1.21,
    },
    "c_dag_c": {
        "output_path": "figures/c_dag_c.png",
        "geodesic_time_scale": 20.0,
    },
    "H_pm": {
        "output_path": "figures/H_pm.png",
        "geodesic_time_scale": 20.0,
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

    if prepared["horizon_case"] not in HORIZON_CASE_ALIASES:
        raise ValueError("horizon_case must be 'chi_plus_wh' or 'chi_minus_bh'")
    prepared["pos"] = HORIZON_CASE_ALIASES[prepared["horizon_case"]]
    if prepared["horizon_case"] in prepared["j0_by_case"]:
        j0_case = prepared["horizon_case"]
    elif prepared["pos"] == "lr":
        j0_case = "chi_plus_wh"
    else:
        j0_case = "chi_minus_bh"

    if prepared["beta_profile"] not in BETA_PROFILE_ALIASES:
        raise ValueError("beta_profile must be 'pos', 'center', or 'flat'")
    prepared["beta_profile"] = BETA_PROFILE_ALIASES[prepared["beta_profile"]]

    direction_sign_by_name = {
        "right": 1,
        "left": -1,
    }
    if prepared["initial_direction"] not in direction_sign_by_name:
        raise ValueError("initial_direction must be 'right' or 'left'")
    prepared["initial_direction_sign"] = direction_sign_by_name[prepared["initial_direction"]]

    L = prepared["L"]
    prepared["epsilon"] = prepared["l"] / L
    prepared["dt"] = prepared["dt_scale"]*(300/L)
    prepared["times"] = np.arange(prepared["t_i"] + prepared["dt"], prepared["t_f"], prepared["dt"])
    prepared["sigma"] = prepared["sigma_fraction"]*L
    prepared["j0"] = prepared["j0_by_case"][j0_case]
    if not prepared["run_name"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prepared["run_name"] = f"{timestamp}_{prepared['horizon_case']}_L{L}"
    prepared["output_dir"] = str(Path(prepared["output_base_dir"]) / prepared["run_name"])
    return prepared
