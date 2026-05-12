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
    "surface_gravity_beta_width": 0.1,
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
    # Physical x position used as the threshold for the stagnation-time diagnostic.
    "stagnation_position": 5.4789375878605995,
    "fft_observables": ["H_p"],
    "fft_remove_spatial_mean": True,
    # Lattice-site fractions drawn as reference markers in density animations.
    # The last two PBC markers are legacy horizon checks for the L=300 setup.
    "animation_marker_fractions_pbc": [1/4, 3/4, 146/300, 154/300],
    # Open-boundary animations only need the main horizon-side reference marker.
    "animation_marker_fractions_open": [1/4],
    # Physical x/l fractions drawn in BdG mode-function plots as horizon markers.
    "mode_function_marker_fractions": [0.235, 0.767],
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


def beta_from_config(j, config):
    """Evaluate beta(j) from a prepared or run-local config dictionary."""
    L_cfg = config["L"]
    epsilon_cfg = config.get("epsilon", config.get("l", 2*np.pi) / L_cfg)

    if config.get("surface_gravity_beta", False):
        width = config.get("surface_gravity_beta_width", DEFAULT_CONFIG["surface_gravity_beta_width"])
        A = 1
        center = int(L_cfg / 2)
        return A * np.tanh(width * (j - center) * epsilon_cfg) + A

    sign_value = config.get("beta_sign_value")
    if sign_value is None:
        sign_value = 1 if config.get("beta_sign", "plus") == "plus" else -1

    profile = BETA_PROFILE_ALIASES.get(config.get("beta_profile", "pos"), config.get("beta_profile", "pos"))
    if profile == "flat":
        return np.zeros_like(j, dtype=float)
    if profile == "pos_horizon":
        amp = config["beta_amplitude"]
        width = config["beta_width"]
        center = config["beta_center_fraction"] * L_cfg
        return sign_value * amp * (np.tanh(3 / width * (j - center) * epsilon_cfg) + 1)
    if profile == "centered_horizon":
        amp = config["centered_beta_amplitude"]
        width = config["centered_beta_width"]
        center = config["centered_beta_center_fraction"] * L_cfg
        return sign_value * (amp * np.tanh(width * (j - center) * epsilon_cfg) + amp)
    raise ValueError("beta_profile must be 'pos', 'center', or 'flat'")


def beta_expression_from_config(config):
    """Return a human-readable beta profile expression for display."""
    if config.get("surface_gravity_beta", False):
        width = config.get("surface_gravity_beta_width", DEFAULT_CONFIG["surface_gravity_beta_width"])
        return rf"$\beta(j)=\tanh({width:g}(j-L/2)\epsilon)+1$"

    sign_value = config.get("beta_sign_value")
    if sign_value is None:
        sign_value = 1 if config.get("beta_sign", "plus") == "plus" else -1
    sign_prefix = "" if sign_value == 1 else "-"

    profile = BETA_PROFILE_ALIASES.get(config.get("beta_profile", "pos"), config.get("beta_profile", "pos"))
    if profile == "flat":
        return r"$\beta(j)=0$"
    if profile == "pos_horizon":
        amp = config["beta_amplitude"]
        width = config["beta_width"]
        center_fraction = config["beta_center_fraction"]
        return rf"$\beta(j)={sign_prefix}{amp:g}\left[\tanh\left(\frac{{3}}{{{width:g}}}(j-{center_fraction:g}L)\epsilon\right)+1\right]$"
    if profile == "centered_horizon":
        amp = config["centered_beta_amplitude"]
        width = config["centered_beta_width"]
        center_fraction = config["centered_beta_center_fraction"]
        return rf"$\beta(j)={sign_prefix}\left[{amp:g}\tanh\left({width:g}(j-{center_fraction:g}L)\epsilon\right)+{amp:g}\right]$"
    return r"$\beta(j)$: unknown profile"


def compute_horizon_positions_from_config(config, num_samples=10000):
    """Return lattice-index positions where abs(beta)=1 for a config."""
    L_cfg = config.get("L")
    if L_cfg is None:
        return []
    xs = np.linspace(0, L_cfg - 1, num_samples)
    vals = np.abs(beta_from_config(xs, config)) - 1
    positions = []
    for idx in range(len(xs) - 1):
        v0, v1 = vals[idx], vals[idx + 1]
        if v0 == 0:
            positions.append(xs[idx])
        elif v0 * v1 < 0:
            x0, x1 = xs[idx], xs[idx + 1]
            positions.append(x0 - v0 * (x1 - x0) / (v1 - v0))
    if vals[-1] == 0:
        positions.append(xs[-1])

    unique_positions = []
    for position in positions:
        if not unique_positions or abs(position - unique_positions[-1]) > 1e-3:
            unique_positions.append(float(position))
    return unique_positions


def beta_derivative_at_j(config, j_position, physical_step=0.25):
    """Return d beta / dx at a lattice-index position using centered differences."""
    L_cfg = config.get("L")
    if L_cfg is None:
        return None
    epsilon_cfg = config.get("epsilon", config.get("l", 2*np.pi) / L_cfg)
    step_j = physical_step / epsilon_cfg
    j_left = max(0, j_position - step_j)
    j_right = min(L_cfg - 1, j_position + step_j)
    if j_right == j_left:
        return None
    beta_left = beta_from_config(j_left, config)
    beta_right = beta_from_config(j_right, config)
    return float((beta_right - beta_left) / ((j_right - j_left) * epsilon_cfg))


def ideal_surface_gravity_from_config(config, horizon_positions=None):
    """Return (kappa, horizon_j) with kappa=|d beta/dx| at the relevant horizon."""
    if horizon_positions is None:
        horizon_positions = compute_horizon_positions_from_config(config)
    if not horizon_positions:
        return None, None

    chirality = config.get("chirality")
    target_beta = None
    if chirality == "chi_plus":
        target_beta = -1
    elif chirality == "chi_minus":
        target_beta = 1

    if target_beta is None:
        selected_horizon = horizon_positions[0]
    else:
        selected_horizon = min(
            horizon_positions,
            key=lambda j: abs(float(beta_from_config(j, config)) - target_beta),
        )

    beta_prime = beta_derivative_at_j(config, selected_horizon)
    if beta_prime is None:
        return None, selected_horizon
    return abs(beta_prime), selected_horizon


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
    override_notes = []

    def override(key, value, reason):
        old_value = prepared.get(key)
        if old_value != value:
            override_notes.append((key, old_value, value, reason))
        prepared[key] = value

    if prepared.get("surface_gravity_beta", False):
        override("scenario", "BH_chi_minus", "surface_gravity_beta 用の基準設定を使うため")
        override("j0_fraction", 0.48, "surface_gravity_beta では初期波束を horizon 近くに置くため")
        override("sigma_fraction", 0.003, "surface_gravity_beta では sigma 成長を見るため細い波束を使うため")

    if prepared["scenario"] is not None:
        if prepared["scenario"] not in SCENARIO_ALIASES:
            raise ValueError("scenario must be 'BH_chi_plus', 'WH_chi_plus', 'BH_chi_minus', or null")
        prepared["scenario"] = SCENARIO_ALIASES[prepared["scenario"]]
        for key, value in SCENARIO_CONFIGS[prepared["scenario"]].items():
            override(key, value, f"scenario={prepared['scenario']} が {key} を決めるため")

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
    prepared["config_override_notes"] = [
        {
            "key": key,
            "old": old_value,
            "new": new_value,
            "reason": reason,
        }
        for key, old_value, new_value, reason in override_notes
    ]
    if override_notes:
        print("[設定] config の一部を実行用に上書きしました:")
        for key, old_value, new_value, reason in override_notes:
            print(f"[設定]   {key}: {old_value!r} -> {new_value!r} / {reason}")
    if not prepared["run_name"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prepared["scenario"] is None:
            run_label = f"manual_{prepared['chirality']}_beta_{prepared['beta_sign']}"
        else:
            run_label = prepared["scenario"]
        prepared["run_name"] = f"{timestamp}_{run_label}_L{L}"
    prepared["output_dir"] = str(Path(prepared["output_base_dir"]) / prepared["run_name"])
    return prepared
