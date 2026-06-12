"""Measure Fig. 6(b) stagnation-time scaling by running the lattice model.

This is the calculation-based counterpart to ``make_fig6b_scaling.py``, which
plots digitized manuscript points.  The default run list is capped at L=500 to
avoid the expensive L=800 calculation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .common import DT_DEFAULT, ELL_DEFAULT, REPO
    from .reproduce_panel import PANEL_SPECS, draw_panel
    from .style import configure_figure5_style
except ImportError:
    from common import DT_DEFAULT, ELL_DEFAULT, REPO
    from reproduce_panel import PANEL_SPECS, draw_panel
    from style import configure_figure5_style


if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_ROOT = REPO / "paper_reproduction" / "runs" / "FIG6b_measured"
DATA_PATH = REPO / "paper_data" / "logL" / "T_lat_measured_L500.csv"
OUT_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG6"

DEFAULT_LS = (100, 200, 300, 400, 500)
CURVATURE_THRESHOLD = 0.1
WH_A = -0.6
WH_B = 3.0
WH_X0 = 2 * ELL_DEFAULT / 3

OUTPUTS_OFF = {
    "show_beta_profile": True,
    "heatmaps": False,
    "gifs": False,
    "mode_functions": False,
    "geodesic": False,
    "surface_gravity_fit": False,
    "fft": False,
}


def build_config(L: int) -> dict:
    from config import DEFAULT_CONFIG, deep_merge, prepare_config

    overrides = {
        "L": int(L),
        "l": ELL_DEFAULT,
        "p": 0,
        "m": 0.0,
        "scenario": None,
        "chirality": "chi_plus",
        "initial_direction": "right",
        "beta_sign": "minus",
        "beta_profile": "pos",
        "surface_gravity_beta": False,
        "beta_amplitude": 0.6,
        "beta_width": 1.0,
        "beta_center_fraction": 0.7,
        "j0_fraction": 0.2,
        "sigma_fraction": 0.05,
        "t_i": 0.0,
        "t_f": 8.0,
        "dt_scale": DT_DEFAULT,
        "PBC": False,
        "output_base_dir": str(RUN_ROOT),
        "run_name": f"L{int(L)}",
        "fft_observables": ["H_p"],
        "outputs": OUTPUTS_OFF,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_config(deep_merge(DEFAULT_CONFIG, overrides))


def run_one(L: int, *, rerun: bool) -> dict:
    config = build_config(L)
    out_dir = Path(config["output_dir"])
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not rerun:
        with summary_path.open() as f:
            summary = json.load(f)
        print(f"[fig6b-measure] using cached L={L}: T_lat={summary.get('stagnation_time')}")
        return summary

    from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS
    from simulation import configure, run_simulation

    print(
        "[fig6b-measure] running "
        f"L={L}, dt={config['dt']:.6g}, steps={(config['t_f'] - config['t_i']) / config['dt']:.0f}"
    )
    configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)
    run_simulation()
    with summary_path.open() as f:
        summary = json.load(f)
    print(f"[fig6b-measure] finished L={L}: T_lat={summary.get('stagnation_time')}")
    gc.collect()
    return summary


def write_results(summaries: list[dict]) -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in summaries:
        measured = postprocess_run(int(item["L"]))
        rows.append([
            item["L"],
            np.log(float(item["L"])),
            measured["T_lat"],
            measured["t_in"],
            measured["t_min"],
            measured["H_p_sigma_min"],
            item.get("stagnation_time", np.nan),
            measured["t_in_after_drop"],
            measured["T_lat_after_drop"],
        ])
    rows = np.asarray(rows, dtype=float)
    rows = rows[np.argsort(rows[:, 0])]
    np.savetxt(
        DATA_PATH,
        rows,
        delimiter=",",
        header=(
            "L,lnL,T_lat,t_in,t_min,H_p_sigma_min,"
            "summary_stagnation_time,t_in_after_drop,T_lat_after_drop"
        ),
        comments="",
    )
    print(f"[fig6b-measure] wrote -> {DATA_PATH}")
    return DATA_PATH


def beta_second_derivative(x: np.ndarray) -> np.ndarray:
    u = WH_B * (x - WH_X0)
    sech2 = 1.0 / np.cosh(u) ** 2
    return -2.0 * WH_A * WH_B**2 * sech2 * np.tanh(u)


def postprocess_run(L: int) -> dict:
    csv = RUN_ROOT / f"L{int(L)}" / "H_p_val.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing H_p data for L={L}: {csv}")

    raw = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=complex)
    times = np.real(raw[:, 0])
    values = np.real(raw[:, 1:])
    weights = np.abs(values)
    sites = np.arange(int(L), dtype=float)
    norm = weights.sum(axis=1)
    mean_j = (weights * sites).sum(axis=1) / norm
    var_j = (weights * (sites[None, :] - mean_j[:, None]) ** 2).sum(axis=1) / norm
    std_j = np.sqrt(np.maximum(var_j, 0.0))
    sigma_delta = std_j - std_j[0]

    t_min_idx = int(np.argmin(sigma_delta))
    t_min = float(times[t_min_idx])
    xbar = mean_j * ELL_DEFAULT / float(L)
    curvature = np.abs(beta_second_derivative(xbar))

    in_compression = curvature > CURVATURE_THRESHOLD
    if not np.any(in_compression):
        raise RuntimeError(f"|beta''(xbar)| never exceeds {CURVATURE_THRESHOLD} for L={L}")
    t_in_idx = int(np.argmax(in_compression))
    t_in = float(times[t_in_idx])

    after_drop_idx = np.nan
    t_in_after_drop = np.nan
    T_after_drop = np.nan
    seen_compression = False
    for idx, active in enumerate(in_compression):
        if active:
            seen_compression = True
        elif seen_compression:
            after_drop_idx = idx
            t_in_after_drop = float(times[idx])
            T_after_drop = t_min - t_in_after_drop
            break

    return {
        "L": float(L),
        "T_lat": t_min - t_in,
        "t_in": t_in,
        "t_min": t_min,
        "H_p_sigma_min": float(sigma_delta[t_min_idx]),
        "t_in_after_drop": t_in_after_drop,
        "T_lat_after_drop": T_after_drop,
        "after_drop_idx": after_drop_idx,
    }


def load_results(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(data["L"]), np.asarray(data["lnL"]), np.asarray(data["T_lat"])


def draw_measured_panel(ax, path: Path = DATA_PATH):
    Ls, log_l, t_lat = load_results(path)
    keep = np.isfinite(Ls) & np.isfinite(log_l) & np.isfinite(t_lat)
    Ls, log_l, t_lat = Ls[keep], log_l[keep], t_lat[keep]
    if len(Ls) == 0:
        raise RuntimeError(f"no finite measured Fig. 6(b) points in {path}")
    ax.plot(log_l, t_lat, "o", ms=3.8, color="#1f77b4", label="Numerical simulation")
    ax.set_xlabel(r"$\ln L$")
    ax.set_ylabel(r"$T$", rotation=0, labelpad=10)
    ax.grid(True, color="0.70", lw=0.45, alpha=0.7)
    ax.legend(loc="upper left", frameon=True, handlelength=1.8, borderpad=0.4)
    ax.text(-0.15, 1.03, "(b)", transform=ax.transAxes, fontsize=10)

    x_span = max(float(log_l.max() - log_l.min()), 1.0)
    y_span = max(float(t_lat.max() - t_lat.min()), 1.0)
    ax.set_xlim(float(log_l.min() - 0.05 * x_span), float(log_l.max() + 0.05 * x_span))
    ax.set_ylim(float(t_lat.min() - 0.06 * y_span), float(t_lat.max() + 0.06 * y_span))
    return Ls, t_lat


def save_figures(path: Path = DATA_PATH) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_figure5_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    draw_measured_panel(ax, path)
    for ext in ("pdf", "png"):
        fig.savefig(
            (OUT_DIR / "FIG6b_measured_L500").with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)
    draw_panel(fig, axes[0], PANEL_SPECS["fig6a"])
    draw_measured_panel(axes[1], path)
    for ext in ("pdf", "png"):
        fig.savefig(
            (OUT_DIR / "FIG6_measured_L500").with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"[fig6b-measure] saved -> {OUT_DIR / 'FIG6_measured_L500.png'}")


def parse_l_values(raw: str | None, max_l: int) -> list[int]:
    if raw:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = list(DEFAULT_LS)
    return [value for value in values if value <= max_l]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-L", type=int, default=500)
    parser.add_argument("--Ls", help="Comma-separated L values. Defaults to 100,200,300,400,500.")
    parser.add_argument("--rerun", action="store_true", help="Recompute even if cached summaries exist.")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw figures from the measured CSV.")
    args = parser.parse_args(argv)

    if not args.plot_only:
        L_values = parse_l_values(args.Ls, args.max_L)
        if not L_values:
            raise SystemExit("No L values selected.")
        summaries = [run_one(L, rerun=args.rerun) for L in L_values]
        write_results(summaries)
    save_figures()


if __name__ == "__main__":
    main()
