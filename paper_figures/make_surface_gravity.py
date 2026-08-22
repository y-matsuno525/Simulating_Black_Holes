"""Build Fig. 5: surface-gravity extraction from wave-packet broadening.

The manuscript Fig. 5 settings are
L=500, beta(x)=tanh(x-l/2)+1, j0=245, sigma=0.003 L, and t in [0, 1].
Use --recompute to regenerate the p=0 and p=1 sigma data with those settings
before plotting.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np

try:
    from .common import REPO, PAPER_DATA, ELL_DEFAULT, exp_fit, ensure_fig_dir
    from .style import configure_figure5_style
except ImportError:
    from common import REPO, PAPER_DATA, ELL_DEFAULT, exp_fit, ensure_fig_dir
    from style import configure_figure5_style


RUN_ROOT = REPO / "paper_reproduction" / "runs"
FIT_OUTPUT_DIR = RUN_ROOT / "FIG5_fit"
PANEL_OUTPUT_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG5"
PAPER_FIGURE_PATH = REPO / "paper" / "figure" / "sg.png"

SG_SETTINGS = {
    "L": 500,
    "l": ELL_DEFAULT,
    "m": 0.0,
    "scenario": None,
    "chirality": "chi_minus",
    "initial_direction": "left",
    "beta_sign": "plus",
    "beta_profile": "center",
    "surface_gravity_beta": False,
    "centered_beta_width": 1.0,
    "centered_beta_amplitude": 1.0,
    "centered_beta_center_fraction": 0.5,
    "j0_fraction": 245 / 500,
    "sigma_fraction": 0.003,
    "t_i": 0.0,
    "t_f": 1.0,
    "dt_scale": 0.01,
    "PBC": False,
    "fft_observables": ["H_m"],
    "outputs": {
        "show_beta_profile": True,
        "heatmaps": False,
        "gifs": False,
        "mode_functions": False,
        "geodesic": False,
        "surface_gravity_fit": False,
        "fft": False,
    },
}


def _ensure_repo_on_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def _build_sg_config(p: int) -> dict:
    _ensure_repo_on_path()
    from config import DEFAULT_CONFIG, deep_merge, prepare_config

    overrides = dict(SG_SETTINGS)
    overrides["p"] = p
    overrides["output_base_dir"] = str(RUN_ROOT)
    overrides["run_name"] = f"FIG5_p{p}"
    merged = deep_merge(DEFAULT_CONFIG, overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_config(merged)


def recompute_surface_gravity_data() -> None:
    _ensure_repo_on_path()
    from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS
    from simulation import configure, run_simulation

    PAPER_DATA.mkdir(parents=True, exist_ok=True)
    for p in (0, 1):
        config = _build_sg_config(p)
        print(
            "[sg] recomputing "
            f"p={p}: L={config['L']}, j0={config['j0']}, "
            f"sigma={config['sigma']:.3g}, beta=tanh(x-l/2)+1"
        )
        configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)
        run_simulation()
        src = Path(config["output_dir"]) / "H_m_sigmas.txt"
        dst = PAPER_DATA / f"H_m_sigmas_p{p}.txt"
        data = np.loadtxt(src)
        np.savetxt(dst, data, fmt="%.10e")
        print(f"[sg] wrote {dst}")


def lattice_width_to_physical(delta_sigma_lattice, *, ell, L):
    """Convert a width measured in lattice sites to the physical x coordinate."""
    return np.asarray(delta_sigma_lattice, dtype=float) * (ell / L)


def _panel(ax, sigma_path, *, ell, L, panel_tag):
    data = np.loadtxt(sigma_path)
    times = data[:, 0]
    sig = lattice_width_to_physical(data[:, 1], ell=ell, L=L)
    A, B, t_fine, _ = exp_fit(times, sig)
    ideal = A * np.exp(t_fine) - A
    kappa = B
    rel_err = abs(kappa - 1.0) * 100.0

    ax.plot(times, sig, "o", color="blue", ms=3, label="Numerical simulation")
    ax.plot(t_fine, ideal, "-", color="red", lw=2.5, label="Analytical prediction")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\delta\sigma$", rotation=0, labelpad=18)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.text(-0.13, 1.04, panel_tag, transform=ax.transAxes, fontsize=11)
    return {
        "kappa_fit": float(kappa),
        "relative_error_percent": float(rel_err),
        "prefactor_from_free_fit": float(A),
        "first_time": float(times[0]),
        "last_time": float(times[-1]),
        "sample_count": int(len(times)),
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute p=0 and p=1 data with the manuscript Fig. 5 settings before plotting.",
    )
    args = parser.parse_args([] if argv is None else argv)

    if args.recompute:
        recompute_surface_gravity_data()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_figure5_style()
    p0 = PAPER_DATA / "H_m_sigmas_p0.txt"
    p1 = PAPER_DATA / "H_m_sigmas_p1.txt"

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    results = {}
    if p0.exists():
        results["p0"] = _panel(
            axes[0], p0, ell=ELL_DEFAULT, L=SG_SETTINGS["L"], panel_tag="(a)"
        )
        print(f"[sg] p=0 kappa_fit = {results['p0']['kappa_fit']:.4f}")
    else:
        axes[0].set_title("(a) p=0 (data missing)")
    if p1.exists():
        results["p1"] = _panel(
            axes[1], p1, ell=ELL_DEFAULT, L=SG_SETTINGS["L"], panel_tag="(b)"
        )
        print(f"[sg] p=1 kappa_fit = {results['p1']['kappa_fit']:.4f}")
    else:
        axes[1].set_title("(b) p=1 (data missing)")

    out_base = ensure_fig_dir() / "sg"
    PANEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        fig.savefig(
            PANEL_OUTPUT_DIR / f"FIG5.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
        print(f"[sg] saved -> {out}")
    PAPER_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURE_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    FIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_summary = {
        "settings": SG_SETTINGS,
        "analytical_kappa": 1.0,
        "width_conversion": "delta_sigma_physical=(ell/L)*delta_sigma_lattice",
        "fits": results,
    }
    with (FIT_OUTPUT_DIR / "fit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(fit_summary, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
