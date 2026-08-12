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
    ax.set_ylabel(r"$\delta\sigma_{\mathcal{E}}(t)$", rotation=0, labelpad=22)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.text(-0.13, 1.04, panel_tag, transform=ax.transAxes, fontsize=11)
    return kappa


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
    if p0.exists():
        k0 = _panel(axes[0], p0, ell=ELL_DEFAULT, L=SG_SETTINGS["L"], panel_tag="(a)")
        print(f"[sg] p=0 kappa_fit = {k0:.4f}")
    else:
        axes[0].set_title("(a) p=0 (data missing)")
    if p1.exists():
        k1 = _panel(axes[1], p1, ell=ELL_DEFAULT, L=SG_SETTINGS["L"], panel_tag="(b)")
        print(f"[sg] p=1 kappa_fit = {k1:.4f}")
    else:
        axes[1].set_title("(b) p=1 (data missing)")

    out_base = ensure_fig_dir() / "sg"
    for ext in ("pdf", "png"):
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"[sg] saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
