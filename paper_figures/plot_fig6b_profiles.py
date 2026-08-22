"""Plot the Fig. 6(b) energy profile at t=0 and t=t_min."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .common import ELL_DEFAULT, REPO
    from .measure_fig6b_scaling import WH_A, WH_B, WH_X0, build_config, compute_h_p_profiles, measure_profiles
    from .style import configure_figure5_style
except ImportError:
    from common import ELL_DEFAULT, REPO
    from measure_fig6b_scaling import WH_A, WH_B, WH_X0, build_config, compute_h_p_profiles, measure_profiles
    from style import configure_figure5_style


OUT_DIR = REPO / "paper_figures" / "generated" / "diagnostics"


def white_hole_horizon_j(L: int) -> float:
    """Return the chi-plus horizon beta(x_h)=-1 in lattice coordinates."""
    target_beta = -1.0
    argument = (target_beta - WH_A) / WH_A
    x_h = WH_X0 + np.arctanh(argument) / WH_B
    return float(L * x_h / ELL_DEFAULT)


def make_profile_figure(L: int, output_dir: Path = OUT_DIR) -> tuple[Path, dict]:
    config = build_config(L)
    config["times"] = np.insert(np.asarray(config["times"], dtype=float), 0, 0.0)
    times, profiles = compute_h_p_profiles(config)
    measurement = measure_profiles(L, times, profiles)
    t_min_idx = int(np.argmin(np.abs(times - measurement["t_min"])))
    horizon_j = white_hole_horizon_j(L)

    configure_figure5_style()
    import matplotlib.pyplot as plt

    sites = np.arange(L)
    selected = ((0, float(times[0])), (t_min_idx, float(times[t_min_idx])))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharex=True, sharey=True)

    for panel, (ax, (time_idx, time_value)) in enumerate(zip(axes, selected)):
        ax.plot(sites, profiles[time_idx], color="#1f77b4", lw=1.35)
        ax.axvline(
            horizon_j,
            color="#d62728",
            ls=":",
            lw=1.4,
            label=rf"horizon ($j_h={horizon_j:.2f}$)",
        )
        ax.axhline(0.0, color="0.45", lw=0.55, zorder=0)
        ax.set_xlim(0, L - 1)
        ax.set_xlabel(r"$j$")
        ax.set_title(rf"$t={time_value:g}$", pad=4)
        ax.text(-0.13, 1.03, f"({chr(ord('a') + panel)})", transform=ax.transAxes)
        ax.grid(True, color="0.82", lw=0.45)

    axes[0].set_ylabel(r"$\delta\mathcal{H}_j^+$")
    axes[1].legend(loc="upper left", frameon=True, fontsize=8.5)
    fig.tight_layout(w_pad=1.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"FIG6b_profiles_L{L}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    np.savetxt(
        base.with_suffix(".csv"),
        np.column_stack([sites, profiles[0], profiles[t_min_idx]]),
        delimiter=",",
        header=f"j,delta_Hp_t0,delta_Hp_tmin_{times[t_min_idx]:.12g}",
        comments="",
    )
    measurement["horizon_j"] = horizon_j
    measurement["profile_path"] = str(base.with_suffix(".png"))
    return base.with_suffix(".png"), measurement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=100)
    args = parser.parse_args()
    path, measurement = make_profile_figure(args.L)
    print(f"[fig6b-profile] wrote -> {path}")
    print(
        "[fig6b-profile] "
        f"j_h={measurement['horizon_j']:.6g}, "
        f"t_in={measurement['t_in']:.6g}, "
        f"t_min={measurement['t_min']:.6g}, "
        f"T_lat={measurement['T_lat']:.6g}"
    )


if __name__ == "__main__":
    main()
