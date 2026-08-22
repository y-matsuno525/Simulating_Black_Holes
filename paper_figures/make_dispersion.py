"""Generate Fig. 1 from the homogeneous lattice dispersion relation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
from dispersion_relation import dispersion_bands, light_cone_slopes  # noqa: E402

try:
    from .common import ensure_fig_dir
    from .style import configure_figure5_style
except ImportError:
    from common import ensure_fig_dir
    from style import configure_figure5_style


BETAS = (0.0, 1.0, 2.0)
PAPER_FIGURE_DIR = REPO / "paper" / "figure"
PANEL_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG1"


def _style_dispersion_axis(ax, *, p: int, beta: float, column: int) -> None:
    k, upper, lower = dispersion_bands(beta, p, nk=2401)
    ax.plot(k, lower, color="#0072B2", lw=1.25)
    ax.plot(k, upper, color="#E69F00", lw=1.25)
    ax.axhline(0.0, color="0.45", lw=0.65, ls=":", zorder=0)
    ax.set_xlim(-np.pi, np.pi)
    scale = max(float(np.max(np.abs(upper))), float(np.max(np.abs(lower))), 1.0)
    ax.set_ylim(float(lower.min()) - 0.05 * scale, float(upper.max()) + 0.05 * scale)
    ax.set_xticks([-np.pi, 0.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_yticks([0.0])
    ax.set_yticklabels([r"$0$"])
    ax.set_xlabel(r"$k$", labelpad=-10)
    ax.xaxis.set_label_coords(1.05, -0.02)
    ax.set_ylabel(r"$E$", rotation=0, labelpad=7)
    ax.yaxis.set_label_coords(-0.08, 0.92)
    if p == 0:
        ax.set_title(rf"$\beta={beta:g}$", pad=3)
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.7)
    if column > 0:
        ax.tick_params(labelleft=True)


def _draw_beta_profile(ax) -> None:
    x = np.linspace(-3.2, 3.2, 800)
    beta = np.tanh(1.5 * x) + 1.0
    ax.plot(x, beta, color="0.10", lw=1.35)
    ax.axhline(1.0, color="0.15", lw=0.75, ls="--")
    ax.axvline(0.0, color="red", lw=0.8, ls=":")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.2, 2.25)
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_xticks([])
    ax.set_ylabel(r"$\beta$", rotation=0, labelpad=18)
    ax.set_xlabel(r"$x$", labelpad=-8)
    ax.xaxis.set_label_coords(0.99, -0.04)
    for location in (-2.15, 0.0, 2.15):
        ax.annotate(
            "",
            xy=(location, 1.98),
            xytext=(location, 2.45),
            arrowprops={"arrowstyle": "simple", "color": "black", "mutation_scale": 11},
            annotation_clip=False,
        )
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.7)


def _draw_light_cone(ax, beta: float) -> None:
    vp, vm = light_cone_slopes(beta)
    t_future = np.linspace(0.0, 1.0, 100)
    t_past = np.linspace(-1.0, 0.0, 100)
    color = "#86B6D9"
    edge = "#23384A"

    ax.fill_betweenx(t_future, vm * t_future, vp * t_future, color=color, alpha=0.9)
    ax.fill_betweenx(t_past, vp * t_past, vm * t_past, color=color, alpha=0.9)
    ax.plot(vm * t_future, t_future, color=edge, lw=0.8)
    ax.plot(vp * t_future, t_future, color=edge, lw=0.8)
    ax.plot(vp * t_past, t_past, color=edge, lw=0.8)
    ax.plot(vm * t_past, t_past, color=edge, lw=0.8)

    span = max(abs(vp), abs(vm), 1.0) * 1.18
    ax.annotate("", xy=(span, 0), xytext=(-span, 0), arrowprops={"arrowstyle": "-|>", "lw": 1.0})
    ax.annotate("", xy=(0, 1.15), xytext=(0, -1.15), arrowprops={"arrowstyle": "-|>", "lw": 1.0})
    ax.text(span * 0.98, -0.08, r"$x$", ha="left", va="top")
    ax.text(-0.10, 1.10, r"$t$", ha="right", va="bottom")
    ax.set_xlim(-span, span)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_figure5_style()
    fig = plt.figure(figsize=(7.0, 5.15))
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(1.0, 1.0, 0.92, 0.92),
        left=0.09,
        right=0.98,
        bottom=0.04,
        top=0.97,
        wspace=0.36,
        hspace=0.22,
    )

    axes = []
    for row, p in enumerate((0, 1)):
        for column, beta in enumerate(BETAS):
            ax = fig.add_subplot(grid[row, column])
            _style_dispersion_axis(ax, p=p, beta=beta, column=column)
            axes.append(ax)

    profile_ax = fig.add_subplot(grid[2, :])
    _draw_beta_profile(profile_ax)
    for column, beta in enumerate(BETAS):
        _draw_light_cone(fig.add_subplot(grid[3, column]), beta)

    fig.text(0.008, 0.965, "(a)", fontsize=11, va="top")
    fig.text(0.008, 0.705, "(b)", fontsize=11, va="top")
    fig.text(0.008, 0.445, "(c)", fontsize=11, va="top")
    fig.text(0.008, 0.205, "(d)", fontsize=11, va="top")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    ensure_fig_dir().mkdir(parents=True, exist_ok=True)
    outputs = (
        PANEL_DIR / "FIG1.pdf",
        PANEL_DIR / "FIG1.png",
        ensure_fig_dir() / "dispersion.png",
    )
    for output in outputs:
        fig.savefig(output, dpi=600 if output.suffix == ".png" else None)
        print(f"[dispersion] saved -> {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
