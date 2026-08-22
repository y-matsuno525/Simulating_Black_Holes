from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from paper_figures.common import ELL_DEFAULT, PAPER_DATA
from paper_figures.make_doubler_fft import draw_dispersion_panel, draw_fft_panel
from paper_figures.make_fig6b_scaling import draw_scaling_panel
from paper_figures.make_surface_gravity import SG_SETTINGS, _panel
from paper_figures.reproduce_panel import PANEL_SPECS, configure_matplotlib, draw_panel
from paper_figures.style import configure_figure5_style


OUT = Path(__file__).resolve().parent / "horizontal"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def make_fig4() -> None:
    configure_figure5_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    kd = draw_dispersion_panel(axes[0])
    draw_fft_panel(axes[1], kd)
    save(fig, "doubler_fft")


def make_fig5() -> None:
    configure_figure5_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    _panel(
        axes[0],
        PAPER_DATA / "H_m_sigmas_p0.txt",
        ell=ELL_DEFAULT,
        L=SG_SETTINGS["L"],
        panel_tag="(a)",
    )
    _panel(
        axes[1],
        PAPER_DATA / "H_m_sigmas_p1.txt",
        ell=ELL_DEFAULT,
        L=SG_SETTINGS["L"],
        panel_tag="(b)",
    )
    save(fig, "sg")


def make_fig6() -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    draw_panel(fig, axes[0], PANEL_SPECS["fig6a"])
    draw_scaling_panel(axes[1])
    save(fig, "WH_p=0")


if __name__ == "__main__":
    make_fig4()
    make_fig5()
    make_fig6()
