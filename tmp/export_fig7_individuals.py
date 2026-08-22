from pathlib import Path
import shutil
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_figures.common import ELL_DEFAULT
from paper_figures.make_fig6b_scaling import draw_scaling_panel
from paper_figures.make_surface_gravity import SG_SETTINGS, _panel
from paper_figures.reproduce_panel import (
    PANEL_SPECS,
    configure_matplotlib,
    draw_panel,
)
from paper_figures.style import configure_figure5_style


DOCUMENTS = REPO.parent
DOWNLOADS = REPO.parents[1] / "Downloads"
IMAGE_DIR = DOCUMENTS / "indivisual" / "FIG7"
DATA_DIR = DOCUMENTS / "data" / "FIG7"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    configure_figure5_style()

    surface_sources = []
    for panel, p_value in (("a", 0), ("b", 1)):
        source = REPO / "paper_data" / f"H_m_sigmas_p{p_value}.txt"
        surface_sources.append(source)
        raw = np.loadtxt(source)
        physical = raw[:, 1] * (ELL_DEFAULT / SG_SETTINGS["L"])
        exported = np.column_stack((raw[:, 0], raw[:, 1], physical))
        np.savetxt(
            DATA_DIR / f"FIG7_{panel}.csv",
            exported,
            delimiter=",",
            header="t,delta_sigma_lattice,delta_sigma_physical",
            comments="",
            fmt="%.10e",
        )

        fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
        _panel(
            ax,
            source,
            ell=ELL_DEFAULT,
            L=SG_SETTINGS["L"],
            panel_tag=f"({panel})",
        )
        fig.savefig(IMAGE_DIR / f"FIG7_{panel}.png", dpi=600)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    for ax, panel, source in zip(axes, ("a", "b"), surface_sources):
        _panel(
            ax,
            source,
            ell=ELL_DEFAULT,
            L=SG_SETTINGS["L"],
            panel_tag=f"({panel})",
        )
    fig.savefig(DOWNLOADS / "sg.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig8_image_dir = DOCUMENTS / "indivisual" / "FIG8"
    fig8_data_dir = DOCUMENTS / "data" / "FIG8"
    fig8_image_dir.mkdir(parents=True, exist_ok=True)
    fig8_data_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    draw_panel(fig, ax, PANEL_SPECS["fig6a"])
    fig.savefig(fig8_image_dir / "FIG8_a.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    draw_scaling_panel(ax)
    fig.savefig(fig8_image_dir / "FIG8_b.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    draw_panel(fig, axes[0], PANEL_SPECS["fig6a"])
    draw_scaling_panel(axes[1])
    fig.savefig(DOWNLOADS / "WH_p=0.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(
        REPO / "paper_reproduction" / "runs" / "FIG6a" / "H_p_val.csv",
        fig8_data_dir / "FIG8_a.csv",
    )
    shutil.copy2(
        REPO / "paper_figures" / "measured_data" / "T_lat_measured_L800.csv",
        fig8_data_dir / "FIG8_b.csv",
    )


if __name__ == "__main__":
    main()
