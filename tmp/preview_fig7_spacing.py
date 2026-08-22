"""Render Fig. 7 spacing trials without changing the paper figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_figures.reproduce_panel import (  # noqa: E402
    PANEL_SPECS,
    configure_matplotlib,
    draw_panel,
)


SPACINGS = (0.62, 0.42, 0.34, 0.25, 0.12)
OUT_DIR = ROOT / "tmp" / "fig7_spacing_trials"
BASE_HEIGHT = 5.8
BASE_HSPACE = 0.62
BOTTOM_MARGIN_IN = BASE_HEIGHT * 0.08
TOP_MARGIN_IN = BASE_HEIGHT * (1.0 - 0.96)
ROW_HEIGHT_IN = (
    BASE_HEIGHT - BOTTOM_MARGIN_IN - TOP_MARGIN_IN
) / (2.0 + BASE_HSPACE)


def render_trial(hspace: float) -> Path:
    configure_matplotlib()
    figure_height = (
        BOTTOM_MARGIN_IN
        + TOP_MARGIN_IN
        + ROW_HEIGHT_IN * (2.0 + hspace)
    )
    fig = plt.figure(figsize=(7.0, figure_height))
    grid = fig.add_gridspec(
        2,
        4,
        left=0.08,
        right=0.95,
        bottom=BOTTOM_MARGIN_IN / figure_height,
        top=1.0 - TOP_MARGIN_IN / figure_height,
        wspace=0.75,
        hspace=hspace,
    )
    axes = np.array(
        [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 1:3]),
        ]
    )
    for ax, panel_id in zip(axes, ("fig7a", "fig7b", "fig7c")):
        draw_panel(fig, ax, PANEL_SPECS[panel_id])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fig7_hspace_{hspace:.2f}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [plt.imread(path) for path in paths]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    for ax, image, hspace in zip(axes.flat, images, SPACINGS):
        ax.imshow(image)
        ax.set_title(f"hspace = {hspace:.2f}", fontsize=13)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.04, hspace=0.10)
    out = OUT_DIR / "fig7_spacing_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    paths = [render_trial(hspace) for hspace in SPACINGS]
    comparison = make_contact_sheet(paths)
    for path in paths:
        print(path)
    print(comparison)


if __name__ == "__main__":
    main()
