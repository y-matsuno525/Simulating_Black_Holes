"""Generate the p=1, beta=1.2 dispersion panel with doubler crossings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG3"

sys.path.insert(0, str(REPO / "analysis"))
from dispersion_relation import dispersion_bands  # noqa: E402

try:
    from .style import configure_figure5_style
except ImportError:
    from style import configure_figure5_style


BETA = 1.2
P = 1


def doubler_k(beta: float = BETA) -> float:
    """Return the positive nonzero zero crossing for p=1."""
    return float(2.0 * np.arccos(1.0 / abs(beta)))


def configure_matplotlib() -> None:
    configure_figure5_style()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    k, upper, lower = dispersion_bands(BETA, P, nk=2400)
    kd = doubler_k(BETA)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    ax.plot(k, lower, color="#0072B2", lw=1.7)
    ax.plot(k, upper, color="#E69F00", lw=1.7)
    ax.axhline(0.0, color="0.45", lw=0.8, ls=(0, (1.0, 1.5)), zorder=0)

    ax.scatter(
        [-kd, kd],
        [0.0, 0.0],
        s=96,
        facecolors="none",
        edgecolors="red",
        linewidths=1.1,
        zorder=5,
    )

    ax.set_xlim(-np.pi, np.pi)
    y_pad = 0.10 * float(np.max(np.abs([upper.min(), upper.max(), lower.min(), lower.max()])))
    ax.set_ylim(float(lower.min()) - y_pad, float(upper.max()) + y_pad)
    ax.set_xlabel(r"$k$", fontsize=13)
    ax.set_ylabel(r"$E$", rotation=0, fontsize=13)
    ax.xaxis.set_label_coords(1.035, -0.035)
    ax.yaxis.set_label_coords(-0.055, 1.02)
    ax.set_xticks([-np.pi, -kd, 0.0, kd, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-k_d$", r"$0$", r"$k_d$", r"$\pi$"])
    ax.set_yticks([0.0])
    ax.set_yticklabels([r"$0$"])
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"dispersion_beta1p2_p1.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"[beta12_dispersion] saved -> {out}")
    plt.close(fig)
    print(f"[beta12_dispersion] doubler k = {kd:.6f}")


if __name__ == "__main__":
    main()
