"""Plot Fig. 6(b) stagnation-time scaling from digitized manuscript data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .common import REPO
    from .style import configure_figure5_style
except ImportError:
    from common import REPO
    from style import configure_figure5_style


DATA = REPO / "paper_figures" / "reference_data" / "T_lat_digitized.csv"
OUT_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG6"
DEFAULT_MAX_L: float | None = None


def load_scaling_data(path: Path = DATA, *, max_l: float | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"missing Fig. 6(b) scaling data: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None or len(data.dtype.names) < 2:
        raw = np.loadtxt(path, delimiter=",")
        Ls, times, source = raw[:, 0], raw[:, 1], "T"
    else:
        names = data.dtype.names
        x_name = "L" if "L" in names else names[0]
        if "T_lat" in names:
            y_name = "T_lat"
        else:
            y_name = next(name for name in names if name != x_name)
        Ls, times, source = np.asarray(data[x_name], dtype=float), np.asarray(data[y_name], dtype=float), y_name

    if max_l is not None:
        keep = Ls <= max_l
        Ls = Ls[keep]
        times = times[keep]
    return Ls, times, source


def fit_log_scaling(Ls: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    return tuple(np.polyfit(np.log(Ls), times, 1))


def configure_matplotlib():
    configure_figure5_style()


def draw_scaling_panel(ax, *, max_l: float | None = DEFAULT_MAX_L):
    Ls, times, source_label = load_scaling_data(max_l=max_l)
    x = np.log(Ls)

    ax.plot(x, times, "o", ms=3.8, color="#1f77b4", label="Numerical simulation")
    ax.set_xlabel(r"$\ln L$")
    ax.set_ylabel(r"$T_{\mathrm{lat}}$", rotation=0, labelpad=14)
    ax.grid(True, color="0.70", lw=0.45, alpha=0.7)
    ax.legend(loc="upper left", frameon=True, handlelength=1.8, borderpad=0.4)
    ax.text(-0.15, 1.03, "(b)", transform=ax.transAxes, fontsize=11)

    y_span = max(float(times.max() - times.min()), 1.0)
    ax.set_xlim(float(x.min() - 0.05 * (x.max() - x.min())), float(x.max() + 0.05 * (x.max() - x.min())))
    ax.set_ylim(float(times.min() - 0.06 * y_span), float(times.max() + 0.06 * y_span))
    return Ls, times, source_label


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-L",
        dest="max_l",
        type=float,
        default=0.0,
        help="Largest stored L value to plot. Use 0 to include all digitized data.",
    )
    args = parser.parse_args([] if argv is None else argv)
    max_l = None if args.max_l == 0 else args.max_l

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    Ls, times, source_label = draw_scaling_panel(ax, max_l=max_l)
    out_base = OUT_DIR / "FIG6b"
    for ext in ("pdf", "png"):
        fig.savefig(
            out_base.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    np.savetxt(
        OUT_DIR / "FIG6b_points.csv",
        np.column_stack([Ls, np.log(Ls), times]),
        delimiter=",",
        header=f"L,lnL,{source_label}",
        comments="",
    )
    limit_note = "all stored L" if max_l is None else f"L <= {max_l:g}"
    print(f"[paper] saved FIG6(b) -> {out_base.with_suffix('.png')} ({limit_note})")
    print(f"[paper] FIG6(b) plotted {len(Ls)} digitized points")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
