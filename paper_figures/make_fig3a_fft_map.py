"""Generate a time-resolved spatial FFT map for Fig. 3(a).

Input:
    paper_reproduction/runs/FIG3a/H_p_val.csv

Outputs:
    paper_figures/generated/panels/FIG3/FIG3a_fft_map.{png,pdf}
    paper_figures/generated/panels/FIG3/FIG3a_fft_snapshots.{png,pdf}
    paper_figures/generated/panels/FIG3/FIG3a_fft_k.csv
    paper_figures/generated/panels/FIG3/FIG3a_fft_val.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "paper_reproduction" / "runs" / "FIG3a"
OUT_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG3"

try:
    from .style import configure_figure5_style
except ImportError:
    from style import configure_figure5_style


def load_time_site_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=complex)
    times = np.real(raw[:, 0])
    values = np.real(raw[:, 1:])
    return times, values


def positive_fft(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=1, keepdims=True)
    spectra = np.abs(np.fft.rfft(centered, axis=1))
    norm = float(np.nanmax(spectra))
    if norm == 0.0 or not np.isfinite(norm):
        norm = 1.0
    spectra = spectra / norm
    k = 2.0 * np.pi * np.fft.rfftfreq(values.shape[1], d=1.0)
    return k, spectra


def doubler_k_reference() -> float | None:
    return float(2.0 * np.arccos(1.0 / 1.2))


def configure_matplotlib() -> None:
    configure_figure5_style()


def save_fft_csv(k: np.ndarray, times: np.ndarray, spectra: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUT_DIR / "FIG3a_fft_k.csv",
        k,
        delimiter=",",
        header="k",
        comments="",
    )
    data = np.column_stack([times, spectra])
    header = ",".join(["time"] + [f"k{i}" for i in range(len(k))])
    np.savetxt(
        OUT_DIR / "FIG3a_fft_val.csv",
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def set_positive_k_ticks(ax, kd: float | None) -> None:
    if kd is None:
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, np.pi])
        ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0", "2.5", r"$\pi$"])
        return

    ax.set_xticks([0, 0.5, kd, 1.5, 2.0, 2.5, np.pi])
    ax.set_xticklabels(["0", "0.5", r"$k_d$", "1.5", "2.0", "2.5", r"$\pi$"])


def plot_fft_map(
    k: np.ndarray,
    times: np.ndarray,
    spectra: np.ndarray,
    kd: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    im = ax.imshow(
        spectra,
        aspect="auto",
        origin="lower",
        extent=[k[0], k[-1], times[0], times[-1]],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    if kd is not None:
        ax.axvline(kd, color="black", lw=0.8, ls=":")

    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(times[0], times[-1])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$t$", rotation=0, labelpad=8)
    set_positive_k_ticks(ax, kd)
    cbar = fig.colorbar(im, ax=ax, pad=0.025)
    cbar.set_label("Normalized amplitude", rotation=90, labelpad=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"FIG3a_fft_map.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"[fig3a_fft] saved -> {out}")
    plt.close(fig)


def plot_snapshots(
    k: np.ndarray,
    times: np.ndarray,
    spectra: np.ndarray,
    kd: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    sample_times = (2.0, 3.0, 3.5, 4.0)
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for target, color in zip(sample_times, colors):
        idx = int(np.argmin(np.abs(times - target)))
        ax.plot(k, spectra[idx], lw=1.2, color=color, label=rf"$t={target:.2f}$")

    if kd is not None:
        ax.axvline(kd, color="black", lw=0.8, ls=":")

    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(-0.015, 1.04)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Normalized amplitude")
    set_positive_k_ticks(ax, kd)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"FIG3a_fft_snapshots.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"[fig3a_fft] saved -> {out}")
    plt.close(fig)


def main() -> None:
    csv = RUN_DIR / "H_p_val.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Missing FIG3(a) data: {csv}. Run "
            "`python paper_figures/reproduce_panel.py FIG3a --rerun` first."
        )

    times, values = load_time_site_csv(csv)
    k, spectra = positive_fft(values)
    kd = doubler_k_reference()

    save_fft_csv(k, times, spectra)
    plot_fft_map(k, times, spectra, kd)
    plot_snapshots(k, times, spectra, kd)
    if kd is not None:
        print(f"[fig3a_fft] beta=1.2 doubler reference k = {kd:.6f}")


if __name__ == "__main__":
    main()
