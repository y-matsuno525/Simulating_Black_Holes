"""Generate the Fig. 4(b) FFT spectrum panel.

The input is the black-hole interior p=1 run used in the manuscript:
``paper_reproduction/runs/FIG3a/H_p_val.csv``.  The plotted spectrum is the
positive-k magnitude of the spatial FFT after subtracting the spatial
mean, normalized by the global maximum over the selected snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from dispersion_relation import dispersion_bands  # noqa: E402

try:
    from .common import DT_DEFAULT, ensure_fig_dir
    from .style import configure_figure5_style
except ImportError:
    from common import DT_DEFAULT, ensure_fig_dir
    from style import configure_figure5_style


BETA_INTERIOR = 1.2
P = 1
FFT_TIMES = (0.0, 3.0, 3.5, 4.0)
FFT_RUN_DIR = Path(__file__).resolve().parent.parent / "paper_reproduction" / "runs" / "FIG3a"


def doubler_k() -> float:
    """Positive low-energy doubler crossing for beta=1.2, p=1."""
    k, _, em = dispersion_bands(BETA_INTERIOR, P, nk=12000)
    mask = k > 0.3
    return float(k[mask][np.argmin(np.abs(em[mask]))])


def fft_snapshots(data: np.ndarray):
    """Return positive-k FFT spectra for the selected manuscript times."""
    n_sites = data.shape[1]
    k = 2.0 * np.pi * np.fft.rfftfreq(n_sites, d=1.0)
    spectra = []
    for t in FFT_TIMES:
        row = min(int(round(t / DT_DEFAULT)), data.shape[0] - 1)
        profile = data[row].astype(float)
        profile = profile - profile.mean()
        spectra.append(np.abs(np.fft.rfft(profile)))
    spectra = np.asarray(spectra)
    norm = float(np.max(spectra))
    if norm == 0.0 or not np.isfinite(norm):
        norm = 1.0
    return k, spectra / norm


def load_time_site_data(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=complex)
    return np.real(raw[:, 1:])


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_figure5_style()

    csv = FFT_RUN_DIR / "H_p_val.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv}. Run `python paper_figures/reproduce_panel.py FIG3a --rerun` first.")

    data = load_time_site_data(csv)
    k, spectra = fft_snapshots(data)
    kd = doubler_k()

    fig, ax = plt.subplots(figsize=(3.375, 2.35))
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    for t, spec, color in zip(FFT_TIMES, spectra, colors):
        ax.plot(k, spec, lw=1.25, color=color, label=rf"$t={t:g}$")

    ax.axvline(kd, color="0.25", lw=0.7, ls=(0, (2.0, 2.0)), zorder=0)
    ax.text(kd + 0.045, 0.93, r"$k_{\rm d}$", ha="left", va="top", fontsize=7)

    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(-0.015, 1.06)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Normalized amplitude")
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, np.pi])
    ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0", "2.5", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=True, handlelength=1.8)
    ax.text(-0.18, 1.04, r"(b)", transform=ax.transAxes, fontsize=10)

    fig.tight_layout(pad=0.35)
    out_dir = ensure_fig_dir()
    for ext in ("pdf", "png"):
        out = out_dir / f"fig4b_fft.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"[fig4b_fft] saved -> {out}")
    plt.close(fig)
    print(f"[fig4b_fft] doubler k = {kd:.6f}")


if __name__ == "__main__":
    main()
