"""Generate Fig. 4: p=1 doubler crossings and FFT spectrum."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))
from dispersion_relation import dispersion_bands  # noqa: E402

try:
    from .common import ensure_fig_dir
    from .exact_profiles import compute_h_plus_profiles, save_profile_provenance
    from .reproduce_panel import PANEL_SPECS, build_prepared_config
    from .style import configure_figure5_style
except ImportError:
    from common import ensure_fig_dir
    from exact_profiles import compute_h_plus_profiles, save_profile_provenance
    from reproduce_panel import PANEL_SPECS, build_prepared_config
    from style import configure_figure5_style


BETA_INTERIOR = 1.2
P = 1
FFT_TIMES = (0.0, 3.0, 3.5, 4.0)
GENERATED_DIR = REPO / "paper_figures" / "generated" / "panels" / "FIG4"
PAPER_FIGURE_DIR = REPO / "paper" / "figure"
FFT_RUN_DIR = REPO / "paper_reproduction" / "runs" / "FIG4b_exact"


def doubler_k(beta: float = BETA_INTERIOR) -> float:
    """Positive nonzero zero-energy doubler crossing for p=1."""
    return float(2.0 * np.arccos(1.0 / abs(beta)))


def fft_snapshots(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform profiles already evaluated at ``FFT_TIMES``."""
    if data.shape[0] != len(FFT_TIMES):
        raise ValueError(f"expected {len(FFT_TIMES)} exact snapshots, got {data.shape[0]}")
    n_sites = data.shape[1]
    k = 2.0 * np.pi * np.fft.rfftfreq(n_sites, d=1.0)
    spectra = []
    for profile in data:
        profile = profile.astype(float)
        profile = profile - profile.mean()
        spectra.append(np.abs(np.fft.rfft(profile)))
    spectra = np.asarray(spectra)
    norm = float(np.nanmax(spectra))
    if norm == 0.0 or not np.isfinite(norm):
        norm = 1.0
    return k, spectra / norm


def load_time_site_data(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=complex)
    return np.real(raw[:, 1:])


def _style_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.9)


def draw_dispersion_panel(ax) -> float:
    k, upper, lower = dispersion_bands(BETA_INTERIOR, P, nk=2400)
    kd = doubler_k(BETA_INTERIOR)

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

    scale = float(np.nanmax(np.abs([upper.min(), upper.max(), lower.min(), lower.max()])))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(float(lower.min()) - 0.10 * scale, float(upper.max()) + 0.10 * scale)
    ax.set_xticks([-np.pi, -kd, 0.0, kd, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-k_d$", r"$0$", r"$k_d$", r"$\pi$"])
    ax.set_yticks([0.0])
    ax.set_yticklabels([r"$0$"])
    ax.set_xlabel(r"$k$", fontsize=13)
    ax.set_ylabel(r"$E$", rotation=0, fontsize=13)
    ax.xaxis.set_label_coords(1.035, -0.035)
    ax.yaxis.set_label_coords(-0.060, 0.985)
    ax.text(-0.12, 1.10, r"(a)", transform=ax.transAxes, fontsize=11)
    _style_spines(ax)
    return kd


def set_positive_k_ticks(ax, kd: float) -> None:
    ax.set_xticks([0.0, 0.5, kd, 1.5, 2.0, 2.5, np.pi])
    ax.set_xticklabels(["0", "0.5", r"$k_d$", "1.5", "2.0", "2.5", r"$\pi$"])


def draw_fft_panel(ax, kd: float) -> None:
    config = build_prepared_config(PANEL_SPECS["fig3a"].overrides)
    sample_times = np.asarray(FFT_TIMES, dtype=float)
    data = compute_h_plus_profiles(config, sample_times)
    save_profile_provenance(FFT_RUN_DIR, config, sample_times, data)
    k, spectra = fft_snapshots(data)
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    for tval, spectrum, color in zip(FFT_TIMES, spectra, colors):
        ax.plot(k, spectrum, lw=1.35, color=color, label=rf"$t={tval:g}$")

    ax.axvline(kd, color="black", lw=0.9, ls=":", zorder=0)
    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(-0.015, 1.06)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Normalized amplitude")
    set_positive_k_ticks(ax, kd)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="upper right", frameon=True, handlelength=1.8)
    ax.text(-0.12, 1.04, r"(b)", transform=ax.transAxes, fontsize=11)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_figure5_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), constrained_layout=True)
    kd = draw_dispersion_panel(axes[0])
    draw_fft_panel(axes[1], kd)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = GENERATED_DIR / f"FIG4.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
        print(f"[doubler_fft] saved -> {out}")

    ensure_fig_dir().mkdir(parents=True, exist_ok=True)
    aux_out = ensure_fig_dir() / "doubler_fft.png"
    fig.savefig(aux_out, dpi=600, bbox_inches="tight")
    print(f"[doubler_fft] saved -> {aux_out}")

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGURE_DIR / "doubler_fft.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[doubler_fft] saved -> {out}")
    print(f"[doubler_fft] finite-momentum crossing kappa_* = {kd:.6f}")


if __name__ == "__main__":
    main()
