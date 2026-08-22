"""Generate the Fig. 4(b) FFT spectrum panel.

The profiles are evaluated directly at the four times stated in the
manuscript caption.  The plotted spectrum is the positive-k magnitude of the
spatial FFT after subtracting the spatial mean, normalized by the global
maximum over the selected snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from dispersion_relation import dispersion_bands  # noqa: E402

try:
    from .common import ensure_fig_dir
    from .exact_profiles import compute_h_plus_profiles, save_profile_provenance
    from .make_doubler_fft import FFT_RUN_DIR, FFT_TIMES, fft_snapshots
    from .reproduce_panel import PANEL_SPECS, build_prepared_config
    from .style import configure_figure5_style
except ImportError:
    from common import ensure_fig_dir
    from exact_profiles import compute_h_plus_profiles, save_profile_provenance
    from make_doubler_fft import FFT_RUN_DIR, FFT_TIMES, fft_snapshots
    from reproduce_panel import PANEL_SPECS, build_prepared_config
    from style import configure_figure5_style


BETA_INTERIOR = 1.2
P = 1
def doubler_k() -> float:
    """Positive low-energy doubler crossing for beta=1.2, p=1."""
    k, _, em = dispersion_bands(BETA_INTERIOR, P, nk=12000)
    mask = k > 0.3
    return float(k[mask][np.argmin(np.abs(em[mask]))])


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_figure5_style()

    config = build_prepared_config(PANEL_SPECS["fig3a"].overrides)
    sample_times = np.asarray(FFT_TIMES, dtype=float)
    data = compute_h_plus_profiles(config, sample_times)
    save_profile_provenance(FFT_RUN_DIR, config, sample_times, data)
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
