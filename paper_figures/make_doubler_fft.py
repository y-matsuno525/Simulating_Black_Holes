"""make_doubler_fft.py  ->  paper/figure/doubler_fft.png

BH 内部での波束振動がダブラーモード励起に由来することを示す図。
(a) beta=1.2（BH 内部相当）の分散。低エネルギー交差（ダブラーのゼロモード）を丸で示す。
(b) BH 内部の波束を空間 FFT したスペクトルの時間発展（t=0,3,3.5,4）。
    k=0 近傍が主モード、二次ピークがダブラー（(a) の交差波数付近で成長）。

分散: analysis/dispersion_relation.py。FFT: paper_data の内部 run（ur_p_2 H_p）を空間 FFT。
由来: 本文 Fig. doubler_fft / simulation.py の FFT 機能。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from dispersion_relation import dispersion_bands  # noqa: E402

try:
    from .common import PAPER_DATA, DT_DEFAULT, load_val, ensure_fig_dir
except ImportError:
    from common import PAPER_DATA, DT_DEFAULT, load_val, ensure_fig_dir

BETA_INTERIOR = 1.2
P = 1
FFT_TIMES = [0.0, 3.0, 3.5, 4.0]


def _doubler_k():
    """beta=1.2, p=1 で E_-(k)=0 となる正の k（ダブラーのゼロモード）。"""
    k = np.linspace(0.05, np.pi, 4000)
    _, _, em = dispersion_bands(BETA_INTERIOR, P, nk=4000)
    kk = np.linspace(-np.pi, np.pi, 4000)
    pos = kk > 0
    em_pos = em[pos]
    kp = kk[pos]
    idx = np.argmin(np.abs(em_pos))
    return kp[idx]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # (a) dispersion at beta=1.2
    k, ep, em = dispersion_bands(BETA_INTERIOR, P)
    axes[0].plot(k, ep, color="C0", lw=2, label=r"$p=1$")
    axes[0].plot(k, em, color="C0", lw=2)
    axes[0].axhline(0, color="gray", lw=0.5)
    kd = _doubler_k()
    axes[0].scatter([kd, -kd], [0, 0], s=120, facecolors="none",
                    edgecolors="red", linewidths=1.8, label="doubler crossing", zorder=5)
    axes[0].set_xticks([-np.pi, 0, np.pi])
    axes[0].set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes[0].set_xlabel("$k$")
    axes[0].set_ylabel(r"$\varepsilon E$", rotation=0, labelpad=10)
    axes[0].set_title(rf"(a) dispersion at $\beta={BETA_INTERIOR}$", fontsize=10)
    axes[0].legend(fontsize=8)

    # (b) FFT spectrum of interior wave packet
    csv = PAPER_DATA / "p1" / "ur_p_2" / "H_p_val.csv"
    if csv.exists():
        data = load_val(csv)
        L = data.shape[1]
        kgrid = np.fft.fftfreq(L, d=1.0 / L) * (2 * np.pi / L)
        order = np.argsort(kgrid)
        for tval in FFT_TIMES:
            row = min(int(round(tval / DT_DEFAULT)), data.shape[0] - 1)
            prof = data[row].astype(float)
            prof = prof - prof.mean()               # 空間平均を除去
            spec = np.abs(np.fft.fft(prof))
            axes[1].plot(kgrid[order], spec[order], lw=1.3, label=f"$t={tval:g}$")
        axes[1].axvline(_doubler_k(), color="red", ls="--", lw=0.8)
        axes[1].axvline(-_doubler_k(), color="red", ls="--", lw=0.8)
        axes[1].set_xlabel("$k$")
        axes[1].set_ylabel(r"$|\tilde{\mathcal{H}}^{+}(k)|$")
        axes[1].set_title("(b) wave-packet FFT (interior)", fontsize=10)
        axes[1].set_xticks([-np.pi, 0, np.pi])
        axes[1].set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes[1].legend(fontsize=8)
    else:
        axes[1].set_title("(b) interior run missing (p1/ur_p_2)"); axes[1].axis("off")

    out = ensure_fig_dir() / "doubler_fft.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[doubler_fft] saved -> {out}  (doubler k ~ {_doubler_k():.3f})")


if __name__ == "__main__":
    main()
