"""make_surface_gravity.py  ->  paper/figure/sg.png

波束幅 delta_sigma(t) の near-horizon 指数成長から表面重力 kappa を抽出する図。
(a) p=0, (b) p=1 の 2 パネル。数値データ (青点) と解析予測 kappa=1 (赤線) を比較。

由来: std ブランチ surface_gravity.py / simulation.py:fit_and_plot_surface_gravity。
データ: paper_data/H_m_sigmas_p0.txt, H_m_sigmas_p1.txt （列: time, sigma）。
"""

from __future__ import annotations

import numpy as np

try:
    from .common import PAPER_DATA, EPS_DEFAULT, ELL_DEFAULT, exp_fit, ensure_fig_dir
except ImportError:  # スクリプト直接実行時
    from common import PAPER_DATA, EPS_DEFAULT, ELL_DEFAULT, exp_fit, ensure_fig_dir


def _panel(ax, sigma_path, eps, ell, title):
    data = np.loadtxt(sigma_path)
    times = data[:, 0]
    sig = data[:, 1] * eps          # 物理スケールの幅
    A, B, t_fine, _ = exp_fit(times, sig)
    # 解析予測 kappa=1（フィット振幅 A はそのまま、成長率のみ 1 に固定）
    ideal = A * np.exp(1.0 * t_fine) - A
    kappa = B
    rel_err = abs(kappa - 1.0) * 100.0

    ax.plot(times, sig * ell, "o", color="blue", ms=3, label="Numerical simulation")
    ax.plot(t_fine, ideal * ell, "-", color="red", lw=2.5, label="Analytical prediction")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\delta\sigma$", rotation=0, labelpad=12)
    ax.set_title(f"{title}: $\\kappa_\\mathrm{{fit}}={kappa:.3f}$ (err {rel_err:.1f}%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return kappa


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p0 = PAPER_DATA / "H_m_sigmas_p0.txt"
    p1 = PAPER_DATA / "H_m_sigmas_p1.txt"

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    if p0.exists():
        k0 = _panel(axes[0], p0, EPS_DEFAULT, ELL_DEFAULT, "(a) $p=0$")
        print(f"[sg] p=0 kappa_fit = {k0:.4f}")
    else:
        axes[0].set_title("(a) p=0 (data missing)")
    if p1.exists():
        k1 = _panel(axes[1], p1, EPS_DEFAULT, ELL_DEFAULT, "(b) $p=1$")
        print(f"[sg] p=1 kappa_fit = {k1:.4f}")
    else:
        axes[1].set_title("(b) p=1 (data missing)")

    out = ensure_fig_dir() / "sg.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[sg] saved -> {out}")


if __name__ == "__main__":
    main()
