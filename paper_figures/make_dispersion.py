"""make_dispersion.py  ->  paper/figure/dispersion.png, paper/figure/p0_dispersion.png

格子ハミルトニアンの局所分散関係と有効光円錐の対応図。
(a) 代表的な beta での分散 eps*E(k)（p=1 実線 / p=0 破線）
(b) beta(x) プロファイルと代表位置
(c) 低運動量の群速度から決まる有効光円錐（傾き beta±1）

分散は analysis/dispersion_relation.py の解析式（数値対角化と一致を確認済み）。
由来: マヨラナ分散.ipynb / painleve_k.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
# 正準 BH beta プロファイルと分散は analysis/dispersion_relation に一本化。
from dispersion_relation import (  # noqa: E402
    beta_horizon_x, beta_profile_x, dispersion_bands, light_cone_slopes,
)

try:
    from .common import ELL_DEFAULT, L_DEFAULT, ensure_fig_dir
except ImportError:
    from common import ELL_DEFAULT, L_DEFAULT, ensure_fig_dir


BETAS = [0.0, 0.6, 1.2]   # 代表値: 外側 / horizon 近傍 / 内側


def _make(out_name, emphasize_p0=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # (a) 分散 3 枚
    for col, b in enumerate(BETAS):
        ax = fig.add_subplot(gs[0, col])
        k, ep1, em1 = dispersion_bands(b, 1)
        k, ep0, em0 = dispersion_bands(b, 0)
        ax.plot(k, ep1, color="C0", lw=2, label="$p=1$")
        ax.plot(k, em1, color="C0", lw=2)
        ax.plot(k, ep0, color="C3", lw=1.5, ls="--", label="$p=0$")
        ax.plot(k, em0, color="C3", lw=1.5, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(rf"$\beta={b:g}$", fontsize=10)
        ax.set_xlabel("$k$")
        ax.set_ylabel(r"$\varepsilon E$", rotation=0, labelpad=10)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        if col == 0:
            ax.legend(fontsize=8, loc="upper center")

    # (b) beta profile
    axb = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0, ELL_DEFAULT, 400)
    axb.plot(xs, beta_profile_x(xs), color="black")
    axb.axhline(1, color="red", ls="--", lw=0.8, label=r"$\beta=1$ (horizon)")
    for b in BETAS:
        xb = beta_horizon_x(b)
        if xb is not None:
            axb.plot(xb, b, "o", color="C0", ms=5)
            axb.annotate(rf"$\beta={b:g}$", (xb, b), fontsize=7,
                         textcoords="offset points", xytext=(4, -8))
    axb.set_xlabel("$x$")
    axb.set_ylabel(r"$\beta(x)$", rotation=0, labelpad=12)
    axb.set_title("(b) $\\beta$ profile", fontsize=10)
    axb.legend(fontsize=7)

    # (c) light cones
    axc = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 1, 50)
    for b in BETAS:
        vp, vm = light_cone_slopes(b)
        axc.plot(vp * t, t, color="C0", lw=1.5)
        axc.plot(vm * t, t, color="C3", lw=1.5)
        axc.annotate(rf"$\beta={b:g}$", (vp, 1.0), fontsize=7)
    axc.axvline(0, color="gray", lw=0.5)
    axc.set_xlabel(r"$x$ (slope $=\beta\pm1$)")
    axc.set_ylabel(r"$t$", rotation=0, labelpad=8)
    axc.set_title("(c) effective light cones", fontsize=10)

    # 注記
    axn = fig.add_subplot(gs[1, 2])
    axn.axis("off")
    axn.text(0.0, 0.9,
             "slope near k=0  = beta +/- 1  (light cone)\n"
             "k=pi : p=0 -> zero mode (doubler)\n"
             "         p=1 -> gap +/- 2|p|\n"
             "beta=1.2 (interior): both speeds > 0\n"
             "  => one-way (black-hole interior)",
             fontsize=8, va="top", family="monospace")

    sub = "p=0" if emphasize_p0 else "p=1"
    fig.suptitle(f"Local dispersion & light-cone structure ({sub} emphasis)", fontsize=12)
    out = ensure_fig_dir() / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[dispersion] saved -> {out}")


def main():
    _make("dispersion.png", emphasize_p0=False)
    _make("p0_dispersion.png", emphasize_p0=True)


if __name__ == "__main__":
    main()
