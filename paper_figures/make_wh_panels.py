"""make_wh_panels.py  ->  paper/figure/WH_p=0.png, paper/figure/WH_p=1.png

ホワイトホール背景での波束ダイナミクス。

WH_p=0.png:
  (a) ur_m |H_m| 時空図（horizon で圧縮 → 有限サイズ反射）
  (b) 停滞時間 T_lat の log L スケーリング
      （paper_data/logL/T_lat.csv があれば実測点、無ければ T∝logL の理論線のみ）

WH_p=1.png:
  (a) ur_p H_p, (b) ur_m H_m, (c) ur_p H_pm
      p=1 では左右モード結合により horizon を部分透過。

由来: master_thesis/figure.ipynb, thesis/plot.ipynb, various_beta.ipynb(logL)。
horizon は解析値 J_WH≈87.18。
"""

from __future__ import annotations

import numpy as np

try:
    from .common import (PAPER_DATA, J_WH, EPS_DEFAULT, load_val, load_geodesic,
                         plot_heatmap, ensure_fig_dir)
except ImportError:
    from common import (PAPER_DATA, J_WH, EPS_DEFAULT, load_val, load_geodesic,
                        plot_heatmap, ensure_fig_dir)


def _logL_panel(ax):
    """停滞時間の logL スケーリング。実測 CSV があれば点 + 線形フィット。"""
    csv = PAPER_DATA / "logL" / "T_lat.csv"
    if csv.exists():
        d = np.loadtxt(csv, delimiter=",", skiprows=1)
        Ls, T = d[:, 0], d[:, 1]
        x = np.log(Ls)
        a, b = np.polyfit(x, T, 1)
        ax.plot(x, T, "o", color="blue", label="Numerical simulation")
        xf = np.linspace(x.min(), x.max(), 100)
        ax.plot(xf, a * xf + b, "-", color="red", label=f"fit: {a:.2f} log L + {b:.2f}")
        note = ""
    else:
        x = np.log(np.array([100, 200, 300, 400, 600]))
        ax.plot(x, 0.9 * x - 2.0, "--", color="red", label=r"$T\propto\log L$ (theory)")
        note = "\n(run analysis/stagnation_logL.py)"
    ax.set_xlabel(r"$\log L$")
    ax.set_ylabel(r"$T_{\mathrm{lat}}$", rotation=0, labelpad=12)
    ax.set_title("(b) stagnation time scaling" + note, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def make_wh_p0():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = PAPER_DATA / "p0"
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    csv = base / "ur_m" / "H_m_val.csv"
    if csv.exists():
        data = load_val(csv)
        geo_path = base / "geodesic" / "ur_m.csv"
        geo = load_geodesic(geo_path, EPS_DEFAULT) if geo_path.exists() else None
        im, lbl = plot_heatmap(axes[0], data, t_f=8.0, jh=J_WH, geodesic=geo,
                               label=r"$|\mathcal{H}_j^{-}|$", cmap="binary",
                               use_abs=True, geo_color="black",
                               title="(a) $|H^{-}|$: compression at WH horizon")
        cbar = fig.colorbar(im, ax=axes[0], pad=0.02)
        cbar.set_label(lbl, rotation=0, labelpad=12)
        axes[0].legend(fontsize=7, loc="upper left")
    else:
        axes[0].set_title("(a) ur_m H_m missing"); axes[0].axis("off")

    _logL_panel(axes[1])

    out = ensure_fig_dir() / "WH_p=0.png"
    fig.suptitle("White hole dynamics (p=0)", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wh] saved -> {out}")


def make_wh_p1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = PAPER_DATA / "p1"
    panels = [
        ("ur_p", "H_p", r"$\mathcal{H}_j^{+}$", "(a) $H^{+}$"),
        ("ur_m", "H_m", r"$\mathcal{H}_j^{-}$", "(b) $H^{-}$"),
        ("ur_p", "H_pm", r"$\mathcal{H}_j^{\pm}$", "(c) $H^{\\pm}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (scen, obs, label, title) in zip(axes, panels):
        csv = base / scen / f"{obs}_val.csv"
        if not csv.exists():
            ax.set_title(f"{title} (missing)"); ax.axis("off"); continue
        data = load_val(csv)
        geo_path = base / "geodesic" / f"{scen}.csv"
        geo = load_geodesic(geo_path, EPS_DEFAULT) if geo_path.exists() else None
        im, lbl = plot_heatmap(ax, data, t_f=8.0, jh=J_WH, geodesic=geo,
                               label=label, title=title)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(lbl, rotation=0, labelpad=10)
        ax.legend(fontsize=7, loc="upper left")

    out = ensure_fig_dir() / "WH_p=1.png"
    fig.suptitle("White hole dynamics (p=1): partial transmission", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wh] saved -> {out}")


def main():
    make_wh_p0()
    make_wh_p1()


if __name__ == "__main__":
    main()
