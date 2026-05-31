"""make_bh_panels.py  ->  paper/figure/p=0_BH.png, paper/figure/p=1_BH.png

ブラックホール背景での波束ダイナミクス（時空 heatmap + horizon + geodesic）。
p=0 と p=1 それぞれについて、保存済みデータから 3 つの観測量パネルを合成する:

  (a) lr_p H_p   : 右進行成分（horizon を通過してBH内部へ）
  (b) lr_p H_m   : 左進行成分（horizon に停滞）
  (c) ur_p_2 H_p : chi_plus を BH 内側近傍に置いた場合の右進行

由来: master_thesis/figure.ipynb, master_thesis_p=0/figure.ipynb（imshow+seismic+geodesic）。
horizon は解析値 J_BH≈212.82。データは paper_data/p{0,1}/。

注: 論文 PDF の p=*_BH.png は j0=30/40/180 など特定 run の 4 パネルで、本スクリプトは
保存データからの「同等図」。厳密再現は simulation.py を該当 config で実行（README 参照）。
"""

from __future__ import annotations

try:
    from .common import (PAPER_DATA, J_BH, EPS_DEFAULT, load_val, load_geodesic,
                         plot_heatmap, ensure_fig_dir)
except ImportError:
    from common import (PAPER_DATA, J_BH, EPS_DEFAULT, load_val, load_geodesic,
                        plot_heatmap, ensure_fig_dir)


PANELS = [
    # (scenario, observable, label, t_f, geodesic_scenario, title)
    ("lr_p", "H_p", r"$\mathcal{H}_j^{+}$", 8.0, "lr_p", "(a) $H^{+}$: right-mover crosses horizon"),
    ("lr_p", "H_m", r"$\mathcal{H}_j^{-}$", 8.0, "lr_p", "(b) $H^{-}$: left-mover stalls at horizon"),
    ("ur_p_2", "H_p", r"$\mathcal{H}_j^{+}$", 3.0, "ur_p_2", "(c) $H^{+}$: packet near interior"),
]


def make_one(p_tag):
    """p_tag in {'p0','p1'} -> paper/figure/p=0_BH.png or p=1_BH.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = PAPER_DATA / p_tag
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (scen, obs, label, t_f, geo_scen, title) in zip(axes, PANELS):
        csv = base / scen / f"{obs}_val.csv"
        if not csv.exists():
            ax.set_title(f"{title}\n(missing {scen}/{obs})", fontsize=8)
            ax.axis("off")
            continue
        data = load_val(csv)
        geo_path = base / "geodesic" / f"{geo_scen}.csv"
        geo = load_geodesic(geo_path, EPS_DEFAULT) if geo_path.exists() else None
        im, lbl = plot_heatmap(ax, data, t_f=t_f, jh=J_BH, geodesic=geo,
                               label=label, title=title)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(lbl, rotation=0, labelpad=10)
        ax.legend(fontsize=7, loc="upper left")

    suffix = "p=0_BH" if p_tag == "p0" else "p=1_BH"
    out = ensure_fig_dir() / f"{suffix}.png"
    fig.suptitle(f"Black hole dynamics ({'p=0' if p_tag=='p0' else 'p=1'})", fontsize=11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[bh] saved -> {out}")


def main():
    make_one("p0")
    make_one("p1")


if __name__ == "__main__":
    main()
