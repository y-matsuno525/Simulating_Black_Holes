"""
replot_paper_figs.py

論文 FIG2/3/6/7 の全パネルを正しいホライズン位置で再描画する。
既存 CSV データはそのまま使用し、ホライズン線の位置のみ修正する。

ホライズン正解値（L=300, l=2π, ε=2π/300）:
  BH: j_h = 2/3*L + arctanh(2/3) / (3*ε) ≈ 212.82
  WH: j_h = 1/3*L - arctanh(2/3) / (3*ε) ≈  87.18
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
# 注意: このスクリプトは旧アーカイブ（master_thesis/ thesis/ 等）のデータを参照する
# レガシー再描画ツール。現行の論文図は paper_figures/make_all.py を使うこと。
# アーカイブの場所は環境変数 SBH_ARCHIVE_BASE で指定可能（既定はホームディレクトリ）。
REPO = Path(__file__).resolve().parent
# ホライズン位置・格子定数は paper_figures/common.py に一本化（重複定義の排除）。
sys.path.insert(0, str(REPO / "paper_figures"))
from common import EPS_DEFAULT, J_BH, J_WH, L_DEFAULT  # noqa: E402

BASE = Path(os.environ.get("SBH_ARCHIVE_BASE", Path.home()))
# 生成図の出力先は paper_figures/generated/ に統一。
OUT = REPO / "paper_figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 物理パラメータ（common と共有）
# ---------------------------------------------------------------------------
L = L_DEFAULT
eps = EPS_DEFAULT
DT = 0.01

print(f"BH horizon: j = {J_BH:.4f}")
print(f"WH horizon: j = {J_WH:.4f}")

# ---------------------------------------------------------------------------
# 全パネル定義
# (出力名, CSVパス, 観測量LaTeXラベル, j_h, t_max, geodesicパス, cmap, 絶対値?)
# ---------------------------------------------------------------------------
PANELS = [
    (
        "BH_Hm",
        BASE / "master_thesis/data/lr_p/H_m_val.csv",
        r"$\mathcal{H}_j^-$",
        J_BH, 8.0,
        BASE / "master_thesis/geodesic/lr_p.csv",
        "seismic", False,
    ),
    (
        "BH_Hp",
        BASE / "master_thesis/data/lr_p/H_p_val.csv",
        r"$\mathcal{H}_j^+$",
        J_BH, 8.0,
        BASE / "master_thesis/geodesic/lr_p.csv",
        "seismic", False,
    ),
    (
        "BH_Hpm",
        BASE / "master_thesis/data/lr_p/H_pm_val.csv",
        r"$\mathcal{H}_j^{+-}$",
        J_BH, 8.0,
        BASE / "master_thesis/geodesic/lr_p.csv",
        "seismic", False,
    ),
    (
        "BH2_Hp",  # (d): 旧図は j≈163 (誤)、正しくは j≈213
        BASE / "master_thesis/data/ur_p_2/H_p_val.csv",
        r"$\mathcal{H}_j^+$",
        J_BH, 3.0,
        BASE / "master_thesis/geodesic/ur_p_2.csv",
        "seismic", False,
    ),
    (
        "WH_Hp",
        BASE / "master_thesis/data/ur_p/H_p_val.csv",
        r"$\mathcal{H}_j^+$",
        J_WH, 0.8,
        BASE / "master_thesis/geodesic/ur_p.csv",
        "seismic", False,
    ),
    (
        "WH_Hm",
        BASE / "master_thesis/data/ur_m/H_m_val.csv",
        r"$\mathcal{H}_j^-$",
        J_WH, 8.0,
        BASE / "master_thesis/geodesic/ur_m.csv",
        "seismic", False,
    ),
    (
        "BH_p0_Hp",
        BASE / "master_thesis_p=0/data/lr_p/H_p_val.csv",
        r"$\mathcal{H}_j^+$",
        J_BH, 8.0,
        BASE / "master_thesis_p=0/geodesic/lr_p.csv",
        "seismic", False,
    ),
    (
        "WH_p0_Hm",
        BASE / "master_thesis_p=0/data/ur_m/H_m_val.csv",
        r"$|\mathcal{H}_j^-|$",
        J_WH, 8.0,
        BASE / "master_thesis_p=0/geodesic/ur_m.csv",
        "binary", True,
    ),
    (
        "WH2_p0_Hm",
        BASE / "thesis/data/p=0/ur_2/H_m_val.csv",
        r"$\mathcal{H}_j^-$",
        J_BH, 2.0,
        BASE / "thesis/geodesic/ur2.csv",
        "seismic", False,
    ),
]


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def load_csv(path):
    """ヘッダなし CSV を (N_time, L) の ndarray として読み込む。"""
    return np.loadtxt(path, delimiter=",")


def load_geodesic(path):
    """x,t ヘッダ付き geodesic CSV を読み込む。t>=0 の部分を返す。"""
    geo = np.genfromtxt(path, delimiter=",", names=True)
    mask = geo["t"] >= 0
    return geo["x"][mask] / eps, geo["t"][mask]   # (j_scaled, t)


def plot_panel(name, csv_path, ylabel, j_h, t_max, geo_path, cmap, use_abs):
    if not csv_path.exists():
        print(f"  SKIP (not found): {csv_path}")
        return

    data = load_csv(csv_path)          # (999, 300)
    n_rows = min(int(t_max / DT), data.shape[0])
    data_plot = data[:n_rows, :]
    if use_abs:
        data_plot = np.abs(data_plot)

    clim = float(np.nanmax(np.abs(data_plot)))
    if clim == 0 or not np.isfinite(clim):
        clim = 1.0

    fig, ax = plt.subplots(figsize=(3.375, 2.8), facecolor="white")
    ax.set_facecolor("white")

    vmin = 0 if use_abs else -clim
    im = ax.imshow(
        data_plot,
        aspect="auto",
        origin="lower",
        extent=[0, L, 0, t_max],
        cmap=cmap,
        vmin=vmin,
        vmax=clim,
        interpolation="nearest",
    )

    ax.axvline(j_h, color="black", linestyle="--", linewidth=0.8, label="Event horizon")

    if geo_path is not None and geo_path.exists():
        j_geo, t_geo = load_geodesic(geo_path)
        mask = t_geo <= t_max
        ax.plot(j_geo[mask], t_geo[mask], color="cyan", linestyle="--", linewidth=0.8, label="Geodesic")

    ax.set_xlim(0, L)
    ax.set_ylim(0, t_max)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$t$", rotation=0, labelpad=8)
    ax.legend(loc="upper left", frameon=True)

    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label(ylabel, rotation=0, labelpad=10)

    plt.tight_layout()
    out_path = OUT / f"{name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}  (j_h={j_h:.2f})")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
print(f"\nOutput directory: {OUT}\n")
for args in PANELS:
    name = args[0]
    print(f"[{name}]")
    plot_panel(*args)

print("\nDone.")
