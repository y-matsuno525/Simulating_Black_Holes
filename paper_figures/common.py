"""paper_figures/common.py

論文 (paper/main.tex) の図を、集約済みデータ paper_data/ から再生成するための共通処理。

データ規約（paper_data/ 配下の実ファイルを確認した結果）:
  - エネルギー密度 ``*_val.csv`` は **ヘッダ無し・time 列無し** の ``(T, L)`` 配列。
    行 i が時刻 ``t = i * dt``、列 j が格子点。``np.loadtxt(delimiter=",")`` で直読みする
    （旧 notebook の ``skiprows=1`` は誤りで先頭時刻を落としてしまうため踏襲しない）。
  - geodesic CSV は ``x,t`` ヘッダ付き。``x`` は物理座標 (0..ell)、``t`` は負値も含む。
    格子 index へは ``j = x / epsilon`` で変換する。

注意: 本スクリプト群は「保存データからの同等図」を再現する。論文 PDF の各パネルは
特定の初期条件 (j0=30/40/180 など) の run を使っており、それらを厳密再現したい場合は
simulation.py を該当 config で回す（README の対応表参照）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --- パス -----------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
PAPER_DATA = REPO / "paper_data"
# 生成図の出力先。論文本体の figure（paper/figure/, main.tex が参照）は上書きせず、
# 保存データからの「同等再現図」をここに出力する。
PAPER_FIG = REPO / "paper_figures" / "generated"
PAPER_FIG_ORIG = REPO / "paper" / "figure"

# --- 既定の物理パラメータ（論文 Sec. III の標準設定）---------------------
L_DEFAULT = 300
ELL_DEFAULT = 2.0 * np.pi
EPS_DEFAULT = ELL_DEFAULT / L_DEFAULT
DT_DEFAULT = 0.01

# 正準ホライズン位置（replot_paper_figs.py と同じ解析値, L=300, ell=2pi）。
#   BH: j_h = 2/3 L + arctanh(2/3)/(3 eps) ≈ 212.82
#   WH: j_h = 1/3 L - arctanh(2/3)/(3 eps) ≈  87.18
J_BH = 2 / 3 * L_DEFAULT + np.arctanh(2 / 3) / (3 * EPS_DEFAULT)
J_WH = 1 / 3 * L_DEFAULT - np.arctanh(2 / 3) / (3 * EPS_DEFAULT)


# --- データ読み込み -------------------------------------------------------
def load_val(path) -> np.ndarray:
    """エネルギー密度時系列 ``(T, L)`` を読み込む（ヘッダ無し・time列無し）。"""
    return np.loadtxt(path, delimiter=",")


def load_geodesic(path, epsilon: float = EPS_DEFAULT):
    """``x,t`` 形式の geodesic CSV を読み、``(j, t)`` を返す。``j = x / epsilon``。"""
    geo = np.genfromtxt(path, delimiter=",", names=True)
    return geo["x"] / epsilon, geo["t"]


# --- ホライズン位置 -------------------------------------------------------
def horizon_j(A, B, C, x0, *, target=1.0, ell=ELL_DEFAULT, L=L_DEFAULT):
    """beta(x)=A tanh(B(x-x0))+C が ``beta=target`` となる格子位置 j_h を返す。

    chi^+ 成分のホライズンは target=-1、chi^- 成分は target=+1。
    ``(target-C)/A`` が (-1,1) の外なら（その成分のホライズンが無い場合）None。
    """
    u = (target - C) / A
    if not -1.0 < u < 1.0:
        return None
    x_h = x0 + np.arctanh(u) / B
    return L * x_h / ell


# --- 描画 -----------------------------------------------------------------
def plot_heatmap(
    ax,
    data,
    *,
    t_f,
    jh=None,
    geodesic=None,
    label=r"$\mathcal{H}_j$",
    cmap="seismic",
    use_abs=False,
    dt=DT_DEFAULT,
    L=L_DEFAULT,
    title=None,
    horizon_color="red",
    geo_color="cyan",
):
    """時空 heatmap を 1 つの Axes に描く。imshow + horizon 線 + geodesic 重畳。

    戻り値: imshow の戻り (colorbar 用)。
    """
    from matplotlib.colors import TwoSlopeNorm

    nt = min(int(round(t_f / dt)), data.shape[0])
    data = data[:nt, :]
    if use_abs:
        data = np.abs(data)

    m = float(np.nanmax(np.abs(data)))
    if m == 0 or not np.isfinite(m):
        m = 1.0
    ny, nx = data.shape

    if use_abs:
        norm = None
        vmin, vmax = 0.0, m
    else:
        norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
        vmin = vmax = None

    im = ax.imshow(
        data,
        cmap=cmap,
        origin="lower",
        extent=[0, nx, 0, ny * dt],
        aspect="auto",
        interpolation="nearest",
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    if jh is not None:
        ax.axvline(jh, color=horizon_color, lw=1.5, ls="--", label="Event horizon")
    if geodesic is not None:
        jg, tg = geodesic
        mask = (tg >= 0) & (tg <= t_f)
        ax.plot(jg[mask], tg[mask], color=geo_color, lw=1.5, ls="--", label="Geodesic")
    ax.set_xlim(0, L)
    ax.set_ylim(0, t_f)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$t$", rotation=0, labelpad=8)
    if title:
        ax.set_title(title, fontsize=10)
    return im, label


# --- 指数フィット（表面重力）---------------------------------------------
def exp_fit(times, sigmas):
    """``A*exp(B t) - A`` でフィットし ``(A, B, t_fine, curve)`` を返す。"""
    from scipy.optimize import curve_fit

    def f(t, A, B):
        return A * np.exp(B * t) - A

    A0 = sigmas[0] if sigmas[0] != 0 else 1e-3
    (A, B), _ = curve_fit(f, times, sigmas, p0=[A0, 1.0], maxfev=10000)
    t_fine = np.linspace(times.min(), times.max(), 2000)
    return A, B, t_fine, f(t_fine, A, B)


def ensure_fig_dir():
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    return PAPER_FIG
