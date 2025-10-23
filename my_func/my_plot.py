import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
#複素数の偏角を計算するのに必要
import cmath
import scipy.linalg
from matplotlib.animation import FuncAnimation, PillowWriter
'''
def plot_density_evolution(density, diff, PBC, times, value_name):
    """
    density: shape (num_time_steps, L)
    diff:    差分表示フラグ
    PBC:     周期境界条件フラグ
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ——— 配列化＋実部抽出 ———
    arr = np.array(density, dtype=complex)
    density_array = np.real(arr).astype(float)

    # NaN を含む可能性に備えて min/max を安全に取得
    data_min = np.nanmin(density_array)
    data_max = np.nanmax(density_array)

    # vmin==vmax（定数配列）の場合に備えて微小幅を確保
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        # すべて NaN 等、可視化不能な場合は早期リターン（必要に応じて挙動を変更）
        raise ValueError("density_array に有効な数値がありません。")
    if data_min == data_max:
        eps = 1e-12 if data_min == 0 else abs(data_min) * 1e-12
        data_min -= eps
        data_max += eps

    nt, L = density_array.shape

    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 25,
        'axes.titlesize': 22,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    plt.figure(figsize=(8, 6))

    if diff:
        cmap = 'seismic'  # 正負が分かりやすい赤青配色
        vmin = data_min
        vmax = data_max
    else:
        cmap = 'viridis'  # 連続値に強い配色
        vmin = data_min
        vmax = data_max

    # times は単調増加が望ましい（imshow の extent 表示のため）
    # 必要に応じて並べ替えや検証を追加可能
    plt.imshow(
        density_array,
        aspect='auto',
        origin='lower',
        extent=[0, L, times[0], times[-1]],
        cmap=cmap,
        vmin=float(vmin),  # ← タプルではなく float を渡す
        vmax=float(vmax)
    )

    # カラーバー
    cbar = plt.colorbar(location='left')
    cbar.ax.tick_params(labelsize=16)

    plt.xlabel('j', fontweight='bold')
    plt.ylabel('t', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure/' + value_name + '.png', dpi=300, bbox_inches='tight')
'''
def plot_density_evolution(density, diff, PBC, times, value_name):
    """
    density: shape (num_time_steps, L)
    diff:    差分表示フラグ
    PBC:     周期境界条件フラグ
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # ——— 配列化＋実部抽出 ———
    arr = np.array(density, dtype=complex)
    density_array = np.real(arr).astype(float)

    data_min = np.nanmin(density_array)
    data_max = np.nanmax(density_array)
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise ValueError("density_array に有効な数値がありません。")
    if data_min == data_max:
        eps = 1e-12 if data_min == 0 else abs(data_min) * 1e-12
        data_min -= eps
        data_max += eps

    nt, L = density_array.shape

    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 25,
        'axes.titlesize': 22,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    plt.figure(figsize=(8, 6))
    cmap = 'seismic' if diff else 'viridis'

    # 背景の密度プロット
    plt.imshow(
        density_array,
        aspect='auto',
        origin='lower',
        extent=[0, L, times[0], times[-1]],
        cmap=cmap,
        vmin=float(data_min),
        vmax=float(data_max)
    )

    # —— geodesic.dat を重ねる（物理範囲→表示範囲の対応） ——
    try:
        # 実行スクリプト基準で geodesic.dat を探す
        file_path = "geodesic.dat"
        geodesic_data = np.loadtxt(file_path, delimiter=",")

        x = geodesic_data[:, 0]  # 物理空間座標（0〜2π）
        t = geodesic_data[:, 1]  # 物理時間（0〜20）

        # スケーリング変換
        x_scaled = (x / (2 * np.pi)) * L
        t_scaled = (t / 20.0) * (times[-1] - times[0]) + times[0]

        #plt.plot(x_scaled, t_scaled, color='white', linewidth=2, label='geodesic')
        #plt.legend(loc='upper right', fontsize=12)
    except Exception as e:
        print(f"Warning: geodesic.dat の重ね描画に失敗しました: {e}")
    # ————————————————————————————————

    cbar = plt.colorbar(location='left')
    cbar.ax.tick_params(labelsize=16)

    plt.xlabel('j', fontweight='bold')
    plt.ylabel('t', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure/' + value_name + '.png',
                dpi=300, bbox_inches='tight', transparent=True)



def save_density_animation(
    density,
    times,
    gif_path,
    *,
    horizon_positions=None,
    fps: int = 20,

    xlabel: str = "Lattice Site Index",
    ylabel: str = "Density",
    line_label: str = 'δ' + r'$\langle c_j^\dagger c_j\rangle$',

    cmap_line: str = "blue",
    PBC
):
    """
    density           : 2 次元配列 (N, L) あるいは同形状の list。行＝時刻，列＝格子サイト
    times             : 1 次元配列 (N,)   ─ 対応する時間点
    gif_path          : 生成した GIF を保存するファイルパス
    horizon_positions : 破線を引く x 座標のシーケンス（既定 None → [L/4, 3L/4]）
    fps               : GIF のフレーム毎秒数（既定 20）
    xlabel, ylabel    : 軸ラベル
    line_label        : 凡例ラベル
    cmap_line         : 折れ線の色（matplotlib が解釈できる任意指定）
    """
    # ---------- 前処理 ------------------------------------------------------
    # 1. ndarray 化
    density_arr = np.asarray(density)
    #print("density_arr.shape =", density_arr.shape)
    # 2. 実部のみを使用（十分小さい虚部は無視）
    density_arr = np.real_if_close(density_arr, tol=1000)  # tol は 10^(-tol) 判定
    density_arr = density_arr.astype(float)                # 明示的に float32/64 へ
    N, L = density_arr.shape
    if len(times) != N:
        raise ValueError("times の長さと density の行数が一致していません。")
    if horizon_positions is None:
        if PBC:
            horizon_positions = [L / 4, 3 * L / 4, L*(146/300), L*(154/300)]
        else:
            horizon_positions = [L / 4]

    # ---------- 図オブジェクトの初期化 --------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    line, = ax.plot([], [], lw=1.8, color=cmap_line)#, label=line_label)
    ax.set_xlim(1, L)
    ax.set_ylim(density_arr.min(), density_arr.max())
    ax.set_xlabel(xlabel, fontsize=16)#ax.set_ylim(-0.01*(300/L),0.01*(300/L))
    #ax.set_ylabel(ylabel, fontsize=16)
    #ax.set_title("Time Evolution of " + r'δ$\langle c_j^\dagger c_j \rangle$')
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":")

    #for pos in horizon_positions:
    #    ax.axvline(x=pos, color="red", ls="--", lw=1)

    # ---------- アニメーション用コールバック -------------------------------
    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(np.arange(1, L + 1), density_arr[frame])
        #ax.set_title(
        #    f"Time Evolution of "+r'δ$\langle c_j^\dagger c_j \rangle$'+"  (t = {times[frame]:.3f})"
        #)
        return (line,)

    # ---------- アニメーション生成と保存 -----------------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=N,
        init_func=init,
        blit=True,
        interval=1000 / fps,     # ミリ秒
    )
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    print("保存しました")
    plt.close(fig)  # 余分なウインドウを閉じる
