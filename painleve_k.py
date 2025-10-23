import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
#複素数の偏角を計算するのに必要
import cmath
import math
import scipy.linalg
from matplotlib.animation import FuncAnimation, PillowWriter

import my_func.my_operator as my_operator
import my_func.fix_eigenvectors as fix_eigenvectors
import my_func.my_plot as my_plot
import my_func.beta as beta

pos = "lr" #lr, ur, ll, ul
BH = False
PBC = False
gaussian = True
PBC_center_is_left_side = True
p_const = True
p_0_to_1 = True
p_val = 0

l = 2
L = 100
epsilon = l/L
m = 0
t_i = 0
t_f = 10
dt = 0.01*(300/L)

times = np.arange(t_i + dt, t_f, dt)

def u_and_v():
    L = 300
    p = 1
    beta = 0
    l = 2*np.pi
    epsilon = l/L
    m = 0
    vecs_list = []
    k_vals = np.arange(-np.pi/2, np.pi/2, 2*np.pi/L)
    bands = np.zeros((len(k_vals), 4), dtype=float)

    #BdG行列の対角化
    for idx, k in enumerate(k_vals):
        h00 = -(p - epsilon*m)
        h01 = p*np.cos(k) - beta*np.sin(k)
        h32 = -(p*np.cos(k) + beta*np.sin(k))
        h03 = 1j*np.sin(k)
        h12 = 1j*np.sin(k)

        h = np.array([
            [h00,           h01,           0,            h03],
            [np.conj(h01),  h00,           h12,          0  ],
            [0,             np.conj(h12),  -h00,         np.conj(h32)],
            [np.conj(h03),  0,             h32,          -h00]
        ], dtype=complex)

        vals, vecs = LA.eigh(h*(-1/epsilon))
        bands[idx, :] = vals
        vecs_list.append(vecs.conj().T)

    k_vals_2 = np.arange(0, np.pi/2, 2*np.pi/L)
    for band_num in range(4):
        for i,k in enumerate(k_vals_2):
            #print("|k| = " + str(k))
            #print("・k = " + str(k_vals[-i-1]) + " and " + str(k_vals[i+1]))
            for j in range(2):
                tmp1 = vecs_list[-i-1][band_num, j].conj() / vecs_list[i+1][3-band_num, j+2]
                tmp2 = vecs_list[-i-1][band_num, j+2].conj() / vecs_list[i+1][3-band_num, j]
                vecs_list[i+1][3-band_num, j+2] *= tmp1
                vecs_list[i+1][3-band_num, j] *= tmp2

    #u_jl, v_jlの計算
    u = np.zeros((L,2), dtype=complex)
    v = np.zeros((L,2), dtype=complex)
    k0 = 0.5
    x0 = int(L/2)
    dk = 0.1

    weights = np.exp(-(k_vals - k0)**2 / (2*dk**2)) * np.exp(-1j*k_vals*x0)
    A = 1 / np.sqrt(np.sum(np.abs(weights)**2)) #確認済み

    for j in range(int(L)):
        for i, k in enumerate(k_vals):
            if k<0:
                continue
            else:
                band_num = 0
            coeff = (1/np.sqrt(L/2)) * A * weights[i]
            if j%2 == 0:
                u[j,0] += coeff * vecs_list[i][band_num, 0].conj() * np.exp(1j*k*j)
                v[j,0] += coeff * vecs_list[i][band_num, 0].conj() * np.exp(-1j*k*j)
            else:
                u[j,1] += coeff * vecs_list[i][band_num, 1].conj() * np.exp(1j*k*j)
                v[j,1] += coeff * vecs_list[i][band_num, 1].conj() * np.exp(-1j*k*j)
    return u, v

def generate_BdG_matrix():

    H = np.zeros((2*L, 2*L), dtype=complex)

    def p(j):
        if p_const:
            return p_val
        else:
            if not PBC:
                if p_0_to_1:
                    if j < int(L/4):
                        return 10**-5
                    else:
                        return 1
                else:
                    if j < int(L/4):
                        return 1
                    else:
                        return 10**-5
            else:
                if p_0_to_1:
                    if int(L/4) < j < int(3*L/4):
                        return 10**-5
                    else:
                        return 1
                else:
                    if int(L/4) < j < int(3*L/4):
                        return 1
                    else:
                        return 10**-5

    def dif_x_p(j):
        return (p(j+1) - p(j-1))/2
    for j in range(L):
        print(str(j)+" : "+str(beta.beta(j,L,pos,epsilon)))
    # beta(j) の値
    beta_vals = [beta.beta(j,L,pos,epsilon) for j in range(L)]

    # プロット
    plt.figure(figsize=(12, 8))

    # beta(j)
    plt.subplot(3, 1, 1)
    plt.plot(range(L), beta_vals, label='beta(j)')
    plt.ylim(-2,2)
    plt.title("beta(j)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    for i in range(2*L):
        for j in range(2*L):
            #左上
            if i < L and j < L:
                if i == j:
                    H[i, j] = -1*(p(i)-epsilon*(0.5*dif_x_p(i) + m))
                elif i-j == 1:
                    H[i, j] = (p(i) - 1j*beta.beta(i,L,pos,epsilon))/2
                elif j-i == 1:
                    H[i, j] = (p(j) + 1j*beta.beta(j,L,pos,epsilon))/2
            #右上
            elif i < L and j >= L:
                if j-i == L-1:
                    H[i, j] = -1/2
                elif j-i == L+1:
                    H[i, j] = 1/2
            #左下
            elif i >= L and j < L:
                if i-j == L-1:
                    H[i, j] = -1/2
                elif i-j == L+1:
                    H[i, j] = 1/2
            #右下
            else:
                if i == j:
                    H[i, j] = p(i-L)-epsilon*(0.5*dif_x_p(i-L) + m)
                elif i-j == 1:
                    H[i, j] = -1*(p(i-L) + 1j*beta.beta(i-L,L,pos,epsilon))/2
                elif j-i == 1:
                    H[i, j] = -1*(p(j-L) - 1j*beta.beta(j-L,L,pos,epsilon))/2
    if PBC:
        #周期境界条件
        #orange
        H[0,L-1] = 0.5*(p(L) - 1j*beta.beta(L,L,pos,epsilon))
        H[2*L-1,L] = -0.5*(p(L) - 1j*beta.beta(L,L,pos,epsilon))
        #blue
        H[L-1,0] =  0.5*(p(L) + 1j*beta.beta(L,L,pos,epsilon))
        H[L,2*L-1] = -0.5*(p(L) + 1j*beta.beta(L,L,pos,epsilon))
        #black
        H[0,2*L-1] = -0.5
        H[L-1,L] = 0.5
        #red
        H[2*L-1,0] = -0.5
        H[L,L-1] = 0.5

    return (-1/epsilon)*H

def arrange_eigenvectors(eigenvalues, eigenvectors, tol=1e-8):
    # インデックス抽出
    idx_pos = np.where(eigenvalues > tol)[0]
    idx_neg = np.where(eigenvalues < -tol)[0]
    idx_zero = np.where(np.abs(eigenvalues) <= tol)[0]

    # ソート
    idx_pos_sorted = idx_pos[np.argsort(eigenvalues[idx_pos])]
    idx_neg_sorted = idx_neg[np.argsort(np.abs(eigenvalues[idx_neg]))]

    # ゼロの振り分け
    idx_combined = []
    if idx_zero.size > 0:
        # 先頭に１つ
        idx_combined.append(idx_zero[0])
    # 正側を続ける
    idx_combined.extend(idx_pos_sorted.tolist())
    if idx_zero.size > 1:
        # 残りのゼロを中ほどに
        idx_combined.extend(idx_zero[1:].tolist())
    # 負側を末尾に
    idx_combined.extend(idx_neg_sorted.tolist())

    # 並べ替え適用
    eigenvalues_sorted  = eigenvalues[idx_combined]
    eigenvectors_sorted = eigenvectors[:, idx_combined]
    return eigenvalues_sorted, eigenvectors_sorted

def fix_phase(eigenvectors):
    for i in range(2*L):
        phase = cmath.phase(eigenvectors[i, i])
        eigenvectors[:,i] = np.exp(-1j*phase) * eigenvectors[:,i]
    return eigenvectors

def is_hermitian(matrix):

    return np.allclose(matrix, np.conj(matrix.T), atol=1e-100)

def is_unitary(matrix):

    identity_matrix = np.eye(matrix.shape[0])

    return np.allclose(np.dot(np.conj(matrix.T), matrix), identity_matrix,atol=1e-13)

def plot_mode_function(U):

    f = U[:L,L:]
    x = np.linspace(0, l, L)
    plt.figure()
    for i in range(L):
        if i < 5:
            plt.plot(x, f[:, i]/np.sqrt(epsilon), label='f_'+str(i))
            plt.xlabel('x')
            plt.ylabel('f')
            plt.title('Mode Function')
            plt.legend()
            plt.grid(True)
            plt.show()
    return

def generate_U(eigenvectors):
    U = np.zeros((2*L, 2*L), dtype=complex)
    for i in range(L):
        U[i,:] = (eigenvectors[i,:] + eigenvectors[i+L,:])/np.sqrt(2)
        U[i+L,:] = (eigenvectors[i,:] - eigenvectors[i+L,:])/(1j*np.sqrt(2))
    return U

def generate_c_dag_c(eigenvectors):
    c_dag_c_list = []
    for i in range(L):
        c_dag_c_tmp = np.zeros((L, L), dtype=complex)
        for n in range(L):
            for m in range(L):
                c_dag_c_tmp[n,m] = np.conj(eigenvectors[i,n]) * eigenvectors[i,m]
                c_dag_c_tmp[n,m] += -1*np.conj(eigenvectors[i,m+L]) * eigenvectors[i,n+L]
                if n == m:
                    for k in range(L):
                        c_dag_c_tmp[n,m] += np.conj(eigenvectors[i,k+L]) * eigenvectors[i,k+L]
        c_dag_c_list.append(c_dag_c_tmp)
        print("c†c作成中:" + str(int(i/L*100))+"%")
    return c_dag_c_list

def generate_time_evolution_operator(eigenvalues):
    H = np.zeros((L, L), dtype=complex)
    for i in range(L):
        H[i,i] = eigenvalues[i]
    return scipy.linalg.expm(-1j*H*dt)

def initialize_state_vector(eigenvectors, gaussian, PBC):
    psi_tmp = np.zeros((L, 1), dtype=complex)

    if gaussian and PBC:
        # “中心”を L (= 0 と同じ) に設定
        center = L
        # 幅と sigma の設定は従来通り
        width = int(L/8)
        sigma = width / 3

        # 重み付け計算
        for j in range(-width, width + 1):
            # 周期境界条件を考慮してモジュロ演算
            idx = (center + j) % L
            weight = np.exp(- (j ** 2) / (2 * sigma ** 2))
            # 波動関数への加算
            for i in range(L):
                psi_tmp[i, 0] += weight * np.conj(eigenvectors[idx, i])

    elif gaussian and (not PBC):
        center = int(0.8*L)
        width = int(0.15*L) -2
        sigma = width/3
        k0 = -1
        for j in range(-width, width + 1):
            idx = center + j
            if int(L*0.4) <= idx < L:  # 境界チェック
                weight = np.exp(- (j ** 2) / (2 * sigma ** 2)) #* np.exp(-1j*k0*idx)
                for i in range(L):
                    psi_tmp[i, 0] += weight * np.conj(eigenvectors[idx, i])

    elif (not gaussian) and PBC:
        if PBC_center_is_left_side:
            center = int(2)
        else:
            center = int(L-2)
        for i in range(L):
            psi_tmp[i, 0] += np.conj(eigenvectors[center, i])

    else:
        center = int((150/300)*L)
        for i in range(L):
            psi_tmp[i, 0] += np.conj(eigenvectors[center, i])
    psi_tmp /= np.linalg.norm(psi_tmp)
    return psi_tmp

def initialize_state_vector_ab(eigenvectors):
    psi_tmp = np.zeros((L, 1), dtype=complex)
    u, v = u_and_v()
    for j in range(0,L,2):
            for n in range(L):
                psi_tmp[n, 0] += u[j, 0] * eigenvectors[j,n].conj()
                psi_tmp[n, 0] += v[j, 0] * eigenvectors[j,n+L]
                psi_tmp[n, 0] += u[j+1, 1] * eigenvectors[j+1,n].conj()
                psi_tmp[n, 0] += v[j+1, 1] * eigenvectors[j+1,n+L]
    psi_tmp /= np.linalg.norm(psi_tmp)
    return psi_tmp

def plot_density_evolution(density, diff, PBC):
    """
    density: shape (num_time_steps, L)
    diff:    差分表示フラグ
    PBC:     周期境界条件フラグ
    """
    # ——— 配列化＋実部抽出 ———
    # まず complex 型で受け取り
    arr = np.array(density, dtype=complex)
    # 実部だけを取り出して float 配列に
    density_array = np.real(arr).astype(float)

    nt, L = density_array.shape

    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 25,
        'axes.titlesize': 22,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    plt.figure(figsize=(8, 6))

    cmap = 'hot'  # カラーマップを画像に合わせる

    if diff:
        plt.imshow(
            density_array, aspect='auto', origin='lower',
            extent=[0, L, times[0], times[-1]],
        vmin=-0.01*(300/L), vmax=0.01*(300/L),
        cmap=cmap
        )
    else:

        plt.imshow(density_array, aspect='auto', origin='lower',
                extent=[0, L, times[0], times[-1]], cmap=cmap)

    # カラーバー
    cbar = plt.colorbar(location='left')
    cbar.set_label("δ" + r'$\langle c_j^\dagger c_j\rangle$', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('j', fontweight='bold')
    plt.ylabel('t', fontweight='bold')

    # 境界線
    '''
    line_kwargs = dict(color='deepskyblue', linestyle='--', linewidth=5)
    if PBC:
        plt.axvline(x=L*0.25, **line_kwargs)
        plt.axvline(x=L*0.75, **line_kwargs)
        plt.axvline(x=L*(146/300), **line_kwargs)
        plt.axvline(x=L*(154/300), **line_kwargs)
    else:
        plt.axvline(x=L/4, **line_kwargs)
    '''

    plt.tight_layout()
    plt.savefig('density_evolution.png', dpi=300, bbox_inches='tight')
    plt.subplots_adjust(left=0.2)
    plt.show()

def c_dag_c_vacuum(eigenvectors):
    c_dag_c_v = []
    for i in range(L):
        total = 0.0
        for k in range(L):
            total += abs(eigenvectors[i, k+L])**2
        c_dag_c_v.append(total)
    return c_dag_c_v

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
    line, = ax.plot([], [], lw=1.8, color=cmap_line, label=line_label)
    ax.set_xlim(1, L)
    ax.set_ylim(density_arr.min(), density_arr.max())
    ax.set_xlabel(xlabel, fontsize=16)#ax.set_ylim(-0.01*(300/L),0.01*(300/L))
    ax.set_ylabel(ylabel, fontsize=16)
    #ax.set_title("Time Evolution of " + r'δ$\langle c_j^\dagger c_j \rangle$')
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":")

    for pos in horizon_positions:
        ax.axvline(x=pos, color="red", ls="--", lw=1)

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

def adjust_eigenvectors(eigenvactors):
        V = np.zeros((2*L, 2*L), dtype=complex)
        for i in range(L):
            V[:,i] = eigenvactors[:,i]
            V[:L,2*L-1-i] = np.conj(eigenvactors[L:,i])
            V[L:,2*L-1-i] = np.conj(eigenvactors[:L,i])
        return V

def calculate_energy(eigenvalues, psi):

    H = np.zeros((L, L), dtype=complex)
    for i in range(L):
        H[i,i] = eigenvalues[i]
    return (psi.conj().T @ H @ psi).item()  # ⟨psi|H|psi⟩

def save_density_pair_average_animation(
    density,
    times,
    gif_path,
    *,
    horizon_positions=None,
    fps: int = 20,
    xlabel: str = "Pair Index",
    ylabel: str = "Average Density",
    line_label: str = r"$\overline{\rho}_{j,j+1}(t)$",
    cmap_line: str = "blue",
    PBC
):
    """
    隣り合うペアごとに平均をとった密度を1点としてプロットするアニメーションを GIF として保存します。

    density           : 2 次元配列 (N, L) あるいは同形状の list。行＝時刻，列＝格子サイト
    times             : 1 次元配列 (N,)   ─ 対応する時間点
    gif_path          : 生成した GIF を保存するファイルパス
    horizon_positions : 破線を引く x 座標のシーケンス（既定 None → [L/8, 3L/8] など）
    fps               : GIF のフレーム毎秒数（既定 20）
    xlabel, ylabel    : 軸ラベル
    line_label        : 凡例ラベル
    cmap_line         : 折れ線の色（matplotlib が解釈できる任意指定）
    PBC               : 周期境界条件を考慮する場合 True
    """
    # ---------- 前処理 ------------------------------------------------------
    density_arr = np.asarray(density)
    density_arr = np.real_if_close(density_arr, tol=1000).astype(float)
    N, L = density_arr.shape
    if len(times) != N:
        raise ValueError("times の長さと density の行数が一致していません。")

    # 隣り合うペアごとに平均を取る
    # L が奇数の場合は最後のサイトを無視
    pair_count = L // 2
    # shape: (N, pair_count)
    density_pair = 0.5 * (
        density_arr[:, 0:2*pair_count:2] + density_arr[:, 1:2*pair_count:2]
    )
    # 更新
    density_arr = density_pair
    L2 = pair_count

    # horizon_positions のデフォルト
    if horizon_positions is None:
        if PBC:
            # 元の L に対する割合を new インデックスに変換
            horizon_positions = [L2/4, 3*L2/4]
        else:
            horizon_positions = [L2 / 4]

    # ---------- 図オブジェクトの初期化 --------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    x_vals = np.arange(1, L2 + 1)
    line, = ax.plot([], [], lw=1.8, color=cmap_line, label=line_label)
    ax.set_xlim(1, L2)
    ax.set_ylim(density_arr.min(), density_arr.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Time Evolution of Pair-Averaged Density")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":")

    for pos in horizon_positions:
        ax.axvline(x=pos, color="red", ls="--", lw=1)

    # ---------- アニメーション用コールバック -------------------------------
    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(x_vals, density_arr[frame])
        ax.set_title(
            f"Time Evolution of Pair-Averaged Density  (t = {times[frame]:.3f})"
        )
        return (line,)

    # ---------- アニメーション生成と保存 -----------------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=N,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    print("保存しました: {}".format(gif_path))
    plt.close(fig)

def initialize_state_vector_a_plus(eigenvectors):
    psi_tmp = np.zeros((L, 1), dtype=complex)
    if pos == "ur" or pos == "lr":
        j0 = int(0.2*L)
    else:
        j0 = int(0.8*L)
    print(pos)
    print(j0)
    sigma = 0.05*L
    # j0を中心としたガウシアン波束の重み
    weights = np.exp(-((np.arange(L) - j0) ** 2) / (2 * sigma ** 2))
    weights /= np.linalg.norm(weights)  # 規格化

    for j in range(L):
        for n in range(L):
            #+
            if pos == "ur" or pos == "lr":
                psi_tmp[n, 0] += weights[j] * (
                1/np.sqrt(2) * (np.exp(1j*np.pi/4) * eigenvectors[j, n+L] + np.exp(-1j*np.pi/4) * eigenvectors[j, n].conj())
            )
            #-
            else:
                psi_tmp[n, 0] += weights[j] * (
                1/np.sqrt(2) * (np.exp(-1j*np.pi/4) * eigenvectors[j, n+L] + np.exp(1j*np.pi/4) * eigenvectors[j, n].conj())
                )

    # 最終的な規格化
    psi_tmp /= np.linalg.norm(psi_tmp)
    return psi_tmp

#BdG行列を作成
h = generate_BdG_matrix()
#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(h)
eigenvectors = adjust_eigenvectors(eigenvectors)

c_dag_c_list = generate_c_dag_c(eigenvectors)
cj1_cj_list = my_operator.generate_cj_dag_cj1_dag(eigenvectors,L)
cj1_dag_cj_list = my_operator.generate_cj_dag_cj1(eigenvectors,L)
H_ps, H_ms = my_operator.H_p_m_K(cj1_cj_list, cj1_dag_cj_list, L, epsilon, pos)
U_dt = generate_time_evolution_operator(eigenvalues)

density = []
density_diff = []

#a+から移植
H_p_list = []
H_m_list = []
H_pm_list = []

#psi = initialize_state_vector(eigenvectors,gaussian=gaussian, PBC=PBC)
psi = initialize_state_vector_a_plus(eigenvectors)

#真空の期待値

#a+から移植
H_p0 = []
H_m0 = []
H_pm0 = []

c_dag_c_v = c_dag_c_vacuum(eigenvectors)
Hp_v_k, Hm_v_k = my_operator.H_vacuum_k(eigenvectors,L, epsilon,p_val, pos)

#a+から移植
'''
for j, H_p in enumerate(H_ps):

    val = np.conj(psi.T) @ H_p @ psi
    H_p0.append(val.item())

for j, H_m in enumerate(H_ms):
    val = np.conj(psi.T) @ H_m @ psi
    H_m0.append(val.item())

for j, H_pm in enumerate(H_pms):
    val = np.conj(psi.T) @ H_pm @ psi
    H_pm0.append(val.item())
'''

# 時間発展
for i, _ in enumerate(times, start=1):
    psi = U_dt @ psi
    psi /= np.linalg.norm(psi)

    #energy = calculate_energy(eigenvalues, psi)
    #energy_t.append(energy)

    current_density = []
    current_diff = []

    #a+から移植
    current_H_p = []
    current_H_m = []
    current_H_pm = []

    #a+から移植
    #H_pの期待値
    for j, H_p in enumerate(H_ps):
        val = np.conj(psi.T) @ H_p @ psi
        current_H_p.append(val.item() - Hp_v_k[j])
    H_p_list.append(current_H_p)

    #H_mの期待値
    for j, H_m in enumerate(H_ms):
        val = np.conj(psi.T) @ H_m @ psi
        current_H_m.append(val.item() - Hm_v_k[j])
    H_m_list.append(current_H_m)

    #H_pmの期待値
    # for j, H_pm in enumerate(H_pms):
    #     val = np.conj(psi.T) @ H_pm @ psi
    #     current_H_pm.append(val.item() - Hpm_v[j])
    # H_pm_list.append(current_H_pm)

    for j, c_dag_c in enumerate(c_dag_c_list):
        val = np.conj(psi.T) @ c_dag_c @ psi
        current_density.append(val.item())
        diff = val.item() - c_dag_c_v[j]
        current_diff.append(diff)

    density.append(current_density)
    density_diff.append(current_diff)

    print("時間発展中:"+str(int(i/len(times)*100)) + "%")

#a+から移植
my_plot.plot_density_evolution(H_p_list, diff=False, PBC=PBC, times=times, value_name="H_p")
my_plot.plot_density_evolution(H_m_list, diff=False, PBC=PBC, times=times, value_name="H_m")
#my_plot.plot_density_evolution(H_pm_list, diff=False, PBC=PBC, times=times, value_name="H_pm")
my_plot.plot_density_evolution(density_diff, diff=True, PBC=PBC,times=times, value_name="δc†c")
my_plot.save_density_animation(H_p_list, times, "figure/H_p.gif", PBC=PBC)
my_plot.save_density_animation(H_m_list, times, "figure/H_m.gif", PBC=PBC)
#my_plot.save_density_animation(H_pm_list, times, "figure/H_pm.gif", PBC=PBC)
my_plot.save_density_animation(density, times, "figure/c†c.gif", PBC=PBC)
my_plot.save_density_animation(density_diff, times, "figure/δc†c.gif", PBC=PBC)