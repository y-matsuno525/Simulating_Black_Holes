import json
from pathlib import Path

import numpy as np
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

CONFIG = None
DENSITY_PLOT_CONFIGS = {}
ANIMATION_CONFIGS = {}


def configure(config, density_plot_configs, animation_configs):
    """Install prepared configuration for the numerical pipeline."""
    global CONFIG, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS
    global L, l, epsilon, p, m, chirality, beta_sign, beta_sign_value, t_i, t_f, dt, PBC, beta_profile
    global surface_gravity_beta, beta_width, beta_amplitude, beta_center_fraction
    global centered_beta_width, centered_beta_amplitude, centered_beta_center_fraction
    global j0, sigma, initial_direction_sign, times, mode_function_count
    global geodesic_points, surface_gravity_output_path, stagnation_position, outputs, output_dir
    global fft_observables, fft_remove_spatial_mean

    CONFIG = config
    DENSITY_PLOT_CONFIGS = density_plot_configs
    ANIMATION_CONFIGS = animation_configs
    L = CONFIG["L"]
    l = CONFIG["l"]
    epsilon = CONFIG["epsilon"]
    p = CONFIG["p"]
    m = CONFIG["m"]
    chirality = CONFIG["chirality"]
    beta_sign = CONFIG["beta_sign"]
    beta_sign_value = CONFIG["beta_sign_value"]
    t_i = CONFIG["t_i"]
    t_f = CONFIG["t_f"]
    dt = CONFIG["dt"]
    PBC = CONFIG["PBC"]
    beta_profile = CONFIG["beta_profile"]
    surface_gravity_beta = CONFIG.get("surface_gravity_beta", False)
    beta_width = CONFIG["beta_width"]
    beta_amplitude = CONFIG["beta_amplitude"]
    beta_center_fraction = CONFIG["beta_center_fraction"]
    centered_beta_width = CONFIG["centered_beta_width"]
    centered_beta_amplitude = CONFIG["centered_beta_amplitude"]
    centered_beta_center_fraction = CONFIG["centered_beta_center_fraction"]
    j0 = CONFIG["j0"]
    sigma = CONFIG["sigma"]
    initial_direction_sign = CONFIG["initial_direction_sign"]
    times = CONFIG["times"]
    mode_function_count = CONFIG["mode_function_count"]
    geodesic_points = CONFIG["geodesic_points"]
    surface_gravity_output_path = CONFIG["surface_gravity_output_path"]
    stagnation_position = CONFIG["stagnation_position"]
    fft_observables = CONFIG.get("fft_observables", ["H_p"])
    fft_remove_spatial_mean = CONFIG.get("fft_remove_spatial_mean", True)
    outputs = CONFIG["outputs"]
    output_dir = Path(CONFIG["output_dir"])

def is_hermitian(matrix):
    """Return True when a matrix is Hermitian to numerical precision."""
    is_hermitian = np.allclose(matrix, np.conj(matrix.T), atol=1e-10)

    if is_hermitian:
        return True
    else:
        return False

def beta_flat(j, L, beta_sign, epsilon):
    """Flat beta profile used as a no-horizon baseline."""
    return 0

def beta_centered_horizon(j, L, beta_sign, epsilon):
    """Centered smooth tanh beta profile."""
    # j は格子 index だが、tanh の引数では epsilon を掛けて物理座標スケールへ戻す。
    jh = int(centered_beta_center_fraction*L)
    return beta_sign_value*(centered_beta_amplitude*np.tanh(centered_beta_width*(j - jh)*epsilon) + centered_beta_amplitude)

def beta_pos_horizon(j, L, beta_sign, epsilon):
    """Positioned beta profile whose sign is explicit in config."""
    jh = int(beta_center_fraction*L)
    return beta_sign_value*beta_amplitude*np.tanh(3/beta_width*(j - jh)*epsilon) + beta_sign_value*beta_amplitude

def beta(j,L,beta_sign,epsilon):
    """Dispatch to the configured beta profile."""
    if surface_gravity_beta:
        width = 0.1
        A = 1
        jh = int(L/2)
        return A*np.tanh(width*(j - jh)*epsilon) + A
    beta_functions = {
        "flat": beta_flat,
        "centered_horizon": beta_centered_horizon,
        "pos_horizon": beta_pos_horizon,
    }
    return beta_functions[beta_profile](j, L, beta_sign, epsilon)

def compute_horizon_positions(num_samples=10000):
    """Return lattice-index positions where abs(beta)=1 by linear interpolation."""
    xs = np.linspace(0, L - 1, num_samples)
    vals = np.array([abs(beta(x, L, beta_sign, epsilon)) - 1 for x in xs])
    positions = []
    for idx in range(len(xs) - 1):
        v0 = vals[idx]
        v1 = vals[idx + 1]
        if v0 == 0:
            positions.append(xs[idx])
        elif v0 * v1 < 0:
            x0 = xs[idx]
            x1 = xs[idx + 1]
            positions.append(x0 - v0 * (x1 - x0) / (v1 - v0))
    if vals[-1] == 0:
        positions.append(xs[-1])

    unique_positions = []
    for position in positions:
        if not unique_positions or abs(position - unique_positions[-1]) > 1e-3:
            unique_positions.append(float(position))
    return unique_positions

def save_horizon_positions():
    """Save horizon positions in lattice and physical coordinates."""
    positions = compute_horizon_positions()
    if positions:
        data = np.column_stack([positions, np.asarray(positions)*epsilon])
    else:
        data = np.empty((0, 2))
    np.savetxt(
        resolve_output_path("horizon_positions.txt"),
        data,
        fmt="%.10e",
        header="j x",
        comments="",
    )
    return positions

def build_bdg_matrix(L, p, m, beta_sign, epsilon, PBC):
    """Build the 2L x 2L BdG Hamiltonian in particle-hole block form."""
    #BdGハミルトニアンの作成(符号関係は確認済み)
    H_BdG = np.zeros((2*L, 2*L), dtype=complex)

    for i in range(2*L):
        for j in range(2*L):
            # i,j < L は粒子ブロック、i,j >= L は正孔ブロック。
            # 片方だけ L をまたぐ成分は pairing / off-diagonal ブロックに対応する。
            #左上
            if i < L and j < L:
                if i == j:
                    # 粒子ブロックの onsite 項。m は質量項として対角成分に入る。
                    H_BdG[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))*(-1)
                elif i-j == 1:
                    # 左隣との有限差分。beta はリンク中央の値として両端平均を使う。
                    H_BdG[i, j] = -1/(2*epsilon) * (p - 1j*(beta(j+1/2,L,beta_sign,epsilon)+beta(i+1/2,L,beta_sign,epsilon))/2)
                elif j-i == 1:
                   # 右隣との有限差分。Hermiticity が保たれるよう複素共役側の符号になる。
                   H_BdG[i, j] = -1/(2*epsilon) * (p + 1j*(beta(i+1/2,L,beta_sign,epsilon)+beta(j+1/2,L,beta_sign,epsilon))/2)
            #右上
            elif i < L and j >= L:
                # 粒子 -> 正孔の pairing ブロック。最近接だけが非ゼロ。
                if j-i == L-1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-1)
                elif j-i == L+1:
                    H_BdG[i, j] = -1/(2*epsilon) * (1)
            #左下
            elif i >= L and j < L:
                # 正孔 -> 粒子の pairing ブロック。上のブロックと対応する位置を埋める。
                if i-j == L-1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-1)
                elif i-j == L+1:
                    H_BdG[i, j] = -1/(2*epsilon) * (1)
            #右下
            else:
                # 正孔ブロック。粒子ブロックと p, beta の符号が反転した形になる。
                if i == j:
                    H_BdG[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))
                elif i-j == 1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-p - 1j*(beta(j+1/2-L,L,beta_sign,epsilon)+beta(i+1/2-L,L,beta_sign,epsilon))/2)
                elif j-i == 1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-p + 1j*(beta(i+1/2-L,L,beta_sign,epsilon)+beta(j+1/2-L,L,beta_sign,epsilon))/2)

    if PBC == True:
        # 周期境界条件では j=L-1 と j=0 の間の BdG 行列要素を追加する。
        # 色名は元ノート/図の対応を残した目印で、各行は境界をまたぐ成分。
        #red
        H_BdG[0,L-1] = -1/(2*epsilon) * (p - 1j*beta(L-1,L,beta_sign,epsilon)) * (1)
        H_BdG[2*L-1,L] = -1/(2*epsilon) * (p - 1j*beta(L-1,L,beta_sign,epsilon)) * (-1)
        #blue
        H_BdG[L-1,0] = -1/(2*epsilon) * (p + 1j*beta(L-1,L,beta_sign,epsilon)) * (1)
        H_BdG[L,2*L-1] = -1/(2*epsilon) * (p + 1j*beta(L-1,L,beta_sign,epsilon)) * (-1)
        #orange
        H_BdG[0,2*L-1] = -1/(2*epsilon) * (-1)
        H_BdG[L-1,L] = -1/(2*epsilon) * (1)
        #black
        H_BdG[2*L-1,0] = -1/(2*epsilon) * (-1)
        H_BdG[L,L-1] = -1/(2*epsilon) * (1)

    return H_BdG

def diagonalize_bdg_matrix(H_BdG, L):
    """Diagonalize the BdG Hamiltonian and reorder paired eigenmodes."""
    #BdG行列を対角化
    eigenvalues, eigenvectors = LA.eigh(H_BdG)

    # LA.eigh は昇順で返す。ここでは先に正エネルギー側 L 本を並べ、
    # 後半に負エネルギー側を反転して置き、粒子-正孔ペアを扱いやすくする。
    #固有値、固有ベクトルのソート(確認済み)
    eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
    eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)
    return eigenvalues, eigenvectors

def enforce_particle_hole_symmetry(eigenvectors, L):
    """Rebuild the second half of the basis from particle-hole partners."""
    #粒子-反粒子対称性を満たすように固有ベクトルを調整(列方向に調整しないといけないらしい。行方向だとうまくいかない。固有ベクトルを横切るからか？)
    V = np.zeros((2*L, 2*L), dtype=complex)
    for i in range(L):
        # 前半 L 本は対角化で得た正エネルギー側を採用し、
        # 後半 L 本は粒子成分と正孔成分を入れ替えた複素共役として作り直す。
        V[:,i] = eigenvectors[:,i]
        V[:L,i+L] = np.conj(eigenvectors[L:,i])
        V[L:,i+L] = np.conj(eigenvectors[:L,i])
    return V

def build_operator_lists(eigenvectors, L, PBC):
    """Construct local bilinear operators in the quasiparticle basis."""
    # ここで作る operator はすべて L x L 行列で、psi.T.conj() @ O_j @ psi により
    # site j の期待値を評価できる形にしておく。
    #cj_dag_cj(作り方は以前と変わらない)
    cj_dag_cj_list = []
    for j in range(L):
        # c_j^\dagger c_j: local number density.  対角の真空項もここで含める。
        cj_dag_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj_dag_cj_tmp[k,l] = eigenvectors[j,k].conj() * eigenvectors[j,l]
                cj_dag_cj_tmp[k,l] += -1*eigenvectors[j,l+L].conj() * eigenvectors[j,k+L]
                if k == l:
                    for n in range(L):
                        cj_dag_cj_tmp[k,l] += eigenvectors[j,n+L].conj() * eigenvectors[j,n+L]
        isHermitian = is_hermitian(cj_dag_cj_tmp)
        assert isHermitian, "cj_dag_cj(j=" + str(j) + ") is not Hermitian!"
        cj_dag_cj_list.append(cj_dag_cj_tmp)
        print("cj†cj作成中:" + str(int(j/L*100))+"%")

    #cj1_cj
    cj1_cj_list = []
    for j in range(L-1):
        # c_{j+1} c_j: energy density の最近接 pairing 成分に使う。
        cj1_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj1_cj_tmp[k,l] = eigenvectors[j+1,k+L] * eigenvectors[j,l]
                cj1_cj_tmp[k,l] -= eigenvectors[j+1,l] * eigenvectors[j,k+L]
                if k == l:
                    for n in range(L):
                        cj1_cj_tmp[k,l] += eigenvectors[j+1,n] * eigenvectors[j,n+L]
        cj1_cj_list.append(cj1_cj_tmp)
        print("cj1_cj作成中:" + str(int(j/L*100))+"%")
    #PBCの場合、右端は非ゼロ
    if PBC == True:
        # 周期境界条件では最後の bond (L-1 -> 0) も最近接として追加する。
        # 開境界の場合はこの bond が存在しないので、下の else でゼロ行列を入れる。
        cj1_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj1_cj_tmp[k,l] = eigenvectors[0,k+L] * eigenvectors[L-1,l]
                cj1_cj_tmp[k,l] -= eigenvectors[0,l] * eigenvectors[L-1,k+L]
                if k == l:
                    for n in range(L):
                        cj1_cj_tmp[k,l] += eigenvectors[0,n] * eigenvectors[L-1,n+L]
        cj1_cj_list.append(cj1_cj_tmp)
    else:
        cj1_cj_list.append(np.zeros((L, L), dtype=complex))

    #cj1_dag_cj
    cj1_dag_cj_list = []
    for j in range(L-1):
        # c_{j+1}^\dagger c_j: energy density の hopping 成分に使う。
        cj1_dag_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj1_dag_cj_tmp[k,l] = eigenvectors[j+1,k].conj() * eigenvectors[j,l]
                cj1_dag_cj_tmp[k,l] -= eigenvectors[j+1,l+L].conj() * eigenvectors[j,k+L]
                if k == l:
                    for n in range(L):
                        cj1_dag_cj_tmp[k,l] += eigenvectors[j+1,n+L].conj() * eigenvectors[j,n+L]
        cj1_dag_cj_list.append(cj1_dag_cj_tmp)
        print("cj1†_cj作成中:" + str(int(j/L*100))+"%")
    #PBCの場合、右端は非ゼロ
    if PBC == True:
        # hopping 成分でも最後の bond (L-1 -> 0) を追加する。
        cj1_dag_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj1_dag_cj_tmp[k,l] = eigenvectors[0,k].conj() * eigenvectors[L-1,l]
                cj1_dag_cj_tmp[k,l] -= eigenvectors[0,l+L].conj() * eigenvectors[L-1,k+L]
                if k == l:
                    for n in range(L):
                        cj1_dag_cj_tmp[k,l] += eigenvectors[0,n+L].conj() * eigenvectors[L-1,n+L]
        cj1_dag_cj_list.append(cj1_dag_cj_tmp)
    else:
        cj1_dag_cj_list.append(np.zeros((L, L), dtype=complex))

    return cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list

def build_initial_state(L, eigenvectors, j0, sigma, PBC, direction_sign):
    """Create a normalized Gaussian wave packet in the quasiparticle basis."""
    psi = np.zeros((L, 1), dtype=complex)

    if PBC == True:
        # 周期境界では j=0 と j=L-1 が隣接するので、距離も ring 上の最短距離で測る。
        idx = np.arange(L)
        delta = np.abs(idx - j0)
        periodic_delta = np.minimum(delta, L - delta)
        weights = np.exp(-(periodic_delta**2) / (2 * sigma**2))
        mask = periodic_delta <= 0.5*L
    else:
        # 開境界では通常の直線距離で Gaussian packet を作る。
        weights = np.exp(-((np.arange(L) - j0) ** 2) / (2 * sigma ** 2))
        mask = np.abs(np.arange(L) - j0) <= 0.5*L

    # 反対側の境界から回り込む tail を避けるため、半周より遠い成分は切る。
    weights[~mask] = 0

    x_ = np.arange(L)
    p_ = weights/weights.sum()
    mean_pos = np.sum(p_ * x_)
    var_pos = np.sum(p_ * (x_ - mean_pos)**2)
    std_pos_ = np.sqrt(var_pos)
    print("初期状態のweightの標準偏差:", std_pos_)

    for j in range(L):
        for n in range(L):
            # site basis の Gaussian weight を BdG 固有ベクトルへ射影する。
            # exp(+- i*pi/4) の相対位相が右向き/左向きの初期 packet を選ぶ。
            psi[n, 0] += weights[j] * (
            1/np.sqrt(2) * (np.exp(direction_sign*1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(-direction_sign*1j*np.pi/4) * eigenvectors[j,n].conj())
            )
    #状態ベクトルの規格化
    psi /= np.linalg.norm(psi)
    return psi, weights

#ハミルトニアン密度作成###############################################################################################################
def build_energy_densities(cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list, L, epsilon, p, m, beta_sign, PBC):
    """Assemble H_+, H_-, and mixed local energy-density operators."""
    H_p = []
    H_m = []
    H_pm = []

    # 以降の式では c_j c_{j+1}, c_j c_{j+1}^\dagger なども必要になる。
    # 直接作った c_{j+1} c_j 系から、反交換関係と Hermitian conjugate で向きをそろえる。
    #cj_cj1
    cj_cj1_list = []
    for cj1_cj in cj1_cj_list:
        cj_cj1_list.append(-1*cj1_cj)

    #cj_cj1_dag
    cj_cj1_dag_list = []
    for cj1_dag_cj in cj1_dag_cj_list:
        cj_cj1_dag_list.append(-1*cj1_dag_cj)

    #cj_dag_cj1
    cj_dag_cj1_list = []
    for cj1_dag_cj in cj1_dag_cj_list:
        cj_dag_cj1_list.append(cj1_dag_cj.T.conj())

    #cj_dag_cj1_dag
    cj_dag_cj1_dag_list = []
    for cj1_cj in cj1_cj_list:
        cj_dag_cj1_dag_list.append(cj1_cj.T.conj())

    for j in range(L):
        # 各 site j に局所 energy density operator を作る。
        # j-1 と j の bond を平均することで、site 中心の密度として扱う。
        H_p_j = np.zeros((L, L), dtype=complex)
        H_m_j = np.zeros((L, L), dtype=complex)
        H_pm_j = np.zeros((L, L), dtype=complex)

        #ハミルトニアン密度作成
        H_p_j = -1j/(4*epsilon) * (1+beta(j+1/2,L,beta_sign,epsilon))  * (1j*(cj_cj1_list[j-1] + cj_cj1_list[j])/2 + (cj_cj1_dag_list[j-1]+cj_cj1_dag_list[j])/2 + (cj_dag_cj1_list[j-1]+cj_dag_cj1_list[j])/2 - 1j*(cj_dag_cj1_dag_list[j-1]+cj_dag_cj1_dag_list[j])/2)
        H_m_j = -1j/(4*epsilon) * (-1+beta(j+1/2,L,beta_sign,epsilon))  * (-1j*(cj_cj1_list[j-1] + cj_cj1_list[j])/2 + (cj_cj1_dag_list[j-1]+cj_cj1_dag_list[j])/2 + (cj_dag_cj1_list[j-1]+cj_dag_cj1_list[j])/2 + 1j*(cj_dag_cj1_dag_list[j-1]+cj_dag_cj1_dag_list[j])/2)
        H_pm_j = -1j/(2*epsilon) * (p*(1j*(cj_cj1_dag_list[j-1]+cj_cj1_dag_list[j])/2 - 1j*(cj_dag_cj1_list[j-1]+cj_dag_cj1_list[j])/2) - (p - epsilon*m)*(-2*1j*cj_dag_cj_list[j]))

        #エルミートか確認
        isHermitian = is_hermitian(H_p_j)
        assert isHermitian, "H_p(j=" + str(j) + ") is not Hermitian!"
        isHermitian = is_hermitian(H_m_j)
        assert isHermitian, "H_m(j=" + str(j) + ") is not Hermitian!"
        isHermitian = is_hermitian(H_pm_j)
        assert isHermitian, "H_pm(j=" + str(j) + ") is not Hermitian!"

        H_p.append(H_p_j)
        H_m.append(H_m_j)
        H_pm.append(H_pm_j)
        if not PBC:
            if j == L-1 or j == 0:
                # 開境界では端点の両側に bond がそろわないため、端の密度はゼロにして除外する。
                H_p[-1] = np.zeros((L, L), dtype=complex)
                H_m[-1] = np.zeros((L, L), dtype=complex)
                H_pm[-1] = np.zeros((L, L), dtype=complex)

    return {
        "H_p": H_p,
        "H_m": H_m,
        "H_pm": H_pm,
    }

#真空の量を計算##########################################################################################
def compute_vacuum_values(eigenvectors, L, epsilon, p, m, beta_sign, PBC):
    """Compute vacuum expectation values subtracted from excited profiles."""
    Hp_v = []
    Hm_v = []
    Hpm_v = []

    # Excited packet の寄与だけを見るため、各 observable の vacuum expectation を先に計算する。
    # 後段の compute_expectation_profile() で <psi|O_j|psi> からこの値を引く。
    #真空のc_dag_cを計算
    c_dag_c_v = []
    for j in range(L):
        total = 0.0
        for n in range(L):
            total += abs(eigenvectors[j, n+L])**2
        c_dag_c_v.append(total)

    #真空のcj1_cjを計算
    cj1_cj_v = []
    for j in range(L-1):
        total = 0.0
        for n in range(L):
            total += eigenvectors[j+1, n] * eigenvectors[j, n+L]
        cj1_cj_v.append(total)
    cj1_cj_v.append(0.0) #PBCなしの場合

    #真空のcj1_dag_cjを計算
    cj1_dag_cj_v = []
    for j in range(L-1):
        total = 0.0
        for n in range(L):
            total += eigenvectors[j+1, n+L].conj() * eigenvectors[j, n+L]
        cj1_dag_cj_v.append(total)
    cj1_dag_cj_v.append(0.0) #PBCなしの場合

    F1_list = []
    F2_list = []

    for j in range(L):
        # F1, F2 は隣接 site 間の vacuum contraction。
        # local energy density は j-1, j の bond 平均を使うため、履歴として list に保存する。
        Hp_v_j = 0
        Hm_v_j = 0
        H_pm_v_j = 0
        F1 = 0
        F2 = 0
        for n in range(L):
            # PBC では最後の bond が site 0 に戻る。開境界では最後の bond は存在しない。
            if PBC == True and j == L-1:
                F1 += eigenvectors[0,n] * eigenvectors[L-1,n+L]
                F2 += eigenvectors[0,n+L].conj() * eigenvectors[L-1,n+L]
            elif not PBC and j == L-1:
                F1 += 0
                F2 += 0
            else:
                # 通常の内部 bond では site j と j+1 の contraction を足す。
                F1 += eigenvectors[j+1,n] * eigenvectors[j,n+L]
                F2 += eigenvectors[j+1,n+L].conj() * eigenvectors[j,n+L]
        F1_list.append(F1)
        F2_list.append(F2)
        if j == 0:
            # Ensure complex type so .conj() exists for the Hermiticity check below
            # j=0 は左側 bond がないため、まずゼロの complex 値として扱う。
            Hp_v_j = 0+0j
            Hm_v_j = 0+0j
            H_pm_v_j = 0+0j
        else:
            Hp_v_j = -1/(2*epsilon) * 1j * (1+beta(j+1/2,L,beta_sign,epsilon)) * 1/2 * (1j * (-1*(F1_list[-1]+F1_list[-2])/2) + (-1*(F2_list[-1]+F2_list[-2])/2) + ((F2_list[-1]+F2_list[-2])/2).conj() - 1j * (F1_list[-1]+F1_list[-2]).conj()/2)
            Hm_v_j = -1/(2*epsilon) * 1j * (-1+beta(j+1/2,L,beta_sign,epsilon)) * 1/2 * (-1j * (-1*(F1_list[-1]+F1_list[-2])/2) + (-1*(F2_list[-1]+F2_list[-2])/2) + ((F2_list[-1]+F2_list[-2])/2).conj() + 1j * (F1_list[-1]+F1_list[-2]).conj()/2)
            H_pm_v_j = -1j/(2*epsilon) * (1j * p * (-1*(F2_list[-1]+F2_list[-2])/2 - (F2_list[-1]+F2_list[-2]).conj()) - (p - epsilon*m) * (-2j * c_dag_c_v[j]))
        assert abs(Hp_v_j - np.conj(Hp_v_j)) < 10**-5, "Hp_v_j(j=" + str(j) + ") is not Hermitian!"
        assert abs(Hm_v_j - np.conj(Hm_v_j)) < 10**-5, "Hm_v_j(j=" + str(j) + ") is not Hermitian!"
        Hp_v.append(Hp_v_j)
        Hm_v.append(Hm_v_j)
        Hpm_v.append(H_pm_v_j)
        if not PBC:
            if j == L-1:
                Hp_v[-1] = 0+0j
                Hm_v[-1] = 0+0j
                Hpm_v[-1] = 0+0j

    return {
        "H_p": Hp_v,
        "H_m": Hm_v,
        "H_pm": Hpm_v,
        "c_dag_c": c_dag_c_v,
        "cj1_cj": cj1_cj_v,
        "cj1_dag_cj": cj1_dag_cj_v,
    }

#初期状態の量###############################################################################################################
def compute_std(weights):
    """Return the position standard deviation of a one-dimensional profile."""
    weights = np.abs(weights)
    # profile を確率分布として扱うため、絶対値を正規化してから平均と分散を取る。
    x = np.arange(len(weights))
    p = weights/weights.sum()
    mean = np.sum(p * x)
    var = np.sum(p * (x - mean)**2)
    var = max(var, 0.0)
    std = np.sqrt(var)
    return std

def compute_expectation_profile(psi, operators, vacuum_values):
    """Evaluate <psi|O_j|psi> for each site and subtract vacuum offsets."""
    profile = []
    raw_values = []
    for j, operator in enumerate(operators):
        # raw_values は虚部の大きさを後で診断するため、vacuum subtraction 前の値も残す。
        val = psi.T.conj() @ operator @ psi
        profile.append(val.item() - vacuum_values[j])
        raw_values.append(val)
    return profile, raw_values

def compute_initial_observables(psi, energy_densities, vacuum_values, operator_lists):
    """Measure all observables at t=0 and record initial packet widths."""
    initial_values = {}

    # H_p, H_m は packet の幅 sigma を追跡する対象なので、初期幅を基準値として保存する。
    H_p_0, _ = compute_expectation_profile(psi, energy_densities["H_p"], vacuum_values["H_p"])
    weights = np.array([x.real for x in H_p_0])
    std_pos_p = compute_std(weights)
    print("H_pの標準偏差:", std_pos_p)
    initial_values["H_p"] = H_p_0

    H_m_0, _ = compute_expectation_profile(psi, energy_densities["H_m"], vacuum_values["H_m"])
    weights = np.array([x.real for x in H_m_0])
    std_pos_m = compute_std(weights)
    print("H_mの標準偏差:", std_pos_m)
    initial_values["H_m"] = H_m_0

    H_pm_0, _ = compute_expectation_profile(psi, energy_densities["H_pm"], vacuum_values["H_pm"])
    initial_values["H_pm"] = H_pm_0

    # 以下の bilinear observables は density や相関の確認用として初期値を保存する。
    cdc_0, _ = compute_expectation_profile(psi, operator_lists["cj_dag_cj"], vacuum_values["c_dag_c"])
    initial_values["c_dag_c"] = cdc_0

    cj1_cj_0, _ = compute_expectation_profile(psi, operator_lists["cj1_cj"], vacuum_values["cj1_cj"])
    initial_values["cj1_cj"] = cj1_cj_0

    cj1_dag_cj_0, _ = compute_expectation_profile(psi, operator_lists["cj1_dag_cj"], vacuum_values["cj1_dag_cj"])
    initial_values["cj1_dag_cj"] = cj1_dag_cj_0

    return initial_values, std_pos_p, std_pos_m

def log_imag(name, arr):
    """Warn when an observable that should be real has imaginary residue."""
    max_im = np.max(np.abs(np.imag(arr)))
    if max_im > 1e-8:  # 目安
        print(f"[Warn] {name} has non-negligible imaginary part: max={max_im:.2e}")

def run_time_evolution(
    psi,
    eigenvalues,
    times,
    dt,
    L,
    energy_densities,
    vacuum_values,
    cj_dag_cj_list,
    std_pos_p,
    std_pos_m,
):
    """Evolve quasiparticle amplitudes in time and collect observables."""
    values = {
        "H_p": [],
        "H_m": [],
        "H_pm": [],
        "c_dag_c": [],
    }
    sigmas = {
        "H_p": [],
        "H_m": [],
        "H_pm": [],
        "c_dag_c": [],
    }
    x_ave_list = []
    psi_initial = psi.copy()

    for i, _ in enumerate(times):
        # In the diagonal basis, each eigenmode only receives a phase factor.
        for n,E in enumerate(eigenvalues[:L]):
            psi[n] = np.exp(-1j*E*(dt*(i+1))) * psi_initial[n]

        # 各時刻で local density profile を評価し、heatmap 用に時系列として積む。
        #H_pの期待値
        H_p_t_val, H_p_test = compute_expectation_profile(psi, energy_densities["H_p"], vacuum_values["H_p"])
        values["H_p"].append(H_p_t_val)

        #H_mの期待値
        H_m_t_val, H_m_test = compute_expectation_profile(psi, energy_densities["H_m"], vacuum_values["H_m"])
        values["H_m"].append(H_m_t_val)

        #位置の平均値
        # 規格化: H_m_t_val は list の場合があるため、安全に正規化して numpy 配列にする
        # H_m_t_val の要素は複素数になる場合があるので、実部を使う（imag が無視できない場合は
        # log_imag() で警告が出る）。ここでは明示的に実部を取り出す。
        H_m_t_val = np.array([x.real for x in H_m_t_val], dtype=float)
        total = H_m_t_val.sum()
        if total != 0:
            # H_m profile を重み分布として正規化し、packet 中心の平均位置を計算する。
            H_m_t_val = H_m_t_val / total
        x = np.arange(L)
        mean = np.sum(H_m_t_val * x)
        x_ave_list.append(mean)

        #H_pmの期待値
        H_pm_t_val, H_pm_test = compute_expectation_profile(psi, energy_densities["H_pm"], vacuum_values["H_pm"])
        values["H_pm"].append(H_pm_t_val)

        #c_dag_cの期待値
        c_dag_c_t_val, c_dag_c_test = compute_expectation_profile(psi, cj_dag_cj_list, vacuum_values["c_dag_c"])
        values["c_dag_c"].append(c_dag_c_t_val)

        log_imag("H_p", H_p_test)
        log_imag("H_m", H_m_test)
        log_imag("H_pm", H_pm_test)
        log_imag("c_dag_c", c_dag_c_test)
        # 初期幅との差分を記録する。絶対幅ではなく広がりの変化量を見るため。
        weights = np.array([x.real for x in H_p_t_val])
        std_pos = compute_std(weights)
        if not std_pos_p is None:
            sigmas["H_p"].append(std_pos - std_pos_p)
        weights = np.array([x.real for x in H_m_t_val])
        std_pos = compute_std(weights)
        if not std_pos_m is None:
            sigmas["H_m"].append(std_pos - std_pos_m)

        # 次の時刻は必ず初期状態から exp(-iEt) を掛け直す。
        # これにより累積丸め誤差を避ける。
        psi = psi_initial.copy()

        print("時間発展中:"+str(int(i/len(times)*100)) + "%")

    return values, sigmas, x_ave_list

def plot_density_map(
    values,
    times,
    output_path,
    *,
    beta_sign,
    horizon_positions,
    geodesic_time_scale,
    colorbar_label=None,
    geodesic_color="cyan",
    geodesic_linestyle="--",
    geodesic_linewidth=1.1,
):
    """Save a time-site density heatmap and overlay geodesic.dat if present."""
    value_arr = np.array(values, dtype=complex)
    # Hermitian observable の期待値なので本来は実数。小さな数値誤差の虚部はここで落とす。
    value_array = np.real(value_arr).astype(float) #ここで実数にしていることに注意

    color_limit = float(np.nanmax(np.abs(value_array)))
    if color_limit == 0 or not np.isfinite(color_limit):
        color_limit = 1.0

    nt, num_sites = value_array.shape #時間ステップ数(使わない)とサイト数を取得

    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 30,
        'axes.titlesize': 22,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
    })

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")

    # 背景の密度プロット
    im = ax.imshow(
        value_array,
        aspect='auto',
        origin='lower',
        extent=[0, num_sites, t_i, t_f],
        cmap='seismic',
        vmin=-color_limit,
        vmax=color_limit,
        interpolation='nearest',
    )

    for idx, horizon_position in enumerate(horizon_positions):
        ax.axvline(
            horizon_position,
            linestyle="--",
            color="black",
            linewidth=1.7,
            label="Event horizon" if idx == 0 else None,
        )

    # —— geodesic.dat を重ねる（物理範囲→表示範囲の対応） ——
    try:
        # run directory に保存した geodesic.dat を探す
        file_path = resolve_output_path("geodesic.dat")
        geodesic_data = np.loadtxt(file_path, delimiter=",")

        x = geodesic_data[:, 0]  # 物理空間座標（0〜2π）
        t = geodesic_data[:, 1]  # 物理時間（0〜20）

        # geodesic.dat は物理座標 x と simulation time t。heatmap の横軸だけ lattice index j へ変換する。
        x_scaled = (x / (2 * np.pi)) * num_sites
        t_scaled = t

        ax.plot(
            x_scaled,
            t_scaled,
            linestyle=geodesic_linestyle,
            color=geodesic_color,
            linewidth=geodesic_linewidth,
        )
    except Exception as e:
        print(f"Warning: geodesic.dat の重ね描画に失敗しました: {e}")
    # ————————————————————————————————

    ax.set_xlim(0, num_sites)
    ax.set_ylim(t_i, t_f)
    ax.set_xlabel(r"$j$", fontweight='bold')
    ax.set_ylabel(r"$t$", fontweight='bold', rotation=0, labelpad=22)
    if horizon_positions:
        ax.legend(loc='upper left', fontsize=20, frameon=True)

    cbar = fig.colorbar(im, ax=ax, location='right', pad=0.06)
    cbar.ax.tick_params(labelsize=15)
    if colorbar_label:
        cbar.set_label(colorbar_label, rotation=0, labelpad=32, fontsize=26)

    fig.tight_layout()
    fig.savefig(resolve_output_path(output_path),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

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
    Save an animated line plot for a time-dependent lattice density.

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
    # 2. 実部のみを使用（十分小さい虚部は無視）
    density_arr = np.real_if_close(density_arr, tol=1000)  # tol は 10^(-tol) 判定
    density_arr = density_arr.astype(float)                # 明示的に float32/64 へ
    N, L = density_arr.shape
    if len(times) != N:
        raise ValueError("times の長さと density の行数が一致していません。")
    if horizon_positions is None:
        # 現在は horizon_positions を描画していないが、将来 axvline を戻すときの既定値として残す。
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
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":")

    # ---------- アニメーション用コールバック -------------------------------
    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        # FuncAnimation から渡される frame 番号に対応する profile だけを線へ反映する。
        line.set_data(np.arange(1, L + 1), density_arr[frame])
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
    ani.save(resolve_output_path(gif_path), writer=PillowWriter(fps=fps))
    print("保存しました")
    plt.close(fig)  # 余分なウインドウを閉じる

def compute_fft_spectrum(values, *, remove_spatial_mean=True):
    """Return shifted FFT amplitudes for each time slice of a time-site profile."""
    value_array = np.real(np.asarray(values, dtype=complex)).astype(float)
    if value_array.ndim != 2:
        raise ValueError("FFT input must be a time-site array")
    if remove_spatial_mean:
        value_array = value_array - value_array.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(value_array, axis=1), axes=1))
    k_values = np.fft.fftshift(np.fft.fftfreq(value_array.shape[1], d=1.0)) * 2 * np.pi
    return k_values, spectrum

def save_fft_csv(name, k_values, spectrum, times):
    """Save FFT k values and a time-k spectrum CSV."""
    np.savetxt(
        resolve_output_path(f"{name}_fft_k.csv"),
        k_values,
        delimiter=",",
        header="k",
        comments="",
    )
    data = np.column_stack([times, spectrum])
    header = ",".join(["time"] + [f"k{i}" for i in range(len(k_values))])
    np.savetxt(
        resolve_output_path(f"{name}_fft_val.csv"),
        data,
        delimiter=",",
        header=header,
        comments="",
    )

def plot_fft_map(name, k_values, spectrum, times):
    """Save a time-k FFT amplitude heatmap."""
    color_limit = float(np.nanmax(spectrum))
    if color_limit == 0 or not np.isfinite(color_limit):
        color_limit = 1.0
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(
        spectrum,
        aspect="auto",
        origin="lower",
        extent=[k_values[0], k_values[-1], times[0], times[-1]],
        cmap="magma",
        vmin=0,
        vmax=color_limit,
        interpolation="nearest",
    )
    ax.set_xlabel(r"$k$", fontweight="bold")
    ax.set_ylabel(r"$t$", fontweight="bold", rotation=0, labelpad=22)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    cbar = fig.colorbar(im, ax=ax, location="right", pad=0.06)
    cbar.set_label(r"$|\mathrm{FFT}|$", rotation=0, labelpad=32, fontsize=22)
    fig.tight_layout()
    fig.savefig(resolve_output_path(f"figures/{name}_fft.png"), dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)

def save_fft_animation(name, k_values, spectrum, times, fps=20):
    """Save an animated k-space FFT spectrum."""
    positive_k = k_values >= 0
    k_values = k_values[positive_k]
    spectrum = spectrum[:, positive_k]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    line, = ax.plot([], [], lw=1.8, color="purple")
    y_max = float(np.nanmax(spectrum))
    if y_max == 0 or not np.isfinite(y_max):
        y_max = 1.0
    ax.set_xlim(0, k_values[-1])
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"$k$", fontsize=16)
    ax.set_ylabel(r"$|\mathrm{FFT}|$", fontsize=16)
    major_ticks = np.linspace(0, np.pi, 17)
    minor_ticks = np.linspace(0, np.pi, 65)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels([f"{tick:.2f}" for tick in major_ticks])
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.35)
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        title.set_text("")
        return (line, title)

    def update(frame):
        line.set_data(k_values, spectrum[frame])
        title.set_text(f"{name} FFT, t={times[frame]:.3f}")
        return (line, title)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),
        init_func=init,
        blit=True,
        interval=1000 / fps,
    )
    ani.save(resolve_output_path(f"figures/{name}_fft.gif"), writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_fft_outputs(density_values, times):
    """Save FFT spectra, heatmaps, and GIFs for configured observables."""
    for name in fft_observables:
        if name not in density_values:
            print(f"FFT skipped: unknown observable {name}")
            continue
        k_values, spectrum = compute_fft_spectrum(
            density_values[name],
            remove_spatial_mean=fft_remove_spatial_mean,
        )
        save_fft_csv(name, k_values, spectrum, times)
        plot_fft_map(name, k_values, spectrum, times)
        save_fft_animation(name, k_values, spectrum, times)

def save_beta_profile():
    """Save the configured beta profile for notebook-side inspection."""
    sites = np.arange(L)
    values = np.array([float(beta(j, L, beta_sign, epsilon)) for j in sites])
    data = np.column_stack([sites, sites * epsilon, values])
    np.savetxt(
        resolve_output_path("beta_profile.csv"),
        data,
        delimiter=",",
        header="j,x,beta",
        comments="",
    )

def check_particle_hole_pairs(eigenvectors, L):
    """Print modes that fail the expected particle-hole pairing check."""
    for i in range(L):
        threshold = 1e-10
        ans = eigenvectors[:,i+L] + np.concatenate((eigenvectors[L:,i].conj(), eigenvectors[:L,i].conj()), 0)
        if np.all(np.abs(ans) < threshold):
            continue
            print(str(i))
            print(np.where(np.abs(ans) < threshold, 0.0, ans))
        else:
            ans = eigenvectors[:,i+L] - np.concatenate((eigenvectors[L:,i].conj(), eigenvectors[:L,i].conj()), 0)
            if np.all(np.abs(ans) < threshold):
                continue
                print(np.where(np.abs(ans) < threshold, 0.0, ans))
            else:
                print(str(i)+"''")
                ans = eigenvectors[:,i+L].T.conj() @ np.concatenate((eigenvectors[L:,i].conj(), eigenvectors[:L,i].conj()), 0)
                print(np.linalg.norm(np.where(np.abs(ans) < threshold, 0.0, ans)))

def ensure_output_dirs():
    """Create the run output directory and its figure subdirectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

def resolve_output_path(path):
    """Resolve a run-local output path and create its parent directory."""
    path = output_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def json_ready(value):
    """Convert config values such as numpy arrays/scalars into JSON-safe data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

def save_run_config():
    """Store the exact prepared config used for this run."""
    with resolve_output_path("config.json").open("w") as f:
        json.dump(json_ready(CONFIG), f, indent=2)

def save_summary(summary):
    """Store a compact machine-readable summary for the run."""
    with resolve_output_path("summary.json").open("w") as f:
        json.dump(json_ready(summary), f, indent=2)

def save_time_site_csv(name, values, times):
    """Save a time-site profile with time in the first column."""
    values = np.real(np.asarray(values, dtype=complex)).astype(float)
    data = np.column_stack([times, values])
    header = ",".join(["time"] + [f"j{j}" for j in range(values.shape[1])])
    np.savetxt(resolve_output_path(f"{name}_val.csv"), data, delimiter=",", header=header, comments="")

def save_profile_csv(name, values):
    """Save a single site profile."""
    values = np.real(np.asarray(values, dtype=complex)).astype(float)
    data = np.column_stack([np.arange(len(values)), values])
    np.savetxt(resolve_output_path(f"{name}.csv"), data, delimiter=",", header="j,value", comments="")

def generate_mode_transform(eigenvectors, L):
    """Convert BdG eigenvectors to the f/g mode-function basis."""
    U = np.zeros((2*L, 2*L), dtype=complex)
    for i in range(L):
        U[i, :] = (eigenvectors[i, :] + eigenvectors[i+L, :]) / np.sqrt(2)
        U[i+L, :] = (eigenvectors[i, :] - eigenvectors[i+L, :]) / (1j*np.sqrt(2))
    return U

def save_mode_functions(eigenvectors, eigenvalues, count):
    """Save the first few BdG mode functions as PNG files."""
    ensure_output_dirs()
    U = generate_mode_transform(eigenvectors, L)
    f_modes = U[:L, L:]
    x = np.linspace(0, l, L)
    for i in range(min(count, L)):
        plt.figure()
        plt.plot(x, f_modes[:, i] / np.sqrt(epsilon))
        plt.xlabel(r'$x_j$', fontsize=25)
        plt.xticks([0, 0.5*l, l], ["0", r"$\pi$", r"$2\pi$"], fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel(rf"$f_{{j{i+1}}}$", rotation=0, fontsize=25, labelpad=15)
        plt.axvline(x=0.235*l, color="red", linewidth=1, linestyle="--")
        plt.axvline(x=0.767*l, color="red", linewidth=1, linestyle="--")
        plt.title(rf"E={eigenvalues[i]:.3f}", fontsize=20)
        plt.grid(True)
        plt.savefig(
            resolve_output_path(f"figures/mode_function_p={p}_k={i+1}.png"),
            dpi=300,
            bbox_inches="tight",
            transparent=False,
        )
        plt.close()

def beta_continuous(x):
    """Evaluate the configured beta profile at a physical coordinate x."""
    x_eval = np.mod(x, l) if PBC else x
    return beta(x_eval / epsilon, L, beta_sign, epsilon)

def dxdt_packet_geodesic(t, x):
    """Null geodesic velocity for the packet branch selected by the simulation."""
    return [beta_continuous(x[0]) + initial_direction_sign]

def hit_left_boundary(_t, x):
    """Stop an open-boundary geodesic at x=0."""
    return x[0]

def hit_right_boundary(_t, x):
    """Stop an open-boundary geodesic at x=l."""
    return l - x[0]

hit_left_boundary.terminal = True
hit_left_boundary.direction = -1
hit_right_boundary.terminal = True
hit_right_boundary.direction = -1

def save_geodesic_dat():
    """Generate geodesic.dat for heatmap overlays."""
    ensure_output_dirs()
    x_start = j0 * epsilon
    events = None if PBC else (hit_left_boundary, hit_right_boundary)
    sol = solve_ivp(
        fun=dxdt_packet_geodesic,
        t_span=(t_i, t_f),
        y0=[x_start],
        dense_output=True,
        events=events,
    )
    t_stop = sol.t[-1]
    ts = np.linspace(t_i, t_stop, geodesic_points)
    xs = sol.sol(ts)[0]
    if PBC:
        xs = np.mod(xs, l)
    np.savetxt(resolve_output_path("geodesic.dat"), np.column_stack([xs, ts]), fmt="%.10f, %.10f")
    plt.figure()
    plt.plot(xs, ts, color="cyan", linestyle="--", linewidth=1.1)
    plt.xlabel("x")
    plt.ylabel("t(x)")
    plt.xlim(0, l)
    plt.title("packet-center null geodesic")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(resolve_output_path("figures/geodesic.png"), dpi=300, bbox_inches="tight")
    plt.close()

def exp_func(t, A, B):
    """Exponential model used for fitting sigma growth."""
    return A * np.exp(B * t) - A

def fit_and_plot_surface_gravity(times, H_m_sigmas):
    """Fit H_m sigma growth and save the comparison plot."""
    ensure_output_dirs()
    H_m_sigmas = np.asarray(H_m_sigmas, dtype=float) * epsilon
    if len(times) < 2 or len(H_m_sigmas) < 2:
        print("surface gravity fit skipped: not enough sigma samples")
        return

    A0 = H_m_sigmas[0] if H_m_sigmas[0] != 0 else 1e-12
    B0 = 1.0
    try:
        params, _ = curve_fit(exp_func, times, H_m_sigmas, p0=[A0, B0], maxfev=10000)
    except (RuntimeError, ValueError) as exc:
        print(f"surface gravity fit skipped: {exc}")
        return
    A_fit, B_fit = params
    print("Fitted function: H_m_sigma(t) = A * exp(B t) - A")
    print("A =", A_fit)
    print("B =", B_fit)

    t_fine = np.linspace(times.min(), times.max(), 2000)
    ideal_curve = exp_func(t_fine, A_fit, 1.0)
    plt.figure(figsize=(12, 8))
    plt.plot(times, H_m_sigmas*l, "o", label="Numerical simulation", color="blue", lw=1)
    plt.plot(t_fine, ideal_curve*l, label="Analytical prediction", color="red", lw=3)
    plt.xlabel(r"$t$", fontsize=25, fontweight="bold")
    plt.ylabel(r"$\delta \sigma$", fontsize=25, fontweight="bold", rotation=0, labelpad=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(resolve_output_path(surface_gravity_output_path), dpi=300, bbox_inches="tight")
    plt.close()

def resolve_geodesic_time_scale(value):
    """Allow plot config to refer to the final simulation time symbolically."""
    if value == "t_f":
        return t_f
    return value

def save_outputs(time_values, sigmas, x_ave_list, times):
    """Write numerical outputs, heatmaps, and animations produced by the run."""
    ensure_output_dirs()
    horizon_positions = save_horizon_positions()
    # run_time_evolution() の辞書から、保存・描画に使う observable を取り出す。
    H_p_val = time_values["H_p"]
    H_m_val = time_values["H_m"]
    H_pm_val = time_values["H_pm"]
    c_dag_c_val = time_values["c_dag_c"]
    H_p_sigmas = sigmas["H_p"]
    H_m_sigmas = sigmas["H_m"]

    plt.figure()
    plt.plot(times, x_ave_list, label="x average")
    plt.xlabel("time")
    plt.ylabel("x average")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(resolve_output_path("figures/x_ave.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # 後で解析しやすいよう、時間と対応する量を2列のテキストとして保存する。
    np.savetxt(resolve_output_path("x_ave.txt"), np.column_stack([times, x_ave_list]), fmt="%.10e")
    np.savetxt(resolve_output_path("H_m_sigmas.txt"), np.column_stack([times, H_m_sigmas]), fmt="%.10e")
    if outputs.get("surface_gravity_fit", True):
        fit_and_plot_surface_gravity(times, H_m_sigmas)

    density_values = {
        "H_p": H_p_val,
        "H_m": H_m_val,
        "H_pm": H_pm_val,
        "c_dag_c": c_dag_c_val,
    }
    for name, values in density_values.items():
        save_time_site_csv(name, values, times)
    save_profile_csv("H_p_0", H_p_val[0])
    save_profile_csv("H_m_0", H_m_val[0])
    save_profile_csv("H_pm_0", H_pm_val[0])
    save_profile_csv("c_dag_c_0", c_dag_c_val[0])

    if outputs.get("fft", True):
        save_fft_outputs(density_values, times)

    if outputs.get("heatmaps", True):
        for name, plot_config in DENSITY_PLOT_CONFIGS.items():
            # DENSITY_PLOT_CONFIGS に追加すれば、新しい observable も同じ heatmap 関数で保存できる。
            plot_density_map(
                density_values[name],
                times,
                plot_config["output_path"],
                beta_sign=beta_sign,
                horizon_positions=horizon_positions,
                geodesic_time_scale=resolve_geodesic_time_scale(plot_config["geodesic_time_scale"]),
                colorbar_label=plot_config.get("colorbar_label"),
                geodesic_color=plot_config.get("geodesic_color", "cyan"),
                geodesic_linestyle=plot_config.get("geodesic_linestyle", "--"),
                geodesic_linewidth=plot_config.get("geodesic_linewidth", 1.1),
            )

    plt.close('all')
    idx_p = np.argmin(H_p_sigmas)
    # sigma が最小になる時刻の profile を個別に表示して、packet の収束/拡散を確認する。
    t_p_min = times[idx_p]
    x_ave_physical = np.asarray(x_ave_list, dtype=float) * epsilon
    stagnation_candidates = np.where(x_ave_physical > stagnation_position)[0]
    if len(stagnation_candidates) > 0:
        idx_0 = stagnation_candidates[0]
        t_line_0 = times[idx_0]
        stagnation_time = t_p_min - t_line_0
        print("線形部分通過時間 t_line_0:", t_line_0)
        print("停滞時間:", stagnation_time)
    else:
        t_line_0 = None
        stagnation_time = None
        print("停滞時間: x 平均が指定位置を超えなかったため未計算")
    print("H_p sigma が最小になる t:", t_p_min)
    print("そのときの H_p sigma:", H_p_sigmas[idx_p])
    plt.figure()
    plt.plot(H_p_val[idx_p])
    plt.title("H_p at t = {:.3f}".format(t_p_min))
    plt.grid()
    plt.tight_layout()
    plt.savefig(resolve_output_path("figures/H_p_sigma_min.png"), dpi=300, bbox_inches="tight")
    plt.close()

    idx_m = np.argmin(H_m_sigmas)
    t_m_min = times[idx_m]
    print("H_m sigma が最小になる t:", t_m_min)
    print("そのときの H_m sigma:", H_m_sigmas[idx_m])
    plt.figure()
    plt.plot(H_m_val[idx_m])
    plt.title("H_m at t = {:.3f}".format(t_m_min))
    plt.grid()
    plt.tight_layout()
    plt.savefig(resolve_output_path("figures/H_m_sigma_min.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if outputs.get("gifs", True):
        for name, animation_config in ANIMATION_CONFIGS.items():
            # GIF は heatmap では見えにくい profile の形状変化を確認するために出力する。
            save_density_animation(
                density=density_values[name],
                times=times,
                gif_path=animation_config["gif_path"],
                xlabel="Lattice Site Index j",
                ylabel=r'δ$\langle c_j^\dagger c_j \rangle$',
                line_label=r'δ$\langle c_j^\dagger c_j \rangle$',
                cmap_line=animation_config["cmap_line"],
                PBC=PBC,
            )

    save_summary({
        "run_name": CONFIG["run_name"],
        "output_dir": str(output_dir),
        "L": L,
        "scenario": CONFIG["scenario"],
        "chirality": chirality,
        "beta_sign": beta_sign,
        "beta_profile": beta_profile,
        "p": p,
        "m": m,
        "t_f": t_f,
        "dt": dt,
        "H_p_sigma_min_time": t_p_min,
        "H_p_sigma_min": H_p_sigmas[idx_p],
        "H_m_sigma_min_time": t_m_min,
        "H_m_sigma_min": H_m_sigmas[idx_m],
        "t_line_0": t_line_0,
        "stagnation_time": stagnation_time,
        "horizon_positions_j": horizon_positions,
        "horizon_positions_x": [position*epsilon for position in horizon_positions],
    })

def run_simulation():
    """Run the full black-hole lattice simulation pipeline."""
    ensure_output_dirs()
    save_run_config()
    print("output_dir:", output_dir)
    if outputs.get("geodesic", True):
        save_geodesic_dat()

    # 1. 背景 beta profile を保存する。表示は view_outputs.ipynb 側で行う。
    if outputs.get("show_beta_profile", True):
        save_beta_profile()

    # 2. BdG 行列を作り、固有モードを粒子-正孔対称な形へ整える。
    H_BdG = build_bdg_matrix(L, p, m, beta_sign, epsilon, PBC)
    eigenvalues, eigenvectors = diagonalize_bdg_matrix(H_BdG, L)
    check_particle_hole_pairs(eigenvectors, L)
    eigenvectors = enforce_particle_hole_symmetry(eigenvectors, L)
    if outputs.get("mode_functions", True) and mode_function_count > 0:
        save_mode_functions(eigenvectors, eigenvalues, mode_function_count)

    # 3. 観測量を quasiparticle basis の L x L operator として準備する。
    cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list = build_operator_lists(eigenvectors, L, PBC)
    operator_lists = {
        "cj_dag_cj": cj_dag_cj_list,
        "cj1_cj": cj1_cj_list,
        "cj1_dag_cj": cj1_dag_cj_list,
    }

    # 4. 初期波束、局所エネルギー密度、真空 subtraction 用の値を準備する。
    psi, _ = build_initial_state(L, eigenvectors, j0, sigma, PBC, initial_direction_sign)
    energy_densities = build_energy_densities(cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list, L, epsilon, p, m, beta_sign, PBC)
    vacuum_values = compute_vacuum_values(eigenvectors, L, epsilon, p, m, beta_sign, PBC)
    _, std_pos_p, std_pos_m = compute_initial_observables(psi, energy_densities, vacuum_values, operator_lists)

    # 5. 時間発展を回して、最後にテキスト・画像・GIF を保存する。
    time_values, sigmas, x_ave_list = run_time_evolution(
        psi,
        eigenvalues,
        times,
        dt,
        L,
        energy_densities,
        vacuum_values,
        cj_dag_cj_list,
        std_pos_p,
        std_pos_m,
    )
    save_outputs(time_values, sigmas, x_ave_list, times)
