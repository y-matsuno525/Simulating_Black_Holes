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
p_const = True
p_val = 0

l = 2*np.pi
L = 100
epsilon = l/L
m = 0
t_i = 0
t_f = 20
dt = 0.01*(300/L)

times = np.arange(t_i + dt, t_f, dt)

def is_hermitian(matrix):

    return np.allclose(matrix, np.conj(matrix.T), atol=1e-100)

def is_unitary(matrix):

    identity_matrix = np.eye(matrix.shape[0])

    return np.allclose(np.dot(np.conj(matrix.T), matrix), identity_matrix,atol=1e-13)

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
    for j in range(L):
        total = 0.0
        for n in range(L):
            total += abs(eigenvectors[j, n])**2
        c_dag_c_v.append(total)
    return c_dag_c_v

def adjust_eigenvectors(eigenvactors):
        V = np.zeros((2*L, 2*L), dtype=complex)
        for i in range(L):
            V[:,i] = eigenvactors[:,i]
            V[:L,2*L-1-i] = np.conj(eigenvactors[L:,i])
            V[L:,2*L-1-i] = np.conj(eigenvactors[:L,i])
        return V

def initialize_state_vector_a_plus(eigenvectors):
    psi_tmp = np.zeros((L, 1), dtype=complex)
    if pos == "ur" or pos == "lr":
        j0 = int(0.2*L)
    else:
        j0 = int(0.8*L)
    print("pos : "+ pos)
    print("j0 : " + str(j0))
    sigma = 0.05*L
    # j0を中心としたガウシアン波束の重み
    weights = np.exp(-((np.arange(L) - j0) ** 2) / (2 * sigma ** 2))
    weights /= np.linalg.norm(weights)  # 規格化

    for j in range(L):
        for n in range(L):
            #+
            if pos == "ur" or pos == "lr":
                psi_tmp[n, 0] += weights[j] * (
                1/np.sqrt(2) * (np.exp(1j*np.pi/4) * eigenvectors[j, n] + np.exp(-1j*np.pi/4) * eigenvectors[j, (2*L-1)-n].conj())
            )
            #-
            else:
                psi_tmp[n, 0] += weights[j] * (
                1/np.sqrt(2) * (np.exp(-1j*np.pi/4) * eigenvectors[j, n] + np.exp(1j*np.pi/4) * eigenvectors[j, (2*L-1)-n].conj())
                )

    # 最終的な規格化
    psi_tmp /= np.linalg.norm(psi_tmp)
    return psi_tmp

#BdG行列を作成
h = my_operator.generate_BdG_matrix(L, m, p_val, PBC, p_const, pos, epsilon)
#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(h)
eigenvectors = adjust_eigenvectors(eigenvectors)

c_dag_c_list = my_operator.generate_c_dag_c(eigenvectors,L)
cj1_cj_list = my_operator.generate_cj1_cj(eigenvectors,L)
cj1_dag_cj_list = my_operator.generate_cj1_dag_cj(eigenvectors,L)
H_ps, H_ms = my_operator.H_p_m_K(cj1_cj_list, cj1_dag_cj_list, L, epsilon, pos)
U_dt = my_operator.generate_time_evolution_operator(eigenvalues, dt, L)

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