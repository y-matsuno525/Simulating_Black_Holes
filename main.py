import numpy as np
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#パラメータ
L = 100
l = 2*np.pi
epsilon = l / L
p = 1
m = 0.0001
pos = "lr" #lr, ur
t_i = 0
width = 1
A = 1
t_f = 5
dt = 0.01*(300/L) #この値は後で検討
PBC = False

#betaプロファイル
beta_profile = "pos" #pos/pos_horizon, center/centered_horizon, flat
beta_width = 1
beta_amplitude = 0.6
beta_center_fraction = 2/3
centered_beta_width = 1
centered_beta_amplitude = 1
centered_beta_center_fraction = 1/2

#初期状態パラメータ
j0_by_pos = {
    "ur": 25,#245,
    "lr": 25#245,
}
if pos not in j0_by_pos:
    raise ValueError("pos must be 'ur' or 'lr'")
j0 = j0_by_pos[pos]
sigma = 0.05*L #c_0はsigmaのLの係数に反比例傾向(完全反比例ではない)
initial_direction = "right" #right, left
direction_sign_by_name = {
    "right": 1,
    "left": -1,
}
if initial_direction not in direction_sign_by_name:
    raise ValueError("initial_direction must be 'right' or 'left'")
initial_direction_sign = direction_sign_by_name[initial_direction]

BETA_PROFILE_ALIASES = {
    "pos": "pos_horizon",
    "pos_horizon": "pos_horizon",
    "center": "centered_horizon",
    "centered_horizon": "centered_horizon",
    "flat": "flat",
}
if beta_profile not in BETA_PROFILE_ALIASES:
    raise ValueError("beta_profile must be 'pos', 'center', or 'flat'")
beta_profile = BETA_PROFILE_ALIASES[beta_profile]

times = np.arange(t_i + dt, t_f, dt)

def is_hermitian(matrix):
    is_hermitian = np.allclose(matrix, np.conj(matrix.T), atol=1e-10)

    if is_hermitian:
        return True
    else:
        return False

def beta_flat(j, L, pos, epsilon):
    return 0

def beta_centered_horizon(j, L, pos, epsilon):
    jh = int(centered_beta_center_fraction*L)
    return centered_beta_amplitude*np.tanh(centered_beta_width*(j - jh)*epsilon) + centered_beta_amplitude

def beta_pos_horizon(j, L, pos, epsilon):
    jh = int(beta_center_fraction*L)
    if pos == "lr":
        return -beta_amplitude*np.tanh(3/beta_width*(j - jh)*epsilon) - beta_amplitude
    elif pos == "ur":
        return  beta_amplitude*np.tanh(3/beta_width*(j - jh)*epsilon) + beta_amplitude
    raise ValueError("pos must be 'ur' or 'lr'")

def beta(j,L,pos,epsilon):
    beta_functions = {
        "flat": beta_flat,
        "centered_horizon": beta_centered_horizon,
        "pos_horizon": beta_pos_horizon,
    }
    return beta_functions[beta_profile](j, L, pos, epsilon)

def build_bdg_matrix(L, p, m, pos, epsilon, PBC):
    #BdGハミルトニアンの作成(符号関係は確認済み)
    H_BdG = np.zeros((2*L, 2*L), dtype=complex)

    for i in range(2*L):
        for j in range(2*L):
            #左上
            if i < L and j < L:
                if i == j:
                    H_BdG[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))*(-1)
                elif i-j == 1:
                    H_BdG[i, j] = -1/(2*epsilon) * (p - 1j*(beta(j+1/2,L,pos,epsilon)+beta(i+1/2,L,pos,epsilon))/2)
                elif j-i == 1:
                   H_BdG[i, j] = -1/(2*epsilon) * (p + 1j*(beta(i+1/2,L,pos,epsilon)+beta(j+1/2,L,pos,epsilon))/2)
            #右上
            elif i < L and j >= L:
                if j-i == L-1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-1)
                elif j-i == L+1:
                    H_BdG[i, j] = -1/(2*epsilon) * (1)
            #左下
            elif i >= L and j < L:
                if i-j == L-1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-1)
                elif i-j == L+1:
                    H_BdG[i, j] = -1/(2*epsilon) * (1)
            #右下
            else:
                if i == j:
                    H_BdG[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))
                elif i-j == 1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-p - 1j*(beta(j+1/2-L,L,pos,epsilon)+beta(i+1/2-L,L,pos,epsilon))/2)
                elif j-i == 1:
                    H_BdG[i, j] = -1/(2*epsilon) * (-p + 1j*(beta(i+1/2-L,L,pos,epsilon)+beta(j+1/2-L,L,pos,epsilon))/2)

    if PBC == True:
        #red
        H_BdG[0,L-1] = -1/(2*epsilon) * (p - 1j*beta(L-1,L,pos,epsilon)) * (1)
        H_BdG[2*L-1,L] = -1/(2*epsilon) * (p - 1j*beta(L-1,L,pos,epsilon)) * (-1)
        #blue
        H_BdG[L-1,0] = -1/(2*epsilon) * (p + 1j*beta(L-1,L,pos,epsilon)) * (1)
        H_BdG[L,2*L-1] = -1/(2*epsilon) * (p + 1j*beta(L-1,L,pos,epsilon)) * (-1)
        #orange
        H_BdG[0,2*L-1] = -1/(2*epsilon) * (-1)
        H_BdG[L-1,L] = -1/(2*epsilon) * (1)
        #black
        H_BdG[2*L-1,0] = -1/(2*epsilon) * (-1)
        H_BdG[L,L-1] = -1/(2*epsilon) * (1)

    return H_BdG

def diagonalize_bdg_matrix(H_BdG, L):
    #BdG行列を対角化
    eigenvalues, eigenvectors = LA.eigh(H_BdG)

    #固有値、固有ベクトルのソート(確認済み)
    eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
    eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)
    return eigenvalues, eigenvectors

def enforce_particle_hole_symmetry(eigenvectors, L):
    #粒子-反粒子対称性を満たすように固有ベクトルを調整(列方向に調整しないといけないらしい。行方向だとうまくいかない。固有ベクトルを横切るからか？)
    V = np.zeros((2*L, 2*L), dtype=complex)
    for i in range(L):
        V[:,i] = eigenvectors[:,i]
        V[:L,i+L] = np.conj(eigenvectors[L:,i])
        V[L:,i+L] = np.conj(eigenvectors[:L,i])
    return V

H_BdG = build_bdg_matrix(L, p, m, pos, epsilon, PBC)

bs = []
for j in range(L):
    bs.append(float(beta(j,L,pos,epsilon)))
plt.plot(bs)
plt.grid()
plt.show()

#bogoliubov変換行列の作成##########################################################################################################################
eigenvalues, eigenvectors = diagonalize_bdg_matrix(H_BdG, L)
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

eigenvectors = enforce_particle_hole_symmetry(eigenvectors, L)

def build_operator_lists(eigenvectors, L, PBC):
    #cj_dag_cj(作り方は以前と変わらない)
    cj_dag_cj_list = []
    for j in range(L):
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

#演算子の作成########################################################################################################################################
cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list = build_operator_lists(eigenvectors, L, PBC)


def build_initial_state(L, eigenvectors, j0, sigma, PBC, direction_sign):
    psi = np.zeros((L, 1), dtype=complex)

    if PBC == True:
        idx = np.arange(L)
        delta = np.abs(idx - j0)
        periodic_delta = np.minimum(delta, L - delta)
        weights = np.exp(-(periodic_delta**2) / (2 * sigma**2))
        mask = periodic_delta <= 0.5*L
    else:
        weights = np.exp(-((np.arange(L) - j0) ** 2) / (2 * sigma ** 2))
        mask = np.abs(np.arange(L) - j0) <= 0.5*L

    weights[~mask] = 0

    x_ = np.arange(L)
    p_ = weights/weights.sum()
    mean_pos = np.sum(p_ * x_)
    var_pos = np.sum(p_ * (x_ - mean_pos)**2)
    std_pos_ = np.sqrt(var_pos)
    print("初期状態のweightの標準偏差:", std_pos_)

    for j in range(L):
        for n in range(L):
            psi[n, 0] += weights[j] * (
            1/np.sqrt(2) * (np.exp(direction_sign*1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(-direction_sign*1j*np.pi/4) * eigenvectors[j,n].conj())
            )
    #状態ベクトルの規格化
    psi /= np.linalg.norm(psi)
    return psi, weights

#初期状態作成###################################################################################################################
psi, weights = build_initial_state(L, eigenvectors, j0, sigma, PBC, initial_direction_sign)

#ハミルトニアン密度作成###############################################################################################################
def build_energy_densities(cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list, L, epsilon, p, m, pos, PBC):
    H_p = []
    H_m = []
    H_pm = []

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
        H_p_j = np.zeros((L, L), dtype=complex)
        H_m_j = np.zeros((L, L), dtype=complex)
        H_pm_j = np.zeros((L, L), dtype=complex)

        #ハミルトニアン密度作成
        H_p_j = -1j/(4*epsilon) * (1+beta(j+1/2,L,pos,epsilon))  * (1j*(cj_cj1_list[j-1] + cj_cj1_list[j])/2 + (cj_cj1_dag_list[j-1]+cj_cj1_dag_list[j])/2 + (cj_dag_cj1_list[j-1]+cj_dag_cj1_list[j])/2 - 1j*(cj_dag_cj1_dag_list[j-1]+cj_dag_cj1_dag_list[j])/2)
        H_m_j = -1j/(4*epsilon) * (-1+beta(j+1/2,L,pos,epsilon))  * (-1j*(cj_cj1_list[j-1] + cj_cj1_list[j])/2 + (cj_cj1_dag_list[j-1]+cj_cj1_dag_list[j])/2 + (cj_dag_cj1_list[j-1]+cj_dag_cj1_list[j])/2 + 1j*(cj_dag_cj1_dag_list[j-1]+cj_dag_cj1_dag_list[j])/2)
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
                H_p[-1] = np.zeros((L, L), dtype=complex)
                H_m[-1] = np.zeros((L, L), dtype=complex)
                H_pm[-1] = np.zeros((L, L), dtype=complex)

    return {
        "H_p": H_p,
        "H_m": H_m,
        "H_pm": H_pm,
    }

energy_densities = build_energy_densities(cj_dag_cj_list, cj1_cj_list, cj1_dag_cj_list, L, epsilon, p, m, pos, PBC)
H_p = energy_densities["H_p"]
H_m = energy_densities["H_m"]
H_pm = energy_densities["H_pm"]

#真空の量を計算##########################################################################################
def compute_vacuum_values(eigenvectors, L, epsilon, p, m, pos, PBC):
    Hp_v = []
    Hm_v = []
    Hpm_v = []

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
        Hp_v_j = 0
        Hm_v_j = 0
        H_pm_v_j = 0
        F1 = 0
        F2 = 0
        for n in range(L):
            if PBC == True and j == L-1:
                F1 += eigenvectors[0,n] * eigenvectors[L-1,n+L]
                F2 += eigenvectors[0,n+L].conj() * eigenvectors[L-1,n+L]
            elif not PBC and j == L-1:
                F1 += 0
                F2 += 0
            else:
                F1 += eigenvectors[j+1,n] * eigenvectors[j,n+L]
                F2 += eigenvectors[j+1,n+L].conj() * eigenvectors[j,n+L]
        F1_list.append(F1)
        F2_list.append(F2)
        if j == 0:
            # Ensure complex type so .conj() exists for the Hermiticity check below
            Hp_v_j = 0+0j
            Hm_v_j = 0+0j
            H_pm_v_j = 0+0j
        else:
            Hp_v_j = -1/(2*epsilon) * 1j * (1+beta(j+1/2,L,pos,epsilon)) * 1/2 * (1j * (-1*(F1_list[-1]+F1_list[-2])/2) + (-1*(F2_list[-1]+F2_list[-2])/2) + ((F2_list[-1]+F2_list[-2])/2).conj() - 1j * (F1_list[-1]+F1_list[-2]).conj()/2)
            Hm_v_j = -1/(2*epsilon) * 1j * (-1+beta(j+1/2,L,pos,epsilon)) * 1/2 * (-1j * (-1*(F1_list[-1]+F1_list[-2])/2) + (-1*(F2_list[-1]+F2_list[-2])/2) + ((F2_list[-1]+F2_list[-2])/2).conj() + 1j * (F1_list[-1]+F1_list[-2]).conj()/2)
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

vacuum_values = compute_vacuum_values(eigenvectors, L, epsilon, p, m, pos, PBC)
Hp_v = vacuum_values["H_p"]
Hm_v = vacuum_values["H_m"]
Hpm_v = vacuum_values["H_pm"]
c_dag_c_v = vacuum_values["c_dag_c"]
cj1_cj_v = vacuum_values["cj1_cj"]
cj1_dag_cj_v = vacuum_values["cj1_dag_cj"]

#初期状態の量###############################################################################################################

H_p_0 = []
H_m_0 = []
H_pm_0 = []
cdc_0 = []
cj1_cj_0 = []
cj1_dag_cj_0 = []
def compute_std(weights):
    weights = np.abs(weights)
    x = np.arange(len(weights))
    p = weights/weights.sum()
    mean = np.sum(p * x)
    var = np.sum(p * (x - mean)**2)
    var = max(var, 0.0)
    std = np.sqrt(var)
    return std

def compute_expectation_profile(psi, operators, vacuum_values):
    profile = []
    raw_values = []
    for j, operator in enumerate(operators):
        val = psi.T.conj() @ operator @ psi
        profile.append(val.item() - vacuum_values[j])
        raw_values.append(val)
    return profile, raw_values

#H_pの期待値
H_p_0, _ = compute_expectation_profile(psi, H_p, Hp_v)
#H_pの標準偏差
weights = np.array([x.real for x in H_p_0])
std_pos_p = compute_std(weights)
print("H_pの標準偏差:", std_pos_p)
#H_mの期待値
H_m_0, _ = compute_expectation_profile(psi, H_m, Hm_v)
#H_mの標準偏差
x_ = np.arange(L)
weights = np.array([x.real for x in H_m_0])
std_pos_m = compute_std(weights)
print("H_mの標準偏差:", std_pos_m)

#H_pmの期待値
H_pm_0, _ = compute_expectation_profile(psi, H_pm, Hpm_v)

#c_dag_cの期待値
cdc_0, _ = compute_expectation_profile(psi, cj_dag_cj_list, c_dag_c_v)

#cj1_cjの期待値
cj1_cj_0, _ = compute_expectation_profile(psi, cj1_cj_list, cj1_cj_v)

#cj1_dag_cjの期待値
cj1_dag_cj_0, _ = compute_expectation_profile(psi, cj1_dag_cj_list, cj1_dag_cj_v)

def log_imag(name, arr):
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
        for n,E in enumerate(eigenvalues[:L]):
            psi[n] = np.exp(-1j*E*(dt*(i+1))) * psi_initial[n]

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
        weights = np.array([x.real for x in H_p_t_val])
        std_pos = compute_std(weights)
        if not std_pos_p is None:
            sigmas["H_p"].append(std_pos - std_pos_p)
        weights = np.array([x.real for x in H_m_t_val])
        std_pos = compute_std(weights)
        if not std_pos_m is None:
            sigmas["H_m"].append(std_pos - std_pos_m)

        psi = psi_initial.copy()

        print("時間発展中:"+str(int(i/len(times)*100)) + "%")

    return values, sigmas, x_ave_list

#時間発展###########################################################################################################################
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
H_p_val = time_values["H_p"]
H_m_val = time_values["H_m"]
H_pm_val = time_values["H_pm"]
c_dag_c_val = time_values["c_dag_c"]
H_p_sigmas = sigmas["H_p"]
H_m_sigmas = sigmas["H_m"]
H_pm_sigmas = sigmas["H_pm"]
c_dag_c_sigmas = sigmas["c_dag_c"]

def plot_density_map(
    values,
    times,
    output_path,
    *,
    pos,
    geodesic_time_scale,
    geodesic_color="white",
    geodesic_linestyle="-",
    geodesic_linewidth=2,
):
    value_arr = np.array(values, dtype=complex)
    value_array = np.real(value_arr).astype(float) #ここで実数にしていることに注意

    data_min = np.nanmin(value_array)
    data_max = np.nanmax(value_array)

    nt, num_sites = value_array.shape #時間ステップ数(使わない)とサイト数を取得

    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 25,
        'axes.titlesize': 22,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    plt.figure(figsize=(8, 6))

    # 背景の密度プロット
    plt.imshow(
        value_array,
        aspect='auto',
        origin='lower',
        extent=[0, num_sites, times[0], times[-1]],
        cmap='viridis',
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
        x_scaled = (x / (2 * np.pi)) * num_sites
        t_scaled = (t / geodesic_time_scale) * (times[-1] - times[0]) + times[0]

        plt.plot(
            x_scaled,
            t_scaled,
            linestyle=geodesic_linestyle,
            color=geodesic_color,
            linewidth=geodesic_linewidth,
            label='geodesic'
        )
        plt.legend(loc='upper right', fontsize=12)
    except Exception as e:
        print(f"Warning: geodesic.dat の重ね描画に失敗しました: {e}")
    # ————————————————————————————————

    cbar = plt.colorbar(location='left')
    cbar.ax.tick_params(labelsize=16)

    plt.xlabel('j', fontweight='bold')
    plt.ylabel('t', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path,
                dpi=300, bbox_inches='tight', transparent=True)

#################################################################################################################################
#プロット
#位置平均
plt.plot(times, x_ave_list, label="x average")
plt.xlabel("time")
plt.ylabel("x average")
plt.legend()
plt.grid()
plt.show()
np.savetxt("x_ave.txt", np.column_stack([times, x_ave_list]), fmt="%.10e")
#標準偏差のプロット
np.savetxt("H_m_sigmas.txt", np.column_stack([times, H_m_sigmas]), fmt="%.10e")

density_plot_configs = {
    "H_p": {
        "values": H_p_val,
        "output_path": "figure/H_p.png",
        "geodesic_time_scale": t_f,
        "geodesic_color": "red",
        "geodesic_linestyle": "--",
        "geodesic_linewidth": 1,
    },
    "H_m": {
        "values": H_m_val,
        "output_path": "figure/H_m.png",
        "geodesic_time_scale": 1.21,
    },
    "c_dag_c": {
        "values": c_dag_c_val,
        "output_path": "figure/c_dag_c.png",
        "geodesic_time_scale": 20.0,
    },
    "H_pm": {
        "values": H_pm_val,
        "output_path": "figure/H_pm.png",
        "geodesic_time_scale": 20.0,
    },
}

for plot_config in density_plot_configs.values():
    plot_density_map(
        plot_config["values"],
        times,
        plot_config["output_path"],
        pos=pos,
        geodesic_time_scale=plot_config["geodesic_time_scale"],
        geodesic_color=plot_config.get("geodesic_color", "white"),
        geodesic_linestyle=plot_config.get("geodesic_linestyle", "-"),
        geodesic_linewidth=plot_config.get("geodesic_linewidth", 2),
    )

plt.close('all')
# H_p の標準偏差が最小になるときの t
idx_p = np.argmin(H_p_sigmas)   # 最小値を取るインデックス
t_p_min = times[idx_p]
print("H_p sigma が最小になる t:", t_p_min)
print("そのときの H_p sigma:", H_p_sigmas[idx_p])
weights = np.abs(np.array([x.real for x in H_p_val[idx_p]]))
plt.plot(H_p_val[idx_p])
plt.title("H_p at t = {:.3f}".format(t_p_min))
plt.grid()
plt.show()

# H_m の標準偏差が最小になるときの t
idx_m = np.argmin(H_m_sigmas)
t_m_min = times[idx_m]
print("H_m sigma が最小になる t:", t_m_min)
print("そのときの H_m sigma:", H_m_sigmas[idx_m])
plt.plot(H_m_val[idx_m])
plt.title("H_m at t = {:.3f}".format(t_m_min))
plt.grid()
plt.show()

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
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":")

    # ---------- アニメーション用コールバック -------------------------------
    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
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
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    print("保存しました")
    plt.close(fig)  # 余分なウインドウを閉じる

animation_configs = {
    "H_p": {
        "density": H_p_val,
        "gif_path": "figure/H_p.gif",
        "cmap_line": "blue",
    },
    "H_m": {
        "density": H_m_val,
        "gif_path": "figure/H_m.gif",
        "cmap_line": "orange",
    },
    "H_pm": {
        "density": H_pm_val,
        "gif_path": "figure/H_pm.gif",
        "cmap_line": "green",
    },
}

for animation_config in animation_configs.values():
    save_density_animation(
        density=animation_config["density"],
        times=times,
        gif_path=animation_config["gif_path"],
        xlabel="Lattice Site Index j",
        ylabel=r'δ$\langle c_j^\dagger c_j \rangle$',
        line_label=r'δ$\langle c_j^\dagger c_j \rangle$',
        cmap_line=animation_config["cmap_line"],
        PBC=PBC,
    )
