#import
import numpy as np
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt

#パラメータ
L = 50
l = 2*np.pi
epsilon = l / L
p = 1
m = 0.0001
dt = 0.01*(300/L)
PBC = True

#BdGハミルトニアンの作成(符号関係は確認済み)
H_BdG = np.zeros((2*L, 2*L), dtype=complex)

def is_hermitian(matrix):
    is_hermitian = np.allclose(matrix, np.conj(matrix.T), atol=1e-10)

    if is_hermitian:
        return True
    else:
        return False

def beta(j,L,epsilon):
    return 0

for i in range(2*L):
    for j in range(2*L):
        #左上
        if i < L and j < L:
            if i == j:
                H_BdG[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))*(-1)
            elif i-j == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (p - 1j*beta(j,L,epsilon))
            elif j-i == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (p + 1j*beta(i,L,epsilon))
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
                H_BdG[i, j] = -1/(2*epsilon) * (-p - 1j*beta(j-L,L,epsilon))
            elif j-i == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (-p + 1j*beta(i-L,L,epsilon))

if PBC == True:
    #red
    H_BdG[0,L-1] = -1/(2*epsilon) * (p - 1j*beta(L-1,L,epsilon)) * (1)
    H_BdG[2*L-1,L] = -1/(2*epsilon) * (p - 1j*beta(L-1,L,epsilon)) * (-1)
    #blue
    H_BdG[L-1,0] = -1/(2*epsilon) * (p + 1j*beta(L-1,L,epsilon)) * (1)
    H_BdG[L,2*L-1] = -1/(2*epsilon) * (p + 1j*beta(L-1,L,epsilon)) * (-1)
    #orange
    H_BdG[0,2*L-1] = -1/(2*epsilon) * (-1)
    H_BdG[L-1,L] = -1/(2*epsilon) * (1)
    #black
    H_BdG[2*L-1,0] = -1/(2*epsilon) * (-1)
    H_BdG[L,L-1] = -1/(2*epsilon) * (1)

#bogoliubov変換行列の作成##########################################################################################################################
#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(H_BdG)

#固有値、固有ベクトルのソート(確認済み)
eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)

V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V

#演算子の作成########################################################################################################################################
#cj_dag_cj
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


#時間発展演算子作成
def generate_time_evolution_operator(eigenvalues, n):
    H = np.zeros((L, L), dtype=complex)
    for i in range(L):
        H[i,i] = eigenvalues[i]
    U_dt = scipy.linalg.expm(-1j*H*(dt*(n+1)))
    return U_dt



#初期状態作成###################################################################################################################
psi = np.zeros((L, 1), dtype=complex)
j0 = int(0.5*L)
sigma = 0.05*L

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

for j in range(L):
    for n in range(L):
        #+
        psi[n, 0] += weights[j] * (
        1/np.sqrt(2) * (np.exp(1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(-1j*np.pi/4) * eigenvectors[j,n].conj())
        )
        #-
        # psi[n, 0] += weights[j] * (
        # 1/np.sqrt(2) * (np.exp(-1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(1j*np.pi/4) * eigenvectors[j,n].conj())
        # )
#状態ベクトルの規格化
psi /= np.linalg.norm(psi)

#ハミルトニアン密度作成###############################################################################################################
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
    H_p_j = -1j/(4*epsilon) * (1+beta(j,L,epsilon))  * (1j*cj_cj1_list[j] + cj_cj1_dag_list[j] + cj_dag_cj1_list[j] - 1j*cj_dag_cj1_dag_list[j])
    H_m_j = -1j/(4*epsilon) * (-1+beta(j,L,epsilon))  * (-1j*cj_cj1_list[j] + cj_cj1_dag_list[j] + cj_dag_cj1_list[j] + 1j*cj_dag_cj1_dag_list[j])
    H_pm_j = -1j/(2*epsilon) * (p*(1j*cj_cj1_dag_list[j] - 1j*cj_dag_cj1_list[j]) - (p - epsilon*m)*(-2*1j*cj_dag_cj_list[j]))

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
        if j == L-1:
            H_p[-1] = np.zeros((L, L), dtype=complex)
            H_m[-1] = np.zeros((L, L), dtype=complex)
            H_pm[-1] = np.zeros((L, L), dtype=complex)

#真空の量を計算##########################################################################################
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
        else:
            F1 += eigenvectors[j+1,n] * eigenvectors[j,n+L]
            F2 += eigenvectors[j+1,n+L].conj() * eigenvectors[j,n+L]
    Hp_v_j = -1/(2*epsilon) * 1j * (1+beta(j,L,epsilon)) * 1/2 * (1j * (-1*F1) + (-1*F2) + (F2.conj()) - 1j * (F1.conj()))
    Hm_v_j = -1/(2*epsilon) * 1j * (-1+beta(j,L,epsilon)) * 1/2 * (-1j * (-1*F1) + (-1*F2) + (F2.conj()) + 1j * (F1.conj()))
    H_pm_v_j = -1j/(2*epsilon) * (1j * p * (-1*F2 - F2.conj()) - (p - epsilon*m) * (-2j * c_dag_c_v[j]))
    assert abs(Hp_v_j - Hp_v_j.conj()) < 10**-5, "Hp_v_j(j=" + str(j) + ") is not Hermitian!"
    assert abs(Hm_v_j - Hm_v_j.conj()) < 10**-5, "Hm_v_j(j=" + str(j) + ") is not Hermitian!"
    Hp_v.append(Hp_v_j)
    Hm_v.append(Hm_v_j)
    Hpm_v.append(H_pm_v_j)
    if not PBC:
        if j == L-1:
            Hp_v[-1] = 0
            Hm_v[-1] = 0
            Hpm_v[-1] = 0

#初期状態の量###############################################################################################################

H_p_0 = []
H_m_0 = []
H_pm_0 = []
#H_pの期待値
for j, H_p_j in enumerate(H_p):
    val = psi.T.conj() @ H_p_j @ psi
    H_p_0.append(val.item() - Hp_v[j])
#H_mの期待値
for j, H_m_j in enumerate(H_m):
    val = psi.T.conj() @ H_m_j @ psi
    H_m_0.append(val.item() - Hm_v[j])
#H_pmの期待値
for j, H_pm_j in enumerate(H_pm):
    val = psi.T.conj() @ H_pm_j @ psi
    H_pm_0.append(val.item() - Hpm_v[j])

plt.plot(H_p_0, label="H_p_0")
plt.plot(H_m_0, label="H_m_0")
plt.plot(H_pm_0, label="H_pm_0")
plt.legend()
plt.grid()
plt.show()