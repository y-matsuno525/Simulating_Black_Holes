#import
import numpy as np
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt

#パラメータ
L = 300
l = 2*np.pi
epsilon = l / L
p = 1#0**(-5)
m = 0
pos = "ll" #lr, ur, ll, ul
t_i = 0
t_f = 10
dt = 0.01*(300/L) #この値は後で検討
PBC = False

times = np.arange(t_i + dt, t_f, dt)

#BdGハミルトニアンの作成(符号関係は確認済み)
H_BdG = np.zeros((2*L, 2*L), dtype=complex)

def is_hermitian(matrix):
    is_hermitian = np.allclose(matrix, np.conj(matrix.T), atol=1e-10)

    if is_hermitian:
        return True
    else:
        return False

def beta(j,L,pos,epsilon):
    #return 0.5
    width = 1
    A = 0.6
    jh = int(L/3)
    if pos == "lr":
        # β = -1 を j = 71 で踏むように調整
        return -A*np.tanh(3/width*(j - 2*jh - 0.730833344)*epsilon) - 0.6
    elif pos == "ur":
        # β = +1 を j = 70 で踏むように調整
        return  A*np.tanh(3/width*(j - 2*jh + 0.269166656056467)*epsilon) + 0.6
        # （注）式は (j - center - c) なので c = -0.269... は “+0.269...” と等価
    elif pos == "ll":
        # β = -1 を j = 29 で踏むように調整
        return  A*np.tanh(3/width*(j - jh - 0.269166656056467)*epsilon) - 0.6
    elif pos == "ul":
        # β = +1 を j = 29 で踏むように調整
        return -A*np.tanh(3/width*(j - jh - 0.269166656056467)*epsilon) + 0.6

for i in range(2*L):
    for j in range(2*L):
        #左上
        if i < L and j < L:
            if i == j:
                H_BdG[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))*(-1)
            elif i-j == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (p - 1j*beta(i,L,pos,epsilon))
            elif j-i == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (p + 1j*beta(i,L,pos,epsilon))
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
                H_BdG[i, j] = -1/(2*epsilon) * (-p - 1j*beta(i-L,L,pos,epsilon))
            elif j-i == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (-p + 1j*beta(j-L,L,pos,epsilon))

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

bs = []
for j in range(L):
    bs.append(float(beta(j,L,pos,epsilon)))
plt.plot(bs)
plt.grid()
plt.show()

#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(H_BdG)

#固有値、固有ベクトルのソート(確認済み)
eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)

#粒子-反粒子対称性を満たすように調整(列方向に調整しないといけないらしい。行方向だとうまくいかない。固有ベクトルを横切るからか？)
V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V
print(eigenvalues)
print(eigenvectors)

# #粒子-反粒子対称性の確認(確認済み)
# for j in range(L):
#     print(eigenvectors[j,:L] - eigenvectors[j+L,L:].conj())
#     print(eigenvectors[j,L:] - eigenvectors[j+L,:L].conj())
# print()
# for j in range(L):
#     print(eigenvectors[j,:L].T.conj() - eigenvectors[j+L,L:].T.conj().conj())
#     print(eigenvectors[j,L:].T.conj() - eigenvectors[j+L,:L].T.conj().conj())

#演算子の作成
#cj_dag_cj(作り方は以前と変わらない)
cj_dag_cj_list = []
for j in range(L):
    cj_dag_cj_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_dag_cj_tmp[k,l] = eigenvectors[j,k].conj() * eigenvectors[j,l]
            cj_dag_cj_tmp[k,l] -= eigenvectors[j,l+L].conj() * eigenvectors[j,k+L]
            if k == l:
                for n in range(L):
                    cj_dag_cj_tmp[k,l] += eigenvectors[j,n+L].conj() * eigenvectors[j,n+L]
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
H = np.zeros((L, L), dtype=complex)
for i in range(L):
    H[i,i] = eigenvalues[i]
U_dt = scipy.linalg.expm(-1j*H*dt)



#初期状態作成###################################################################################################################
psi = np.zeros((L, 1), dtype=complex)
if pos == "ur" or pos == "lr":
    j0 = int(0.2*L)
else:
    j0 = int(0.8*L)
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

weights /= np.linalg.norm(weights)
#plt.plot(weights)
plt.plot(np.r_[weights, weights[0]])
plt.title("weights")
plt.grid()
plt.show()

for j in range(L):
    for n in range(L):
        #cを置く場合
        #psi[n, 0] += weights[j] * (eigenvectors[j,n].conj())
        #+
        if pos == "ur" or pos == "lr":
            psi[n, 0] += weights[j] * (
            1/np.sqrt(2) * (np.exp(1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(-1j*np.pi/4) * eigenvectors[j,n].conj())
        )
        #-
        else:
            psi[n, 0] += weights[j] * (
            1/np.sqrt(2) * (np.exp(-1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(1j*np.pi/4) * eigenvectors[j,n].conj())
        )
#状態ベクトルの規格化
psi /= np.linalg.norm(psi)



#ハミルトニアン密度作成###############################################################################################################
H_p = []
H_m = []

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

    #ハミルトニアン密度作成
    H_p_j = -1j/(4*epsilon) * (1+beta(j,L,pos,epsilon))  * (1j*cj_cj1_list[j] + cj_cj1_dag_list[j] + cj_dag_cj1_list[j] - 1j*cj_dag_cj1_dag_list[j])
    H_m_j = -1j/(4*epsilon) * (-1+beta(j,L,pos,epsilon))  * (-1j*cj_cj1_list[j] + cj_cj1_dag_list[j] + cj_dag_cj1_list[j] + 1j*cj_dag_cj1_dag_list[j])

    #エルミートか確認
    isHermitian = is_hermitian(H_p_j)
    assert isHermitian, "H_p(j=" + str(j) + ") is not Hermitian!"
    isHermitian = is_hermitian(H_m_j)
    assert isHermitian, "H_m(j=" + str(j) + ") is not Hermitian!"

    H_p.append(H_p_j)
    H_m.append(H_m_j)



#真空の量を計算##########################################################################################
Hp_v = []
Hm_v = []
for j in range(L):
    Hp_v_j = 0
    Hm_v_j = 0
    F1 = 0
    F2 = 0
    for n in range(L):
        if PBC == True and j == L-1:
            F1 += eigenvectors[0,n] * eigenvectors[L-1,n+L]
            F2 += eigenvectors[0,n+L].conj() * eigenvectors[L-1,n+L]
        else:
            F1 += eigenvectors[j+1,n] * eigenvectors[j,n+L]
            F2 += eigenvectors[j+1,n+L].conj() * eigenvectors[j,n+L]
    Hp_v_j = -1/(2*epsilon) * 1j * (1+beta(j,L,pos,epsilon)) * 1/2 * (1j * (-1*F1) + (-1*F2) + (F2.conj()) - 1j * (F1.conj()))
    Hm_v_j = -1/(2*epsilon) * 1j * (-1+beta(j,L,pos,epsilon)) * 1/2 * (-1j * (-1*F1) + (-1*F2) + (F2.conj()) + 1j * (F1.conj()))
    assert abs(Hp_v_j - Hp_v_j.conj()) < 10**-5, "Hp_v_j(j=" + str(j) + ") is not Hermitian!"
    assert abs(Hm_v_j - Hm_v_j.conj()) < 10**-5, "Hm_v_j(j=" + str(j) + ") is not Hermitian!"
    Hp_v.append(Hp_v_j)
    Hm_v.append(Hm_v_j)
    if not PBC:
        if j == L-1:
            Hp_v[-1] = 0
            Hm_v[-1] = 0

plt.plot(Hp_v, label="Hp_v")
plt.plot(Hm_v, label="Hm_v")
plt.legend()
plt.show()

#真空のc_dag_cを計算
c_dag_c_v = []
for j in range(L):
    total = 0.0
    for n in range(L):
        total += abs(eigenvectors[j, n+L])**2
    c_dag_c_v.append(total)

plt.plot(c_dag_c_v)
plt.title("c_dag_c_v")
plt.grid()
plt.show()

#初期状態の量###############################################################################################################

H_p_0 = []
H_m_0 = []
c_0 = []
#H_pの期待値
for j, H_p_j in enumerate(H_p):
    val = psi.T.conj() @ H_p_j @ psi
    H_p_0.append(val.item() - Hp_v[j])

#H_mの期待値
for j, H_m_j in enumerate(H_m):
    val = psi.T.conj() @ H_m_j @ psi
    H_m_0.append(val.item() - Hm_v[j])

#c_dag_cの期待値
for j, cj_dag_cj in enumerate(cj_dag_cj_list):
    val = psi.T.conj() @ cj_dag_cj @ psi
    c_0.append(val.item() - c_dag_c_v[j])

plt.plot(H_p_0, label="H_p_0")
plt.plot(H_m_0, label="H_m_0")
plt.legend()
plt.grid()
plt.show()
plt.plot(c_0, label="c_0")
plt.legend()
plt.grid()
plt.show()


#時間発展###########################################################################################################################
H_p_val = []
H_m_val = []
c_dag_c_val = []
for i, _ in enumerate(times):
    psi = U_dt @ psi
    psi /= np.linalg.norm(psi)

    H_p_t_val = []
    H_m_t_val = []
    c_dag_c_t_val = []

    #H_pの期待値
    for j, H_p_j in enumerate(H_p):
        val = psi.T.conj() @ H_p_j @ psi
        H_p_t_val.append(val.item() - Hp_v[j])
    H_p_val.append(H_p_t_val)

    #H_mの期待値
    for j, H_m_j in enumerate(H_m):
        val = psi.T.conj() @ H_m_j @ psi
        H_m_t_val.append(val.item() - Hm_v[j])
    H_m_val.append(H_m_t_val)

    #c_dag_cの期待値
    for j, cj_dag_cj in enumerate(cj_dag_cj_list):
        val = psi.T.conj() @ cj_dag_cj @ psi
        c_dag_c_t_val.append(val.item() - c_dag_c_v[j])
    c_dag_c_val.append(c_dag_c_t_val)

    print("時間発展中:"+str(int(i/len(times)*100)) + "%")

#################################################################################################################################
#プロット
#+
H_p_arr = np.array(H_p_val, dtype=complex)
H_p_array = np.real(H_p_arr).astype(float) #ここで実数にしていることに注意

data_min = np.nanmin(H_p_array)
data_max = np.nanmax(H_p_array)

nt, L = H_p_array.shape #時間ステップ数(使わない)とサイト数を取得

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
    H_p_array,
    aspect='auto',
    origin='lower',
    extent=[0, L, times[0], times[-1]],
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
    x_scaled = (x / (2 * np.pi)) * L
    t_scaled = (t / 20.0) * (times[-1] - times[0]) + times[0]

    plt.plot(x_scaled, t_scaled, color='white', linewidth=2, label='geodesic')
    plt.legend(loc='upper right', fontsize=12)
except Exception as e:
    print(f"Warning: geodesic.dat の重ね描画に失敗しました: {e}")
# ————————————————————————————————

cbar = plt.colorbar(location='left')
cbar.ax.tick_params(labelsize=16)

plt.xlabel('j', fontweight='bold')
plt.ylabel('t', fontweight='bold')
plt.tight_layout()
plt.savefig('figure/H_p.png',
            dpi=300, bbox_inches='tight', transparent=True)

#-
H_m_arr = np.array(H_m_val, dtype=complex)
H_m_array = np.real(H_m_arr).astype(float) #ここで実数にしていることに注意

data_min = np.nanmin(H_m_array)
data_max = np.nanmax(H_m_array)

nt, L = H_m_array.shape #時間ステップ数(使わない)とサイト数を取得

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
    H_m_array,
    aspect='auto',
    origin='lower',
    extent=[0, L, times[0], times[-1]],
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
    x_scaled = (x / (2 * np.pi)) * L
    t_scaled = (t / 20.0) * (times[-1] - times[0]) + times[0]

    plt.plot(x_scaled, t_scaled, color='white', linewidth=2, label='geodesic')
    plt.legend(loc='upper right', fontsize=12)
except Exception as e:
    print(f"Warning: geodesic.dat の重ね描画に失敗しました: {e}")
# ————————————————————————————————

cbar = plt.colorbar(location='left')
cbar.ax.tick_params(labelsize=16)

plt.xlabel('j', fontweight='bold')
plt.ylabel('t', fontweight='bold')
plt.tight_layout()
plt.savefig('figure/H_m.png',
            dpi=300, bbox_inches='tight', transparent=True)

#c_dag_c
c_dag_c_arr = np.array(c_dag_c_val, dtype=complex)
c_dag_c_array = np.real(c_dag_c_arr).astype(float) #ここで実数にしていることに注意

data_min = np.nanmin(c_dag_c_array)
data_max = np.nanmax(c_dag_c_array)

nt, L = c_dag_c_array.shape #時間ステップ数(使わない)とサイト数を取得

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
    c_dag_c_array,
    aspect='auto',
    origin='lower',
    extent=[0, L, times[0], times[-1]],
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
    x_scaled = (x / (2 * np.pi)) * L
    t_scaled = (t / 20.0) * (times[-1] - times[0]) + times[0]

    plt.plot(x_scaled, t_scaled, color='white', linewidth=2, label='geodesic')
    plt.legend(loc='upper right', fontsize=12)
except Exception as e:
    print(f"Warning: geodesic.dat の重ね描画に失敗しました: {e}")
# ————————————————————————————————

cbar = plt.colorbar(location='left')
cbar.ax.tick_params(labelsize=16)

plt.xlabel('j', fontweight='bold')
plt.ylabel('t', fontweight='bold')
plt.tight_layout()
plt.savefig('figure/c_dag_c.png',
            dpi=300, bbox_inches='tight', transparent=True)

