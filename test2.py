#cがgaussiasnで置けてるか確認
#import
import numpy as np
import random
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt
import random
#パラメータ
L = 20
l = 2*np.pi
epsilon = l / L
p = 1
m = 0.0001
pos = "lr" #lr, ur, ll, ul
t_i = 0
t_f = 10
dt = 0.01*(300/L) #この値は後で検討
PBC = True

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
    return 0
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
                H_BdG[i, j] = -1/(2*epsilon) * (p - 1j*beta(j,L,pos,epsilon))
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
                H_BdG[i, j] = -1/(2*epsilon) * (-p - 1j*beta(j-L,L,pos,epsilon))
            elif j-i == 1:
                H_BdG[i, j] = -1/(2*epsilon) * (-p + 1j*beta(i-L,L,pos,epsilon))

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

# bs = []
# for j in range(L):
#     bs.append(float(beta(j,L,pos,epsilon)))
# plt.plot(bs)
# plt.grid()
# plt.show()

#bogoliubov変換行列の作成##########################################################################################################################
#BdG行列を対角化
#print(H_BdG)
eigenvalues, eigenvectors = LA.eigh(H_BdG)
threshold = 1e-14
eigenvectors = np.where(np.abs(eigenvectors) < threshold, 0.0, eigenvectors)
#固有値、固有ベクトルのソート(確認済み)
eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)

for i in range(1,L-1,2):
    if random.random() < 0.5:
        print(str(i) + "番目と" + str(i+1) + "番目を入れ替え")
        tmp = eigenvectors[:,i].copy()
        eigenvectors[:,i] = -1*eigenvectors[:,i+1].copy()
        eigenvectors[:,i+1] = -1*tmp
        print(str(L + i) + "番目と" + str(L + i + 1) + "番目を入れ替え")
        tmp = eigenvectors[:,L+i].copy()
        eigenvectors[:,L+i] = -1*eigenvectors[:,L+i+1].copy()
        eigenvectors[:,L+i+1] = -1*tmp
# print("0番目と"+str(L)+"番目を入れ替え")
# tmp = eigenvectors[:,0].copy()
# eigenvectors[:,0] = -1*eigenvectors[:,L].copy()
# eigenvectors[:,L] = -1*tmp

#粒子-反粒子対称性を満たすように固有ベクトルを調整(列方向に調整しないといけないらしい。行方向だとうまくいかない。固有ベクトルを横切るからか？)
V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V

print("固有値")
print(eigenvalues)
print("ゼロモード固有ベクトル")
print(eigenvectors[:,0])
print(eigenvectors[:,L])
print("ゼロモード固有ベクトル確認")
print([(H_BdG @ eigenvectors[:,0])[i] / eigenvectors[:,0][i] for i in range(2*L)])
print([(H_BdG @ eigenvectors[:,L])[i] / eigenvectors[:,L][i] for i in range(2*L)]) 
# print("入れ替え前")
# print(eigenvectors)
# eigenvectors[:,1] = -1*eigenvectors[:,1]
# eigenvectors[:,2] = -1*eigenvectors[:,2]
# eigenvectors[:,6] = -1*eigenvectors[:,6]
# eigenvectors[:,7] = -1*eigenvectors[:,7]

# print("入れ替え後")
# print(eigenvalues)
# print(eigenvectors)

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
    cj_dag_cj_list.append(cj_dag_cj_tmp)
    print("cj†cj作成中:" + str(int(j/L*100))+"%")

# # #cj_cj_dag
# cj_cj_dag_list = []
# for j in range(L):
#     cj_cj_dag_tmp = np.zeros((L, L), dtype=complex)
#     for k in range(L):
#         for l in range(L):
#             cj_cj_dag_tmp[k,l] = eigenvectors[j,k+L] * eigenvectors[j,l+L].conj()
#             cj_cj_dag_tmp[k,l] += -1*eigenvectors[j,l] * eigenvectors[j,k].conj()
#             if k == l:
#                 for n in range(L):
#                     cj_cj_dag_tmp[k,l] += eigenvectors[j,n] * eigenvectors[j,n].conj()
#     cj_cj_dag_list.append(cj_cj_dag_tmp)
#     print("cj cj†作成中:" + str(int(j/L*100))+"%")

# for j in range(L):
#     print("確認中:" + str(j))
#     print(cj_dag_cj_list[j] + cj_cj_dag_list[j])
#     print()

#初期状態作成
psi = np.zeros((L, 1), dtype=complex)
if pos == "ur" or pos == "lr":
    j0 = int(0.5*L)
else:
    j0 = int(0.5*L)
sigma = 0.05*L #c_0はsigmaのLの係数に反比例傾向(完全反比例ではない)

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
# print("左右非対称箇所")
# print(eigenvectors[2,:])
# print(eigenvectors[2+L,:])
# print(eigenvectors[4,:])
# print(eigenvectors[4+L,:])
for j in range(L):
    for n in range(L):
        # #cを置く場合
        # psi[n, 0] += weights[j] * (eigenvectors[j,n].conj())
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

#真空のc_dag_cを計算
c_dag_c_v = []
for j in range(L):
    total = 0.0
    for n in range(L):
        total += abs(eigenvectors[j, n+L])**2
    c_dag_c_v.append(total)

#c_dag_cの期待値
c_0 = []
for j, cj_dag_cj in enumerate(cj_dag_cj_list):
    val = psi.T.conj() @ cj_dag_cj @ psi
    c_0.append(val.item() - c_dag_c_v[j])

plt.plot(c_0)
plt.title("<c†c>(t=0)")
plt.grid()
plt.show()