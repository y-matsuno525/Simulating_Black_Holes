#cがgaussiasnで置けてるか確認
#import
import numpy as np
import random
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt
import random
#パラメータ
L = 10
l = 2*np.pi
epsilon = l / L
p = 1
m = 0.0001
pos = "lr" #lr, ur, ll, ul
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

#bogoliubov変換行列の作成##########################################################################################################################

#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(H_BdG)

#固有値、固有ベクトルのソート(確認済み)
eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)
# print("固有ベクトル（旧）")
# for i in range(L):
#     print(eigenvalues[i])
#     print(eigenvectors[:,i])
#     print(eigenvalues[L+i])
#     print(eigenvectors[:,L+i])
# #ゼロモード(0とL)入れ替え
# tmp = eigenvectors[:,0].copy()
# eigenvectors[:,0] = eigenvectors[:,L].copy()
# eigenvectors[:,L] = tmp

#粒子-反粒子対称性を満たすように固有ベクトルを調整(0~L-1成分でL~2L-1成分を作る)
# cnt = []
# for i in range(L):
#     threshold = 1e-10
#     ans = eigenvectors[:,i+L] + np.concatenate((eigenvectors[L:,i].conj(), eigenvectors[:L,i].conj()), 0)
#     if np.all(np.abs(ans) < threshold):
#         #continue
#         print(str(i))
#         cnt.append(1)
#         #print(np.where(np.abs(ans) < threshold, 0.0, ans))
#     else:
#         print(str(i)+"'")
#         ans = eigenvectors[:,i+L] - np.concatenate((eigenvectors[L:,i].conj(), eigenvectors[:L,i].conj()), 0)
#         if np.all(np.abs(ans) < threshold):
#             cnt.append(-1)
#             continue
#             print(np.where(np.abs(ans) < threshold, 0.0, ans))
#         else:
#             cnt.append(0)
#             print(str(i)+"''")
#             ans = eigenvectors[:,i+L].T.conj() @ np.concatenate((eigenvectors[L:,i].conj(), eigenvectors[:L,i].conj()), 0)
#             print(np.where(np.abs(ans) < threshold, 0.0, ans))
# plt.plot(cnt)
# plt.show()
# bad_idx = []
# for i in range(L-1):
#     # i と i+1 が 1,-1 または -1,1 のペアならOK
#     if cnt[i] in (1, -1) and cnt[i+1] in (1, -1) and cnt[i+1] == -cnt[i]:
#         continue
#     # それ以外は「交互じゃない」とみなして記録
#     bad_idx.append(i)

# print("1,-1 が交互になっていない位置 i (ペア: i, i+1):", bad_idx)
# d_cnt = []
# for i in range(L-1):
#     if abs(cnt[i+1] - cnt[i]) -2 < 1e-5:
#         d_cnt.append(1)
#     else:
#         d_cnt.append(0)
# plt.plot(d_cnt)
# plt.show()
V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V
for i in range(2*L):
    print(eigenvalues[i])
    print(eigenvectors[:,i])
# print()
# print("固有ベクトル（新）")
# for i in range(L):
#     print(eigenvalues[i])
#     print(eigenvectors[:,i])
#     print(eigenvalues[L+i])
#     print(eigenvectors[:,L+i])
# import sys
# sys.exit()
# for i in range(L):
#     print("固有値")
#     print(eigenvalues[i])
#     print("固有ベクトル")
#     print(eigenvectors[:,i])
#     print("固有値")
#     print(eigenvalues[i+L])
#     print("固有ベクトル")
#     print(eigenvectors[:,i+L])

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

#初期状態作成
psi = np.zeros((L, 1), dtype=complex)
if pos == "ur" or pos == "lr":
    j0 = int(0.5*L)
else:
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

weights /= np.linalg.norm(weights)


# plt.plot(np.r_[weights, weights[0]])
# plt.title("weights")
# plt.grid()
# plt.show()

for j in range(L):
    for n in range(L):
        #cを置く場合
        psi[n, 0] += weights[j] * (eigenvectors[j,n].conj())
        #+
        # if pos == "ur" or pos == "lr":
        #     psi[n, 0] += weights[j] * (
        #     1/np.sqrt(2) * (np.exp(1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(-1j*np.pi/4) * eigenvectors[j,n].conj())
        # )
        # #-
        # else:
        #     psi[n, 0] += weights[j] * (
        #     1/np.sqrt(2) * (np.exp(-1j*np.pi/4) * eigenvectors[j,n+L] + np.exp(1j*np.pi/4) * eigenvectors[j,n].conj())
        # )
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

import statistics
print(statistics.stdev(weights))
print(statistics.stdev([x.real for x in c_0]))
plt.plot(c_0)
plt.title("<c†c>(t=0)")
plt.grid()
plt.show()