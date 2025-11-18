#cがgaussiasnで置けてるか確認
#import
import numpy as np
import random
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt
import random
#パラメータ
L = 100
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

V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V
for i in range(2*L):
    print(eigenvalues[i])
    print(eigenvectors[:,i])

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