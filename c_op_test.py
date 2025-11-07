#import
import numpy as np
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt

#パラメータ
L = 10
l = 2*np.pi
epsilon = l / L
p = 10**(-5)
m = 0
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

bs = []
for j in range(L):
    bs.append(float(beta(j,L,pos,epsilon)))
plt.plot(bs)
plt.grid()
plt.show()

#bogoliubov変換行列の作成##########################################################################################################################
#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(H_BdG)
print(eigenvalues)
print(eigenvectors)

#固有値、固有ベクトルのソート(確認済み)
eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)

#粒子-反粒子対称性を満たすように固有ベクトルを調整(列方向に調整しないといけないらしい。行方向だとうまくいかない。固有ベクトルを横切るからか？)
V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V
print(eigenvalues)
print(eigenvectors)

# #粒子-反粒子対称性の確認(確認済み)
# #c = sum gamma
# for j in range(L):
#     print(eigenvectors[j,:L] - eigenvectors[j+L,L:].conj())
#     print(eigenvectors[j,L:] - eigenvectors[j+L,:L].conj())
# print()
# #gamma = sum c
# for j in range(L):
#     print(eigenvectors[j,:L].T.conj() - eigenvectors[j+L,L:].T.conj().conj())
#     print(eigenvectors[j,L:].T.conj() - eigenvectors[j+L,:L].T.conj().conj())
# import sys
# sys.exit()
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
    cj_dag_cj_list.append(cj_dag_cj_tmp)
    print("cj†cj作成中:" + str(int(j/L*100))+"%")

#cj_cj_dag
cj_cj_dag_list = []
for j in range(L):
    cj_cj_dag_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_cj_dag_tmp[k,l] = eigenvectors[j,k+L] * eigenvectors[j,l+L].conj()
            cj_cj_dag_tmp[k,l] += -1*eigenvectors[j,l] * eigenvectors[j,k].conj()
            if k == l:
                for n in range(L):
                    cj_cj_dag_tmp[k,l] += eigenvectors[j,n] * eigenvectors[j,n].conj()
    cj_cj_dag_list.append(cj_cj_dag_tmp)
    print("cj cj†作成中:" + str(int(j/L*100))+"%")

for j in range(L):
    print("確認中:" + str(j))
    print(cj_dag_cj_list[j] + cj_cj_dag_list[j])
    print()

#################################################################################################
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

#cj_cj1
cj_cj1_list = []
for j in range(L-1):
    cj_cj1_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_cj1_tmp[k,l] = eigenvectors[j,k+L] * eigenvectors[j+1,l]
            cj_cj1_tmp[k,l] -= eigenvectors[j,l] * eigenvectors[j+1,k+L]
            if k == l:
                for n in range(L):
                    cj_cj1_tmp[k,l] += eigenvectors[j,n] * eigenvectors[j+1,n+L]
    cj_cj1_list.append(cj_cj1_tmp)
    print("cj_cj1作成中:" + str(int(j/L*100))+"%")
#PBCの場合、右端は非ゼロ
if PBC == True:
    cj_cj1_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_cj1_tmp[k,l] = eigenvectors[L-1,k+L] * eigenvectors[0,l]
            cj_cj1_tmp[k,l] -= eigenvectors[L-1,l] * eigenvectors[0,k+L]
            if k == l:
                for n in range(L):
                    cj_cj1_tmp[k,l] += eigenvectors[L-1,n] * eigenvectors[0,n+L]
    cj_cj1_list.append(cj_cj1_tmp)
else:
    cj_cj1_list.append(np.zeros((L, L), dtype=complex))

#cj1_dag_cj_dag
cj1_dag_cj_dag_list = []
for j in range(L-1):
    cj1_dag_cj_dag_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj1_dag_cj_dag_tmp[k,l] = eigenvectors[j+1,k].conj() * eigenvectors[j,l+L].conj()
            cj1_dag_cj_dag_tmp[k,l] -= eigenvectors[j+1,l+L].conj() * eigenvectors[j,k].conj()
            if k == l:
                for n in range(L):
                    cj1_dag_cj_dag_tmp[k,l] += eigenvectors[j+1,n+L].conj() * eigenvectors[j,n].conj()
    cj1_dag_cj_dag_list.append(cj1_dag_cj_dag_tmp)
    print("cj1† cj†作成中:" + str(int(j/L*100))+"%")
#PBCの場合、右端は非ゼロ
if PBC == True:
    cj1_dag_cj_dag_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj1_dag_cj_dag_tmp[k,l] = eigenvectors[0,k].conj() * eigenvectors[L-1,l+L].conj()
            cj1_dag_cj_dag_tmp[k,l] -= eigenvectors[0,l+L].conj() * eigenvectors[L-1,k].conj()
            if k == l:
                for n in range(L):
                    cj1_dag_cj_dag_tmp[k,l] += eigenvectors[0,n+L].conj() * eigenvectors[L-1,n].conj()
    cj1_dag_cj_dag_list.append(cj1_dag_cj_dag_tmp)
else:
    cj1_dag_cj_dag_list.append(np.zeros((L, L), dtype=complex))

#cj_dag_cj1_dag
cj_dag_cj1_dag_list = []
for j in range(L-1):
    cj_dag_cj1_dag_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_dag_cj1_dag_tmp[k,l] = eigenvectors[j,k].conj() * eigenvectors[j+1,l+L].conj()
            cj_dag_cj1_dag_tmp[k,l] -= eigenvectors[j,l+L].conj() * eigenvectors[j+1,k].conj()
            if k == l:
                for n in range(L):
                    cj_dag_cj1_dag_tmp[k,l] += eigenvectors[j,n+L].conj() * eigenvectors[j+1,n].conj()
    cj_dag_cj1_dag_list.append(cj_dag_cj1_dag_tmp)
    print("cj† cj1†作成中:" + str(int(j/L*100))+"%")
#PBCの場合、右端は非ゼロ
if PBC == True:
    cj_dag_cj1_dag_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_dag_cj1_dag_tmp[k,l] = eigenvectors[L-1,k].conj() * eigenvectors[0,l+L].conj()
            cj_dag_cj1_dag_tmp[k,l] -= eigenvectors[L-1,l+L].conj() * eigenvectors[0,k].conj()
            if k == l:
                for n in range(L):
                    cj_dag_cj1_dag_tmp[k,l] += eigenvectors[L-1,n+L].conj() * eigenvectors[0,n].conj()
    cj_dag_cj1_dag_list.append(cj_dag_cj1_dag_tmp)
else:
    cj_dag_cj1_dag_list.append(np.zeros((L, L), dtype=complex))

for j in range(L):
    print("確認中:" + str(j))
    print("1")
    print(cj1_cj_list[j] + cj_cj1_list[j])
    print("2")
    print(cj1_cj_list[j] + cj1_dag_cj_dag_list[j].conj().T)
    print("3")
    print(cj_cj1_list[j] - cj_dag_cj1_dag_list[j].conj().T)