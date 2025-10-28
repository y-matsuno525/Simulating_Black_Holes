#import
import numpy as np
from numpy import linalg as LA

#パラメータ
L = 10
l = 2*np.pi
epsilon = l / L
p = 1
m = 0
pos = "lr" #lr, ur, ll, ul

#BdGハミルトニアンの作成
H = np.zeros((2*L, 2*L), dtype=complex)

def is_hermitian(matrix):
    is_hermitian = np.allclose(matrix, np.conj(matrix.T), atol=1e-10)

    if is_hermitian:
        return True
    else:
        return False

def beta(j,L,pos,epsilon):
    #return 0
    width = 1
    A = 0.6
    if pos == "lr":
        # β = -1 を j = 71 で踏むように調整
        return -A*np.tanh(3/width*(j - int(2*L/3) - 0.730833344)*epsilon) - 0.6
    elif pos == "ur":
        # β = +1 を j = 70 で踏むように調整
        return  A*np.tanh(3/width*(j - int(2*L/3) + 0.269166656056467)*epsilon) + 0.6
        # （注）式は (j - center - c) なので c = -0.269... は “+0.269...” と等価
    elif pos == "ll":
        # β = -1 を j = 29 で踏むように調整
        return  A*np.tanh(3/width*(j - int(L/3) - 0.269166656056467)*epsilon) - 0.6
    elif pos == "ul":
        # β = +1 を j = 29 で踏むように調整
        return -A*np.tanh(3/width*(j - int(L/3) - 0.269166656056467)*epsilon) + 0.6

for i in range(2*L):
    for j in range(2*L):
        #左上
        if i < L and j < L:
            if i == j:
                H[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))*(-1)
            elif i-j == 1:
                H[i, j] = -1/(2*epsilon) * (p - 1j*beta(i,L,pos,epsilon))
            elif j-i == 1:
                H[i, j] = -1/(2*epsilon) * (p + 1j*beta(i,L,pos,epsilon))
        #右上
        elif i < L and j >= L:
            if j-i == L-1:
                H[i, j] = -1/(2*epsilon) * (-1)
            elif j-i == L+1:
                H[i, j] = -1/(2*epsilon) * (1)
        #左下
        elif i >= L and j < L:
            if i-j == L-1:
                H[i, j] = -1/(2*epsilon) * (-1)
            elif i-j == L+1:
                H[i, j] = -1/(2*epsilon) * (1)
        #右下
        else:
            if i == j:
                H[i, j] = -1/(2*epsilon) * (2*p - epsilon*(2*m))
            elif i-j == 1:
                H[i, j] = -1/(2*epsilon) * (-p - 1j*beta(i-L,L,pos,epsilon))
            elif j-i == 1:
                H[i, j] = -1/(2*epsilon) * (-p + 1j*beta(j-L,L,pos,epsilon))

#BdG行列を対角化
eigenvalues, eigenvectors = LA.eigh(H)
print("固有値")
print(eigenvalues)
print()

#粒子-反粒子対称性を満たすように調整
V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,2*L-1-i] = np.conj(eigenvectors[L:,i])
    V[L:,2*L-1-i] = np.conj(eigenvectors[:L,i])
eigenvectors = V
for j in range(L):
    assert np.allclose(eigenvectors[L:,j].conj(), eigenvectors[:L,2*L-1-j]), print(str(j)+"番目の固有ベクトルがだめ"
    "")
    assert np.allclose(eigenvectors[:L,j].conj(), eigenvectors[L:,2*L-1-j]), print(str(j)+"番目の固有ベクトルがだめ")


#演算子の作成
#cj_dag_cj
cj_dag_cj_list = []
for j in range(L):
    cj_dag_cj_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj_dag_cj_tmp[k,l] = np.conj(eigenvectors[j,(2*L-1)-k]) * eigenvectors[j,(2*L-1)-l]
            cj_dag_cj_tmp[k,l] -= np.conj(eigenvectors[j,l]) * eigenvectors[j,k]
            if k == l:
                for n in range(L):
                    cj_dag_cj_tmp[k,l] += np.conj(eigenvectors[j,n]) * eigenvectors[j,n]
    assert is_hermitian(cj_dag_cj_tmp), print("cj†cjがエルミートじゃない")
    cj_dag_cj_list.append(cj_dag_cj_tmp)
    print("cj†cj作成中:" + str(int(j/L*100))+"%")

# #cj_cj_dag
# cj_cj_dag_list = []
# for j in range(L):
#     cj_cj_dag_tmp = np.zeros((L, L), dtype=complex)
#     for k in range(L):
#         for l in range(L):
#             cj_cj_dag_tmp[k,l] = eigenvectors[j,k] * eigenvectors[j,l].conj()
#             cj_cj_dag_tmp[k,l] -= eigenvectors[j,(2*L-1)-l] * eigenvectors[j,(2*L-1)-k].conj()
#             if k == l:
#                 for n in range(L):
#                     cj_cj_dag_tmp[k,l] += eigenvectors[j,(2*L-1)-n] * eigenvectors[j,(2*L-1)-n].conj()
#     assert is_hermitian(cj_cj_dag_tmp), print("cjcj†がエルミートじゃない")
#     cj_cj_dag_list.append(cj_cj_dag_tmp)
#     print("cjcj†作成中:" + str(int(j/L*100))+"%")

#cj1_cj
cj1_cj_list = []
for j in range(L-1):
    cj1_cj_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj1_cj_tmp[k,l] = eigenvectors[j+1,k] * eigenvectors[j,(2*L-1)-l]
            cj1_cj_tmp[k,l] -= eigenvectors[j+1,(2*L-1)-l] * eigenvectors[j,k]
            if k == l:
                for n in range(L):
                    cj1_cj_tmp[k,l] += eigenvectors[j+1,(2*L-1)-n] * eigenvectors[j,n]
    cj1_cj_list.append(cj1_cj_tmp)
    print("cj1_cj作成中:" + str(int(j/L*100))+"%")

#j = L-1 の場合は0にする
cj1_cj_list.append(np.zeros((L,L), dtype=complex))

# #cj1_dag_cj_dag
# cj1_dag_cj_dag_list = []
# for j in range(L-1):
#     cj1_dag_cj_dag_tmp = np.zeros((L, L), dtype=complex)
#     for k in range(L):
#         for l in range(L):
#             cj1_dag_cj_dag_tmp[k,l] = eigenvectors[j+1,(2*L-1)-k].conj() * eigenvectors[j,l].conj()
#             cj1_dag_cj_dag_tmp[k,l] -= eigenvectors[j+1,l].conj() * eigenvectors[j,(2*L-1)-k].conj()
#             if k == l:
#                 for n in range(L):
#                     cj1_dag_cj_dag_tmp[k,l] += eigenvectors[j+1,n].conj() * eigenvectors[j,(2*L-1)-n].conj()
#     cj1_dag_cj_dag_list.append(cj1_dag_cj_dag_tmp)
#     print("cj1_cj作成中:" + str(int(j/L*100))+"%")

# #j = L-1 の場合は0にする
# cj1_dag_cj_dag_list.append(np.zeros((L,L), dtype=complex))

# #cj_cj1
# cj_cj1_list = []
# for j in range(L-1):
#     cj_cj1_tmp = np.zeros((L, L), dtype=complex)
#     for k in range(L):
#         for l in range(L):
#             cj_cj1_tmp[k,l] = eigenvectors[j,k] * eigenvectors[j+1,(2*L-1)-l]
#             cj_cj1_tmp[k,l] -= eigenvectors[j,(2*L-1)-l] * eigenvectors[j+1,k]
#             if k == l:
#                 for n in range(L):
#                     cj_cj1_tmp[k,l] += eigenvectors[j,(2*L-1)-n] * eigenvectors[j+1,n]
#     cj_cj1_list.append(cj_cj1_tmp)
#     print("cj_cj1作成中:" + str(int(j/L*100))+"%")

# #j = L-1 の場合は0にする
# cj_cj1_list.append(np.zeros((L,L), dtype=complex))

#cj1_dag_cj
cj1_dag_cj_list = []
for j in range(L-1):
    cj1_dag_cj_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj1_dag_cj_tmp[k,l] = eigenvectors[j+1,(2*L-1)-k].conj() * eigenvectors[j,(2*L-1)-l]
            cj1_dag_cj_tmp[k,l] -= eigenvectors[j+1,l].conj() * eigenvectors[j,k]
            if k == l:
                for n in range(L):
                    cj1_dag_cj_tmp[k,l] += np.conj(eigenvectors[j+1,n]) * eigenvectors[j,n]
    cj1_dag_cj_list.append(cj1_dag_cj_tmp)
    print("cj1†_cj作成中:" + str(int(j/L*100))+"%")

#j = L-1 の場合は0にする
cj1_dag_cj_list.append(np.zeros((L,L), dtype=complex))

# #cj_cj1_dag
# cj_cj1_dag_list = []
# for j in range(L-1):
#     cj_cj1_dag_tmp = np.zeros((L, L), dtype=complex)
#     for k in range(L):
#         for l in range(L):
#             cj_cj1_dag_tmp[k,l] = eigenvectors[j,k] * eigenvectors[j+1,l].conj()
#             cj_cj1_dag_tmp[k,l] -= eigenvectors[j,(2*L-1)-l] * eigenvectors[j+1,(2*L-1)-k].conj()
#             if k == l:
#                 for n in range(L):
#                     cj_cj1_dag_tmp[k,l] += eigenvectors[j,(2*L-1)-n] * eigenvectors[j+1,(2*L-1)-n].conj()
#     cj_cj1_dag_list.append(cj_cj1_dag_tmp)
#     print("cj_cj1†作成中:" + str(int(j/L*100))+"%")

# #j = L-1 の場合は0にする
# cj_cj1_dag_list.append(np.zeros((L,L), dtype=complex))

#cj_cj1_dag
cj1_cj_dag_list = []
for j in range(L-1):
    cj1_cj_dag_tmp = np.zeros((L, L), dtype=complex)
    for k in range(L):
        for l in range(L):
            cj1_cj_dag_tmp[k,l] = eigenvectors[j+1,k] * eigenvectors[j,l].conj()
            cj1_cj_dag_tmp[k,l] -= eigenvectors[j+1,(2*L-1)-l] * eigenvectors[j,(2*L-1)-k].conj()
            if k == l:
                for n in range(L):
                    cj1_cj_dag_tmp[k,l] += eigenvectors[j+1,(2*L-1)-n] * eigenvectors[j,(2*L-1)-n].conj()
    cj1_cj_dag_list.append(cj1_cj_dag_tmp)
    print("cj_cj1†作成中:" + str(int(j/L*100))+"%")

#j = L-1 の場合は0にする
cj1_cj_dag_list.append(np.zeros((L,L), dtype=complex))

for j in range(L):
        print(str(j) + "番目")
        print(cj1_dag_cj_list[j] + cj1_cj_dag_list[j].T.conj())
# for j in range(L):
#     is_all_close = np.allclose(cj_dag_cj_list[j],-1*cj_cj_dag_list[j].conj().T)
#     if not is_all_close:
#         print(str(j) + "番目が違う")