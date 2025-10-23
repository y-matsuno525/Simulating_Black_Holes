import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from .my_check import is_hermitian
from .beta import beta

def generate_c_dag_c(eigenvectors, L):
    c_dag_c_list = []
    for j in range(L):
        c_dag_c_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                c_dag_c_tmp[k,l] = np.conj(eigenvectors[j,(2*L-1)-k]) * eigenvectors[j,(2*L-1)-l]
                c_dag_c_tmp[k,l] -= np.conj(eigenvectors[j,l]) * eigenvectors[j,k]
                if k == l:
                    for n in range(L):
                        c_dag_c_tmp[k,l] += np.conj(eigenvectors[j,n]) * eigenvectors[j,n]
        c_dag_c_list.append(c_dag_c_tmp)
        print("cj†cj作成中:" + str(int(i/L*100))+"%")
    return c_dag_c_list

def generate_time_evolution_operator(eigenvalues, dt, L):
    H = np.zeros((L, L), dtype=complex)
    for i in range(L):
        H[i,i] = eigenvalues[i]
    return scipy.linalg.expm(-1j*H*dt)

def H_vacuum(eigenvectors,L, epsilon,p_val,pos):

    Hp_v = []
    Hm_v = []
    Hpm_v = []
    for j in range(L):
        Hp_v_j = 0
        Hm_v_j = 0
        Hpm_v_j = 0
        for n in range(L):
            Hp_v_j += -1/(8*epsilon) * (1j*(1+beta(j,L,pos,epsilon))*(-2*1j*eigenvectors[j,n+L].conj() * eigenvectors[j+1,n].conj() \
                                                        + 2*eigenvectors[j,n+L].conj()*eigenvectors[j+1,n+L] \
                                                        + 2*(-1*eigenvectors[j,n+L]*eigenvectors[j+1,n+L].conj())\
                                                        +2*1j*(-1*eigenvectors[j,n+L]*eigenvectors[j+1,n])))
            Hm_v_j += -1/(8*epsilon) * (1j*(-1+beta(j,L,pos,epsilon))*(2*1j*eigenvectors[j,n+L].conj() * eigenvectors[j+1,n].conj() \
                                                        + 2*eigenvectors[j,n+L].conj()*eigenvectors[j+1,n+L] \
                                                        + 2*(-1*eigenvectors[j,n+L]*eigenvectors[j+1,n+L].conj())\
                                                        -2*1j*(-1*eigenvectors[j,n+L]*eigenvectors[j+1,n])))
            Hpm_v_j += -1*p_val/(8*epsilon) * ((4*eigenvectors[j,n+L].conj()*eigenvectors[j+1,n+L] \
                                                -4*(-1*eigenvectors[j,n+L]*eigenvectors[j+1,n+L].conj()))\
                                        +2*1j*(-2*(-1*eigenvectors[j,n].conj()*eigenvectors[j,n+L].conj()) \
                                                -2*eigenvectors[j,n]*eigenvectors[j,n+L] \
                                                +4*1j*eigenvectors[j,n+L].conj()*eigenvectors[j,n+L]))

        #print(Hp_v_j)
        #print(Hm_v_j)
        #print(Hpm_v_j)

        #assert abs(Hp_v_j - Hp_v_j.conj()) < 10**-5, "Hp_v_j(j=" + str(j) + ") is not Hermitian!"
        #assert abs(Hm_v_j - Hm_v_j.conj()) < 10**-5, "Hm_v_j(j=" + str(j) + ") is not Hermitian!"
        #assert abs(Hpm_v_j - Hpm_v_j.conj()) < 10**-5, "Hpm_v_j(j=" + str(j) + ") is not Hermitian!"

        Hp_v.append(Hp_v_j)
        Hm_v.append(Hm_v_j)
        Hpm_v.append(Hpm_v_j)
    return Hp_v, Hm_v, Hpm_v

def generate_cj1_cj(eigenvectors, L):

    cj1_cj_list = []
    for j in range(L-1):
        cj1_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj1_cj_tmp[k,l] = eigenvectors[j+1,k+L] * eigenvectors[j,l]
                cj1_cj_tmp[k,l] += -1*eigenvectors[j+1,l] * eigenvectors[j,k+L]
                if k == l:
                    for n in range(L):
                        cj1_cj_tmp[k,l] += eigenvectors[j+1,n] * eigenvectors[j,n+L]
        cj1_cj_list.append(cj1_cj_tmp)
        print("cj1_cj作成中:" + str(int(j/L*100))+"%")

    #j = L-1 の場合は0にする
    cj1_cj_list.append(np.zeros((L,L), dtype=complex))

    return cj1_cj_list

def generate_cj1_dag_cj(eigenvectors, L):

    cj1_dag_cj_list = []
    for j in range(L-1):
        cj1_dag_cj_tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                cj1_dag_cj_tmp[k,l] = eigenvectors[j+1,k].conj() * eigenvectors[j,l]
                cj1_dag_cj_tmp[k,l] += -1*eigenvectors[j+1,l+L].conj() * eigenvectors[j,k+L]
                if k == l:
                    for n in range(L):
                        cj1_dag_cj_tmp[k,l] += np.conj(eigenvectors[j+1,n+L]) * eigenvectors[j,n+L]
        cj1_dag_cj_list.append(cj1_dag_cj_tmp)
        print("cj1†_cj作成中:" + str(int(j/L*100))+"%")

    #j = L-1 の場合は0にする
    cj1_dag_cj_list.append(np.zeros((L,L), dtype=complex))

    return cj1_dag_cj_list

def H_p_m_K(cj1_cj_list, cj1_dag_cj_list, L, epsilon, pos):

    H_p = []
    H_m = []

    for j in range(L):
        H_p_j = np.zeros((L, L), dtype=complex)
        H_m_j = np.zeros((L, L), dtype=complex)
    
        #cj+1_cjからcj_cj+1を作る
        cj_cj1_list = []
        for cj1_cj in cj1_cj_list:
            cj_cj1_list.append(-1*cj1_cj)
        
        #cj+1†_cjからcj_cj+1†を作る
        cj_cj1_dag_list = []
        for cj1_dag_cj in cj1_dag_cj_list:
            cj_cj1_dag_list.append(-1*cj1_dag_cj)

        #cj+1_cjからcj†_cj+1†を作る
        cj_dag_cj1_dag_list = []
        for cj1_cj in cj1_cj_list:
            cj_dag_cj1_dag_list.append(cj1_cj.T.conj())

        #cj+1†_cjからcj†_cj+1を作る
        cj_dag_cj1_list = []
        for cj1_dag_cj in cj1_dag_cj_list:
            cj_dag_cj1_list.append(cj1_dag_cj.T.conj())

        #ハミルトニアン密度作成
        H_p_j = -1j/(4*epsilon) * (1+beta(j,L,pos,epsilon))  *(1j*cj_cj1_list[j] + cj_cj1_dag_list[j] + cj_dag_cj1_list[j] - 1j*cj_dag_cj1_dag_list[j])
        H_m_j = -1j/(4*epsilon) * (-1+beta(j,L,pos,epsilon))  *(-1j*cj_cj1_list[j] + cj_cj1_dag_list[j] + cj_dag_cj1_list[j] + 1j*cj_dag_cj1_dag_list[j])

        #エルミートか確認
        isHermitian = is_hermitian(H_p_j)
        assert isHermitian, "H_p(j=" + str(j) + ") is not Hermitian!"
        isHermitian = is_hermitian(H_m_j)
        assert isHermitian, "H_m(j=" + str(j) + ") is not Hermitian!"

        H_p.append(H_p_j)
        H_m.append(H_m_j)

    return H_p, H_m

def H_vacuum_k(eigenvectors,L, epsilon,p_val,pos):

    Hp_v = []
    Hm_v = []
    for j in range(L):
        Hp_v_j = 0
        Hm_v_j = 0
        G1 = 0
        G2 = 0
        for n in range(L):
            G1 += eigenvectors[j+1, n] * eigenvectors[j, n+L]
            G2 += eigenvectors[j+1, n+L].conj() * eigenvectors[j, n+L]
        Hp_v_j = -1/(2*epsilon) * 1j * (1+beta(j,L,pos,epsilon)) * 1/2 * (1j * (-1*G1) + (-1*G2) + (G2.conj()) - 1j * (G1.conj()))
        Hm_v_j = -1/(2*epsilon) * 1j * (-1+beta(j,L,pos,epsilon)) * 1/2 * (-1j * (-1*G1) + (-1*G2) + (G2.conj()) + 1j * (G1.conj()))
        assert abs(Hp_v_j - Hp_v_j.conj()) < 10**-5, "Hp_v_j(j=" + str(j) + ") is not Hermitian!"
        assert abs(Hm_v_j - Hm_v_j.conj()) < 10**-5, "Hm_v_j(j=" + str(j) + ") is not Hermitian!"
        Hp_v.append(Hp_v_j)
        Hm_v.append(Hm_v_j)
    return Hp_v, Hm_v

#テスト用
def generate_c_c_dag(eigenvectors, L):
    c_c_dag_list = []
    for j in range(L):
        tmp = np.zeros((L, L), dtype=complex)
        for k in range(L):
            for l in range(L):
                tmp[k,l] = np.conj(eigenvectors[j,k+L]) * eigenvectors[j,l+L]
                tmp[k,l] += -1 * eigenvectors[j,l] * eigenvectors[j,k].conj()
                if k == l:
                    for n in range(L):
                        tmp[k,l] += eigenvectors[j,n] * eigenvectors[j,n].conj()
        c_c_dag_list.append(tmp)
        print("cjcj†作成中:" + str(int(j/L*100))+"%")
    return c_c_dag_list