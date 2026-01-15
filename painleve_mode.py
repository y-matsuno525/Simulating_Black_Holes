#import
import numpy as np
from numpy import linalg as LA #BdGハミルトニアンの作成で利用
import scipy.linalg #時間発展演算子の作成で利用
import matplotlib.pyplot as plt
#標準偏差の計算に使う
import statistics
import math

#パラメータ
L = 100
l = 2*np.pi
epsilon = l / L
p = 1
m = 0.0001
pos = "lr" #lr, ur, ll, ul
t_i = 0
width = 1
A = 1
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
    return  np.tanh(3*(j - int(L/2) )*epsilon) + 1
    #width = 0.1
    # A = 1
    # jh = int(L/2)
    # return  A*np.tanh(width*(j - jh)*epsilon) + A# - (A-1)
    # #return 0
    width = 1
    A = 0.6
    jh = int(L/3)
    c1=0#.730833344
    if pos == "lr":
        # β = -1 を j = 71 で踏むように調整
        return -A*np.tanh(3/width*(j - 1*jh - c1)*epsilon) - A
    elif pos == "ur":
        # β = +1 を j = 70 で踏むように調整
        return  A*np.tanh(3/width*(j - 2*jh - c1)*epsilon) + A
        # （注）式は (j - center - c) なので c = -0.269... は “+0.269...” と等価
    elif pos == "ll":
        # β = -1 を j = 29 で踏むように調整
        return  A*np.tanh(3/width*(j - jh + c1)*epsilon) - A
    elif pos == "ul":
        # β = +1 を j = 29 で踏むように調整
        return -A*np.tanh(3/width*(j - jh + c1)*epsilon) + A

#print("surface gravity:", (beta(int(2*),L,pos,epsilon) - beta(int(2*L/3)+ 0.730833344-1,L,pos,epsilon)) / (2*epsilon) * 1/2)
# import sys
# sys.exit()
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

#固有値、固有ベクトルのソート(確認済み)
eigenvalues = np.concatenate((eigenvalues[L:], eigenvalues[:L][::-1]), 0)
eigenvectors = np.concatenate((eigenvectors[:,L:], eigenvectors[:,:L][:,::-1]), 1)

V = np.zeros((2*L, 2*L), dtype=complex)
for i in range(L):
    V[:,i] = eigenvectors[:,i]
    V[:L,i+L] = np.conj(eigenvectors[L:,i])
    V[L:,i+L] = np.conj(eigenvectors[:L,i])
eigenvectors = V

def generate_U(eigenvectors):
    U = np.zeros((2*L, 2*L), dtype=complex)
    for i in range(L):
        U[i,:] = (eigenvectors[i,:] + eigenvectors[i+L,:])/np.sqrt(2)
        U[i+L,:] = (eigenvectors[i,:] - eigenvectors[i+L,:])/(1j*np.sqrt(2))
    return U
def plot_mode_function(U):

    f = U[:L,L:]
    x = np.linspace(0, l, L)
    plt.figure()
    for i in range(L):
        if i < 5:
            plt.plot(x, f[:, i]/np.sqrt(epsilon))
            plt.xlabel(r'$x_j$',fontsize=25)
            index = "j"+str(i+1)
            plt.xticks([0,0.5*l,l],["0","$\pi$","2$\pi$"],fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylabel(rf"$f_{{j{i+1}}}$", rotation=0, fontsize=25, labelpad=15)
            plt.axvline(x=0.5*l, color="red", linewidth=1, linestyle="--")
            plt.grid(True)
            plt.savefig('figure/mode_function_' + 'p='+str(p) + '_k=' + str(i+1) + '.png',dpi=300,bbox_inches='tight',transparent=False)
            plt.close()
    return
U = generate_U(eigenvectors)
plot_mode_function(U)
