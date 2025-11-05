import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
def compute_eigenvalues():
    l = np.pi
    L = 100
    epsilon = l / L

    n = np.linspace(-(L/2),(L/2)-1,L)
    k = (2*np.pi/L)*(n)
    alpha = 1
    beta = 3
    gamma = 1
    p = 1

    zeta = np.pi/2
    m = 0

    A = (-1/(2*epsilon))*(p*np.cos(k) - beta*np.sin(k) - 0.5*(2*p - epsilon*(2*m*alpha)))
    B = (-1/(2*epsilon))*(alpha/gamma)*(np.cos(zeta) - 1j*np.sin(zeta))*1j*np.sin(k)
    C = (-1/(2*epsilon))*(-1)*(alpha/gamma)*(np.cos(zeta) + 1j*np.sin(zeta))*1j*np.sin(k)
    D = -1*(-1/(2*epsilon))*(p*np.cos(k) + beta*np.sin(k) - 0.5*(2*p - epsilon*(2*m*alpha)))

    Hk = np.array([[A, B],[C, D]])
    Hk = np.transpose(Hk, (2,0,1))

    vals = LA.eigvals(Hk).real.flatten()

    tol = 1e-12
    pos = vals[vals>tol]
    neg = vals[vals<-tol]
    zero = vals[np.abs(vals)<=tol]

    pos_sorted = pos[np.argsort(np.abs(pos))]
    neg_sorted = neg[np.argsort(np.abs(neg))]

    # zero が 1 個しか無い等の安全対策
    if len(zero)==0:
        final_vals = np.concatenate([pos_sorted, neg_sorted])
    elif len(zero)==1:
        final_vals = np.concatenate([ [zero[0]], pos_sorted, neg_sorted ])
    else:
        final_vals = np.concatenate([ [zero[0]], pos_sorted, [zero[1]], neg_sorted ])

    return final_vals