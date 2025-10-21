import numpy as np

class Parameter:
    L = 100
    l = 2*np.pi
    epsilon = l / L
    m = 0

    def p(j):
        return 0

    def zeta(j):
        return 0

    def alpha(j):
        return 1

    def gamma(j):
        return 1

    def beta(j):
        return np.tanh(j)

    def diff_func(func, j):
        return func(j+1) - func(j-1) / 2