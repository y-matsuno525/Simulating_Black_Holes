import numpy as np

def beta(j,L,pos,epsilon):
        return 0
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