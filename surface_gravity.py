import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

l = 2*np.pi
L = 250
epsilon = l / L
dt = 0.01*(300/L)

# データ読み込み
data = np.loadtxt("H_m_sigmas.txt")

times = data[:,0]
H_m_sigmas = data[:,1] * epsilon

# N = len(times)
# cut = int(N * 1/2)
# times = times[:cut]
# H_m_sigmas = H_m_sigmas[:cut]

# ===============================
#  指数関数フィット: A * exp(B t)
# ===============================
def exp_func(t, A, B):
    return A * np.exp(B * t) - A

# 初期推定値（ある程度適当でOK）
A0 = H_m_sigmas[0]
B0 = 1.0

params, cov = curve_fit(exp_func, times, H_m_sigmas, p0=[A0, B0])
A_fit, B_fit = params

print("Fitted function:  H_m_sigma(t) = A * exp(B t)")
print("A =", A_fit)
print("B =", B_fit)

# フィット曲線
t_fine = np.linspace(times.min(), times.max(), 2000)
fit_curve = exp_func(t_fine, A_fit, B_fit)

# ===============================
#  プロット
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(times, H_m_sigmas, "b.", label="data")
plt.plot(t_fine, fit_curve, "r-", label=f"fit: {A_fit:.3e} exp({B_fit:.3e} t)")
ideal = A_fit*13 * np.exp(0.165 * times)
#plt.plot(times, ideal, "g--", label="ideal: 0.1 * exp(0.165 t)")
plt.xlabel("time")
plt.ylabel("H_m sigma")
plt.grid()
plt.legend()
plt.show()
