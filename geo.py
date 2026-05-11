"""Generate a null geodesic table used as an overlay in density plots."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#---------------------------------
# パラメータ
#---------------------------------
l = 2.0 * np.pi   # x 方向の長さ l = 2π

#---------------------------------
# 1. β(x) を定義
#---------------------------------
def beta(x: np.ndarray) -> np.ndarray:
    """Continuous beta profile for the geodesic ODE."""
    A = 1.0
    width = 1.0
    return A * np.tanh(width * (x - l/2.0)) + A
    # ↑ ここだけを書き換えれば好きな β(x) にできる

#---------------------------------
# 2. 左向きモードの ODE : dt/dx = -1/(1+β(x))
#   （式に合わせて + にしておくね）
#---------------------------------
def dtdx_left(x, t):
    """ODE right-hand side dt/dx for a left-moving null ray."""
    return -1.0 / (1.0 - beta(x))

def geodesic_left(x_start, x_end, t_start=0.0, n_points=200):
    """
    Integrate the left-moving null geodesic from x_start to x_end.

    左向きモードの測地線を (x_start, t_start) から x_end まで計算する。
    dt/dx = -1/(1+β(x))
    """
    sol = solve_ivp(
        fun=dtdx_left,
        t_span=(x_start, x_end),
        y0=[t_start],
        dense_output=True
    )

    xs = np.linspace(x_start, x_end, n_points)
    ts = sol.sol(xs)[0]
    return xs, ts

# 例: x = 0.75 l から l まで左向きモードの測地線
x_start = 0.75 * l
x_end   = l - 0.4
t_start = 0.0

xs, ts = geodesic_left(x_start, x_end, t_start=t_start, n_points=500)

#---------------------------------
# 3. 測地線をプロット
#---------------------------------
plt.figure()
plt.plot(xs, ts)
plt.xlabel("x")
plt.ylabel("t(x)")
plt.xlim(0, l)
plt.title("left-moving null geodesic")
plt.grid(True)
plt.tight_layout()
plt.show()

#---------------------------------
# 4. dat ファイルに書き出し
#   例に合わせて「x, t(x)」の形（カンマ区切り）で保存
#---------------------------------
data = np.column_stack([xs, ts])
np.savetxt(
    "geodesic.dat",   # 出力ファイル名
    data,
    fmt="%.10f, %.10f"     # 例と同じ小数10桁 & カンマ区切り
)
