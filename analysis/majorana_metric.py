"""analysis/majorana_metric.py

格子模型が模擬する有効計量（Painleve--Gullstrand 型）の幾何量を計算する。

    ds^2 = -(1 - beta(x)^2) dt^2 - 2 beta(x) dt dx + dx^2.

- Christoffel 記号・Riemann・Ricci・Ricci スカラーを **numpy の数値微分** で計算（依存追加なしで動く）。
- 零測地線の特性速度 dx/dt = beta(x) ± 1（= 格子分散の k=0 群速度、有効光円錐）。
- OGRePy が入っていれば、シンボリックな計量定義例も表示（任意）。

由来: master_thesis/マヨラナ分散.ipynb（OGRePy で計量・Christoffel・Riemann を計算していた部分）。
本ファイルはそれを依存無しの数値版として再構成し、格子分散 (dispersion_relation.py) と接続する。
"""

from __future__ import annotations

import numpy as np

# 正準 BH beta プロファイルは dispersion_relation に一本化（重複定義の排除）。
from dispersion_relation import ELL_DEFAULT as ELL
from dispersion_relation import beta_profile_x as beta_profile


def metric(x, beta_func=beta_profile):
    """座標 (t,x) の計量 g_{mu nu}（2x2）。index 0=t, 1=x。"""
    b = beta_func(x)
    return np.array([[-(1 - b ** 2), -b],
                     [-b, 1.0]])


def inverse_metric(x, beta_func=beta_profile):
    g = metric(x, beta_func)
    return np.linalg.inv(g)


def christoffel(x, beta_func=beta_profile, h=1e-5):
    """Gamma^a_{bc}（数値微分）。計量は x のみに依存（t 非依存・定常）。"""
    # g の偏微分: d/dt = 0, d/dx を中心差分で
    def g(xx):
        return metric(xx, beta_func)

    dg = np.zeros((2, 2, 2))           # dg[mu,nu, c]  (c: 微分方向 0=t,1=x)
    dg[:, :, 1] = (g(x + h) - g(x - h)) / (2 * h)   # d/dx
    ginv = inverse_metric(x, beta_func)
    Gamma = np.zeros((2, 2, 2))        # Gamma[a,b,c]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                s = 0.0
                for d in range(2):
                    s += ginv[a, d] * (dg[d, b, c] + dg[d, c, b] - dg[b, c, d])
                Gamma[a, b, c] = 0.5 * s
    return Gamma


def ricci_scalar(x, beta_func=beta_profile, h=1e-4):
    """Ricci スカラー R（数値）。Gamma の x 微分から Riemann→Ricci を組む。"""
    def G(xx):
        return christoffel(xx, beta_func, h=1e-5)

    dG = (G(x + h) - G(x - h)) / (2 * h)    # dGamma/dx  -> 成分 c=1 のみ非ゼロ
    Gx = G(x)
    # Riemann R^a_{b c d} = d_c Gamma^a_{b d} - d_d Gamma^a_{b c} + Gamma^a_{c e}Gamma^e_{b d} - Gamma^a_{d e}Gamma^e_{b c}
    dGamma = np.zeros((2, 2, 2, 2))         # dGamma[a,b,d, c] = d_c Gamma^a_{b d}
    dGamma[:, :, :, 1] = dG                  # x 微分のみ
    R = np.zeros((2, 2, 2, 2))               # R[a,b,c,d]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    term = dGamma[a, b, d, c] - dGamma[a, b, c, d]
                    for e in range(2):
                        term += Gx[a, c, e] * Gx[e, b, d] - Gx[a, d, e] * Gx[e, b, c]
                    R[a, b, c, d] = term
    # Ricci R_{bd} = R^a_{b a d}
    Ric = np.einsum("abad->bd", R)
    ginv = inverse_metric(x, beta_func)
    return float(np.einsum("bd,bd->", ginv, Ric))


def null_speeds(x, beta_func=beta_profile):
    """零測地線の特性速度 (beta+1, beta-1)（有効光円錐 = 格子分散 k=0 群速度）。"""
    b = beta_func(x)
    return b + 1.0, b - 1.0


def main():
    xs = np.linspace(0.1, ELL - 0.1, 7)
    print("PG 有効計量 ds^2 = -(1-beta^2)dt^2 - 2 beta dt dx + dx^2")
    print(f"{'x':>6} {'beta':>8} {'v+ =b+1':>9} {'v- =b-1':>9} {'R(Ricci)':>12}")
    for x in xs:
        b = beta_profile(x)
        vp, vm = null_speeds(x)
        R = ricci_scalar(x)
        print(f"{x:6.3f} {b:8.3f} {vp:9.3f} {vm:9.3f} {R:12.4e}")
    # horizon: beta=1
    print("\nhorizon (beta=1) で v- = 0 -> 左進行成分が停滞（ブラックホール地平面）。")

    # OGRePy があればシンボリック定義例（任意）
    try:
        import OGRePy  # noqa: F401
        print("\n[OGRePy 検出] シンボリック計算も利用可能（majorana_metric.symbolic_demo 参照）。")
    except Exception:
        print("\n[OGRePy 未導入] 数値版で計算しました（pip install OGRePy で symbolic 版が使えます）。")


if __name__ == "__main__":
    main()
