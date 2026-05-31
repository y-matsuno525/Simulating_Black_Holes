"""analysis/dispersion_relation.py

均一 beta（定数）の BdG ハミルトニアンを周期境界で構成・対角化し、局所分散関係 E(k) を返す。
行列要素は simulation.py:build_bdg_matrix と同一規約（定数 beta・PBC）。

用途:
  - 論文 Fig. dispersion（低運動量の群速度 = 有効光円錐の傾き beta±1、k=pi 近傍のダブラー分岐）
  - 論文 Fig. doubler_fft の (a) beta=1.2 でのダブラー分岐

由来: std ブランチ painleve_k.py / master_thesis のマヨラナ分散.ipynb。
"""

from __future__ import annotations

import numpy as np
from numpy import linalg as LA


def build_homogeneous_bdg(beta, p, m, L, epsilon, PBC=True):
    """定数 beta の 2L x 2L BdG 行列（build_bdg_matrix と同規約）。"""
    H = np.zeros((2 * L, 2 * L), dtype=complex)
    pref = -1.0 / (2 * epsilon)
    for i in range(2 * L):
        for j in range(2 * L):
            if i < L and j < L:                       # 粒子ブロック
                if i == j:
                    H[i, j] = pref * (2 * p - epsilon * 2 * m) * (-1)
                elif i - j == 1:
                    H[i, j] = pref * (p - 1j * beta)
                elif j - i == 1:
                    H[i, j] = pref * (p + 1j * beta)
            elif i < L and j >= L:                    # pairing 右上
                if j - i == L - 1:
                    H[i, j] = pref * (-1)
                elif j - i == L + 1:
                    H[i, j] = pref * (1)
            elif i >= L and j < L:                    # pairing 左下
                if i - j == L - 1:
                    H[i, j] = pref * (-1)
                elif i - j == L + 1:
                    H[i, j] = pref * (1)
            else:                                     # 正孔ブロック
                if i == j:
                    H[i, j] = pref * (2 * p - epsilon * 2 * m)
                elif i - j == 1:
                    H[i, j] = pref * (-p - 1j * beta)
                elif j - i == 1:
                    H[i, j] = pref * (-p + 1j * beta)
    if PBC:
        H[0, L - 1] = pref * (p - 1j * beta)
        H[2 * L - 1, L] = pref * (p - 1j * beta) * (-1)
        H[L - 1, 0] = pref * (p + 1j * beta)
        H[L, 2 * L - 1] = pref * (p + 1j * beta) * (-1)
        H[0, 2 * L - 1] = pref * (-1)
        H[L - 1, L] = pref * (1)
        H[2 * L - 1, 0] = pref * (-1)
        H[L, L - 1] = pref * (1)
    return H


def dispersion_bands(beta, p, nk=801):
    """均一 beta の 2 バンド分散を解析的に返す: ``(k, epsE_plus, epsE_minus)``。

    2x2 Bloch BdG を build_homogeneous_bdg と同規約でフーリエ変換して得られる厳密式:

        eps * E_pm(k) = beta sin k  ±  sqrt( p^2 (1-cos k)^2 + sin^2 k ).

    返すのは eps*E（無次元化済み）。k=0 近傍の傾きは beta±1（有効光円錐）、
    k=pi では p=1 で ±2|p|（ギャップ）、p=0 で 0（ダブラーのゼロモード）。
    質量 m=0 を仮定（論文 Sec. III）。
    """
    k = np.linspace(-np.pi, np.pi, nk)
    root = np.sqrt(p ** 2 * (1 - np.cos(k)) ** 2 + np.sin(k) ** 2)
    base = beta * np.sin(k)
    return k, base + root, base - root


def dispersion(beta, p, m=0.0, L=200, ell=2 * np.pi):
    """数値版（検証用）: 定数 beta の BdG を対角化し ``(k, eps*E)`` を返す。

    各固有ベクトルの粒子成分を FFT して主運動量 k を割り当てる。
    解析式 dispersion_bands の妥当性確認に用いる。
    """
    epsilon = ell / L
    H = build_homogeneous_bdg(beta, p, m, L, epsilon, PBC=True)
    evals, evecs = LA.eigh(H)
    ks = np.fft.fftfreq(L, d=1.0 / L) * (2 * np.pi / L)
    out_k, out_E = [], []
    for n in range(2 * L):
        spec = np.abs(np.fft.fft(evecs[:L, n]))
        out_k.append(ks[int(np.argmax(spec))])
        out_E.append(evals[n].real * epsilon)   # eps*E に揃える
    order = np.argsort(out_k)
    return np.array(out_k)[order], np.array(out_E)[order]


def light_cone_slopes(beta):
    """有効光円錐の特性速度 (beta+1, beta-1)。"""
    return beta + 1.0, beta - 1.0


if __name__ == "__main__":
    for b in [0.0, -0.6, -1.2, 1.2]:
        for p in [1, 0]:
            k, ep, em = dispersion_bands(b, p)
            i0 = len(k) // 2
            # k=0 近傍の傾き（中心差分）
            slope_p = (ep[i0 + 5] - ep[i0]) / (k[i0 + 5] - k[i0])
            print(f"beta={b:+.1f} p={p}: upper-band slope@k=0 ~ {slope_p:+.3f} (expect beta+1={b+1:+.1f})")
