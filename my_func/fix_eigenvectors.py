import numpy as np
#複素数の偏角を計算するのに必要
import cmath

def adjust_eigenvectors(eigenvactors,L):
        """Fill particle-hole partner columns from the first half of eigenvectors."""
        V = np.zeros((2*L, 2*L), dtype=complex)
        for i in range(L):
            V[:,i] = eigenvactors[:,i]
            V[:L,2*L-1-i] = np.conj(eigenvactors[L:,i])
            V[L:,2*L-1-i] = np.conj(eigenvactors[:L,i])
        return V

def arrange_eigenvectors(eigenvalues, eigenvectors, tol=1e-8):
    """Sort eigenmodes into zero, positive, remaining zero, then negative sectors."""
    """
    固有値を以下の順で並べ替える：
    1) “ゼロ” のうち最初の１つ
    2) 正の固有値（小さい順）
    3) “ゼロ” の残り
    4) 負の固有値（絶対値の小さい順）
    tol を境界として「ほぼゼロ」を定義。
    """
    # インデックス抽出
    idx_pos = np.where(eigenvalues > tol)[0]
    idx_neg = np.where(eigenvalues < -tol)[0]
    idx_zero = np.where(np.abs(eigenvalues) <= tol)[0]

    # ソート
    idx_pos_sorted = idx_pos[np.argsort(eigenvalues[idx_pos])]
    idx_neg_sorted = idx_neg[np.argsort(np.abs(eigenvalues[idx_neg]))]

    # ゼロの振り分け
    idx_combined = []
    if idx_zero.size > 0:
        # 先頭に１つ
        idx_combined.append(idx_zero[0])
    # 正側を続ける
    idx_combined.extend(idx_pos_sorted.tolist())
    if idx_zero.size > 1:
        # 残りのゼロを中ほどに
        idx_combined.extend(idx_zero[1:].tolist())
    # 負側を末尾に
    idx_combined.extend(idx_neg_sorted.tolist())

    # 並べ替え適用
    eigenvalues_sorted  = eigenvalues[idx_combined]
    eigenvectors_sorted = eigenvectors[:, idx_combined]
    return eigenvalues_sorted, eigenvectors_sorted

def fix_phase(eigenvectors,L):
    """Rotate each eigenvector so its diagonal component has zero phase."""
    for i in range(2*L):
        phase = cmath.phase(eigenvectors[i, i])
        eigenvectors[:,i] = np.exp(-1j*phase) * eigenvectors[:,i]
    return eigenvectors
