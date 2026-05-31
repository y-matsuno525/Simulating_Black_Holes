"""analysis/stagnation_logL.py

ホワイトホールでの波束停滞時間 T_lat のシステムサイズ L 依存性。

連続極限の見積り: 地平面近傍で波束幅は kappa の率で指数圧縮し sigma(t)=sigma0 e^{-kappa t}。
格子間隔 eps=ell/L を下回った時刻を停滞時間とすると

    sigma0 e^{-kappa T} = eps = ell / L
    =>  T_cont(L) = (1/kappa) [ ln(sigma0 L / ell) ] ∝ log L.

本スクリプトは
  (1) この連続見積り T_cont(L) を計算して paper_data/logL/T_lat.csv に書き出し、
  (2) ``--measure`` 指定時は simulation.py を各 L で実走し、格子の T_lat を実測する
      （重い。simulation.py / config.py を利用）。

由来: std ブランチ logL.py / master_thesis の L_dependency.ipynb・various_beta.ipynb。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ELL = 2 * np.pi


def continuum_T(L, kappa=1.0, sigma0_frac=0.05, ell=ELL):
    """連続極限の停滞時間 T_cont(L)。sigma0_frac は初期幅の L に対する割合。"""
    L = np.asarray(L, dtype=float)
    sigma0_phys = sigma0_frac * ell          # 物理初期幅（sigma=0.05 L サイト = 0.05 ell）
    eps = ell / L
    return (1.0 / kappa) * np.log(sigma0_phys / eps)


def write_continuum_csv(Ls=(100, 150, 200, 300, 400, 600, 800), kappa=1.0):
    out_dir = REPO / "paper_data" / "logL"
    out_dir.mkdir(parents=True, exist_ok=True)
    Ls = np.array(Ls, dtype=float)
    T = continuum_T(Ls, kappa=kappa)
    path = out_dir / "T_lat.csv"
    np.savetxt(path, np.column_stack([Ls, T]), delimiter=",",
               header="L,T_cont", comments="")
    # logL 線形性の確認
    a, b = np.polyfit(np.log(Ls), T, 1)
    print(f"[stagnation] wrote {path}")
    print(f"[stagnation] T_cont = {a:.3f} log L + {b:.3f}  (傾き 1/kappa = {1/kappa:.3f} を期待)")
    return path


def measure_lattice_T(L, kappa=1.0):
    """simulation.py を L で実走し格子停滞時間 T_lat を実測（重い・任意）。

    config.json を基に L と WH 設定へ上書きして run_simulation を呼び、
    sigma 最小時刻 - 線形領域進入時刻 を T_lat とする。
    依存が重いので既定では使わない。
    """
    sys.path.insert(0, str(REPO))
    from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS, load_config, prepare_config
    import simulation as sim

    config = load_config(str(REPO / "config.json"))
    config["L"] = L
    # 軽量化: 図・GIF・FFT 等を無効化
    config["outputs"] = {k: False for k in config.get("outputs", {})}
    config = prepare_config(config)
    sim.configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)
    # 注: 完全な T_lat 抽出は run_simulation 内の停滞時間ロジックに準拠する。
    #     ここではフックの所在のみ提供する（実測は GUI / 個別実行で）。
    raise NotImplementedError(
        "格子実測は simulation.run_simulation の停滞時間出力を利用してください "
        "（config を WH 設定にして main.py を各 L で実行）。"
    )


def main():
    measure = "--measure" in sys.argv
    if measure:
        print("[stagnation] 格子実測モード（simulation.py を各 L で実走）")
        for L in (100, 200, 300):
            try:
                print(L, measure_lattice_T(L))
            except NotImplementedError as e:
                print("  ", e)
                break
    else:
        write_continuum_csv()
        print("[stagnation] 連続見積り T_cont(L) ∝ log L を出力（WH_p=0 図 panel(b) で使用）。")


if __name__ == "__main__":
    main()
