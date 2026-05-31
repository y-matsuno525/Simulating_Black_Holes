"""make_all.py  —  論文 paper/figure/ の全図を一括再生成する。

実行: ``python paper_figures/make_all.py``
依存データ: paper_data/（gitignore、ディスク上）。各スクリプトは欠損時にそのパネルをスキップ。
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = [
    "make_dispersion",
    "make_doubler_fft",
    "make_bh_panels",
    "make_wh_panels",
    "make_surface_gravity",
]


def main():
    ok, ng = [], []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            mod.main()
            ok.append(name)
        except Exception:
            print(f"[make_all] ERROR in {name}:")
            traceback.print_exc()
            ng.append(name)
    print(f"\n[make_all] 成功: {ok}")
    if ng:
        print(f"[make_all] 失敗: {ng}")
        sys.exit(1)


if __name__ == "__main__":
    main()
