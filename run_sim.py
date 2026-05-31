"""run_sim.py — config.json を基に上書きパラメータでシミュレーションを実行する。

GUI (gui.py) や CLI から、config.json を破壊せずにパラメータを差し替えて回すための薄い入口。

使い方:
    python run_sim.py                      # config.json のまま実行（main.py と同じ）
    python run_sim.py overrides.json       # config.json に overrides.json を上書きして実行

overrides.json は config.json の一部キーだけを持つ JSON（例: {"L":200,"p":0,"t_f":5}）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import (ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS, DEFAULT_CONFIG,
                    deep_merge, prepare_config)
from simulation import configure, run_simulation

REPO = Path(__file__).resolve().parent


def build_config(overrides_path=None):
    base = {}
    cfg_json = REPO / "config.json"
    if cfg_json.exists():
        base = json.loads(cfg_json.read_text(encoding="utf-8"))
    merged = deep_merge(DEFAULT_CONFIG, base)
    if overrides_path:
        ov = json.loads(Path(overrides_path).read_text(encoding="utf-8"))
        merged = deep_merge(merged, ov)
    return prepare_config(merged)


def main():
    overrides = sys.argv[1] if len(sys.argv) > 1 else None
    config = build_config(overrides)
    print(f"[run_sim] L={config['L']} p={config['p']} scenario={config.get('scenario')} "
          f"t_f={config['t_f']} run_name={config.get('run_name')}")
    configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)
    run_simulation()
    print("[run_sim] done.")


if __name__ == "__main__":
    main()
