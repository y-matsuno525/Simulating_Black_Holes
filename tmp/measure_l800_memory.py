from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_figures.measure_fig6b_scaling import (  # noqa: E402
    build_config,
    compute_h_p_profiles,
    measure_profiles,
)


config = build_config(800)
times, profiles = compute_h_p_profiles(config)
result = measure_profiles(800, times, profiles)
print(f"T_lat={result['T_lat']:.15g}")
