"""Animate the Fig. 6(b) energy profile through stagnation and reflection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .measure_fig6b_scaling import build_config, compute_h_p_profiles, measure_profiles
    from .plot_fig6b_profiles import OUT_DIR, white_hole_horizon_j
    from .style import configure_figure5_style
except ImportError:
    from measure_fig6b_scaling import build_config, compute_h_p_profiles, measure_profiles
    from plot_fig6b_profiles import OUT_DIR, white_hole_horizon_j
    from style import configure_figure5_style


BEFORE_COLOR = "#1f77b4"
AFTER_COLOR = "#e67e22"


def make_animation(L: int, output_dir: Path = OUT_DIR, max_frames: int = 267) -> tuple[Path, dict]:
    config = build_config(L)
    config["times"] = np.insert(np.asarray(config["times"], dtype=float), 0, 0.0)
    times, profiles = compute_h_p_profiles(config)
    measurement = measure_profiles(L, times, profiles)
    t_in = float(measurement["t_in"])
    t_min = float(measurement["t_min"])
    horizon_j = white_hole_horizon_j(L)
    if len(times) <= max_frames:
        frame_indices = np.arange(len(times), dtype=int)
    else:
        frame_indices = np.unique(
            np.concatenate(
                [
                    np.linspace(0, len(times) - 1, max_frames, dtype=int),
                    [int(np.argmin(np.abs(times - t_in))), int(np.argmin(np.abs(times - t_min)))],
                ]
            )
        )

    configure_figure5_style()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sites = np.arange(L)
    y_min = float(profiles.min())
    y_max = float(profiles.max())
    y_pad = 0.06 * max(y_max - y_min, 1.0)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    line, = ax.plot([], [], lw=1.6, color=BEFORE_COLOR)
    ax.axvline(
        horizon_j,
        color="#d62728",
        ls=":",
        lw=1.5,
        label=rf"horizon ($j_h={horizon_j:.2f}$)",
    )
    ax.axhline(0.0, color="0.45", lw=0.55, zorder=0)
    ax.axvspan(0, 0, color=AFTER_COLOR, alpha=0.0)
    status = ax.set_title(r"$t=0.00$   before stagnation end: approach", pad=5, fontsize=10)
    ax.text(
        0.98,
        0.96,
        rf"$t_{{\min}}={t_min:g}$",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
    )
    ax.set_xlim(0, L - 1)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$\delta\mathcal{H}_j^+$")
    ax.grid(True, color="0.82", lw=0.45)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def update(frame: int):
        time = float(times[frame])
        after = time > t_min + 1e-12
        color = AFTER_COLOR if after else BEFORE_COLOR
        if after:
            phase = "after stagnation end: reflection"
        elif time >= t_in:
            phase = "before stagnation end: compression"
        else:
            phase = "before stagnation end: approach"
        line.set_data(sites, profiles[frame])
        line.set_color(color)
        status.set_text(rf"$t={time:.2f}$   " + phase)
        status.set_color(color)
        return line, status

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=45,
        blit=True,
        repeat=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"FIG6b_profile_animation_L{L}.gif"
    animation.save(output, writer=PillowWriter(fps=22), dpi=120)
    plt.close(fig)

    measurement["horizon_j"] = horizon_j
    measurement["animation_path"] = str(output)
    return output, measurement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=267)
    args = parser.parse_args()
    output, measurement = make_animation(args.L, max_frames=args.max_frames)
    print(f"[fig6b-animation] wrote -> {output}")
    print(
        "[fig6b-animation] "
        f"j_h={measurement['horizon_j']:.6g}, "
        f"t_in={measurement['t_in']:.6g}, "
        f"t_min={measurement['t_min']:.6g}, "
        f"T_lat={measurement['T_lat']:.6g}"
    )


if __name__ == "__main__":
    main()
