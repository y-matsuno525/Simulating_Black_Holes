"""Recalculate and plot individual manuscript panels.

Usage examples:

    python paper_figures/reproduce_panel.py FIG2c --rerun
    python paper_figures/reproduce_panel.py FIG2(c)
    python paper_figures/reproduce_panel.py FIG2

The script is intentionally panel-first.  Each panel has an explicit run
configuration matching the corrected paper caption, and the generated CSV/config are
kept under ``paper_reproduction/runs/<panel-id>/``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .common import DT_DEFAULT, ELL_DEFAULT, L_DEFAULT, ensure_fig_dir
    from .style import configure_figure5_style
except ImportError:
    from common import DT_DEFAULT, ELL_DEFAULT, L_DEFAULT, ensure_fig_dir
    from style import configure_figure5_style


REPO = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO / "paper_reproduction" / "runs"
PANEL_ROOT = REPO / "paper_figures" / "generated" / "panels"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


BASE_OUTPUTS_OFF = {
    "show_beta_profile": True,
    "heatmaps": False,
    "gifs": False,
    "mode_functions": False,
    "geodesic": True,
    "surface_gravity_fit": False,
    "fft": False,
}


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    figure_id: str
    tag: str
    observable: str
    label: str
    caption_title: str
    t_plot: float
    overrides: dict
    use_abs: bool = False
    cmap: str = "seismic"
    horizon_color: str = "black"
    horizon_linestyle: str | tuple = ":"
    horizon_linewidth: float = 0.9
    geodesic_color: str = "cyan"
    geodesic_linestyle: str | tuple = "--"
    geodesic_linewidth: float = 0.9
    show_geodesic: bool = True
    show_legend: bool = True
    run_id: str | None = None

    @property
    def run_dir(self) -> Path:
        return RUN_ROOT / (self.run_id or self.panel_id)


def _run_overrides(
    *,
    panel_id: str,
    p: int,
    chirality: str,
    initial_direction: str,
    beta_sign: str,
    beta_center_fraction: float,
    j0_fraction: float,
    t_f: float,
    m: float = 0.0,
) -> dict:
    return {
        "L": L_DEFAULT,
        "l": ELL_DEFAULT,
        "p": p,
        "m": m,
        "scenario": None,
        "chirality": chirality,
        "initial_direction": initial_direction,
        "beta_sign": beta_sign,
        "beta_profile": "pos",
        "surface_gravity_beta": False,
        "beta_amplitude": 0.6,
        "beta_width": 1.0,
        "beta_center_fraction": beta_center_fraction,
        "j0_fraction": j0_fraction,
        "sigma_fraction": 0.05,
        "t_i": 0.0,
        "t_f": t_f,
        "dt_scale": DT_DEFAULT,
        "PBC": False,
        "output_base_dir": str(RUN_ROOT),
        "run_name": panel_id,
        "fft_observables": ["H_p"],
        "outputs": BASE_OUTPUTS_OFF,
    }


def _bh_panel(
    panel_id: str,
    figure_id: str,
    tag: str,
    *,
    p: int,
    observable: str,
    chirality: str,
    initial_direction: str,
    beta_center_fraction: float,
    j0_fraction: float,
    t_plot: float,
    title: str,
) -> PanelSpec:
    return PanelSpec(
        panel_id=panel_id,
        figure_id=figure_id,
        tag=tag,
        observable=observable,
        label=(
            r"$\delta\mathcal{E}_{j,+}$"
            if observable == "H_p"
            else r"$\delta\mathcal{E}_{j,-}$"
        ),
        caption_title=title,
        t_plot=t_plot,
        overrides=_run_overrides(
            panel_id=panel_id,
            p=p,
            chirality=chirality,
            initial_direction=initial_direction,
            beta_sign="plus",
            beta_center_fraction=beta_center_fraction,
            j0_fraction=j0_fraction,
            t_f=t_plot,
        ),
    )


def _wh_panel(
    panel_id: str,
    figure_id: str,
    tag: str,
    *,
    p: int,
    observable: str,
    label: str,
    t_plot: float,
    title: str,
    run_id: str | None = None,
) -> PanelSpec:
    return PanelSpec(
        panel_id=panel_id,
        figure_id=figure_id,
        tag=tag,
        observable=observable,
        label=label,
        caption_title=title,
        t_plot=t_plot,
        overrides=_run_overrides(
            panel_id=run_id or panel_id,
            p=p,
            chirality="chi_plus",
            initial_direction="right",
            beta_sign="minus",
            # main_revised.tex fixes the white-hole profile at x0 = 2*ell/3
            # (j_h ~ 213); the older 0.7*ell runs are kept for comparison in
            # paper_reproduction/runs/*_x0_0p7ell.
            beta_center_fraction=2 / 3,
            j0_fraction=0.2,
            t_f=t_plot,
        ),
        run_id=run_id,
    )


PANEL_SPECS: dict[str, PanelSpec] = {
    # Fig. 2: p=0 black-hole dynamics.
    "fig2a": _bh_panel(
        "FIG2a", "FIG2", "a", p=0, observable="H_p", chirality="chi_plus",
        initial_direction="right", beta_center_fraction=2 / 3,
        j0_fraction=60 / 300, t_plot=4.0,
        title=r"FIG2(a) $p=0$, outside packet, $\chi^+$",
    ),
    "fig2b": _bh_panel(
        "FIG2b", "FIG2", "b", p=0, observable="H_m", chirality="chi_minus",
        initial_direction="left", beta_center_fraction=2 / 3,
        j0_fraction=120 / 300, t_plot=1.0,
        title=r"FIG2(b) $p=0$, outside packet, $\chi^-$",
    ),
    "fig2c": _bh_panel(
        "FIG2c", "FIG2", "c", p=0, observable="H_p", chirality="chi_plus",
        initial_direction="right", beta_center_fraction=1 / 3,
        j0_fraction=150 / 300, t_plot=1.0,
        title=r"FIG2(c) $p=0$, inside packet, $\chi^+$",
    ),
    "fig2d": _bh_panel(
        "FIG2d", "FIG2", "d", p=0, observable="H_m", chirality="chi_minus",
        initial_direction="left", beta_center_fraction=1 / 3,
        j0_fraction=150 / 300, t_plot=10.0,
        title=r"FIG2(d) $p=0$, inside packet, $\chi^-$",
    ),
    # Fig. 3: same panels with p=1.
    "fig3a": _bh_panel(
        "FIG3a", "FIG3", "a", p=1, observable="H_p", chirality="chi_plus",
        initial_direction="right", beta_center_fraction=2 / 3,
        j0_fraction=60 / 300, t_plot=4.0,
        title=r"FIG3(a) $p=1$, outside packet, $\chi^+$",
    ),
    "fig3b": _bh_panel(
        "FIG3b", "FIG3", "b", p=1, observable="H_m", chirality="chi_minus",
        initial_direction="left", beta_center_fraction=2 / 3,
        j0_fraction=120 / 300, t_plot=1.0,
        title=r"FIG3(b) $p=1$, outside packet, $\chi^-$",
    ),
    "fig3c": _bh_panel(
        "FIG3c", "FIG3", "c", p=1, observable="H_p", chirality="chi_plus",
        initial_direction="right", beta_center_fraction=1 / 3,
        j0_fraction=150 / 300, t_plot=1.0,
        title=r"FIG3(c) $p=1$, inside packet, $\chi^+$",
    ),
    "fig3d": _bh_panel(
        "FIG3d", "FIG3", "d", p=1, observable="H_m", chirality="chi_minus",
        initial_direction="left", beta_center_fraction=1 / 3,
        j0_fraction=150 / 300, t_plot=10.0,
        title=r"FIG3(d) $p=1$, inside packet, $\chi^-$",
    ),
    # Fig. 6/7 white-hole panels from the manuscript captions.
    "fig6a": _wh_panel(
        "FIG6a", "FIG6", "a", p=0, observable="H_p",
        label=r"$\delta\mathcal{E}_{j,+}$", t_plot=8.0,
        title=r"FIG6(a) $p=0$ white-hole compression",
    ),
    "fig7a": _wh_panel(
        "FIG7a", "FIG7", "a", p=1, observable="H_p",
        label=r"$\delta\mathcal{E}_{j,+}$", t_plot=8.0,
        title=r"FIG7(a) $p=1$, $\mathcal{H}^+$",
        run_id="FIG7",
    ),
    "fig7b": _wh_panel(
        "FIG7b", "FIG7", "b", p=1, observable="H_m",
        label=r"$\delta\mathcal{E}_{j,-}$", t_plot=8.0,
        title=r"FIG7(b) $p=1$, $\mathcal{H}^-$",
        run_id="FIG7",
    ),
    "fig7c": _wh_panel(
        "FIG7c", "FIG7", "c", p=1, observable="H_pm",
        label=r"$\delta\mathcal{E}_{j,\mathrm{int}}$", t_plot=8.0,
        title=r"FIG7(c) $p=1$, $\mathcal{H}^{\mathrm{int}}$",
        run_id="FIG7",
    ),
}


FIGURE_GROUPS = {
    "fig2": ["fig2a", "fig2b", "fig2c", "fig2d"],
    "fig3": ["fig3a", "fig3b", "fig3c", "fig3d"],
    "fig6": ["fig6a"],
    "fig7": ["fig7a", "fig7b", "fig7c"],
}

SPECIAL_GENERATORS = {
    "fig1": ("make_dispersion", "full dispersion figure"),
    "fig4": ("make_doubler_fft", "full doubler/FFT figure"),
    "fig4b": ("make_fig4b_fft", "FFT panel only"),
    "fig3afft": ("make_fig3a_fft_map", "time-resolved FFT of FIG3(a)"),
    "fig3adispersion": ("make_beta12_dispersion", "p=1, beta=1.2 dispersion"),
    "fig5": ("make_surface_gravity", "surface-gravity figure"),
    "fig6b": ("make_fig6b_scaling", "Fig. 6(b) white-hole stagnation scaling"),
}


def normalize_panel_name(name: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    if re.fullmatch(r"\d+[a-z]?", token):
        token = f"fig{token}"
    return token


def build_prepared_config(overrides: dict) -> dict:
    from config import DEFAULT_CONFIG, deep_merge, prepare_config

    merged = deep_merge(DEFAULT_CONFIG, overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return prepare_config(merged)


def run_panel(spec: PanelSpec) -> Path:
    from config import ANIMATION_CONFIGS, DENSITY_PLOT_CONFIGS
    from simulation import configure, run_simulation

    config = build_prepared_config(spec.overrides)
    configure(config, DENSITY_PLOT_CONFIGS, ANIMATION_CONFIGS)
    run_simulation()
    return Path(config["output_dir"])


def csv_exists(spec: PanelSpec) -> bool:
    return (spec.run_dir / f"{spec.observable}_val.csv").exists()


def load_time_site_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=complex)
    times = raw[:, 0]
    values = np.real(raw[:, 1:])
    times = np.real(times)
    return times, values


def load_geodesic(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",")
    x = data[:, 0]
    t = data[:, 1]
    return x / (ELL_DEFAULT / L_DEFAULT), t


def horizon_positions_from_config(spec: PanelSpec) -> list[float]:
    from config import compute_horizon_positions_from_config

    config = build_prepared_config(spec.overrides)
    return compute_horizon_positions_from_config(config)


def load_panel_plot_data(spec: PanelSpec):
    from matplotlib.colors import TwoSlopeNorm

    csv = spec.run_dir / f"{spec.observable}_val.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"missing calculation output for {spec.panel_id}: {csv}. "
            "Run with --rerun first."
        )

    times, data = load_time_site_csv(csv)
    t_max = min(spec.t_plot, float(times[-1])) if len(times) else spec.t_plot
    keep = times <= t_max
    data = data[keep, :]
    times = times[keep]
    if spec.use_abs:
        data = np.abs(data)

    scale = float(np.nanmax(np.abs(data))) if data.size else 1.0
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0

    if spec.use_abs:
        norm = None
        vmin, vmax = 0.0, scale
    else:
        norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
        vmin = vmax = None

    return times, data, t_max, norm, vmin, vmax


def configure_matplotlib():
    configure_figure5_style()


def draw_panel(fig, ax, spec: PanelSpec):
    times, data, t_max, norm, vmin, vmax = load_panel_plot_data(spec)

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[0, data.shape[1], 0, t_max],
        cmap=spec.cmap,
        interpolation="nearest",
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )

    for idx, horizon in enumerate(horizon_positions_from_config(spec)):
        ax.axvline(
            horizon,
            color=spec.horizon_color,
            linestyle=spec.horizon_linestyle,
            linewidth=spec.horizon_linewidth,
            label="Event horizon" if spec.show_legend and idx == 0 else None,
            zorder=5,
        )

    geo = load_geodesic(spec.run_dir / "geodesic.dat") if spec.show_geodesic else None
    if geo is not None:
        j_geo, t_geo = geo
        mask = (t_geo >= 0.0) & (t_geo <= t_max)
        ax.plot(
            j_geo[mask],
            t_geo[mask],
            color=spec.geodesic_color,
            linestyle=spec.geodesic_linestyle,
            linewidth=spec.geodesic_linewidth,
            label="Geodesic" if spec.show_legend else None,
            zorder=4,
        )

    if spec.show_legend:
        ax.legend(loc="upper left", frameon=True, handlelength=1.8, borderpad=0.4)

    ax.set_xlim(0, L_DEFAULT)
    ax.set_ylim(0, t_max)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$t$", rotation=0, labelpad=8)
    ax.text(-0.17, 1.03, f"({spec.tag})", transform=ax.transAxes, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.025)
    cbar.set_label(spec.label, rotation=0, labelpad=10)
    return im


def plot_panel(spec: PanelSpec) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    draw_panel(fig, ax, spec)
    out_dir = PANEL_ROOT / spec.figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / spec.panel_id
    for ext in ("pdf", "png"):
        fig.savefig(
            out_base.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return out_base.with_suffix(".png")


def plot_composite(specs: list[PanelSpec]) -> Path | None:
    if not specs:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    figure_id = specs[0].figure_id
    fig, axes = plt.subplots(
        1,
        len(specs),
        figsize=(3.05 * len(specs), 2.55),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, spec in zip(axes, specs):
        draw_panel(fig, ax, spec)

    out_dir = PANEL_ROOT / figure_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / figure_id
    for ext in ("pdf", "png"):
        fig.savefig(
            out_base.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return out_base.with_suffix(".png")


def plot_fig6_composite() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from .make_fig6b_scaling import draw_scaling_panel
    except ImportError:
        from make_fig6b_scaling import draw_scaling_panel

    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)
    draw_panel(fig, axes[0], PANEL_SPECS["fig6a"])
    draw_scaling_panel(axes[1])

    out_dir = PANEL_ROOT / "FIG6"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "FIG6"
    for ext in ("pdf", "png"):
        fig.savefig(
            out_base.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return out_base.with_suffix(".png")


def reproduce_panel(spec: PanelSpec, *, rerun: bool) -> Path:
    if rerun or not csv_exists(spec):
        print(f"[paper] running {spec.panel_id}: {spec.caption_title}")
        run_panel(spec)
    else:
        print(f"[paper] using cached run: {spec.run_dir}")
    out = plot_panel(spec)
    print(f"[paper] saved panel -> {out}")
    return out


def selected_specs(selection: str) -> list[PanelSpec]:
    key = normalize_panel_name(selection)
    if key in PANEL_SPECS:
        return [PANEL_SPECS[key]]
    if key in FIGURE_GROUPS:
        return [PANEL_SPECS[item] for item in FIGURE_GROUPS[key]]
    valid = ", ".join(sorted([*PANEL_SPECS, *FIGURE_GROUPS, *SPECIAL_GENERATORS]))
    raise SystemExit(f"unknown panel/figure '{selection}'. Valid keys: {valid}")


def is_figure_group(selection: str) -> bool:
    return normalize_panel_name(selection) in FIGURE_GROUPS


def reproduce_special(selection: str, *, rerun: bool = False) -> bool:
    key = normalize_panel_name(selection)
    if key not in SPECIAL_GENERATORS:
        return False
    module_name, description = SPECIAL_GENERATORS[key]
    print(f"[paper] running {key}: {description}")
    module = importlib.import_module(module_name)
    if rerun and key == "fig5":
        module.main(["--recompute"])
    else:
        module.main()
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", help="Panel or figure, e.g. FIG2c, FIG2(c), 2c, or FIG2")
    parser.add_argument("--rerun", action="store_true", help="Recompute the simulation before plotting")
    args = parser.parse_args(argv)

    if reproduce_special(args.selection, rerun=args.rerun):
        return

    specs = selected_specs(args.selection)
    key = normalize_panel_name(args.selection)
    rerun_done: set[Path] = set()
    for spec in specs:
        rerun_this = args.rerun and spec.run_dir not in rerun_done
        reproduce_panel(spec, rerun=rerun_this)
        if rerun_this:
            rerun_done.add(spec.run_dir)
    if is_figure_group(args.selection):
        if key == "fig6":
            reproduce_special("FIG6b")
            out = plot_fig6_composite()
        else:
            out = plot_composite(specs)
        if out is not None:
            print(f"[paper] saved composite -> {out}")


if __name__ == "__main__":
    main()
