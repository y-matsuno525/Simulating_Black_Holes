"""Generate proposed manuscript layouts without touching current paper figures.

The black-hole panels reuse the stored FIG2a, FIG2b, FIG3a, and FIG3b
simulation CSV files.  Plotting changes only the Matplotlib color
normalization; the loaded arrays are never thresholded, rescaled, smoothed, or
otherwise modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    from .common import ELL_DEFAULT, PAPER_DATA, REPO
    from .reproduce_panel import (
        PANEL_SPECS,
        PanelSpec,
        draw_panel,
        horizon_positions_from_config,
        load_geodesic,
        load_panel_plot_data,
        load_time_site_csv,
    )
    from .style import configure_figure5_style
except ImportError:
    from common import ELL_DEFAULT, PAPER_DATA, REPO
    from reproduce_panel import (
        PANEL_SPECS,
        PanelSpec,
        draw_panel,
        horizon_positions_from_config,
        load_geodesic,
        load_panel_plot_data,
        load_time_site_csv,
    )
    from style import configure_figure5_style


OUTPUT_ROOT = REPO / "paper" / "figure" / "new_layout"
COMPOSITE_DIR = OUTPUT_ROOT / "composites"
INDIVIDUAL_DIR = OUTPUT_ROOT / "individual"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
SINGLE_COLUMN_DIR = OUTPUT_ROOT / "single_column_candidates"
PROFILE_CUT_ROOT = OUTPUT_ROOT / "profile_cut_previews"
PROFILE_CUT_COMPOSITE_DIR = PROFILE_CUT_ROOT / "composites"
PROFILE_CUT_INDIVIDUAL_DIR = PROFILE_CUT_ROOT / "individual"
INDEPENDENT_CUT_ROOT = OUTPUT_ROOT / "profile_cut_independent_previews"
INDEPENDENT_CUT_COMPOSITE_DIR = INDEPENDENT_CUT_ROOT / "composites"
INDEPENDENT_CUT_INDIVIDUAL_DIR = INDEPENDENT_CUT_ROOT / "individual"
PROFILE_CUT_TIMES = (2.0, 3.5)
PROFILE_CUT_TIMES_BY_KEY = {
    "bhp0aplus": (2.0, 3.5),
    "bhp0aminus": (0.5, 0.9),
    "bhp1aplus": (2.0, 3.5),
    "bhp1aminus": (0.5, 0.9),
}

OBSERVABLES = ("H_p", "H_m", "H_pm")
OBSERVABLE_LABELS = {
    "H_p": r"$\mathcal{H}_{j}^{+}$",
    "H_m": r"$\mathcal{H}_{j}^{-}$",
    "H_pm": r"$\mathcal{H}_{j}^{\mathrm{int}}$",
}
OBSERVABLE_FILE_LABELS = {
    "H_p": "Hplus",
    "H_m": "Hminus",
    "H_pm": "Hint",
}

# These point sizes are applied at the final physical widths (7.0 inch for a
# double-column composite and 3.35 inch for a single-column figure).  Keeping
# the point sizes identical is what makes the compiled manuscript typography
# match across layouts with different aspect ratios.
DOUBLE_COLUMN_FIGSIZE = (7.0, 2.20)
PUBLICATION_FONT_STYLE = {
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8.0,
}
PANEL_LABEL_FONTSIZE = 9.0
COLORBAR_TICK_FONTSIZE = 7.5
COMPOSITE_LEGEND_FONTSIZE = 6.0

PROFILE_STYLES = {
    "H_p": {"color": "#D55E00", "linestyle": "-"},
    "H_m": {"color": "#0072B2", "linestyle": "--"},
    "H_pm": {"color": "#009E73", "linestyle": "-."},
}


@dataclass(frozen=True)
class LayoutDefinition:
    key: str
    figure_number: int
    stem: str
    source_spec_key: str
    p: int
    packet: str


LAYOUTS = {
    "bhp0aplus": LayoutDefinition(
        "bhp0aplus", 2, "fig02_p0_aplus", "fig2a", 0, "a+"
    ),
    "bhp0aminus": LayoutDefinition(
        "bhp0aminus", 3, "fig03_p0_aminus", "fig2b", 0, "a-"
    ),
    "bhp1aplus": LayoutDefinition(
        "bhp1aplus", 4, "fig04_p1_aplus", "fig3a", 1, "a+"
    ),
    "bhp1aminus": LayoutDefinition(
        "bhp1aminus", 5, "fig05_p1_aminus", "fig3b", 1, "a-"
    ),
}


def profile_cut_times(definition: LayoutDefinition) -> tuple[float, float]:
    return PROFILE_CUT_TIMES_BY_KEY[definition.key]


def configure_matplotlib() -> None:
    """Apply the manuscript style and keep SVG text editable."""
    import matplotlib.pyplot as plt

    configure_figure5_style()
    plt.rcParams["svg.fonttype"] = "none"


def _normalize_figure_typography(fig, panel_axes) -> None:
    """Normalize effective point sizes after helpers with local overrides."""
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(PUBLICATION_FONT_STYLE["axes.labelsize"])
        ax.yaxis.label.set_fontsize(PUBLICATION_FONT_STYLE["axes.labelsize"])
        ax.tick_params(
            axis="both", labelsize=PUBLICATION_FONT_STYLE["xtick.labelsize"]
        )
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(PUBLICATION_FONT_STYLE["legend.fontsize"])
            legend.get_title().set_fontsize(PUBLICATION_FONT_STYLE["legend.fontsize"])

    for ax in np.atleast_1d(panel_axes).flat:
        for item in ax.texts:
            if item.get_text().strip() in {"(a)", "(b)", "(c)", "(d)"}:
                item.set_fontsize(PANEL_LABEL_FONTSIZE)


def _panel_specs(definition: LayoutDefinition) -> tuple[PanelSpec, ...]:
    source = PANEL_SPECS[definition.source_spec_key]
    specs = []
    for tag, observable in zip("abc", OBSERVABLES):
        # Preserve the inherited normalization only for the originally plotted
        # component.  Newly exposed components use the existing symmetric
        # linear normalization.
        color_norm = source.color_norm if observable == source.observable else "linear"
        panel_id = (
            f"fig{definition.figure_number:02d}{tag}_"
            f"{OBSERVABLE_FILE_LABELS[observable]}"
        )
        specs.append(
            replace(
                source,
                panel_id=panel_id,
                figure_id=f"NEW_FIG{definition.figure_number:02d}",
                tag=tag,
                observable=observable,
                label=OBSERVABLE_LABELS[observable],
                caption_title=(
                    f"proposed Fig. {definition.figure_number} "
                    f"p={definition.p}, {definition.packet}, {observable}"
                ),
                color_norm=color_norm,
                run_id=source.run_dir.name,
            )
        )
    return tuple(specs)


def _raw_scale(spec: PanelSpec) -> float:
    """Return max(abs(data)) without changing the loaded values."""
    _, data, _, _, _, _ = load_panel_plot_data(spec)
    return float(np.nanmax(np.abs(data))) if data.size else 0.0


def _display_scale(raw_scale: float) -> float:
    """Return a valid normalization range; this never modifies plot data."""
    return raw_scale if raw_scale > 0.0 and np.isfinite(raw_scale) else 1.0


def _save_formats(fig, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf", "svg"):
        out = out_base.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600 if ext == "png" else None)
        outputs.append(out)
    return outputs


def _plot_composite(
    specs: tuple[PanelSpec, ...],
    out_base: Path,
    *,
    common_scale: float | None,
) -> list[Path]:
    import matplotlib.pyplot as plt
    with plt.rc_context(PUBLICATION_FONT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COLUMN_FIGSIZE)
        fig.subplots_adjust(
            left=0.05,
            right=0.94,
            bottom=0.20,
            top=0.88,
            wspace=0.40,
        )
        for index, (ax, spec) in enumerate(zip(axes, specs)):
            draw_panel(
                fig,
                ax,
                replace(spec, show_legend=True),
                scale_override=common_scale,
                panel_label_fontsize=PANEL_LABEL_FONTSIZE,
                colorbar_label_rotation=90,
                colorbar_tick_labelsize=COLORBAR_TICK_FONTSIZE,
                show_ylabel=index == 0,
                legend_loc="upper left",
                legend_fontsize=COMPOSITE_LEGEND_FONTSIZE,
            )
    outputs = _save_formats(fig, out_base)
    plt.close(fig)
    return outputs


def _plot_individual(
    specs: tuple[PanelSpec, ...],
    out_dir: Path,
    *,
    common_scale: float | None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs = []
    for spec in specs:
        with plt.rc_context(PUBLICATION_FONT_STYLE):
            fig, ax = plt.subplots(figsize=(3.35, 2.45), layout="constrained")
            draw_panel(
                fig,
                ax,
                spec,
                scale_override=common_scale,
                panel_label_fontsize=PANEL_LABEL_FONTSIZE,
                colorbar_tick_labelsize=COLORBAR_TICK_FONTSIZE,
            )
            _normalize_figure_typography(fig, [ax])
            outputs.extend(_save_formats(fig, out_dir / spec.panel_id))
            plt.close(fig)
    return outputs


def _write_manifest(
    definition: LayoutDefinition,
    specs: tuple[PanelSpec, ...],
    raw_scales: dict[str, float],
    common_scale: float,
) -> Path:
    selected_mode = "common" if definition.p == 0 else "component"
    manifest = {
        "target": definition.key,
        "proposed_figure": definition.figure_number,
        "p": definition.p,
        "packet": definition.packet,
        "source_run": specs[0].run_dir.relative_to(REPO).as_posix(),
        "selected_scale_mode": selected_mode,
        "common_symmetric_scale": [-common_scale, common_scale],
        "panels": [],
        "data_processing": "none; only Matplotlib color normalization is changed",
    }
    for spec in specs:
        raw_scale = raw_scales[spec.observable]
        manifest["panels"].append(
            {
                "panel_id": spec.panel_id,
                "observable": spec.observable,
                "source_csv": (spec.run_dir / f"{spec.observable}_val.csv")
                .relative_to(REPO)
                .as_posix(),
                "raw_max_abs": raw_scale,
                "component_symmetric_scale": [
                    -_display_scale(raw_scale),
                    _display_scale(raw_scale),
                ],
                "normalization": spec.color_norm,
                "asinh_linear_width": (
                    spec.asinh_linear_width if spec.color_norm == "asinh" else None
                ),
                "t_plot": spec.t_plot,
            }
        )
    out = INDIVIDUAL_DIR / definition.stem / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return out


def generate_target(key: str) -> list[Path]:
    """Generate both scale previews and the selected candidate for one target."""
    if key not in LAYOUTS:
        raise KeyError(f"unknown new-layout target: {key}")

    import matplotlib

    matplotlib.use("Agg")
    configure_matplotlib()

    definition = LAYOUTS[key]
    specs = _panel_specs(definition)
    raw_scales = {spec.observable: _raw_scale(spec) for spec in specs}
    common_scale = _display_scale(max(raw_scales.values()))
    outputs = []

    outputs.extend(
        _plot_composite(
            specs,
            PREVIEW_DIR / "common_scale" / definition.stem,
            common_scale=common_scale,
        )
    )
    outputs.extend(
        _plot_composite(
            specs,
            PREVIEW_DIR / "component_scale" / definition.stem,
            common_scale=None,
        )
    )

    selected_common_scale = common_scale if definition.p == 0 else None
    outputs.extend(
        _plot_composite(
            specs,
            COMPOSITE_DIR / definition.stem,
            common_scale=selected_common_scale,
        )
    )
    outputs.extend(
        _plot_individual(
            specs,
            INDIVIDUAL_DIR / definition.stem,
            common_scale=selected_common_scale,
        )
    )
    outputs.append(_write_manifest(definition, specs, raw_scales, common_scale))
    for output in outputs:
        print(f"[new-layout] saved -> {output}")
    return outputs


def load_profile_cut_data(
    definition: LayoutDefinition,
) -> tuple[
    tuple[PanelSpec, ...],
    dict[str, np.ndarray],
    dict[float, int],
    dict[float, dict[str, np.ndarray]],
]:
    """Load the stored arrays and select exact fixed-time rows without interpolation."""
    specs = _panel_specs(definition)
    arrays: dict[str, np.ndarray] = {}
    reference_times: np.ndarray | None = None
    for spec in specs:
        times, values = load_time_site_csv(
            spec.run_dir / f"{spec.observable}_val.csv"
        )
        if reference_times is None:
            reference_times = times
        elif not np.array_equal(times, reference_times):
            raise ValueError(
                f"time grid mismatch in stored run {spec.run_dir.name}: "
                f"{spec.observable}"
            )
        arrays[spec.observable] = values

    if reference_times is None:
        raise ValueError(f"no stored times for {definition.key}")

    indices: dict[float, int] = {}
    cuts: dict[float, dict[str, np.ndarray]] = {}
    for cut_time in profile_cut_times(definition):
        matches = np.flatnonzero(
            np.isclose(reference_times, cut_time, rtol=0.0, atol=1.0e-12)
        )
        if len(matches) != 1:
            raise ValueError(
                f"stored run {specs[0].run_dir.name} does not contain exactly "
                f"one row at t={cut_time:g}"
            )
        row = int(matches[0])
        indices[cut_time] = row
        cuts[cut_time] = {
            observable: values[row]
            for observable, values in arrays.items()
        }
    return specs, arrays, indices, cuts


def profile_cut_y_limit(cuts: dict[float, dict[str, np.ndarray]]) -> float:
    """Return one symmetric plotting limit shared by both fixed-time cuts."""
    maximum = max(
        float(np.max(np.abs(values)))
        for cut in cuts.values()
        for values in cut.values()
    )
    return 1.05 * maximum if maximum > 0.0 and np.isfinite(maximum) else 1.0


def _scientific_tex(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent <= 2:
        return f"{value:.3g}"
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


def _profile_legend_handles(spec: PanelSpec):
    from matplotlib.lines import Line2D

    handles = []
    for observable in OBSERVABLES:
        style = PROFILE_STYLES[observable]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.4,
                label=OBSERVABLE_LABELS[observable],
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=spec.horizon_color,
                linestyle=spec.horizon_linestyle,
                linewidth=spec.horizon_linewidth,
                label="Event horizon",
            ),
            Line2D(
                [0],
                [0],
                color=spec.geodesic_color,
                linestyle=spec.geodesic_linestyle,
                linewidth=spec.geodesic_linewidth,
                label="Geodesic",
            ),
        ]
    )
    return handles


def _geodesic_position_at_time(spec: PanelSpec, cut_time: float) -> float | None:
    geo = load_geodesic(spec.run_dir / "geodesic.dat")
    if geo is None:
        return None
    j_geo, t_geo = geo
    order = np.argsort(t_geo)
    t_geo = t_geo[order]
    j_geo = j_geo[order]
    if cut_time < t_geo[0] or cut_time > t_geo[-1]:
        return None
    return float(np.interp(cut_time, t_geo, j_geo))


def _draw_profile_cut_panel(
    ax,
    spec: PanelSpec,
    cut: dict[str, np.ndarray],
    *,
    cut_time: float,
    y_limit: float,
    tag: str,
    show_ylabel: bool,
) -> None:
    sites = np.arange(next(iter(cut.values())).size)
    for observable in OBSERVABLES:
        style = PROFILE_STYLES[observable]
        ax.plot(
            sites,
            cut[observable],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            zorder=3,
        )

    ax.axhline(0.0, color="0.55", linewidth=0.6, zorder=0)
    for horizon in horizon_positions_from_config(spec):
        ax.axvline(
            horizon,
            color=spec.horizon_color,
            linestyle=spec.horizon_linestyle,
            linewidth=spec.horizon_linewidth,
            zorder=2,
        )
    geodesic = _geodesic_position_at_time(spec, cut_time)
    if geodesic is not None:
        ax.axvline(
            geodesic,
            color=spec.geodesic_color,
            linestyle=spec.geodesic_linestyle,
            linewidth=spec.geodesic_linewidth,
            zorder=2,
        )

    main_observable = PANEL_SPECS[spec.panel_id].observable if spec.panel_id in PANEL_SPECS else spec.observable
    peak_site = int(np.argmax(np.abs(cut[main_observable])))
    annotation_on_right = peak_site < sites.size / 2
    annotation = "\n".join(
        rf"$\max_j|{OBSERVABLE_LABELS[observable][1:-1]}|={_scientific_tex(float(np.max(np.abs(cut[observable]))))}$"
        for observable in OBSERVABLES
    )
    ax.text(
        0.97 if annotation_on_right else 0.03,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right" if annotation_on_right else "left",
        va="top",
        fontsize=7.0,
        linespacing=1.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=6,
    )
    ax.set_xlim(0, sites.size)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel(r"$j$")
    if show_ylabel:
        ax.set_ylabel(r"$\mathcal{H}_j$", rotation=90, labelpad=4)
    else:
        ax.set_ylabel("")
    ax.set_title(rf"$t={cut_time:g}$", pad=3, fontsize=8.5)
    ax.text(
        -0.12,
        1.03,
        f"({tag})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
    )


def _profile_cut_main_spec(
    definition: LayoutDefinition, specs: tuple[PanelSpec, ...]
) -> PanelSpec:
    source = PANEL_SPECS[definition.source_spec_key]
    main = next(spec for spec in specs if spec.observable == source.observable)
    return replace(main, tag="a", show_legend=False)


def _plot_profile_cut_composite(
    definition: LayoutDefinition,
    specs: tuple[PanelSpec, ...],
    cuts: dict[float, dict[str, np.ndarray]],
    y_limit: float,
    out_base: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    main_spec = _profile_cut_main_spec(definition, specs)
    with plt.rc_context(PUBLICATION_FONT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
        fig.subplots_adjust(
            left=0.06,
            right=0.985,
            bottom=0.18,
            top=0.76,
            wspace=0.38,
        )
        draw_panel(
            fig,
            axes[0],
            main_spec,
            panel_label_fontsize=PANEL_LABEL_FONTSIZE,
            colorbar_label_rotation=90,
            colorbar_tick_labelsize=COLORBAR_TICK_FONTSIZE,
        )
        for tag, cut_time, ax in zip("bc", tuple(cuts), axes[1:]):
            _draw_profile_cut_panel(
                ax,
                main_spec,
                cuts[cut_time],
                cut_time=cut_time,
                y_limit=y_limit,
                tag=tag,
                show_ylabel=tag == "b",
            )
        fig.legend(
            handles=_profile_legend_handles(main_spec),
            loc="upper center",
            bbox_to_anchor=(0.53, 0.985),
            ncol=5,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.2,
        )
        _normalize_figure_typography(fig, axes)
    outputs = _save_formats(fig, out_base)
    plt.close(fig)
    return outputs


def _plot_profile_cut_individuals(
    definition: LayoutDefinition,
    specs: tuple[PanelSpec, ...],
    cuts: dict[float, dict[str, np.ndarray]],
    y_limit: float,
    out_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    main_spec = _profile_cut_main_spec(definition, specs)
    main_name = OBSERVABLE_FILE_LABELS[main_spec.observable]
    outputs = []
    with plt.rc_context(PUBLICATION_FONT_STYLE):
        fig, ax = plt.subplots(figsize=(3.35, 2.45), layout="constrained")
        draw_panel(
            fig,
            ax,
            replace(main_spec, show_legend=True),
            panel_label_fontsize=PANEL_LABEL_FONTSIZE,
            colorbar_tick_labelsize=COLORBAR_TICK_FONTSIZE,
        )
        _normalize_figure_typography(fig, [ax])
        outputs.extend(
            _save_formats(
                fig,
                out_dir / f"fig{definition.figure_number:02d}a_{main_name}_main_heatmap",
            )
        )
        plt.close(fig)

    for tag, cut_time in zip("bc", tuple(cuts)):
        with plt.rc_context(PUBLICATION_FONT_STYLE):
            fig, ax = plt.subplots(figsize=(3.35, 2.45))
            fig.subplots_adjust(left=0.17, right=0.97, bottom=0.18, top=0.72)
            _draw_profile_cut_panel(
                ax,
                main_spec,
                cuts[cut_time],
                cut_time=cut_time,
                y_limit=y_limit,
                tag=tag,
                show_ylabel=True,
            )
            fig.legend(
                handles=_profile_legend_handles(main_spec),
                loc="upper center",
                bbox_to_anchor=(0.53, 0.99),
                ncol=3,
                frameon=False,
                handlelength=1.8,
                columnspacing=1.0,
            )
            _normalize_figure_typography(fig, [ax])
            time_label = str(cut_time).replace(".", "p")
            outputs.extend(
                _save_formats(
                    fig,
                    out_dir / f"fig{definition.figure_number:02d}{tag}_cut_t{time_label}",
                )
            )
            plt.close(fig)
    return outputs


def _write_profile_cut_manifest(
    definition: LayoutDefinition,
    specs: tuple[PanelSpec, ...],
    indices: dict[float, int],
    cuts: dict[float, dict[str, np.ndarray]],
    y_limit: float,
) -> Path:
    source = PANEL_SPECS[definition.source_spec_key]
    manifest = {
        "target": definition.key,
        "proposed_figure": definition.figure_number,
        "p": definition.p,
        "packet": definition.packet,
        "source_run": specs[0].run_dir.relative_to(REPO).as_posix(),
        "main_heatmap_observable": source.observable,
        "cut_times": list(cuts),
        "cut_row_indices": {f"{time:g}": indices[time] for time in cuts},
        "shared_symmetric_cut_ylim": [-y_limit, y_limit],
        "cuts": [],
        "data_processing": "none; exact stored CSV rows are plotted without rescaling",
    }
    for cut_time in cuts:
        manifest["cuts"].append(
            {
                "time": cut_time,
                "max_abs": {
                    observable: float(np.max(np.abs(cuts[cut_time][observable])))
                    for observable in OBSERVABLES
                },
                "source_csvs": {
                    observable: (
                        specs[0].run_dir / f"{observable}_val.csv"
                    ).relative_to(REPO).as_posix()
                    for observable in OBSERVABLES
                },
            }
        )
    out = PROFILE_CUT_INDIVIDUAL_DIR / definition.stem / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return out


def generate_profile_cut_target(key: str) -> list[Path]:
    """Generate one heatmap-plus-two-profile-cut preview from stored data."""
    if key not in LAYOUTS:
        raise KeyError(f"unknown profile-cut target: {key}")

    import matplotlib

    matplotlib.use("Agg")
    configure_matplotlib()
    definition = LAYOUTS[key]
    specs, _, indices, cuts = load_profile_cut_data(definition)
    y_limit = profile_cut_y_limit(cuts)
    outputs = []
    outputs.extend(
        _plot_profile_cut_composite(
            definition,
            specs,
            cuts,
            y_limit,
            PROFILE_CUT_COMPOSITE_DIR / definition.stem,
        )
    )
    outputs.extend(
        _plot_profile_cut_individuals(
            definition,
            specs,
            cuts,
            y_limit,
            PROFILE_CUT_INDIVIDUAL_DIR / definition.stem,
        )
    )
    outputs.append(
        _write_profile_cut_manifest(definition, specs, indices, cuts, y_limit)
    )
    for output in outputs:
        print(f"[profile-cut] saved -> {output}")
    return outputs


def generate_all_profile_cuts() -> list[Path]:
    outputs = []
    for key in LAYOUTS:
        outputs.extend(generate_profile_cut_target(key))
    return outputs


def independent_component_y_limits(
    cuts: dict[float, dict[str, np.ndarray]],
) -> dict[str, float]:
    """Return a separate symmetric scale for each component across both cuts."""
    limits = {}
    for observable in OBSERVABLES:
        maximum = max(
            float(np.max(np.abs(cuts[cut_time][observable])))
            for cut_time in cuts
        )
        # An identically zero array still needs a finite display range.  This
        # is an axis-only convention and does not change any stored value.
        limits[observable] = 1.05 * maximum if maximum > 0.0 else 1.0e-15
    return limits


def _draw_independent_cut_axis(
    ax,
    spec: PanelSpec,
    values: np.ndarray,
    *,
    observable: str,
    cut_time: float,
    y_limit: float,
    tag: str,
    show_xlabel: bool,
    show_title: bool,
) -> None:
    sites = np.arange(values.size)
    style = PROFILE_STYLES[observable]
    ax.plot(
        sites,
        values,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.25,
        zorder=3,
    )
    ax.axhline(0.0, color="0.55", linewidth=0.55, zorder=0)
    for horizon in horizon_positions_from_config(spec):
        ax.axvline(
            horizon,
            color=spec.horizon_color,
            linestyle=spec.horizon_linestyle,
            linewidth=spec.horizon_linewidth,
            zorder=2,
        )
    geodesic = _geodesic_position_at_time(spec, cut_time)
    if geodesic is not None:
        ax.axvline(
            geodesic,
            color=spec.geodesic_color,
            linestyle=spec.geodesic_linestyle,
            linewidth=spec.geodesic_linewidth,
            zorder=2,
        )
    ax.set_xlim(0, values.size)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xticks([0, values.size // 2, values.size])
    if show_xlabel:
        ax.set_xlabel(r"$j$")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_title:
        ax.set_title(OBSERVABLE_LABELS[observable], pad=3, fontsize=8.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    maximum = float(np.max(np.abs(values)))
    ax.text(
        0.04,
        0.91,
        rf"$\max_j|\mathcal{{H}}_j|={_scientific_tex(maximum)}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        zorder=6,
    )
    ax.text(
        -0.20,
        1.03,
        f"({tag})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
    )


def _save_trial_png(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    return out


def _draw_independent_component_axis(
    ax,
    spec: PanelSpec,
    cuts: dict[float, dict[str, np.ndarray]],
    *,
    observable: str,
    y_limit: float,
    tag: str,
) -> None:
    from matplotlib.lines import Line2D

    sites = np.arange(next(iter(cuts.values()))[observable].size)
    cut_times = tuple(cuts)
    time_styles = {
        cut_times[0]: {"color": "#D55E00", "linestyle": "-"},
        cut_times[1]: {"color": "#0072B2", "linestyle": "--"},
    }
    for cut_time in cut_times:
        values = cuts[cut_time][observable]
        style = time_styles[cut_time]
        ax.plot(
            sites,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            label=rf"$t={cut_time:g}$",
            zorder=3,
        )

    ax.axhline(0.0, color="0.55", linewidth=0.55, zorder=0)
    for horizon in horizon_positions_from_config(spec):
        ax.axvline(
            horizon,
            color=spec.horizon_color,
            linestyle=spec.horizon_linestyle,
            linewidth=spec.horizon_linewidth,
            zorder=2,
        )
    ax.set_xlim(0, sites.size)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(OBSERVABLE_LABELS[observable], rotation=90, labelpad=5)
    ax.set_title(OBSERVABLE_LABELS[observable], pad=3, fontsize=8.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.text(
        -0.12,
        1.03,
        f"({tag})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=time_styles[cut_time]["color"],
                lw=1.35,
                ls=time_styles[cut_time]["linestyle"],
                label=rf"$t={cut_time:g}$",
            )
            for cut_time in cut_times
        ],
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=1.8,
    )


def generate_independent_profile_preview(key: str) -> list[Path]:
    """Generate a trial with one independently scaled graph per component."""
    if key not in LAYOUTS:
        raise KeyError(f"unknown independent profile-cut target: {key}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_matplotlib()
    definition = LAYOUTS[key]
    specs, _, _, cuts = load_profile_cut_data(definition)
    main_spec = _profile_cut_main_spec(definition, specs)
    limits = independent_component_y_limits(cuts)

    with plt.rc_context(PUBLICATION_FONT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
        fig.subplots_adjust(
            left=0.075,
            right=0.965,
            bottom=0.10,
            top=0.88,
            wspace=0.34,
            hspace=0.42,
        )
        heatmap_ax = axes[0, 0]
        draw_panel(
            fig,
            heatmap_ax,
            main_spec,
            panel_label_fontsize=PANEL_LABEL_FONTSIZE,
            colorbar_label_rotation=90,
            colorbar_tick_labelsize=COLORBAR_TICK_FONTSIZE,
            show_geodesic=False,
        )
        component_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
        for tag, observable, ax in zip("bcd", OBSERVABLES, component_axes):
            _draw_independent_component_axis(
                ax,
                main_spec,
                cuts,
                observable=observable,
                y_limit=limits[observable],
                tag=tag,
            )
        _normalize_figure_typography(fig, [heatmap_ax, *component_axes])
        composite = _save_trial_png(
            fig,
            INDEPENDENT_CUT_COMPOSITE_DIR / f"{definition.stem}.png",
        )
        plt.close(fig)

    outputs = [composite]
    individual_dir = INDEPENDENT_CUT_INDIVIDUAL_DIR / definition.stem
    with plt.rc_context(PUBLICATION_FONT_STYLE):
        fig, ax = plt.subplots(figsize=(3.35, 2.45))
        fig.subplots_adjust(left=0.17, right=0.82, bottom=0.18, top=0.85)
        draw_panel(
            fig,
            ax,
            main_spec,
            panel_label_fontsize=PANEL_LABEL_FONTSIZE,
            colorbar_label_rotation=90,
            colorbar_tick_labelsize=COLORBAR_TICK_FONTSIZE,
            show_geodesic=False,
        )
        _normalize_figure_typography(fig, [ax])
        outputs.append(
            _save_trial_png(
                fig,
                individual_dir
                / f"fig{definition.figure_number:02d}_main_heatmap.png",
            )
        )
        plt.close(fig)

    for tag, observable in zip("bcd", OBSERVABLES):
        with plt.rc_context(PUBLICATION_FONT_STYLE):
            fig, ax = plt.subplots(figsize=(3.35, 2.45))
            fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.85)
            _draw_independent_component_axis(
                ax,
                main_spec,
                cuts,
                observable=observable,
                y_limit=limits[observable],
                tag=tag,
            )
            _normalize_figure_typography(fig, [ax])
            out = individual_dir / (
                f"fig{definition.figure_number:02d}_"
                f"{OBSERVABLE_FILE_LABELS[observable]}_both_times.png"
            )
            outputs.append(_save_trial_png(fig, out))
            plt.close(fig)
    for output in outputs:
        print(f"[independent-cut] saved -> {output}")
    return outputs


def _draw_saved_fft_panel(ax, kd: float) -> None:
    try:
        from .make_doubler_fft import (
            FFT_RUN_DIR,
            FFT_TIMES,
            fft_snapshots,
            set_positive_k_ticks,
        )
    except ImportError:
        from make_doubler_fft import (
            FFT_RUN_DIR,
            FFT_TIMES,
            fft_snapshots,
            set_positive_k_ticks,
        )

    raw = np.loadtxt(
        FFT_RUN_DIR / "H_p_val.csv", delimiter=",", skiprows=1, dtype=complex
    )
    times = np.real(raw[:, 0])
    if not np.array_equal(times, np.asarray(FFT_TIMES, dtype=float)):
        raise ValueError("saved FIG4b_exact profiles are not at the caption times")
    data = np.real(raw[:, 1:])
    k, spectra = fft_snapshots(data)
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
    for tval, spectrum, color in zip(FFT_TIMES, spectra, colors):
        ax.plot(k, spectrum, lw=1.35, color=color, label=rf"$t={tval:g}$")
    ax.axvline(kd, color="black", lw=0.9, ls=":", zorder=0)
    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(-0.015, 1.06)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Normalized amplitude")
    set_positive_k_ticks(ax, kd)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="upper right", frameon=True, handlelength=1.8)
    ax.text(-0.12, 1.04, r"(b)", transform=ax.transAxes, fontsize=11)


def _generate_doubler_vertical() -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        from .make_doubler_fft import draw_dispersion_panel
    except ImportError:
        from make_doubler_fft import draw_dispersion_panel

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.0), layout="constrained")
    kd = draw_dispersion_panel(axes[0])
    axes[0].xaxis.label.set_fontsize(11)
    axes[0].yaxis.label.set_fontsize(11)
    axes[0].set_xlabel(r"$k$")
    axes[0].xaxis.set_label_coords(0.5, -0.12)
    _draw_saved_fft_panel(axes[1], kd)
    _normalize_figure_typography(fig, axes)
    outputs = _save_formats(fig, SINGLE_COLUMN_DIR / "doubler_fft_vertical")
    plt.close(fig)
    return outputs


def _generate_surface_gravity_vertical() -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        from .make_surface_gravity import SG_SETTINGS, _panel
    except ImportError:
        from make_surface_gravity import SG_SETTINGS, _panel

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.0), layout="constrained")
    _panel(
        axes[0],
        PAPER_DATA / "H_m_sigmas_p0.txt",
        ell=ELL_DEFAULT,
        L=SG_SETTINGS["L"],
        panel_tag="(a)",
    )
    _panel(
        axes[1],
        PAPER_DATA / "H_m_sigmas_p1.txt",
        ell=ELL_DEFAULT,
        L=SG_SETTINGS["L"],
        panel_tag="(b)",
    )
    _normalize_figure_typography(fig, axes)
    outputs = _save_formats(fig, SINGLE_COLUMN_DIR / "surface_gravity_vertical")
    plt.close(fig)
    return outputs


def _generate_wh_p0_vertical() -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        from .make_fig6b_scaling import draw_scaling_panel
    except ImportError:
        from make_fig6b_scaling import draw_scaling_panel

    fig = plt.figure(figsize=(3.35, 5.0), layout="constrained")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.055),
        height_ratios=(1.0, 1.0),
        wspace=0.08,
        hspace=0.08,
    )
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, :]))
    colorbar_ax = fig.add_subplot(grid[0, 1])
    draw_panel(
        fig,
        axes[0],
        PANEL_SPECS["fig6a"],
        colorbar_ax=colorbar_ax,
    )
    draw_scaling_panel(axes[1])
    _normalize_figure_typography(fig, axes)
    outputs = _save_formats(fig, SINGLE_COLUMN_DIR / "WH_p0_vertical")
    plt.close(fig)
    return outputs


def _generate_wh_p1_vertical() -> list[Path]:
    import matplotlib.pyplot as plt

    specs = tuple(PANEL_SPECS[key] for key in ("fig7a", "fig7b", "fig7c"))
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 7.5), layout="constrained")
    for ax, spec in zip(axes, specs):
        draw_panel(fig, ax, spec)
    _normalize_figure_typography(fig, axes)
    outputs = _save_formats(fig, SINGLE_COLUMN_DIR / "WH_p1_vertical")
    plt.close(fig)
    return outputs


def generate_single_column_candidates() -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    configure_matplotlib()
    outputs = []
    outputs.extend(_generate_doubler_vertical())
    outputs.extend(_generate_surface_gravity_vertical())
    outputs.extend(_generate_wh_p0_vertical())
    outputs.extend(_generate_wh_p1_vertical())
    for output in outputs:
        print(f"[new-layout] saved -> {output}")
    return outputs


def generate_all() -> list[Path]:
    outputs = []
    for key in LAYOUTS:
        outputs.extend(generate_target(key))
    outputs.extend(generate_single_column_candidates())
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "selection",
        nargs="?",
        default="all",
        choices=["all", "profile-cuts", *LAYOUTS, *(f"{key}-cuts" for key in LAYOUTS)],
        help="New-layout target to generate (default: all)",
    )
    args = parser.parse_args(argv)
    if args.selection == "all":
        generate_all()
    elif args.selection == "profile-cuts":
        generate_all_profile_cuts()
    elif args.selection.endswith("-cuts"):
        generate_profile_cut_target(args.selection.removesuffix("-cuts"))
    else:
        generate_target(args.selection)


if __name__ == "__main__":
    main()
