"""Generate visibility variants for the Fig. 3(a) and Fig. 6(a) panels.

This is a visual-comparison utility.  It reads the cached simulation data and
does not overwrite any manuscript figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, SymLogNorm, TwoSlopeNorm

try:
    from .common import L_DEFAULT
    from .reproduce_panel import (
        PANEL_SPECS,
        PanelSpec,
        configure_matplotlib,
        horizon_positions_from_config,
        load_geodesic,
        load_panel_plot_data,
    )
except ImportError:
    from common import L_DEFAULT
    from reproduce_panel import (
        PANEL_SPECS,
        PanelSpec,
        configure_matplotlib,
        horizon_positions_from_config,
        load_geodesic,
        load_panel_plot_data,
    )


REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "paper_figures" / "generated" / "comparisons" / "trajectory_visibility"


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    dark_geodesic: bool = False
    halo: bool = False
    numerical_center: bool = False
    percentile_clip: float | None = None
    symlog: bool = False
    asinh_width: float | None = None
    endpoint_threshold: float | None = None
    contour_threshold: float | None = None


VARIANTS = (
    Variant("baseline", "Baseline"),
    Variant("A_dark_thick", "A: darker, thicker geodesic", dark_geodesic=True),
    Variant("B_halo", "B: geodesic with halo", dark_geodesic=True, halo=True),
    Variant("C_centroid", "C: numerical packet center", numerical_center=True),
    Variant("D_clip", "D: 99.5% color clipping", percentile_clip=99.5),
    Variant("E_symlog", "E: symmetric-log color scale", symlog=True),
)


def _finite_scale(data: np.ndarray) -> float:
    scale = float(np.nanmax(np.abs(data))) if data.size else 1.0
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _normalization(data: np.ndarray, variant: Variant):
    full_scale = _finite_scale(data)
    if variant.endpoint_threshold is not None:
        scale = variant.endpoint_threshold
        return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale), scale
    if variant.percentile_clip is not None:
        finite = np.abs(data[np.isfinite(data)])
        finite = finite[finite > 0.0]
        scale = float(np.percentile(finite, variant.percentile_clip)) if finite.size else full_scale
        return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale), scale
    if variant.symlog:
        return (
            SymLogNorm(
                linthresh=0.5,
                linscale=1.0,
                vmin=-full_scale,
                vmax=full_scale,
                base=10,
            ),
            full_scale,
        )
    if variant.asinh_width is not None:
        return (
            AsinhNorm(
                linear_width=variant.asinh_width,
                vmin=-full_scale,
                vmax=full_scale,
            ),
            full_scale,
        )
    return TwoSlopeNorm(vmin=-full_scale, vcenter=0.0, vmax=full_scale), full_scale


def _energy_centroid(data: np.ndarray) -> np.ndarray:
    weights = np.abs(data)
    sites = np.arange(data.shape[1], dtype=float)
    totals = weights.sum(axis=1)
    centers = np.full(data.shape[0], np.nan)
    valid = totals > np.finfo(float).eps
    centers[valid] = (weights[valid] @ sites) / totals[valid]
    return centers


def draw_variant(fig, ax, spec: PanelSpec, variant: Variant, *, add_title: bool):
    times, data, t_max, _, _, _ = load_panel_plot_data(spec)
    norm, scale = _normalization(data, variant)
    cmap = plt.get_cmap(spec.cmap).copy()
    extend = "neither"
    if variant.endpoint_threshold is not None:
        cmap.set_over("#F0E442")
        cmap.set_under("#7A0177")
        extend = "both"
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[0, data.shape[1], 0, t_max],
        cmap=cmap,
        interpolation="nearest",
        norm=norm,
    )

    for idx, horizon in enumerate(horizon_positions_from_config(spec)):
        ax.axvline(
            horizon,
            color=spec.horizon_color,
            linestyle=spec.horizon_linestyle,
            linewidth=spec.horizon_linewidth,
            label="Event horizon" if idx == 0 else None,
            zorder=5,
        )

    geo = load_geodesic(spec.run_dir / "geodesic.dat")
    if geo is not None:
        j_geo, t_geo = geo
        mask = (t_geo >= 0.0) & (t_geo <= t_max)
        color = "#009FB7" if variant.dark_geodesic else spec.geodesic_color
        linewidth = 1.4 if variant.dark_geodesic else spec.geodesic_linewidth
        line, = ax.plot(
            j_geo[mask],
            t_geo[mask],
            color=color,
            linestyle=spec.geodesic_linestyle,
            linewidth=linewidth,
            label="Geodesic",
            zorder=6,
        )
        if variant.halo:
            line.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.6, foreground="white"),
                    path_effects.Normal(),
                ]
            )

    if variant.numerical_center:
        centers = _energy_centroid(data)
        ax.plot(
            centers,
            times,
            color="#7A0177",
            linewidth=1.2,
            linestyle="-",
            label="Numerical center",
            zorder=7,
        )

    if variant.contour_threshold is not None:
        sites = np.arange(data.shape[1], dtype=float) + 0.5
        ax.contour(
            sites,
            times,
            np.abs(data),
            levels=[variant.contour_threshold],
            colors=["#D89000"],
            linewidths=1.2,
            zorder=7,
        )
        ax.plot(
            [],
            [],
            color="#D89000",
            linewidth=1.2,
            label=rf"$|\delta\mathcal{{E}}|={variant.contour_threshold:g}$",
        )

    ax.legend(loc="upper left", frameon=True, handlelength=1.8, borderpad=0.4)
    ax.set_xlim(0, L_DEFAULT)
    ax.set_ylim(0, t_max)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$t$", rotation=0, labelpad=12)
    if add_title:
        suffix = f" ($S={scale:.2g}$)" if variant.percentile_clip is not None else ""
        ax.set_title(variant.title + suffix, fontsize=9.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.025, fraction=0.055, extend=extend)
    if variant.asinh_width is not None:
        major = 10.0 ** np.floor(np.log10(scale))
        multipliers = (-3, -1, 0, 1, 3) if 3.0 * major <= scale else (-1, 0, 1)
        ticks = [value * major for value in multipliers]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{value:g}" for value in ticks])
    if add_title:
        cbar.set_label(spec.label, rotation=90, labelpad=8)
    else:
        cbar.set_label(spec.label, rotation=0, labelpad=17)
    return im


def save_individuals(panel_key: str) -> None:
    spec = PANEL_SPECS[panel_key]
    for variant in VARIANTS:
        fig, ax = plt.subplots(figsize=(3.35, 2.7), constrained_layout=True)
        draw_variant(fig, ax, spec, variant, add_title=False)
        ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes, fontsize=11, va="bottom")
        stem = OUTPUT_DIR / f"{spec.panel_id}_{variant.key}"
        fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_comparison(panel_key: str) -> None:
    spec = PANEL_SPECS[panel_key]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), constrained_layout=True)
    for ax, variant in zip(axes.flat, VARIANTS, strict=True):
        draw_variant(fig, ax, spec, variant, add_title=True)
    stem = OUTPUT_DIR / f"{spec.panel_id}_visibility_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def three_method_variants(panel_key: str) -> tuple[Variant, ...]:
    threshold = 10.0 if panel_key == "fig3a" else 12.0
    return (
        Variant(
            "F_mild_asinh",
            r"1: mild $\mathrm{asinh}$ scale",
            dark_geodesic=True,
            asinh_width=3.0,
        ),
        Variant(
            "G_endpoint_colors",
            rf"2: endpoint colors ($S={threshold:g}$)",
            dark_geodesic=True,
            endpoint_threshold=threshold,
        ),
        Variant(
            "H_threshold_contour",
            rf"3: threshold contour ($S={threshold:g}$)",
            dark_geodesic=True,
            contour_threshold=threshold,
        ),
    )


def save_three_method_comparison(panel_key: str) -> None:
    spec = PANEL_SPECS[panel_key]
    variants = three_method_variants(panel_key)
    for variant in variants:
        fig, ax = plt.subplots(figsize=(3.35, 2.7), constrained_layout=True)
        draw_variant(fig, ax, spec, variant, add_title=False)
        ax.text(-0.14, 1.03, "(a)", transform=ax.transAxes, fontsize=11, va="bottom")
        stem = OUTPUT_DIR / f"{spec.panel_id}_{variant.key}"
        fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), constrained_layout=True)
    for ax, variant in zip(axes, variants, strict=True):
        draw_variant(fig, ax, spec, variant, add_title=True)
    stem = OUTPUT_DIR / f"{spec.panel_id}_three_new_methods"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_asinh_width_comparison(panel_key: str = "fig3a") -> None:
    """Compare several asinh linear widths without changing paper figures."""
    spec = PANEL_SPECS[panel_key]
    widths = (0.5, 1.0, 2.0, 3.0, 5.0)
    variants = tuple(
        Variant(
            f"asinh_H0_{width:g}",
            rf"$\mathcal{{H}}_0={width:g}$",
            dark_geodesic=True,
            asinh_width=width,
        )
        for width in widths
    )

    for variant in variants:
        fig, ax = plt.subplots(figsize=(3.35, 2.7), constrained_layout=True)
        draw_variant(fig, ax, spec, variant, add_title=True)
        stem = OUTPUT_DIR / f"{spec.panel_id}_{variant.key}"
        fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), constrained_layout=True)
    for ax, variant in zip(axes.flat, variants, strict=False):
        draw_variant(fig, ax, spec, variant, add_title=True)
    axes.flat[-1].axis("off")
    stem = OUTPUT_DIR / f"{spec.panel_id}_asinh_H0_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for panel_key in ("fig3a", "fig6a"):
        save_individuals(panel_key)
        save_comparison(panel_key)
        save_three_method_comparison(panel_key)
    print(f"saved comparisons -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
