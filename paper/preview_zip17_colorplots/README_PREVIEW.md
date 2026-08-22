# Color-plot manuscript preview

- Source manuscript archive: `C:/Users/ymats/Downloads/black_hole_simulation (17).zip`
- Preview TeX: `main_revised_colorplot_preview.tex`
- The source `main_revised.tex`, its prose, and its existing caption wording are unchanged.
- The preview TeX changes only figure placement, figure-file references, and the two explicitly requested dummy captions; the scientific prose and equations are unchanged.
- The newly inserted Fig. 3 and Fig. 5 use clearly marked dummy captions for page-layout confirmation; they must be replaced before submission.
- The four existing composite PDFs were reused from `paper/preview_new_layout_manuscript/figure`; no figure data were recalculated for this preview.
- The preview class selects the standard RevTeX `prb` journal option; manuscript text is unchanged.
- Figs. 2--5 and Fig. 9 are double-column figures. Figs. 6--8 use existing vertical, single-column vector PDFs.
- In Fig. 8, panel (b) spans the same outer width as panel (a) including its colorbar; their left and right boundaries therefore align at the compiled single-column size.
- Fig. 8(b) uses the directly measured values in `paper_figures/measured_data/T_lat_measured_L800.csv` for `L=100,200,...,800`. Each CSV value was checked against the corresponding `paper_reproduction/runs/FIG6b_measured/L*/measurement_summary.json` value.
- The source ZIP's horizontal `WH_p=0.png` instead contains the older image-derived points now recorded in `paper_figures/reference_data/T_lat_digitized.csv`. The old manuscript identifies their profile as `A=C=-1`, `B=0.75`, `x0=9*pi/5`, `j0=0.2L`, and `sigma=0.05L`; no original run log or raw CSV for those old points is present in the repository.
- Figs. 2--5 use the same `0.90\linewidth` scale so their plotted text remains mutually consistent. RevTeX is limited to one double-column top float per page, allowing the two-column manuscript text to continue below each figure. No manual page break is used.
- Figs. 3 and 5 include 24 pt of top clearance inside their preview float environments so that their legends and panel labels remain inside the printable area. This affects placement only, not the source figure PDFs.

## Preview figures

| Figure | Initial packet | Panels | Source run | Display scale |
|---|---|---|---|---|
| Fig. 2 | $p=0$, exterior $a^+$ | $\mathcal H^+$, $\mathcal H^-$, $\mathcal H^{\rm int}$ | `paper_reproduction/runs/FIG2a` | one common symmetric linear scale |
| Fig. 3 | $p=0$, exterior $a^-$ | $\mathcal H^+$, $\mathcal H^-$, $\mathcal H^{\rm int}$ | `paper_reproduction/runs/FIG2b` | one common symmetric linear scale |
| Fig. 4 | $p=1$, exterior $a^+$ | $\mathcal H^+$, $\mathcal H^-$, $\mathcal H^{\rm int}$ | `paper_reproduction/runs/FIG3a` | component scales; $\mathcal H^+$ keeps $\mathcal H_0=1$ asinh display |
| Fig. 5 | $p=1$, exterior $a^-$ | $\mathcal H^+$, $\mathcal H^-$, $\mathcal H^{\rm int}$ | `paper_reproduction/runs/FIG3b` | component-specific symmetric linear scales |

The arrays are plotted without thresholding, smoothing, interpolation, clipping of the stored values, or an additional division by $\epsilon$. Black dotted lines mark the horizon, and cyan dashed lines show the corresponding continuum null trajectories.

The twelve `data/FIG2`--`data/FIG5` component CSV files in this directory were verified byte-for-byte against the corresponding source-run CSV files using SHA-256.
