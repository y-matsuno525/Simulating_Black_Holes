# Manuscript figure reproduction

Use `reproduce_panel.py` as the paper-first entry point.  It does not use the
GUI and does not rely on the mutable `config.json` default run.

Examples:

```powershell
python paper_figures\reproduce_panel.py FIG2c --rerun
python paper_figures\reproduce_panel.py FIG2
python paper_figures\reproduce_panel.py FIG4b
python paper_figures\reproduce_panel.py FIG6a --rerun
python paper_figures\reproduce_panel.py FIG6b
python paper_figures\reproduce_panel.py FIG6
python paper_figures\reproduce_panel.py FIG7 --rerun
```

`--rerun` recomputes the BdG simulation and writes the exact run config, CSV
data, geodesic, and summary to:

```text
paper_reproduction/runs/<PANEL_ID>/
```

The plotted panel is written to:

```text
paper_figures/generated/panels/<FIGURE_ID>/<PANEL_ID>.png
paper_figures/generated/panels/<FIGURE_ID>/<PANEL_ID>.pdf
```

The panel definitions are fixed in `paper_figures/reproduce_panel.py`.
For example, `FIG2c` is:

```text
L = 300, ell = 2*pi, p = 0, m = 0
beta(x) = 0.6 * [tanh(3 * (x - ell/3)) + 1]
j0 = 150, sigma = 0.05 L
initial spinor = chi+
observable = H_p
t in [0, 1]
```

For `FIG2` and `FIG3`, the corrected black-hole initial positions are:

```text
(a) j0 = 60, outside horizon, ingoing/right-moving chi+
(b) j0 = 120, outside horizon, outgoing/left-moving chi-
(c) j0 = 150, inside horizon, right-moving chi+
(d) j0 = 150, inside horizon, left-moving chi-
```

These values replace the stale caption values `j0=30,40,180,180`.

The space-time panels (`FIG2`, `FIG3`, `FIG6a`, and `FIG7`) share the same
visual style as `FIG5`: Matplotlib's sans-serif font family, black dotted
horizon lines, cyan dashed null-geodesic curves, matching reference-line
widths, and the same legend style.

For `FIG6a` and `FIG7`, the white-hole profile is fixed from the manuscript
caption:

```text
L = 300, ell = 2*pi, m = 0
beta(x) = -0.6 * tanh[3 * (x - 7*pi/5)] - 0.6
j_h = 222.808... = 0.743 L
j0 = 0.2 L, sigma = 0.05 L
initial spinor = chi+
t in [0, 8]
```

The panels are:

```text
FIG6a: p = 0, observable = H_p
FIG7a: p = 1, observable = H_p
FIG7b: p = 1, observable = H_m
FIG7c: p = 1, observable = H_pm
```

The three `FIG7` panels share one simulation run in:

```text
paper_reproduction/runs/FIG7/
```

That run writes `H_p_val.csv`, `H_m_val.csv`, and `H_pm_val.csv`, so
`python paper_figures\reproduce_panel.py FIG7 --rerun` recomputes the p=1
white-hole time evolution only once.

`FIG6b` reads the digitized manuscript points stored in
`paper_figures/reference_data/T_lat_digitized.csv` and replots the log-L scaling. It does
not run the BdG calculation. The digitized points include `L=800`, but this is
only an image-derived data point, not a new `L=800` simulation. To plot only a
smaller range, run for example:

```powershell
python paper_figures\make_fig6b_scaling.py --max-L 600
```

`FIG6` combines the cached/recomputed `FIG6a` panel with the `FIG6b` scaling
panel and writes `paper_figures/generated/panels/FIG6/FIG6.png` and `.pdf`.

The independently recomputed diagnostic scaling data are tracked separately in
`paper_figures/measured_data/T_lat_measured_L800.csv`.  They are not the
manuscript FIG6(b) data, but they can be replotted with:

```powershell
python paper_figures\measure_fig6b_scaling.py --plot-only
```

This file is the intended replacement for the old GUI-based workflow when the
task is to reproduce a manuscript figure.
