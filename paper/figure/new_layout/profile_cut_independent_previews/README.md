# Independent-scale profile-cut previews

This directory contains the trial Fig. 2--5 layout in which each energy
component has its own line graph and its own symmetric vertical scale.

## Composite layout

- (a): spacetime heatmap of the launched packet's main component
- (b): $\mathcal H_j^+$ at the two selected times
- (c): $\mathcal H_j^-$ at the two selected times
- (d): $\mathcal H_j^{\rm int}$ at the two selected times

The orange solid line is the earlier time and the blue dashed line is the
later time.  Black dotted lines mark the horizon.  Geodesic guide lines are
intentionally omitted from this preview.

The fixed-time panels do not include in-panel maximum-value annotations.

## Stored runs and cut times

| Proposed figure | Stored run | Parameters | Exact stored cut times |
|---|---|---|---|
| Fig. 2 | `FIG2a` | $p=0$, exterior $a^+$ packet | $t=2.0, 3.5$ |
| Fig. 3 | `FIG2b` | $p=0$, exterior $a^-$ packet | $t=0.5, 0.9$ |
| Fig. 4 | `FIG3a` | $p=1$, exterior $a^+$ packet | $t=2.0, 3.5$ |
| Fig. 5 | `FIG3b` | $p=1$, exterior $a^-$ packet | $t=0.5, 0.9$ |

The $a^-$ runs end at $t=0.99$, so their previews use two exact rows before
the packet reaches the left boundary.  No interpolation or rerun is used.

## Data handling

The plotted profiles are direct rows of `H_p_val.csv`, `H_m_val.csv`, and
`H_pm_val.csv`.  There is no thresholding, zeroing, smoothing, clipping,
renormalization, or additional division by epsilon.  An identically zero
component is displayed on an axis-only range of $\pm10^{-15}$; the data remain
exactly zero.

`composites/` contains the four 2-by-2 previews.  Under `individual/`, files
ending in `_both_times.png` are the corresponding stand-alone component plots.
