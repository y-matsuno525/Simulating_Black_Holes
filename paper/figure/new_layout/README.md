# Proposed manuscript figure layout

This directory contains candidate figure files only.  Nothing here is included
by `paper/main_revised.tex`, and the existing files in `paper/figure/` are not
overwritten.

## Directory map

- `composites/`: selected candidate composites.  The current default is a
  common symmetric color scale for the `p=0` figures and component-wise
  symmetric scales for the `p=1` figures.
- `individual/`: the three panels before composition, suitable for uploading
  separately to Overleaf.  Each figure folder also contains `manifest.json`
  with its source run, CSV paths, raw maximum amplitudes, and display scales.
- `previews/common_scale/`: all three panels use one symmetric scale computed
  from the largest absolute stored value in that figure.
- `previews/component_scale/`: every panel uses its own symmetric scale.
- `single_column_candidates/`: vertical layouts for the remaining multi-panel
  figures.

Every plot is saved as PNG, vector PDF, and editable-text SVG.  SVG text is not
converted to paths.

For consistent apparent type size at the intended manuscript widths, all new
figures use 8.5 pt axis labels, 7.5 pt tick labels, 8 pt legends, and 9 pt panel
labels.  The two-column composites are drawn at 7 inch width, while the
remaining vertical figures are drawn at 3.35 inch single-column width.

## Black-hole source runs

| Proposed figure | Packet | Stored run | Panels |
|---|---|---|---|
| Fig. 2 | `p=0`, exterior `a+` | `paper_reproduction/runs/FIG2a` | `H_p`, `H_m`, `H_pm` |
| Fig. 3 | `p=0`, exterior `a-` | `paper_reproduction/runs/FIG2b` | `H_p`, `H_m`, `H_pm` |
| Fig. 4 | `p=1`, exterior `a+` | `paper_reproduction/runs/FIG3a` | `H_p`, `H_m`, `H_pm` |
| Fig. 5 | `p=1`, exterior `a-` | `paper_reproduction/runs/FIG3b` | `H_p`, `H_m`, `H_pm` |

The CSV arrays are plotted as stored.  No thresholding, numerical zeroing,
smoothing, density renormalization, or division by `epsilon` is applied.  Only
Matplotlib color normalization limits differ between preview variants.

## Commands

Generate everything without rerunning the simulation:

```powershell
python paper_figures\reproduce_panel.py NEW_LAYOUT
```

Generate one proposed figure:

```powershell
python paper_figures\reproduce_panel.py BH_P0_APLUS
python paper_figures\reproduce_panel.py BH_P0_AMINUS
python paper_figures\reproduce_panel.py BH_P1_APLUS
python paper_figures\reproduce_panel.py BH_P1_AMINUS
```

`layout_check.tex` is a separate REVTeX document for checking the figures at
their intended two-column and one-column widths.  It does not import or modify
the manuscript.
