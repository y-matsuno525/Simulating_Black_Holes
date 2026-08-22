# New-layout manuscript preview

This directory is a self-contained layout preview based on the current
`paper/main_revised.tex` snapshot.

- The source manuscript and the existing files under `paper/figure/` are not
  modified by this preview.
- Proposed Figs. 2--5 use the four new double-column composites.
- The doubler/FFT, surface-gravity, white-hole `p=0`, and white-hole `p=1`
  figures use the single-column vertical candidates.
- Existing manuscript prose is retained. The old Fig. 2 and Fig. 3 caption
  text remains in the TeX source inside `\\iffalse ... \\fi`, while all four
  proposed black-hole figures use the same short layout-preview placeholder.
- Figure references and prose have intentionally not been rewritten, so this
  PDF is for visual layout review rather than textual proofreading.

Build with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main_revised_new_layout_preview.tex
```
