# Full calculation audit for `main_revised.tex`

Audit date: 2026-08-12

## Overall result

After the corrections documented below, the Hamiltonian, equations of motion,
figure settings, saved numerical profiles, and plotted quantities are mutually
consistent. All 45 regression tests pass. Every production profile used in
Figs. 2, 3, 5, 6(a), and 7 was independently reconstructed from the current
bond-centered BdG Hamiltonian; the largest discrepancy relative to a profile
peak is below `9e-13`.

Two interpretation qualifications remain and are now stated in the manuscript:

1. The late parts of Figs. 2(a), 3(a), 4(b), and 7 include physical-boundary
   overlap. They must not be interpreted as quantitatively boundary-free.
2. Fig. 6(b) tests an approximately logarithmic dependence, not the continuum
   coefficient `1/kappa`.

## Equation-level checks

| Item | Method | Result |
|---|---|---|
| Spin Hamiltonian to Jordan-Wigner Hamiltonian | Full Fock-space matrices, nonuniform beta | PASS, max error below `1e-12` |
| Fermionic Hamiltonian to `a^+`, `a^-` form | Full Fock-space matrices | PASS, max error below `1e-12` |
| Both lattice Heisenberg equations | Direct commutators `[a_j^s,H]` | PASS, max error below `1e-12` |
| BdG Hermiticity | OBC and PBC, nonuniform beta | PASS |
| Homogeneous dispersion | Finite PBC BdG spectrum vs analytical bands | PASS, max error below `1e-12` |
| Local densities | Full Fock-space operators and quasiparticle matrices | PASS |
| Density sum rule | `epsilon sum_j H_j = H` | PASS |
| Vacuum subtraction | Independent contractions | PASS |
| Horizon position | Analytical tanh solution vs stored value | PASS, error about `9e-6` sites |
| Null geodesics | High-accuracy independent integration | PASS, max error below `8e-9` sites |
| Total energy conservation | All dynamics runs | PASS, relative drift below `7e-12` |

The manuscript site convention was changed to `j=0,...,L-1`, matching the code,
plots, quoted `j_0`, and quoted `j_h`. The dimensionless lattice momentum is now
called `k`, matching Figs. 1 and 4, while the physical momentum is `q=k/epsilon`.

## Figure audit

### Fig. 1

- Caption settings: `p=0,1` and `beta=0,1,2`.
- The retained original raster was digitized by color and compared against the
  analytical dispersion equation. The largest median residual is `4.076` pixels.
- A current-code generator reproducing the same four-part design is available,
  but the retained manuscript raster was kept to preserve the requested design.
- Status: PASS.

### Figs. 2 and 3

- All `L`, `ell`, profile, `x_0`, `j_0`, `sigma`, chirality, direction, and `p`
  settings match the captions.
- Analytical horizons are `j_h=212.8075` and `112.8075`, consistent with the
  caption values 213 and 113.
- Every displayed density profile matches an independent recomputation to
  relative error below `9e-13`.
- Fig. 3(a) uses the stated symmetric asinh normalization with linear width 3.
- Boundary qualification: the selected-branch weight in the outer ten sites
  first exceeds 1 percent at `t=3.01` in Fig. 2(a) and `t=2.88` in Fig. 3(a).
- Status: PASS with the stated late-time boundary qualification.

### Fig. 4

- The dispersion uses `p=1`, `beta=1.2`.
- Analytical doubler wave number: `k_d=1.1713710869`.
- The FFT is the magnitude of the spatial FFT of the mean-subtracted
  `delta E_{j,+}` profile, normalized by the maximum over all four spectra.
- Profiles are now evaluated exactly at `t=0,3,3.5,4`. The former plot selected
  `0.01,3.01,3.51,3.99`; the maximum normalized-curve change was `0.0173`.
- An extended-box control with unchanged lattice spacing and local profile still
  develops a `k_d` peak, so the peak is not created solely by the right boundary.
  Its late-time amplitude in the paper-size system is nevertheless enhanced by
  boundary contact.
- Status: PASS after exact-time correction, with late-time qualification.

### Fig. 5

- Settings match the caption: `L=500`, `A=B=C=1`, `x_0=ell/2`, `j_0=245`,
  `sigma=0.003L`, and fit samples within `0<t<1`.
- The width is the standard deviation using `|delta E_{j,-}|` as a nonnegative
  weight. Lattice-site width is converted by exactly `ell/L`.
- Correct current-data fits:
  - `p=0`: `kappa_fit=1.0332161`, relative error `3.3216%`, `R^2=0.9999566`.
  - `p=1`: `kappa_fit=0.9513801`, relative error `4.8620%`, `R^2=0.9944037`.
- The previous `0.3%` and `9.6%` values came from pre-density-fix data and were
  removed from the image, caption, body, and conclusion.
- Status: PASS after image and numerical-text correction.

### Fig. 6

- Fig. 6(a) settings and horizon agree with the caption. The stated asinh
  normalization with linear width 3 is applied.
- Fig. 6(b) was recalculated sequentially for `L=100,200,...,800`. Each run now
  stores its complete prepared configuration.
- Recalculated `T_lat` values are:
  `0.418796, 0.826766, 1.099951, 1.286794, 1.506524, 1.702435,
  1.868279, 2.048486`.
- A linear fit against `ln L` gives slope `0.77238` and `R^2=0.97808`. The paper
  correctly limits the claim to consistency with logarithmic dependence.
- `t_in` is the rightward crossing into the central region after the compression
  stage, defined by `|beta''(xbar)|<0.1`; `t_min` is the minimum standard deviation.
- Status: PASS.

### Fig. 7

- `L=300`, `p=1`, profile, `x_0`, `j_0`, `sigma`, and all three observables match
  the caption.
- `H_+`, `H_-`, and `H_int` profiles independently match the saved data to
  relative error below `6e-14`.
- Boundary qualification: outer-ten-site weight exceeds 1 percent at `t=6.78`
  and reaches 12.8 percent at the final plotted time. The horizon transmission
  discussed in the text occurs earlier.
- Status: PASS with the stated late-time boundary qualification.

## Generated evidence

- `profile_audit.json`: independent profile-by-profile comparison.
- `metadata_audit.json`: settings, horizons, geodesics, conservation, FFT, fits,
  boundary fractions, and Fig. 6(b) statistics.
- `boundary_control.json`: extended-box Fig. 4 diagnostic.
- `fig1_raster_audit.json`: digitized retained Fig. 1 comparison.
- `paper_reproduction/runs/FIG4b_exact/`: exact FFT snapshots and configuration.
- `paper_reproduction/runs/FIG5_fit/fit_summary.json`: Fig. 5 fit provenance.

## Final manuscript QA

- `python -m unittest discover -s tests -v`: PASS, 45/45 tests.
- The workspace manuscript and the standalone submission directory both build
  successfully with `pdflatex`, `bibtex`, and two final `pdflatex` passes.
- The final PDF has 11 pages. All pages were rasterized and inspected; the
  figures, legends, color bars, panel labels, captions, equations, and references
  show no clipping or overlap.
- The standalone submission PDF and the inspected workspace PDF are pixel-identical
  on all 11 rendered pages.
- There are no undefined citations or references. The remaining TeX diagnostics
  are a 3.7 pt equation overfull warning and a deferred Fig. 1 float warning;
  neither produces a visible defect or changes any calculation.
