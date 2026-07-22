# Energy-density revision progress

## Objective

Make the numerical observables, paper figures, and manuscript consistent with
Eqs. (40)--(43) of `main_revised.tex` in
`C:/Users/ymats/Downloads/black_hole_simulation (13)`.

This file is the progress tracker for that work. Update the checkboxes and the
log after each verified step.

## Source of truth

- Manuscript: `black_hole_simulation (13)/main_revised.tex`
- Field notation: `a^+` and `a^-`
- Mixed contribution in paper text and figures: `H^int`
  (LaTeX: `H^{\mathrm{int}}`)
- Existing internal/output key `H_pm`: retained temporarily for compatibility
- `main_revised.tex` already uses `H^{\mathrm{int}}` in Eqs. (40) and (43).
  Existing figure captions and colorbars that still use `H^\pm` will be
  updated to `H^{\mathrm{int}}`.
- Total Hamiltonian: one term per bond, with `beta` evaluated at that bond's
  center, `beta_{j+1/2}`
- Local continuum density:

  ```text
  H = epsilon * sum_j H_j
  ```

- Horizon: the continuous solution of `beta(x_h) = +/-1`; `j_h = x_h/epsilon`
  is a continuous plotting coordinate and does not need to be an integer

## Confirmed current state

| Item | Status | Note |
|---|---|---|
| BdG bond-centered beta | Done | `build_bdg_matrix()` uses the beta value of each bond |
| `a^- = -chi^-` transformation | Verified | Pure-minus initial states differ only by a global sign |
| Hamiltonian in the `a` basis | Verified | Matches the fermionic Hamiltonian to machine precision |
| Heisenberg equations in `main_revised.tex` | Verified | Both lattice equations match to machine precision |
| Local density beta assignment | Needs change | Current code applies `beta_{j+1/2}` to both adjacent bonds |
| Endpoint density | Needs change | Current code zeros both endpoints |
| Continuum-density normalization | Needs change | Current values carry one fewer factor of `1/epsilon` |
| Vacuum subtraction | Needs change | Must use the same revised local operator |
| Fig. 6/7 horizon parameters | Needs decision | PNG uses `x0=0.7 ell`, manuscript says `x0=2 ell/3` |

## Required local-density definition

First define the complete bond contribution. For the plus branch,

```text
B^+_{j+1/2} = -i/(2 epsilon) * (1 + beta_{j+1/2}) * a^+_j a^+_{j+1}.
```

The site energy is one half of each adjacent complete bond,

```text
h^+_j = (B^+_{j-1/2} + B^+_{j+1/2}) / 2.
```

The continuum energy density plotted in the paper is

```text
H^+_j = h^+_j / epsilon.
```

The same order of operations applies to the minus and mixed contributions:
apply the correct coefficient to each bond first, then distribute one half of
the complete bond to each adjacent site.

At an open boundary, omit only the nonexistent bond:

```text
h_1 = B_{3/2}/2
h_L = B_{L-1/2}/2
```

The endpoint values may be hidden in the figure, but they must not be replaced
by zero in the operator definition if `H = sum_j h_j` is to hold exactly.

## Phase 1: code changes

- [x] Refactor `build_energy_densities()` into explicit left-bond and
      right-bond contributions.
- [x] Use `beta_{j-1/2}` for the left bond and `beta_{j+1/2}` for the right
      bond in `H_p` and `H_m`.
- [x] Keep the on-site part of the mixed contribution unchanged.
- [x] Split the mixed bond contribution equally between adjacent sites.
- [x] Preserve one-sided endpoint contributions for open boundaries.
- [x] Include the additional `1/epsilon` required for continuum density.
- [x] Apply exactly the same bond assignment, endpoint rule, and normalization
      in `compute_vacuum_values()`.
- [x] Keep `H_pm` as the internal key and display it as `H^{\mathrm{int}}` in
      paper-facing output.
- [x] Document whether saved `H_*_val.csv` files contain site energies or
      continuum densities. The revised pipeline should save continuum density.
      Documented: `H_p_val.csv`, `H_m_val.csv`, and `H_pm_val.csv` now contain
      the vacuum-subtracted continuum densities `delta E_{j,s}` of
      Eq. (44) (units of `1/epsilon^2`), so that `epsilon * sum_j` of the
      unsubtracted profile reproduces the total energy.

## Phase 2: tests

- [x] Test the left and right beta coefficients directly for a nonuniform
      profile.
- [x] Test the plus, minus, and mixed local operators for Hermiticity.
- [x] Test open-boundary endpoint contributions.
- [x] Test the sum rule `epsilon * sum_j H_j = H` in a small full-Fock-space
      system, allowing only the explicitly known constant term.
- [x] Test that vacuum subtraction uses the same revised density operator.
- [x] Test that multiplying a complete density profile by a positive constant
      does not change the measured packet center or width.
- [x] Run `python -m unittest discover -s tests` (31 tests OK).

## Phase 3: numerical comparison

- [ ] Save the current figure outputs and numerical summaries as the baseline.
- [ ] Recalculate one representative black-hole panel first (`FIG2c`).
- [ ] Compare the old and revised profiles, including peak position, width,
      integrated energy, and maximum pointwise difference.
- [ ] Recalculate one representative white-hole panel (`FIG6a`) because it is
      most sensitive near `1 + beta = 0`.
- [ ] Confirm that trajectory changes are small and identify any color-scale or
      near-horizon changes before launching all runs.

## Phase 4: affected figures

- [ ] Fig. 2: recalculate all four density panels.
- [ ] Fig. 3: recalculate all four density panels.
- [ ] Fig. 4(b): regenerate the FFT from the revised Fig. 3(a) data.
- [ ] Fig. 5: recalculate widths and update both fitted errors.
- [ ] Fig. 6(a): recalculate the white-hole density panel.
- [ ] Fig. 6(b): separately decide whether to retain the current reference data
      or remeasure `T_lat` with the revised density.
- [ ] Fig. 7: recalculate all three density panels.
- [ ] Confirm that Fig. 1 and Fig. 4(a) remain unchanged because they use the
      homogeneous dispersion rather than local density data.

## Phase 5: paper synchronization

- [ ] Use `a^+` and `a^-` consistently in the simulation section and captions.
- [ ] Use `H^+`, `H^-`, and `H^{\mathrm{int}}` consistently in equations, captions,
      colorbars, and prose.
- [ ] Label plotted vacuum-subtracted quantities as `delta E_{j,s}` where the
      manuscript defines that quantity.
- [ ] Update all numerical values that change after recalculation, especially
      the Fig. 5 surface-gravity errors.
- [ ] Resolve the Fig. 6/7 choice:
      `x0=0.7 ell, j_h about 223` or `x0=2 ell/3, j_h about 213`.
- [ ] Remove the duplicated dynamics section and duplicated `sec:dynamics`
      label in `main_revised.tex`.
- [ ] Copy only verified regenerated figures into the manuscript figure folder.
- [ ] Build the manuscript and check cross-references, captions, and figure
      labels.

## Verification commands

```powershell
python -m unittest discover -s tests
python paper_figures\reproduce_panel.py FIG2 --rerun
python paper_figures\reproduce_panel.py FIG3 --rerun
python paper_figures\reproduce_panel.py FIG4
python paper_figures\reproduce_panel.py FIG5
python paper_figures\reproduce_panel.py FIG6 --rerun
python paper_figures\reproduce_panel.py FIG7 --rerun
```

Run the full figure set only after the representative Fig. 2(c) and Fig. 6(a)
comparisons pass.

## Progress log

| Date | Change | Verification | Result |
|---|---|---|---|
| 2026-07-23 | Created revision plan from `main_revised.tex` and current code inspection | Equation/operator comparison and LaTeX inspection | Planning complete; source code not changed |
| 2026-07-23 | Fixed the paper-facing mixed-term notation | Checked Eqs. (40), (43), and the existing Fig. 7 caption | Use `H^{int}` in the manuscript and figures; retain internal key `H_pm` |
| 2026-07-23 | Checkpoint commit `70c2cdc` on branch `claude/energy-density-revision`; imported `main_revised.tex` into `paper/` | `python -m unittest discover -s tests` (22 tests) | OK |
| 2026-07-23 | Rewrote `build_energy_densities()`: complete bond energies with `beta_{j-1/2}`/`beta_{j+1/2}`, half-bond site assignment, one-sided open-boundary endpoints, `1/epsilon^2` continuum normalization, explicit onsite constant `-(p-eps*m)/(2 eps^2)` in `H_pm` | Old regression suite (22 tests) | OK; commit `56c82f3`; vacuum subtraction updated in the next commit |
| 2026-07-23 | Rewrote `compute_vacuum_values()` with per-bond contractions `F1[b]`, `F2[b]` and the same link betas, endpoint rule, and `1/epsilon^2` normalization as the density operators | Full suite incl. new Fock-space tests (31 tests) | OK; commit `ab937b7` |
| 2026-07-23 | Added `tests/test_energy_density.py`: full-Fock-space (Jordan-Wigner, L=6) comparison of the coded densities and vacuum values against literal Eqs. (41)-(43); sum rule `eps*sum_j H_j = H` incl. the onsite constant; left/right link betas; one-sided endpoints; Hermiticity; scaling invariance of center/width | `python -m unittest discover -s tests` (31 tests) | OK; commit `c96d753` |
| 2026-07-23 | Figure-code notation: colorbars now show the vacuum-subtracted `delta E_{j,+/-}` and `delta E_{j,int}` (`H^{int}` replaces `H^{+-}`/`H^{pm}`); Fig. 6/7 white-hole runs switched to the manuscript profile `x0 = 2 ell/3` (`j_h ~ 212.81`); manifest tests and REPRODUCE.md updated | Full suite (31 tests) | OK |
