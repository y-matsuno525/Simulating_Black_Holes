# Run pattern audit from the archived CSV data

This note records what can be inferred directly from the files under
`paper_data/`.  It intentionally does **not** assume that the text in
`paper/main.tex` is already consistent with those archived runs.

## Key correction

The archived `paper_data` runs do not match a simple reading of the current
White-hole prose in `paper/main.tex`.

For example, the current paper text says the White-hole packet uses
`j0=0.2L=60` and a profile like

```tex
beta(x) = -0.6 tanh[3(x - 7 pi/5)] - 0.6 .
```

But the archived `paper_data/p{0,1}/ur_p` and `ur_m` profiles have their
initial energy-density peaks near `j0 ~= 180`, not 60.  Their geodesic files
also pass through `j ~= 180` at `t=0`.

Therefore, do not use the manuscript prose alone as the source of truth for
regenerating `paper_data` figures.  The archived data must be treated as a
separate set of runs until their original configs are recovered or the runs are
recomputed.

## Directly observed initial centers

The following centers are estimated from the absolute value of the `t=0`
profile in each archived CSV.  The geodesic center is obtained by interpolating
`geodesic/*.csv` to `t=0`.

| data folder | dominant observable | profile center at `t=0` | geodesic `j(t=0)` |
| --- | --- | ---: | ---: |
| `p0/lr_p` | `H_p` | 58.09 | 60.06 |
| `p0/ur_p` | `H_p` | 181.20 | 179.88 |
| `p0/ur_m` | `H_m` | 181.57 | 179.88 |
| `p0/ur_p_2` | `H_p` | 58.54 | 60.06 |
| `p1/lr_p` | `H_p` | 59.98 | 60.06 |
| `p1/ur_p` | `H_p` | 180.55 | 179.88 |
| `p1/ur_m` | `H_m` | 179.56 | 179.88 |
| `p1/ur_p_2` | `H_p` | 59.98 | 60.06 |

This already shows that `ur_p` and `ur_m` are not the `j0=60` White-hole runs
described in the current manuscript text.

## Horizon constants currently used by figure scripts

The archived figure scripts use

```python
J_BH = 2/3 * L + arctanh(2/3) / (3 eps)  # ~= 212.82
J_WH = 1/3 * L - arctanh(2/3) / (3 eps)  # ~= 87.18
```

The `J_WH ~= 87.18` value is therefore an archived-data convention, not the
same as the manuscript candidate profile
`beta=-0.6*tanh[3(x-7*pi/5)]-0.6`, whose `beta=-1` crossing would be near
`j ~= 222.82`.

## What should be fixed next

1. Decide the source of truth for the paper figures:
   existing archived `paper_data`, or newly recomputed runs matching the
   manuscript prose.

2. If using archived `paper_data`, update `paper/main.tex` so its stated
   `j0`, beta profile, and horizon positions match the archived data.

3. If using manuscript prose, regenerate all affected CSV/geodesic files from
   explicit configs and save each run's `config.json` next to the data.

4. Add a `paper_data/manifest.csv` or `paper_data/manifest.json` with, at
   minimum: folder, observable, `p`, `m`, beta formula, `j0`, `sigma`, branch,
   boundary condition, horizon line used for the figure, and the intended paper
   panel.

5. Until that manifest exists, do not infer physics from names such as
   `ur_p`, `ur_m`, `lr_p`, or `ur_p_2` alone.

## Code status

`config.compute_horizon_positions_from_config()` currently returns metric
horizons where `abs(beta)=1`, preserving the archived plotting convention.
Target-specific helper functions are available for diagnostics:

```python
compute_target_horizon_positions_from_config(config, target_beta)
horizon_target_beta_from_config(config)
```

They should not replace the archived plotting convention until the intended
physical convention is fixed explicitly.
