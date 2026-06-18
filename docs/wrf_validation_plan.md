# Validation plan: anchoring the channeled turbine wake with WRF

## Why
Everything downstream of the wake — the marginal shielding (~13% median),
the channeling fraction `f` (0.60–0.92), the "4–6×" persistence claim, the
economics — rests on the **turbine deficit in the channeled regime**, which
is currently **unvalidated**. `calibrate.py` validates only the *no-turbine*
baseline against CARRA; `validation.py` (Horns Rev) validates only the
open-ocean `f=0` limit. A mesoscale run with an explicit wind-farm
parameterization is the single highest-credibility way to close this gap.
The sensitivity analysis (`sensitivity.py`) shows the result hinges most on
`H_eff` and the wake-superposition rule — both of which WRF would resolve
directly rather than assume.

## Approach
**Model:** WRF-ARW (≥ v4.5) with the **Fitch et al. (2012) wind-farm
parameterization** (`windfarm_opt=1`), which adds turbine drag + TKE as a
sink in the momentum equations — the mesoscale-appropriate analogue of the
1D blockage term. (A WRF-LES nest, ~50–100 m, is the higher-fidelity but
much costlier alternative; recommended only if the Fitch run is ambiguous.)

**Domains (one-way nest):**
- d01 9 km (synoptic, ~N-Iceland), d02 3 km, d03 **1 km** over Eyjafjörður.
- ≥ 50 vertical levels, ~10 below 200 m to resolve the channeled layer and
  the inversion the model assumes.
- Real terrain from the project DEM (`data/eyjafjordur_dem.tif`) ingested via
  WPS `geogrid` (replace default topo in d03).

**Forcing / setup:** ERA5 (or CARRA) IC/BC; MYNN or YSU PBL (MYNN pairs with
Fitch); simulate **3–5 documented norðanátt events** spanning stable→very-
stable (select from the 1,716 mouth-anchored events in `observed.py`, e.g.
the strongest deciles), 48 h each with 12 h spin-up.

**Turbine layout:** the JAFNVÆGI / SAMSETT rows (`designs.py`, `rows_of`),
turbine Ct/power curves from `TurbineSpec`. Run each event **with and
without** the farm.

## What to compare (the model plugs straight in)
For each event, extract WRF hub-height wind along the fjord axis (the 7
`observed.py` waypoints) and at Akureyri:
1. **Baseline profile** — WRF no-farm vs the calibrated `u(x)=u0·exp(-(x-x0)/Λ)`
   (cross-check the Λ≈84 km fit).
2. **Marginal reduction at Akureyri** — WRF (with−without) vs the 1D
   `marginal_reduction()`; this is the key number.
3. **Implied `f` and `L`** — fit the 1D model to the WRF deficit profile and
   compare the recovered `f`, `L` to the presets (0.60–0.92, 30–80 km) and to
   the Froude/ridge-N estimate (`ridge_stability.py`).
4. **Superposition** — does the multi-row WRF deficit combine closer to
   `product` or `sos`? (Resolves the dominant UQ axis.)

## Acceptance criteria
- 1D baseline within ±1 m/s of WRF no-farm along the axis (matches current
  CARRA R²≈0.9).
- 1D marginal Akureyri reduction within the WRF value's ±1 P10–P90 band
  (`uncertainty.py`). If WRF lands well below the 1D median, the `f`/`L`
  presets are too high and must be revised downward (and the paper's claim
  softened accordingly).

## Effort / compute
~3–5 events × 2 (with/without) × 3 domains. Order ~a few thousand core-hours
on a modest HPC allocation; days of wall-clock incl. WPS setup. Largest cost
is the WRF/WPS build + DEM ingest, not the runs.

## Deliverables back into the study
- `wrf/` harness (namelists, WPS config, post-processing to the 7-waypoint
  arrays so the existing comparison code can ingest it directly).
- A `wrf_validation.py` that reads WRF output and reuses
  `calibrate.fit_friction` / `model.marginal_reduction` for an apples-to-
  apples comparison.
- A revised §2.5 / §4.3 reporting WRF-anchored `f`, `L`, superposition, and
  (if needed) corrected headline numbers. This converts the "plausibility
  study" framing into a validated result — the main thing a WES reviewer
  will ask for.

## Status: container stood up and verified (2026-06-18)
A working WRF+WPS is now running locally via Docker, sidestepping the
GCC-15/macOS native-build fragility:
- Image `sagunkayastha/wrf:latest` (tagged `windfall/wrf:4.3.3`),
  `/BUILD_WRF`. **WRF V4.3.3** + WPS `geogrid`/`metgrid`.
- **Verified executing under `linux/amd64` emulation on Apple Silicon**:
  `wrf.exe` prints the `WRF V4.3.3 MODEL` banner and initializes (stops
  only on absent met input, as expected). The Fitch wind-farm scheme
  (`windfarm_opt`) is present in 4.3.3.
- **Gaps to close before a run:** (i) `ungrib.exe` is not compiled —
  `cd /BUILD_WRF/WPS && ./compile ungrib` (or a child Dockerfile) is
  needed to ingest CARRA/ERA5 GRIB; (ii) the Docker VM has ~8 GB and runs
  emulated, so it suits a **single small proof case**, not the full
  multi-event 1-km validation (HPC, as above).
- **Next step:** generate the Eyjafjörður run deck (namelist.wps with the
  real DEM, namelist.input with `windfarm_opt=1`, and the Fitch turbine
  table + locations from `designs.py`), then drive one norðanátt case.

## Key references
Fitch et al. (2012, MWR) wind-farm parameterization; Fischereit et al. (2022,
WES) mesoscale-wake validation; Ólafsson & Ágústsson (Icelandic channeled
flow); Schyberg et al. (2020, CARRA).
