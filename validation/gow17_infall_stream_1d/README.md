# `gow17_infall_stream_1d`

This validation case evolves a 1D gas stream exposed to stellar UV irradiation while the stream front free-falls inward toward the star. The chemistry is advanced with the DiskBridge `Gow17TimeStepper`, and the run produces abundance histories and movies for a density scan.

## Purpose

This setup is a lightweight validation and exploration problem for the Gong et al. (2017) reduced chemistry and thermal network in a geometry that is simpler than a full disk or envelope model. It is intended to answer questions such as:

- How do CO, C, C+, and CHx respond when the incident stellar UV field strengthens during infall?
- How does the response depend on density?
- How sensitive are the results to evolving the gas energy equation versus holding the gas temperature fixed?
- How sensitive are the results to using a fixed dust temperature versus a simple stellar-heating estimate?

## What The Model Does

The implementation lives in [driver.py](./driver.py) and is launched from [run.py](./run.py).

The entrypoint does not allow an arbitrary stellar model. Instead, it supports five fixed pre-main-sequence stellar presets keyed by effective temperature. These are anchored to literature values for real PMS systems chosen as rough matches to the desired temperature bins:

- `4000 K`: anchored to `TW Hya`, using `M_star = 0.6 Msun` and `R_star = 1.22 Rsun`
- `6000 K`: anchored to `AK Sco`, using `M_star = 1.2 Msun` and `R_star = 1.3 Rsun`
- `6750 K`: anchored to `CQ Tau`, using `M_star = 1.5 Msun`, `L_star = 6.7 Lsun`, and `R_star = 1.89 Rsun`
- `7500 K`: anchored to `HD 169142`, using `M_star = 1.6 Msun` and `R_star = 1.6 Rsun`
- `10000 K`: anchored to `AB Aur`, using `M_star = 2.4 Msun` and `R_star = 2.3 Rsun`

Important interpretation:

- The code still uses the nominal bin temperatures `4000 K`, `6000 K`, `6750 K`, `7500 K`, and `10000 K`.
- The associated masses and radii are placeholders taken from nearby real PMS stars in the literature.
- These are not a single coeval evolutionary sequence.
- They are intended only as rough, literature-informed anchor points for cool, intermediate, and hotter PMS irradiation cases.
- The preset table in `run.py` also records literature accretion rates, and the current driver uses `mdot` to add a simple accretion-hotspot UV contribution on top of the photospheric blackbody.

For each density in `InfallStream1DConfig.nH_list_cm3`, the code:

1. Builds a 1D Cartesian stream of length `stream_length_au` with `n_cells` cells.
2. Sets a uniform hydrogen nuclei density `nH`.
3. Converts `nH` to gas mass density using `1.4 m_H`.
4. Assigns a constant dust cross section per H nucleus:
   `sigma_d_per_H = 1 / (1.086 * 1.87e21) cm^2`.
5. Computes the line-of-sight column from the illuminated front:
   `N_H(x) = nH * x`.
6. Converts that column to visual extinction using:
   `A_V = N_H / 1.87e21`.
7. Initializes the gas temperature to `Tgas_init_K`.
8. Initializes the dust temperature either:
   - to a constant `Tdust_const_K`, or
   - from a simple grey estimate if `estimate_tdust=True`.
9. Computes the stellar UV luminosity in the band
   `912-2067 Angstrom` from the photospheric blackbody and, if `mdot_msun_yr > 0`, an added accretion hotspot component using the same prescription as DiskBridge's broader UV machinery.
10. Converts that UV luminosity into the local incident UV field at each cell:
    `chi = F_UV / (c U_Draine)`.
11. Multiplies the incident field by 2 before storing it in `rad.chi`.
12. Builds a fixed output/history cadence with `n_steps + 1` times spanning the start of infall to the stopping radius, biased toward late times by `output_time_power`.
13. Evolves the chemistry and, if enabled, the gas temperature with
    `Gow17TimeStepper` on a separate adaptive substep sequence between those output times.
14. Limits each adaptive chemistry substep so the illuminated front cell changes by at most `max_dlnchi` in `ln(chi)`, with midpoint evaluation of the UV field and optional grey dust temperature during that substep.
15. Recomputes the stellar UV field and optional grey dust temperature at the end of each accepted adaptive step as the front moves inward.
16. Records abundance histories, front temperatures, and diagnostic summaries only on the fixed output grid.
17. Optionally solves a second branch to local chemical/thermal equilibrium at each recorded infall snapshot to compare against the time-dependent solution.

Before the infall evolution, the code can also perform an initial fixed-environment relaxation (`init_relax=True`) to bring the chemistry closer to a local steady state at the starting irradiation level.
Its relaxation timestep is tied to the mean infall output interval, not the smallest late-time output spacing, so changing `output_time_power` does not accidentally collapse the relaxation timespan.
The init-relax movie does not record every chemistry step. It records only a bounded set of snapshots, biased toward early relaxation, so the movie stays within the configured runtime limit without generating thousands of unnecessary frames.

This initial relaxation is distinct from the optional infall-equilibrium tracking described below. `init_relax` changes only the starting point of the time-dependent run, while the equilibrium track re-solves the chemistry at each recorded infall snapshot under the instantaneous local environment.

## Radiation And Temperature Treatment

The UV irradiation model in this validation is intentionally simple and fully prescribed. The star is not represented by a stellar atmosphere model, line spectrum, or observed SED. Instead, the code uses a spherical photospheric blackbody plus an optional simple accretion-hotspot blackbody when `mdot_msun_yr > 0`, and computes the UV field seen by each cell from geometric dilution.

### Stellar UV Field

The stellar radiation source is defined by:

- `teff_K`: the stellar effective temperature
- `rstar_rsun`: the stellar radius
- `mdot_msun_yr`: the accretion rate
- `accretion_fill_factor`: the fractional stellar surface area covered by the hotspot

The run script supports two radiation modes:

- `draine_scalar`: the historical validation setup. It converts the broad
  stellar `912-2067 Angstrom` energy flux into a single Draine-normalized
  scalar and gives that same Draine-shaped field to all GOW17 photochannels.
- `stellar_products`: the corrected setup. It keeps the same 1D incident-slab
  columns and dust attenuation, but supplies separate stellar-spectrum UV
  products for CO dissociation, H2 dissociation, C ionization, and CO
  photodesorption.

The old saved validation outputs under `out/energy_0_tdust_0_eqtrack_1_*`
were produced with `draine_scalar`. Corrected outputs are written with a
`_rad_stellar_products` suffix, so they do not overwrite those old runs.

In the current command-line workflow, these are not chosen independently. Selecting one of the supported temperatures from `run.py` automatically selects a fixed PMS-style radius, mass, and accretion rate. So when you pick `4000 K`, `6000 K`, `6750 K`, `7500 K`, or `10000 K`, you are choosing a full stellar preset, not only a temperature.

The temperature sets the spectral shape of the photospheric blackbody, while the radius sets the emitting area and therefore the luminosity normalization. If `mdot_msun_yr > 0`, the model also adds a hotspot UV component whose luminosity scales with accretion power and whose emitting area is set by `accretion_fill_factor`. Changing from one preset to another therefore changes both the hardness of the stellar spectrum and the total UV output in the selected band.

The code converts the stellar radius into cgs units with:

`R_star = rstar_rsun * R_sun`

and then treats the stellar photosphere as a blackbody at temperature `T_star = teff_K`.

If `mdot_msun_yr > 0`, the code also computes an accretion luminosity

`L_acc = G M_star mdot / R_star`

assigns it to a hotspot with radius

`R_acc = sqrt(accretion_fill_factor) R_star`

and treats that hotspot as an additional blackbody with

`T_acc = [L_acc / (4 pi sigma_SB R_acc^2)]^(1/4)`.

The UV field used by the chemistry is not the full bolometric luminosity. Instead, the code integrates only over a fixed ultraviolet wavelength interval:

- `uv_lam_min_cm = 9.12e-6 cm` = `912 Angstrom`
- `uv_lam_max_cm = 2.067e-5 cm` = `2067 Angstrom`

This is done with `planck_band_luminosity(R_star, T_star, lam_min, lam_max, n_points=800)`, which returns the stellar luminosity in that wavelength band:

`L_UV = integral_{lambda_min}^{lambda_max} L_lambda d lambda`

for a blackbody emitter of radius `R_star` and temperature `T_star`.

This means the UV source strength depends on all of the following:

- `teff_K`, through the blackbody shape and normalization
- `rstar_rsun`, through the emitting area
- `mdot_msun_yr`, through the accretion luminosity
- `accretion_fill_factor`, through the hotspot area and temperature
- `uv_lam_min_cm` and `uv_lam_max_cm`, because only radiation inside that band contributes

The stellar mass `mstar_msun` does not enter the blackbody luminosity directly, but it does enter the prescribed free-fall dynamics of the stream front. So changing the stellar preset changes both:

- the irradiation through `teff_K` and `rstar_rsun`
- the infall timing through `mstar_msun`

After the band-limited luminosity is computed, the code assumes the radiation propagates freely from the star with inverse-square dilution. At distance `r` from the star, the band-integrated UV flux is:

`F_UV = L_UV / (4 pi r^2)`

No scattering, limb darkening, shadowing, diffuse re-emission, or frequency-dependent attenuation is included in this step. It is a pure geometric dilution from a point-like central source.

The flux is then converted into the dimensionless UV-field strength `chi` using the Draine interstellar radiation field normalization:

`chi = F_UV / (c U_Draine)`.

Here:

- `c` is the speed of light
- `U_Draine` is the energy density of the Draine (1978) UV field

Since `F = c U` for a freely streaming radiation field, this conversion expresses the local stellar UV flux as a multiple of the standard Draine interstellar UV field. So:

- `chi = 1` means the local UV flux equals the Draine field
- `chi = 10` means it is ten times stronger
- `chi = 1e4` means it is four orders of magnitude stronger

In the current implementation, the computed `chi` is then floored at a configurable minimum value:

`chi_used = max(chi, min_chi)`

with default `min_chi = 0.1`.

This floor is applied before the field is passed to the chemistry. It prevents the initial equilibrium calculation and subsequent evolution from seeing an incident UV field weaker than `0.1` times the Draine field.

### How Distance Is Defined In This Setup

The star is not placed on the grid. Instead, the grid represents a 1D stream segment located some distance away from the star.

Let:

- `r_face_cm` be the distance from the star to the illuminated front of the stream
- `x` be the depth into the stream measured from that front

Then the stellar distance to a cell is:

`r_cell = r_face_cm + x`

This is why the front cell always sees the strongest UV field and deeper cells see weaker direct stellar irradiation even before extinction effects enter the chemistry.

The front position is not fixed. During the run it moves inward from `r_face_start_au` to `r_face_stop_au` according to the prescribed free-fall law. As `r_face_cm` decreases, every cell moves closer to the star, so the geometric UV flux rises with time roughly as `r^-2`.

### What The Chemistry Actually Receives

After computing the cell-by-cell `chi`, the code stores:

`rad.chi = 2 * chi_used`

The factor of 2 is part of the present implementation. It is not derived in the README because the driver itself does not document a physical justification for it. For now, it should be treated as a model-specific normalization choice used by this validation setup.

The chemistry and shielding routines therefore see a UV field that depends on:

- the stellar temperature `teff_K`
- the stellar radius `rstar_rsun`
- the chosen UV wavelength band
- the current front distance from the star
- the cell depth into the stream
- the minimum field floor `min_chi`
- the factor-of-2 normalization applied before assigning `rad.chi`

### Extinction And Shielding

The geometric stellar UV calculation described above gives the incident field before any local chemistry-dependent attenuation is applied.

Separately, the model assigns each cell a visual extinction:

`A_V(x) = N_H(x) / 1.87e21`

with

`N_H(x) = nH * x`

measured from the illuminated front. This is a simple slab estimate for the column density into the stream.

That `A_V` is then used by the Gow17 chemistry and shielding treatment, and also by the optional heuristic dust-temperature estimate. In other words:

- geometric dilution from the star sets the incident UV field level
- the local `A_V` profile sets how chemistry and the dust-temperature estimate respond to depth inside the stream

### Consequences Of The Blackbody Assumption

Because the stellar source is a blackbody, the model makes several simplifying assumptions:

- The star is characterized only by radius and effective temperature.
- Spectral lines, chromospheric emission, and non-thermal components are ignored.
- Accretion UV excess is represented only through a simple blackbody hotspot, not a detailed shock-spectrum model.
- The UV spectrum is determined entirely by Planck’s law.
- Two stars with the same `teff_K` but different radii have the same spectral shape but different UV luminosities.
- Two stars with the same radius but different `teff_K` can have dramatically different UV luminosities because the short-wavelength tail of the Planck function is highly temperature sensitive.

This matters in practice: the chemistry may respond very strongly to `teff_K`, not just because the total luminosity changes, but because the fraction of stellar power that falls inside the adopted UV band changes rapidly with temperature.

### Dust Temperature Options

If `estimate_tdust=False`, the dust temperature is fixed to `Tdust_const_K`.

If `estimate_tdust=True`, the code uses:

`T_thin = T_star * sqrt(R_star / (2 r))`

followed by an attenuation factor:

`T_dust = T_thin * exp(-tau / 4)`

with `tau = A_V / 1.086`, and finally floors the result at `T_CMB`.

Interpretation:

- The `T_thin` term is a standard grey, optically thin radiative-equilibrium scaling for grains heated by a central source.
- The `exp(-tau / 4)` factor is a heuristic attenuation of the absorbed stellar flux, not a full dust radiative-transfer solution.
- This temperature estimate is appropriate as a simple validation-level prescription, not as a high-fidelity dust thermal model.

## Chemistry And Thermal Evolution

Chemistry is evolved with `diskbridge.chemistry.models.gow17_timestep.Gow17TimeStepper`, which uses the reduced network implemented in `diskbridge._gow17`.

The tracked outputs highlighted by this validation are:

- `CO`
- `C`
- `C+`
- `CHx`

The model can be run in two modes:

- `evolve_energy=0`: gas temperature is effectively held fixed.
- `evolve_energy=1`: the Gong et al. thermal balance machinery is allowed to evolve the gas temperature.

The default chemistry options passed into `Gow17TimeStepper` include:

- `shielding_outer_1d = "min"`
- `b_kms = 0.3`
- `ion_rate = "2e-16 s^-1"`
- `Zg = 1.0`
- `enable_co_phase = False`

Cooling and chemistry details beyond this wrapper are determined by the DiskBridge Gow17 implementation.

## Geometry And Dynamics

This is not a self-consistent hydrodynamics calculation. The geometry and motion are prescribed:

- The stream is a 1D slab along `x`.
- The illuminated face is at `x = 0`.
- The star is located outside the grid, at distance `r_face_cm + x`.
- The front position moves inward from `r_face_start_au` to `r_face_stop_au`.
- The motion uses a point-mass free-fall relation around a star of mass `mstar_msun`.

The density inside the stream does not change during the run. The model is therefore best viewed as a chemistry-and-irradiation experiment with prescribed kinematics.

### Infall Timestep Control

During the infall phase, the code now separates the stored output cadence from the actual chemistry timestep.

- `n_steps` sets only the number of recorded output intervals used for `history.npz` and the abundance movie.
- `output_time_power` sets how strongly those recorded outputs are concentrated toward the end of the infall.
- The chemistry solver takes as many internal adaptive substeps as needed between two neighboring output times.

For each adaptive chemistry step, the driver:

1. Uses the current free-fall radius to estimate how fast the illuminated front cell's incident UV field is changing.
2. Chooses a substep so that the front cell changes by at most `max_dlnchi` in natural-log UV field, `ln(chi)`.
3. If the front cell is still below the imposed `min_chi` floor, allows the solver to jump directly to either the floor-crossing time or the next output time.
4. Evaluates the UV field and optional grey dust temperature at the midpoint of the accepted substep and holds that environment fixed while `Gow17TimeStepper` advances the chemistry.
5. Refreshes the end-of-step environment before starting the next adaptive step or recording output.

With `u` uniformly spaced from 0 to 1, the recorded times are built as

`t_output = t_end * [1 - (1 - u)^(output_time_power)]`.

So:

- `output_time_power = 1` gives uniform spacing in time
- `output_time_power > 1` clusters outputs toward the end of infall
- larger values put progressively more movie/history frames into the fast, high-UV final phase

This means `n_steps` controls how many history and movie samples are recorded, `output_time_power` controls where those samples are concentrated in time, and `max_dlnchi` controls how accurately the time-dependent irradiation is integrated.

### Optional Infall Equilibrium Track

If `track_infall_equilibrium=True`, the code also carries a second chemistry branch during the infall.

- The ordinary branch is the real time-dependent solution, advanced only by the physical elapsed time between snapshots.
- The optional branch takes the same density structure, `A_V`, UV field, dust temperature, and gas-energy configuration at each recorded infall snapshot and solves to local equilibrium instead.
- During this equilibrium solve, shielding is always iterated with the chemistry. There is no frozen-shielding single-pass mode in this validation.
- Concretely, the tracker uses the Gow17 `astrochem` coupling strategy with convergence control: it performs at least `20` pseudo-time chemistry macro-updates by default, keeps refreshing self-shielding from the current abundances, and continues up to `100` shielding updates if needed until the shielding-coupled abundances stop changing at the configured tolerance.
- Only after that shielding-coupled warm-up does it run the final equilibrium solve using the last updated shielding field.
- These shielding refreshes happen once per macro-update. The current CVODE wrapper does not expose or control the solver's internal substep count directly, so this is not expressed as "every 100 internal chemistry timesteps."

This produces a snapshot-by-snapshot comparison between:

- the transient chemistry actually reached during infall
- the chemistry that would be present if the gas had enough time to equilibrate at that same instantaneous irradiation level

The purpose of this branch is diagnostic. It lets you estimate how far from local equilibrium the infalling stream remains as the UV field strengthens.

## Running The Case

Example:

```bash
python validation/gow17_infall_stream_1d/run.py \
  --evolve-energy 1 \
  --estimate-tdust 1 \
  --track-infall-equilibrium 1 \
  --teff-k 7500 \
  --min-chi 0.1 \
  --output-time-power 3.0 \
  --max-dlnchi 0.05
```

Supported values for `--teff-k` are only:

- `4000`
- `6000`
- `6750`
- `7500`
- `10000`

Each value selects the corresponding PMS stellar preset listed above.

Smaller `--max-dlnchi` values make the adaptive infall integration more accurate and more expensive. Larger `--output-time-power` values shift more recorded frames toward the end of the infall. Changing `n_steps` changes only how many output/movie samples are stored.

The run directory created under `out/` is:

```text
energy_<0|1>_tdust_<0|1>_eqtrack_<0|1>_teff_<temperature>K
```

Runs using `--radiation-mode stellar_products` append
`_rad_stellar_products` to that directory name.

Corrected stellar-spectrum example without mp4 generation:

```bash
python validation/gow17_infall_stream_1d/run.py \
  --evolve-energy 0 \
  --estimate-tdust 0 \
  --track-infall-equilibrium 0 \
  --teff-k 6750 \
  --radiation-mode stellar_products \
  --make-movies 0
```

Compare an old scalar run against a corrected run:

```bash
python validation/gow17_infall_stream_1d/compare_runs.py \
  --old-dir validation/gow17_infall_stream_1d/out/energy_0_tdust_0_eqtrack_1_teff_6750K \
  --new-dir validation/gow17_infall_stream_1d/out/energy_0_tdust_0_eqtrack_0_teff_6750K_rad_stellar_products
```

## Output Files

Each top-level run directory contains:

- `inputs.json`: full serialized configuration used for the run
- `scan_summary.json`: summary metrics for each density
- `density_scan_co_change.png`: relative CO abundance change versus density

Each density subdirectory contains:

- `history.npz`: output-cadence histories for abundances, front position, `chi`, and temperatures, plus internal adaptive-step diagnostics (`t_internal_s`, `dt_internal_s`, `r_face_internal_cm`, `chi_face_internal`)
  If `track_infall_equilibrium=True`, this file also includes `xco_eq_hist`, `xc_eq_hist`, `xcp_eq_hist`, `xchx_eq_hist`, `Tgas_front_eq`, and `Tdust_front_eq`.
- `abundances_init_relax.mp4`: movie of the initial relaxation phase, if enabled
  This movie records only a bounded set of relaxation snapshots and is capped to the configured runtime limit (`20 s` by default).
- `abundances_vs_time.mp4`: movie of the infall evolution
  If `track_infall_equilibrium=True`, the equilibrium branch is overplotted on the same frames with dashed lines while the time-dependent branch remains solid.
  This movie is subsampled as needed at render time so its runtime stays within the configured limit (`20 s` by default).

If the run fails, the top-level output directory receives:

- `error.png`: exception summary and traceback

## Important Assumptions And Limitations

- The model is 1D and does not solve multidimensional radiative transfer.
- The UV field is geometric dilution of a blackbody in a fixed wavelength band.
- The factor of 2 applied to `chi` is part of the current implementation and should be treated as a modeling choice specific to this validation.
- The dust temperature estimate is grey and heuristic.
- The stream density is static while only the front radius changes.
- The model is designed for comparative tests and diagnostics, not precision dust thermal structure predictions.

## Suggested Citations

The following references are relevant to the physics approximations used here.

### Reduced Chemistry And Thermal Network

- Gong, M., Ostriker, E. C., & Wolfire, M. G. 2017, ApJ, 843, 38, doi:10.3847/1538-4357/aa7561

Use this as the primary citation for the reduced carbon-oxygen chemistry and associated thermal treatment behind the `Gow17` implementation.

### Draine UV Field Normalization

- Draine, B. T. 1978, ApJS, 36, 595, doi:10.1086/190513

Use this for the interstellar UV field normalization underlying `chi` and `U_Draine`.

### Column Density To Extinction Conversion

- Bohlin, R. C., Savage, B. D., & Drake, J. F. 1978, ApJ, 224, 132, doi:10.1086/156357

Use this for the standard Milky Way conversion `N_H / A_V = 1.87e21 cm^-2 mag^-1`.

### Grey Optically Thin Dust Temperature Scaling

- Spitzer, L. 1978, *Physical Processes in the Interstellar Medium*, Wiley
- Kuiper, R., Klahr, H., Beuther, H., & Henning, T. 2013, A&A, 555, A81, doi:10.1051/0004-6361/201321404

The Spitzer reference is the classical source for radiative-equilibrium grain heating arguments. The Kuiper et al. discussion is a convenient modern reference for the `T(r) proportional r^-1/2` optically thin grey scaling around a point source.

### Pre-Main-Sequence Stellar Context

- Baraffe, I., Homeier, D., Allard, F., & Chabrier, G. 2015, A&A, 577, A42, doi:10.1051/0004-6361/201425481
- Siess, L., Dufour, E., & Forestini, M. 2000, A&A, 358, 593

These references are appropriate background for the statement that PMS stars do not obey a single universal `M_star -> (R_star, T_eff)` mapping independent of age. The preset stellar parameters used here are only approximate representative PMS values; they are not taken from a single fixed-age evolutionary track.

### Literature Anchors For The Placeholder Stellar Presets

- `TW Hya` for the cool bin: the `4000 K` preset uses `M_star = 0.6 Msun` and `R_star = 1.22 Rsun` as a TW Hya-like T Tauri anchor.
- `AK Sco` for the warmer intermediate bin: the `6000 K` preset uses `M_star = 1.2 Msun` and `R_star = 1.3 Rsun` as an AK Sco-like placeholder.
- `CQ Tau` for the intermediate Herbig-like bin: the `6750 K` preset uses `M_star = 1.5 Msun`, `L_star = 6.7 Lsun`, and `R_star = 1.89 Rsun`.
- `HD 169142` for the hotter Herbig-like bin: the `7500 K` preset uses `M_star = 1.6 Msun` and `R_star = 1.6 Rsun`.
- `AB Aur` for the hottest bin: the `10000 K` preset uses `M_star = 2.4 Msun` and `R_star = 2.3 Rsun`.

The preset table in [run.py](./run.py) also stores literature accretion-rate estimates for these stars, and [driver.py](./driver.py) now uses those values to include the simple accretion-hotspot UV component.

## Recommended Wording

If you describe this validation in a note, talk, or paper, the most accurate summary is:

"Dust temperatures were either held fixed or estimated with a grey, optically thin radiative-equilibrium scaling relative to the stellar blackbody, with an additional ad hoc extinction attenuation applied to the absorbed stellar flux. The chemistry and optional gas thermal evolution were computed with the Gong et al. reduced network."
