# GOW17 Chemistry Model and DiskBridge Extensions

This note describes how the GOW17 PDR/thermochemistry model works in
DiskBridge, how it differs from the original `other_codes/pdr` version, and
which parts of the implementation are tied to later extensions such as
process-specific UV products, 3D shielding, and CO freeze-out/desorption.

The original chemistry model is the reduced hydrogen/carbon network of
Gong, Ostriker, and Wolfire (2017, hereafter GOW17). The intent of that model
is to keep the main PDR transitions - H/H2 and C+/C/CO - accurate enough for
simulation and post-processing work without carrying a full gas-phase chemical
network.

## Code Map

The relevant implementation is split between the original/ported C++ PDR code
and DiskBridge's Python and pybind11 integration:

- `other_codes/pdr/`: local copy of the original GOW17-style C++ code.
- `external/pdr/`: DiskBridge's modified C++ PDR core.
- `src/diskbridge/_gow17.cpp`: pybind11 bindings and batch solvers.
- `src/diskbridge/chemistry/models/gow17.py`: equilibrium and coupled
  shielding/chemistry driver.
- `src/diskbridge/chemistry/models/gow17_timestep.py`: persistent
  time-dependent stepper.
- `src/diskbridge/radmc3d/uv_products.py`: process-specific UV products from
  RADMC-3D mean intensities.
- `src/diskbridge/chemistry/shielding/`: 1D and 3D shielding machinery.
- `src/diskbridge/config.toml`: default GOW17, UV-product, dust, and CO-phase
  settings.

## Original GOW17 Model

### State Variables

The original local code evolves a compact set of abundances per H nucleus plus
an internal-energy variable. In `other_codes/pdr`, the evolved state is:

- `He+`
- `OHx`
- `CHx`
- `CO`
- `C+`
- `HCO+`
- `H2`
- `H+`
- `H3+`
- `H2+`
- `S+`
- `Si+`
- `O+`
- `E`

Several species are "ghost" species: they are not independent ODE variables,
but are reconstructed from conservation and charge balance every RHS call:

- neutral `H`
- neutral `C`
- neutral `O`
- neutral `He`
- neutral `S`
- neutral `Si`
- electrons `e`

DiskBridge preserves this design but adds one evolved state variable,
`CO_ice`, discussed below.

### Reaction Structure

For each cell or slab grid point, the C++ `gow17::RHS` constructs the ODE
right-hand side in a fixed order:

1. Reconstruct ghost species from the current evolved state.
2. Initialize temperature-dependent and radiation-dependent rates.
3. Apply cosmic-ray reactions.
4. Apply two-body gas-phase reactions.
5. Apply photoreactions.
6. Apply the gas energy equation if constant-temperature mode is disabled.
7. Apply grain-assisted reactions.

The native photoreaction rates retain the GOW17 form:

```text
k_ph,i = k_base,i * G_ph,i
```

where `k_base,i` is the Draine-field-calibrated base rate and `G_ph,i` is the
dimensionless radiation field passed into that photoprocess. In the original
slab code, `G_ph,i` is effectively derived from one Draine-normalized incident
field with dust attenuation and molecular shielding.

### Gas Temperature and Thermal Balance

When the model is not run in constant-temperature mode, the evolved variable
`E` stores the thermal energy per H nucleus in the reduced cold-gas heat
capacity approximation. The gas temperature is recovered as:

```text
T_gas = E / CvCold(x_H2, x_He, x_e)
```

The energy equation is:

```text
dE/dt = total heating - total cooling
```

In plain language, `dE/dt` means "how fast the gas thermal energy is changing
right now." If heating is larger than cooling, `dE/dt` is positive and `E`
increases. If cooling is larger than heating, `dE/dt` is negative and `E`
decreases. If heating and cooling are equal, `dE/dt` is zero and the gas is in
thermal balance.

The model does not normally assume thermal balance at every instant. Instead,
it asks: given the current density, radiation field, dust temperature, and
chemical abundances, how much should each abundance and the thermal energy
change over the next small time interval? CVODE then takes many internal
substeps to update the state smoothly and stably.

At the start of a run, `E` is initialized from a temperature guess:

```text
E_initial = T_initial * CvCold(x_H2, x_He, x_e)
```

In computed-temperature mode, `T_initial` usually comes from an existing
`rad.gas_temperature`; if that is absent, DiskBridge falls back to the dust
temperature. If the model is then initialized in equilibrium, the solver
evolves `E` until heating and cooling are approximately equal for that initial
environment. After that, during time-dependent evolution, `E` is carried
forward from the previous step and continues to evolve according to
`heating - cooling`.

The implemented heating terms include:

- cosmic-ray heating
- photoelectric heating
- H2 formation heating
- H2 UV pumping
- H2 photodissociation heating

The implemented cooling terms include:

- CII fine-structure cooling
- CI fine-structure cooling
- OI fine-structure cooling
- Ly-alpha cooling
- CO rotational cooling
- H2 rovibrational cooling
- gas-dust energy exchange
- electron recombination cooling on grains/PAHs
- H2 collisional dissociation cooling
- HI collisional ionization cooling

### Native Numerical Solver

The C++ model uses CVODE from SUNDIALS with a BDF/Newton method and a dense
linear solver. There are two native solve modes:

- `Solve(t_end)`: integrate the time-dependent ODE system to a requested
  absolute time.
- `SolveEq(...)`: repeatedly integrate forward and test abundance changes until
  an equilibrium residual criterion is met or the maximum pseudo-evolution time
  is reached.

The DiskBridge bindings expose both modes in batch form.

The word "ODE" means ordinary differential equation. In this context it is
just a set of rules for rates of change. For example, the chemistry solver does
not directly say "the CO abundance is X"; it says "given the current state, CO
is being created and destroyed at these rates." The numerical solver uses
those rates to advance the state in time.

For a single cell, the state vector contains all evolved abundances plus `E`.
Schematically:

```text
state = [He+, OHx, CHx, CO, C+, ..., CO_ice, E]
```

At each internal CVODE step, GOW17 evaluates:

```text
d(state)/dt = rates(state, local_environment)
```

where `local_environment` contains density, UV fields, shielding, dust
temperature, metallicity/dust scalings, and cosmic-ray ionization rate. CVODE
then decides how large a step it can safely take, updates the state, and
repeats until it reaches the requested time or equilibrium criterion.

`Solve(t_end)` and `SolveEq(...)` differ in what they ask CVODE to do:

- `Solve(t_end)` means "advance this cell for this physical amount of time."
  This is the mode used by the time stepper.
- `SolveEq(...)` means "keep advancing until the state is barely changing."
  If it converges, heating and cooling should be nearly balanced and chemical
  production/loss rates should also nearly cancel.

## DiskBridge Extensions Beyond the Original Code

### 1. Python Bindings and Batch Cell Solvers

The original code is mostly a standalone C++ slab model. DiskBridge adds a
pybind11 module, `_gow17`, which exposes the native solver to Python and allows
many independent cells to be solved in one call.

The main exported native entry points are:

- `solve_slab_1d_equilibrium`
- `solve_batch_equilibrium`
- `solve_batch_time`
- `eval_rhs_batch`

The batch solvers accept arrays for density, gas temperature, dust temperature,
metallicity, dust abundance, UV products, shielding products, velocity
gradient, CO freeze-out parameters, and solver tolerances. The C++ wrapper
runs cells independently with OpenMP.

This is a major architectural change: GOW17 is no longer only a slab model, but
also a local cell physics kernel driven by DiskBridge geometry, RADMC-3D
radiation fields, and external timesteps.

### 2. Persistent Time-Dependent Solver

DiskBridge adds `Gow17TimeStepper` in
`src/diskbridge/chemistry/models/gow17_timestep.py`.

This class stores the current GOW17 state on the `RadModel`, refreshes
shielding and UV fields from the current abundances, and advances the native
ODE system by a caller-supplied `dt_s`.

The time-dependent step is:

1. Read current `rad.chi`, UV products, dust temperature, and gas temperature.
2. Project the current chemical state back onto elemental budgets.
3. Recompute shielding from the current `H2`, `CO`, and neutral `C`.
4. Assemble the process-specific native `Gph` array.
5. Call `_gow17.solve_batch_time(..., t_end=dt_s)`.
6. Repair failed cells if possible and write updated abundances and `Tgas`.

Within one native chemistry step, the radiation/shielding environment is held
fixed; it is refreshed on the next external step. 

### 3. CO Ice, Freeze-Out, and Desorption

DiskBridge extends the original GOW17 state from 14 to 15 variables by adding
`CO_ice`. This is implemented in `external/pdr/gow17.{h,cpp}` and exposed to
Python as `I_CO_ICE`.

The reason for this extension is that, in cold dense gas, CO does not always
stay in the gas. CO molecules can hit dust grains and stick to them as ice. If
the dust warms up, or if UV photons hit the ice, some of that CO can return to
the gas. The code represents this by moving carbon between gas-phase `CO` and a
single ice reservoir called `CO_ice`:

```text
CO_gas -> CO_ice       freeze-out
CO_ice -> CO_gas       thermal desorption
CO_ice -> CO_gas       UV photodesorption
CO_ice -> CO_gas       optional cosmic-ray desorption
```

Here "freeze-out" means adsorption onto dust grains. "Desorption" means leaving
the grain surface and returning to the gas. "Photodesorption" is desorption
caused by UV photons. The model does not try to follow detailed ice chemistry,
different ice layers, or different ice mixtures; all frozen CO is stored in one
number, `CO_ice`.

The helper functions live in `external/pdr/co_phase.h`. DiskBridge's CO phase
model is intentionally reduced: it borrows rate forms and parameter values from
gas-grain PDR models, meaning photodissociation-region models, and
ice-desorption literature, but it tracks only
`CO <-> CO_ice`. It does not implement the full water/oxygen/carbon
grain-surface network of Hollenbach et al. (2009), nor the full dense-cloud
gas-grain networks of Hasegawa et al. (1992) and Hasegawa & Herbst (1993).

The rates are written as rate coefficients. For example, if
`k_fo = 1e-10 s^-1`, then each gas-phase CO molecule has a probability of about
`1e-10` per second of freezing onto a grain. The corresponding timescale is
roughly `1 / k_fo`. Large rates mean fast exchange between gas and ice.

Here is where each piece comes from and what it means physically:

- **Gas-grain collision / freeze-out form.** Hollenbach et al. (2009) define
  freeze-out as gas particles striking grains and sticking, with freeze-out
  time `t_f,i ~ [n_gr sigma_gr v_i]^-1` and near-unity sticking for heavy
  species at low gas temperature. DiskBridge uses the same collision-rate idea
  as a first-order gas-phase CO loss rate:

  ```text
  k_fo = S_CO * sigma_d_per_H * n_H * v_th(CO, T_gas)
  ```

  The pieces of this formula have direct meanings. `n_H` says how much gas is
  present. `sigma_d_per_H` says how much dust cross-sectional area is available
  per H nucleus. `v_th` is the thermal speed of CO molecules, so warmer gas
  gives more grain impacts per second. `S_CO` is the sticking probability: if
  `S_CO = 1`, every CO molecule that hits a grain sticks.

  Hollenbach et al. used an MRN-motivated projected grain area around
  `2e-21 cm^2/H`; MRN is a standard interstellar grain-size distribution.
  DiskBridge can instead use the RADMC/DiskBridge dust surface-area estimate or
  a configured constant, which is the part that makes the freeze-out rate
  respond to the actual model dust. Bisschop et al. (2006) measured lower
  limits to low-temperature CO sticking probabilities close to unity,
  consistent with the default `S_CO = 1`.

- **Projected area versus total grain surface area.** Hollenbach et al. (2009)
  explicitly distinguish projected grain cross section from the full surface
  area. The projected area is the geometric target seen by an incoming gas
  molecule or photon. The full surface area is the area where molecules can sit
  after they are on the grain. For a spherical grain, the full surface area is
  four times the projected area. DiskBridge uses this factor of four when
  estimating how many CO molecules can sit in the UV-active surface layers:

  ```text
  n_active,max = 4 * sigma_d_per_H * n_H * N_surf * N_lay
  ```

  `N_surf` is the number of adsorption sites per unit grain surface area. One
  adsorption site is roughly one place where one molecule can sit. `N_lay` is
  the number of ice monolayers treated as UV-active. A monolayer means one
  molecule-thick layer of ice.

  In the current native implementation, the CO phase block is applied only when
  `sigma_d_per_H > 0` and `N_lay > 0`; setting `N_lay = 0` therefore turns
  off the implemented CO freeze-out/desorption terms rather than only removing
  the photodesorption active layer.

- **Thermal desorption form.** Hasegawa et al. (1992) use the standard
  first-order thermal evaporation form for surface species, and Hollenbach et
  al. (2009) write the same surface-molecule form as
  `R_td,i = nu_i exp(-E_i/kT_gr)`. DiskBridge uses that first-order form for the
  whole tracked `CO_ice` reservoir:

  ```text
  k_td = nu0_CO * exp(-E_bind_CO / T_dust)
  ```

  This says that warm dust releases CO ice much faster than cold dust. The
  exponential is the important part: a small increase in `T_dust` near the
  temperature where CO changes from mostly ice to mostly gas, often called the
  CO snowline, can make thermal desorption much faster. `E_bind_CO` is the
  binding energy, written in Kelvin, and measures how tightly CO is held to the
  surface. `nu0_CO` is the vibrational attempt frequency, meaning the rate at
  which a surface CO molecule samples the possibility of escaping.

  The default `nu0_CO = 1e12 s^-1` is the usual order-of-magnitude vibrational
  attempt frequency used in gas-grain chemistry. The default
  `E_bind_CO = 855 K` is specifically from the pure-CO temperature-programmed
  desorption fits of Bisschop et al. (2006), not from Hollenbach et al.'s
  table; Hollenbach et al. used `960 K` for CO in their Table 1, citing Aikawa
  et al. for that value. One caveat matters: Bisschop et al. found that pure CO
  ice desorption is better described by zeroth-order kinetics in their lab TPD
  experiments. DiskBridge does not reproduce that full TPD model. It uses the
  Bisschop binding energy inside the simpler first-order desorption rate above.

- **Photodesorption form.** Hollenbach et al. (2009) write the photodesorption
  flux from an ice surface as `F_pd,i = Y_i F_FUV f_s,i`, where `Y_i` is the
  photodesorption yield, `F_FUV` is the incident FUV photon flux, and `f_s,i`
  is the fraction of surface sites occupied by species `i`. DiskBridge uses
  the same photon-counting idea, but expresses it as a per-active-ice-molecule
  surface rate:

  ```text
  k_pd,surf = F_CO_pdes_photon * Y_CO / (4 * N_surf * N_lay)
  R_pd      = k_pd,surf * n_active
  ```

  `F_CO_pdes_photon` is the number of UV photons hitting a unit projected grain
  area per second in the CO photodesorption band. `Y_CO` is the yield: how many
  CO molecules are removed per incoming photon. The factor
  `4 * N_surf * N_lay` spreads those photon hits over the molecules in the
  active surface layers, converting a photon flux into a per-molecule rate.

  The default `N_surf = 1.5e15 cm^-2` is the standard gas-grain site-density
  scale used in Hasegawa-style models. The default `N_lay = 2` is DiskBridge's
  active-layer approximation, motivated by Hollenbach et al.'s discussion that
  photodesorption mainly acts in the first two monolayers. `n_active` is capped
  at `n_active,max`, so buried CO ice does not photodesorb until it reaches the
  active layers. This cap applies to photodesorption only; thermal desorption
  and the optional direct cosmic-ray desorption term are first-order losses from
  the tracked `CO_ice` reservoir.

- **CO photodesorption yield.** Oberg et al. (2007) measured pure CO ice
  photodesorption under UV photons and found a yield of order `10^-3` CO
  molecules per incident photon; their headline 15 K laboratory value is
  `3e-3` for 7-10.5 eV photons. Hollenbach et al. (2009) adopted
  `Y_CO = 1e-3` as their standard CO value after citing those experiments.
  DiskBridge follows the Hollenbach standard default: `Y_CO = 1e-3`.

- **Physical photon flux rather than only `G0`.** Hollenbach et al. parameterize
  the FUV photon flux as a Draine/ISM-like photon flux scaled by `G0` and
  attenuated with depth. DiskBridge keeps the same yield-times-photon-flux rate
  structure but, when UV products are enabled, takes `F_CO_pdes_photon`
  directly from RADMC-3D by integrating the local mean intensity over the CO
  photodesorption band. This is a DiskBridge extension, not something in the
  original GOW17 or Hollenbach implementations. When the code has to convert a
  Draine-normalized field back into a photon flux, it integrates the Draine SED
  over the CO photodesorption band and uses that band-specific photon flux.

  This matters because a real stellar spectrum is not necessarily shaped like
  the standard interstellar Draine field. Two radiation fields can have the same
  broad UV energy but different numbers of photons in the wavelength range that
  actually photodesorbs CO.

- **Cosmic-ray effects.** Hasegawa & Herbst (1993) developed the common
  cosmic-ray impulsive-heating desorption prescription, and Hollenbach et al.
  (2009) discuss both direct cosmic-ray desorption of surface species and
  cosmic-ray-produced FUV photons. DiskBridge separates these two ideas.
  `co_k_crdes_CO` is an optional first-order direct CO-ice desorption rate; by
  default it is disabled. `F_CRUV_CO_pdes` is an optional added photon flux for
  cosmic-ray-induced UV photodesorption, scaled with the cosmic-ray ionization
  rate. The direct desorption option is therefore a configurable effective
  rate, not a hard-coded reproduction of either the Hasegawa & Herbst
  impulsive-heating formula or Hollenbach et al.'s CO cosmic-ray desorption
  timescale formula.

  In plain language, the code has two cosmic-ray settings: one can directly
  move CO ice back to gas, and the other adds a small UV photon background
  produced by cosmic rays. The default keeps direct cosmic-ray CO desorption
  off.

So, in short: Hasegawa-style gas-grain chemistry supplies the generic
rate-equation language, Hollenbach et al. supplies the PDR ice/photodesorption
surface-flux formulation and motivates the two-layer photodesorption cap,
Bisschop et al. supplies the default pure-CO binding energy and supports
near-unity sticking, Oberg et al. supplies the CO photodesorption yield scale,
and DiskBridge supplies the RADMC-derived physical photon flux and
dust-surface-area coupling.

### 4. Per-Cell Dust Temperature in the Thermal Solver

The gas temperature is evolved through the energy variable `E`. When
`const_temp = false`, the native solver computes

```text
dE/dt = total heating - total cooling
```

and converts between `E` and gas temperature using the heat capacity. One of the
cooling terms is gas-dust collisional energy exchange. This term depends on the
difference between the gas temperature and the dust temperature:

```text
gas-dust exchange ~ n_H * sqrt(T_gas) * (T_gas - T_dust)
```

If `T_gas > T_dust`, collisions transfer energy from gas to dust, so this acts
as gas cooling. If `T_gas < T_dust`, the same term becomes negative cooling,
which means dust heats the gas.

The original local code had a default dust temperature stored in the native
object. DiskBridge adds `SetTdust(Tdust)` and passes the local RADMC-3D dust
temperature into each cell before solving. In the C++ code this value is used in
`Thermo::CoolingDustTd(Zd_, nH_, Tgas, Tdust_)` inside `dEdt_()`.

This matters for disk/envelope post-processing because the dust temperature can
vary strongly with position. A cell near a warm star-facing surface should not
use the same gas-dust coupling target as a cold shielded midplane cell. With the
DiskBridge extension, the gas temperature evolves toward balance with the local
heating, line cooling, chemistry, and the local dust temperature, rather than
toward balance against one global dust bath.

The same per-cell `T_dust` is also used by the CO ice extension for thermal
desorption:

```text
k_td = nu0_CO * exp(-E_bind_CO / T_dust)
```

So `T_dust` affects both the gas energy equation through gas-dust coupling and
the CO ice balance through thermal desorption. It is not the same thing as
`T_gas`: dust temperature comes from the continuum radiative-transfer solution,
while gas temperature is solved by the GOW17 heating/cooling equation unless the
run is in constant-temperature mode.

#### Dust Grain Opacities Used by RADMC-3D

The dust opacities are not part of the original GOW17 chemistry network. They
are a DiskBridge/RADMC-3D input needed to compute the continuum radiation field,
the dust temperature, and the dust optical depths used in UV shielding.

For each dust size bin, DiskBridge writes a RADMC-3D opacity file
`dustkappa_<species><bin>.inp`. The input material properties come from a
`.lnk` optical-constants file, for example
`data/opac/mix_2species_60silicates_40carbons.lnk`. Each row gives:

```text
lambda   n(lambda)   k(lambda)
```

where `lambda` is wavelength in microns, `n(lambda)` is the real part of the
refractive index, and `k(lambda)` is the imaginary part. Together they define
the complex refractive index

```text
m(lambda) = n(lambda) + i k(lambda)
```

The real part controls how the electromagnetic wave is refracted by the grain;
the imaginary part controls absorption. DiskBridge interpolates these optical
constants onto the RADMC-3D wavelength grid, using logarithmic interpolation in
wavelength.

Composition enters through this optical-constants file. The runtime Mie
calculation does not separately track "silicate molecules", "ice molecules",
or "carbon grains" inside one dust grain. Instead, the selected `species`
chooses one effective refractive-index table.

We use a single ice-free refractory dust composition for both the disk and
background dust components, based on mixed silicate and amorphous carbon
optical constants. The disk and background components differ only in their
grain-size distributions and spatial distributions. Opacities are computed
for each grain-size bin using the geometric bin-centre radius.

With the current parameter files,
`species = mix_2species_60silicates_40carbons` selects
`data/opac/mix_2species_60silicates_40carbons.lnk`. Other bundled choices
include `mix_2species_ice70.lnk` and `mix_2species_porous_ice70.lnk`. The
composition and any effective-medium mixing used to build those tables have
already been folded into their `n(lambda)` and `k(lambda)` values before
DiskBridge reads them.

The other composition-dependent number is the material density
`rho_grain`. This affects both the Stokes number used for settling and the
conversion from geometric cross section to mass opacity: for the same optical
efficiency `Q`, a lower-density or more porous grain gives a larger opacity
per gram because the same grain radius contains less mass. For known opacity
species, DiskBridge treats the species registry as authoritative and uses the
registered intrinsic density consistently for settling, grain-area diagnostics,
and Mie opacity normalization. `grain_density` is only an override for custom
species that are not in the registry.

The dust size distribution is split into logarithmic bins between `amin` and
`amax`. The bin mass fractions are computed from the configured power law

```text
n(a) da proportional to a^(-p) da
```

where `a` is grain radius and `p` is `pindex`. For a power-law size
distribution, the mass in a bin is proportional to the integral of
`a^3 n(a) da`, so the code uses

```text
f_bin = [a_max,bin^(4-p) - a_min,bin^(4-p)]
        / [a_max^(4-p) - a_min^(4-p)]
```

where `f_bin` is the fraction of the total dust mass assigned to that bin.
This is the same power-law size-bin logic used by fargo2radmc3d, and the
`p = 3.5` case corresponds to the standard MRN interstellar-grain slope
(Mathis et al. 1977), although DiskBridge lets `p` vary.

For the opacity of each bin, the current writer passes the lower radius edge of
the bin, `a_min,bin`, as the representative grain size. The opacity calculator
then applies a small logarithmic smoothing around that radius by default
(`logawidth = 0.05`, `na = 20`). This avoids perfectly sharp single-size Mie
features while still treating each bin as one RADMC-3D dust species.

At each wavelength, the Mie calculation uses the size parameter

```text
x = 2 pi a / lambda
```

where `a` is the grain radius and `lambda` is the wavelength. Using
`x` and `m(lambda)`, the Bohren-Huffman Mie routine computes dimensionless
efficiencies:

- `Q_abs`: absorption efficiency
- `Q_sca`: scattering efficiency
- `Q_ext = Q_abs + Q_sca`: total extinction efficiency
- `g`: scattering asymmetry parameter, equal to `<cos(theta)>`

These `Q` values are dimensionless cross sections in units of the geometric
area of the grain. DiskBridge converts them into mass opacities, in
`cm^2 g^-1` of dust, using

```text
kappa_abs(lambda) = Q_abs(lambda) * pi a^2 / m_grain
kappa_sca(lambda) = Q_sca(lambda) * pi a^2 / m_grain
m_grain = (4 pi / 3) rho_grain a^3
```

Here `rho_grain` is the material density of the grain. Therefore, for a single
grain size,

```text
kappa_abs = 3 Q_abs / (4 rho_grain a)
kappa_sca = 3 Q_sca / (4 rho_grain a)
```

The opacities are per gram of dust in that bin, not per gram of gas. The dust
density field tells RADMC-3D how many grams of dust are present in each cell.

The scalar RADMC-3D opacity file stores:

```text
lambda[micron]   kappa_abs[cm^2/g]   kappa_sca[cm^2/g]   g
```

If full scattering matrices are requested, DiskBridge can also write
`dustkapscatmat_*` files. In the usual workflow here, `scat_mode = 2`, so the
code writes scalar `dustkappa_*` opacity files with absorption, scattering, and
`g`, not full angle-dependent scattering matrices.

The same opacity files are reused later when estimating UV dust attenuation for
shielding. DiskBridge reads `kappa_abs + kappa_sca` from each bin and computes a
band-averaged UV extinction opacity. Along a ray direction `k`,

```text
tau_UV(k) = sum_b kappa_ext,UV(b) * Sigma_dust(b, k)
```

where `b` labels a dust bin, `kappa_ext,UV(b)` is the UV-band-averaged
extinction opacity for that bin, and `Sigma_dust(b, k)` is the dust mass column
of that bin along the ray. This is why the grain opacity calculation matters
not only for the continuum dust temperature, but also for the directionally
weighted shielding factors used by the chemistry.

### 5. Process-Specific UV Products

The original GOW17 photorates are calibrated to a Draine-like UV field and a
single dimensionless UV amplitude. DiskBridge keeps the GOW17 base rates, but
it no longer requires every photoprocess to see the same scalar `chi`.

The reason for doing this is that photochemistry is wavelength-dependent. In
the ideal treatment, the rate for a photoreaction is computed by integrating the
local photon field against that process's cross section:

```text
k_photo = integral sigma_process(lambda) * I_photon(lambda) dlambda
```

Different processes respond to different wavelength ranges. CO
photodissociation, H2 photodissociation, neutral carbon ionization, and CO ice
photodesorption do not all sample the same part of the UV spectrum. This is why
a single broad UV number can be misleading, especially in disks where the
stellar/accretion spectrum, Ly-alpha emission, dust opacity, and external UV
field can make the local spectrum very different from the standard Draine field.
This point is standard in disk photochemistry; see van Dishoeck et al. (2006)
and Heays et al. (2017).

RADMC-3D mean intensities are integrated into process-specific products:

- `chi_broad`: broad FUV energy-density product
- `G_CO_diss`: CO dissociation photon product
- `G_H2_diss`: H2 dissociation photon product
- `G_C_ion`: carbon ionization photon product
- `G_CO_pdes`: CO photodesorption photon product, Draine-normalized
- `F_CO_pdes_photon`: physical CO photodesorption photon flux in
  `photons cm^-2 s^-1`

The configured bands are wavelength integration intervals, not detailed
molecular cross sections:

| Product | Wavelength band | Photon energy band | Weighting | Used for |
| --- | ---: | ---: | --- | --- |
| `chi_broad` | 91.2-206.7 nm | 13.6-6.0 eV | energy | broad GOW17 fallback field, photoelectric heating proxy |
| `G_CO_diss` | 91.2-111.8 nm | 13.6-11.1 eV | photon | CO photodissociation before CO shielding |
| `G_H2_diss` | 91.2-111.8 nm | 13.6-11.1 eV | photon | H2 photodissociation before H2 shielding |
| `G_C_ion` | 91.2-110.1 nm | 13.6-11.26 eV | photon | neutral carbon photoionization |
| `G_CO_pdes` | 91.2-205.0 nm | 13.6-6.05 eV | photon | Draine-normalized CO ice photodesorption |
| `F_CO_pdes_photon` | 91.2-205.0 nm | 13.6-6.05 eV | photon flux | physical CO ice photodesorption rate |

Within each band, RADMC-3D supplies the mean intensity at several sampled
wavelengths. DiskBridge integrates those sampled values over the band using
quadrature weights: energy-weighted products integrate energy density, while
photon-weighted products integrate photon flux. The band edges define which
wavelength range is included; they do not mean the spectrum is evaluated at only
one wavelength or treated as flat inside the band.

The lower wavelength edge, 91.2 nm, is the hydrogen ionization limit
13.6 eV; photons harder than this are not part of the FUV products. The broad
field extends down to about 6 eV, matching the usual FUV energy range used for
PDR-style heating and photochemistry. The C ionization band stops at 110.1 nm
because neutral carbon has an ionization threshold of about 11.26 eV. The CO and
H2 dissociation products use the harder 91.2-111.8 nm photon band as a simple
proxy for the dissociating UV lines. The CO photodesorption products use a wider
91.2-205.0 nm photon band because ice photodesorption is treated as a
photon-counting process over a broad FUV interval.

These products are Draine-normalized over their own wavelength bands. That is,
`G_CO_diss = 1` means the local photon flux over the CO dissociation band
equals the Draine reference photon flux in that same band. It does not mean the
full local spectrum is Draine-shaped.

The native `Gph` array is assembled so that CO, H2, and C photoprocesses can
receive different UV amplitudes before shielding is applied:

```text
Gph[CO] = G_CO_diss * theta_CO
Gph[H2] = G_H2_diss * theta_H2
Gph[C]  = G_C_ion   * theta_C
```

For all other GOW17 photoprocesses, DiskBridge currently falls back to the
broad product. Thus this is a band-product correction, not a full
cross-section-resolved photochemical integration.

This puts DiskBridge between two limits. It is more specific than applying one
global `chi` to every photoprocess, because CO, H2, C, and CO ice can see
different effective UV products. It is still simpler than fully
cross-section-resolved photochemistry, where each photorate would be integrated
over the local spectrum with that process's wavelength-dependent cross section.

The important comparison to disk thermo-chemical codes is not that all of them
do the full integral for every reaction. They usually use a hierarchy of
approximations. ProDiMo, for example, writes the formal cross-section integral
for photorates, but Woitke et al. (2009) state that their 2009 implementation
uses UMIST photorates plus molecular self-shielding for most reactions, with
special handling where cross sections are available. They explicitly describe
this as a compromise rather than full UV-line-resolved radiative transfer. DALI
computes the local continuum radiation field from UV to millimeter wavelengths
and Bruderer (2013) states that photodissociation rates are obtained from
molecular cross sections and the FUV intensity from the continuum transfer
calculation. DALI-based CO isotopologue work also treats CO photodissociation
as a species-specific process with CO shielding rather than as a generic
broad-UV rate (Miotello et al. 2014; Visser et al. 2009).

So the normal practice is not necessarily "full cross sections for every
photoreaction". The normal practice is to avoid pretending that all
photoprocesses respond to the same broad UV scalar. DiskBridge follows that
direction, but in a reduced way: it uses band-integrated, process-specific UV
products for the most important GOW17 photoprocesses instead of recomputing the
full photochemical network from wavelength-dependent cross sections.

The most important special case is `F_CO_pdes_photon`: CO photodesorption is
controlled by photon count, so DiskBridge can pass a physical photon flux into
the CO-ice solver instead of exposing a separate user-selected approximation.
This is the local continuum-attenuated photon flux over the CO
photodesorption band and is not multiplied by the CO or H2 molecular shielding
factors used for gas-phase photodissociation.

### 6. 1D and 3D Shielding Outside the Native Slab

DiskBridge computes molecular and atomic shielding outside the C++ slab class
when running on general model grids.

The shielding driver computes:

- H2 self-shielding
- CO self-shielding and shielding by H2
- neutral carbon shielding
- dust attenuation or directionally weighted UV factors, depending on mode

For effectively 1D models, columns can be accumulated along the 1D radial or
slab ordering. For 3D models, DiskBridge casts HEALPix rays through the mesh and
integrates columns along each direction. Directional shielding factors are then
averaged, optionally using UV/dust-direction weights.

The reason this is needed is that molecular self-shielding is directional. CO
and H2 are dissociated by UV line photons. A cell may be exposed to UV from the
star through one low-column direction, while other directions pass through the
disk midplane and have enormous CO/H2 columns. The shielding factor for a ray is
therefore a function of direction:

```text
theta_CO(k) = shielding_function[ N_CO(k), N_H2(k), b_CO ]
```

where `k` labels a HEALPix direction. A single scalar column cannot represent
this in a disk or streamer, because the column to the star, the column out of
the disk surface, and the column through the midplane can differ by orders of
magnitude.

The symbols in this expression mean:

- `theta_CO(k)`: the CO shielding factor for direction `k`; `1` means no
  shielding and values near `0` mean strong shielding.
- `shielding_function[...]`: the tabulated shielding calculation, using the
  Visser et al. (2009) CO shielding tables.
- `N_CO(k)`: CO column density from the cell to the boundary along direction
  `k`, in `cm^-2`.
- `N_H2(k)`: H2 column density along the same direction, also in `cm^-2`; H2
  matters because H2 lines overlap with and shield CO dissociating lines.
- `b_CO`: CO Doppler line-width parameter, usually in `km s^-1`; broader lines
  change how strongly CO self-shields.
- `k`: the index of one HEALPix direction/pixel on the sky around the cell.

The effective photodissociation rate should average those directional shielding
factors using the directions from which UV photons actually arrive:

```text
theta_eff = sum_k W(k) * theta(k)
```

Here:

- `theta(k)`: any directional shielding factor, such as `theta_CO(k)`,
  `theta_H2(k)`, or `theta_C(k)`.
- `theta_eff`: the single effective shielding factor passed to the local GOW17
  chemistry solver for that cell.
- `W(k)`: the fraction of the relevant UV intensity arriving from direction
  `k`; all `W(k)` values sum to `1`.
- `sum_k`: sum over all HEALPix directions.

Equal weights, `W(k) = 1 / N_pix`, assume the UV intensity is isotropic. That is
reasonable only when the radiation field is truly diffuse or externally
isotropic. It is not generally reasonable in a disk with a central star. If most
of the dissociating UV comes from the star, the relevant shielding column is
mostly the column toward the star. Equal HEALPix weights would give the same
importance to directions that carry almost no UV flux, such as heavily shielded
midplane directions or the far side of the disk.

DiskBridge therefore uses a simple angular UV model when directional weights are
enabled:

```text
C(k) = C_star(k) + C_ext(k) + C_iso
W(k) = C(k) / sum_j C(j)
```

Here:

- `C(k)`: an unnormalized UV contribution assigned to direction `k`; it is used
  only to build the angular weights.
- `C_star(k)`: direct stellar/accretion UV contribution. It is nonzero only in
  the HEALPix pixel that points from the cell toward the star.
- `C_ext(k)`: direct external/background UV contribution entering from the
  domain boundary along direction `k`.
- `C_iso`: residual diffuse/scattered UV contribution. It is assigned equally
  to all directions.
- `sum_j C(j)`: the total unnormalized UV contribution over all directions.
- `W(k)`: the normalized directional weight used to average shielding factors.

`C_star(k)` puts the direct stellar/accretion UV into the HEALPix pixel pointing
toward the star, attenuated by the dust optical depth along that ray. Because
the star is effectively a point source, concentrating that contribution into
one pixel is more physical than spreading it equally over the sky. `C_ext(k)`
represents any direct external field entering from the domain boundary, weighted
by the dust optical depth to the boundary along each HEALPix ray. `C_iso` is the
remaining RADMC-3D UV field not accounted for by those direct components; it is
treated as diffuse/scattered and spread equally over all directions.

This point-source plus isotropic-residual model is more accurate than equal
weights because it preserves the main angular information that controls
shielding. If stellar UV dominates, the shielding is controlled by the column to
the star. If external UV dominates, low-dust-opacity escape directions matter
most. If scattered/diffuse UV dominates, the weights naturally become close to
uniform. Equal weights hard-code only the last of those cases.

This extension is what lets the local GOW17 cell solver behave like a PDR
network in a non-slab geometry.

### 7. Coupled Shielding/Chemistry Iteration

Shielding depends on the abundances of H2, CO, and C, but those abundances
depend on shielding. DiskBridge adds two coupling modes around the native
solver:

- `fixed_point`: solve chemistry to equilibrium, recompute shielding, repeat
  until abundance and temperature changes converge.
- `astrochem`: update shielding over a sequence of pseudo-time chemistry
  integrations, optionally use guarded Aitken acceleration on shielding-driving
  species, then finish with an equilibrium solve.

The `astrochem` mode was added to make difficult 3D shielding/chemistry
couplings less brittle than a pure equilibrium fixed-point loop.

### 8. Solver Diagnostics and Numerical Hardening

DiskBridge adds several practical diagnostics and guardrails:

- per-species absolute tolerances from config
- negative-abundance correction counts
- CVODE failure counts
- `SolveEq` maximum-pseudo-time residual reporting
- native RHS residual diagnostics
- temperature floors/ceilings
- projection back onto elemental budgets
- optional interpolation repair for failed cells in the time stepper

The C++ code also has more finite-value checks around numerically sensitive
rate expressions, especially grain-assisted recombination at very low electron
abundance.

## What DiskBridge Does Not Change

Several important pieces remain inherited from GOW17:

- The gas-phase reaction network is still the reduced GOW17 network.
- The base photorates are still Draine-calibrated constants.
- The native ODE system is still local to a cell once the environment is fixed.
- The thermal balance is still GOW17-style heating/cooling, not a full PDR code
  with detailed level populations for every coolant.
- The CO-ice extension tracks only CO gas/ice partitioning; it is not a full
  grain-surface chemistry network.

## Original vs DiskBridge Summary

| Area | Original local GOW17 code | DiskBridge version |
| --- | --- | --- |
| Geometry | Native 1D slab model | 1D slab plus batch cell solver on DiskBridge grids |
| State vector | 14 variables including `E` | 15 variables: adds `CO_ice` |
| UV field | Single Draine-normalized field in native slab logic | Process-specific Draine-normalized UV products from RADMC-3D |
| Dust opacity | Not part of native GOW17 chemistry | Mie opacities per dust-size bin written for RADMC-3D and reused for UV dust attenuation |
| CO photodesorption | Not present | Physical photon-flux driven CO photodesorption |
| Freeze-out/desorption | Not present | CO freeze-out, thermal desorption, photodesorption, optional CR desorption |
| Dust temperature | Limited/native fixed behavior | Per-cell `Tdust` from RADMC-3D |
| Shielding | Native slab shielding | 1D and HEALPix 3D column/shielding support |
| Solver access | C++ examples/slab | Python API, OpenMP batch solving, persistent time stepper |
| Diagnostics | Basic CVODE behavior | Failure counters, residuals, budget projection, RHS diagnostics |

## Literature Context

Core reduced network:

- Gong, M., Ostriker, E. C., & Wolfire, M. G. 2017, "A Simple and Accurate
  Network for Hydrogen and Carbon Chemistry in the Interstellar Medium",
  ApJ, 843, 38. DOI: https://doi.org/10.3847/1538-4357/aa7561.
  arXiv: https://arxiv.org/abs/1610.09023.

UV reference field and photoprocess calibration:

- Draine, B. T. 1978, "Photoelectric heating of interstellar gas", ApJS, 36,
  595. DOI: https://doi.org/10.1086/190513.
- van Dishoeck, E. F., Jonkheid, B., & van Hemert, M. C. 2006,
  "Photoprocesses in protoplanetary disks", Faraday Discussions, 133, 231.
  DOI: https://doi.org/10.1039/B517564J.
- Heays, A. N., Bosman, A. D., & van Dishoeck, E. F. 2017,
  "Photodissociation and photoionisation of atoms and molecules of
  astrophysical interest", A&A, 602, A105. DOI:
  https://doi.org/10.1051/0004-6361/201628742.

Thermo-chemical disk UV treatments:

- Woitke, P., Kamp, I., & Thi, W.-F. 2009, "Radiation thermo-chemical models
  of protoplanetary disks. I. Hydrostatic disk structure and inner rim", A&A,
  501, 383. DOI: https://doi.org/10.1051/0004-6361/200911821.
- Bruderer, S., van Dishoeck, E. F., Doty, S. D., & Herczeg, G. J. 2012, "The
  warm gas atmosphere of the HD 100546 disk seen by Herschel: Evidence of a
  gas-rich, carbon-poor atmosphere?", A&A, 541, A91. DOI:
  https://doi.org/10.1051/0004-6361/201118218.
- Bruderer, S. 2013, "Survival of molecular gas in cavities of transition
  disks. I. CO", A&A, 559, A46. DOI:
  https://doi.org/10.1051/0004-6361/201321171.
- Miotello, A., Bruderer, S., & van Dishoeck, E. F. 2014, "Protoplanetary disk
  masses from CO isotopologue line emission", A&A, 572, A96. DOI:
  https://doi.org/10.1051/0004-6361/201424712.

Dust opacity and grain-size context:

- Mathis, J. S., Rumpl, W., & Nordsieck, K. H. 1977, "The size distribution of
  interstellar grains", ApJ, 217, 425. DOI: https://doi.org/10.1086/155591.
- Bohren, C. F., & Huffman, D. R. 1983, "Absorption and Scattering of Light by
  Small Particles", Wiley.
- Weingartner, J. C., & Draine, B. T. 2001, "Dust Grain-Size Distributions and
  Extinction in the Milky Way, Large Magellanic Cloud, and Small Magellanic
  Cloud", ApJ, 548, 296. DOI: https://doi.org/10.1086/318651.

H2 shielding:

- Draine, B. T., & Bertoldi, F. 1996, "Structure of Stationary
  Photodissociation Fronts", ApJ, 468, 269. DOI:
  https://doi.org/10.1086/177689.

CO shielding:

- Visser, R., van Dishoeck, E. F., & Black, J. H. 2009, "The
  photodissociation and chemistry of CO isotopologues: applications to
  interstellar clouds and circumstellar disks", A&A, 503, 323. DOI:
  https://doi.org/10.1051/0004-6361/200912129.

CO freeze-out/desorption and gas-grain context:

- Hasegawa, T. I., Herbst, E., & Leung, C. M. 1992, "Models of Gas-Grain
  Chemistry in Dense Interstellar Clouds with Complex Organic Molecules",
  ApJS, 82, 167. DOI: https://doi.org/10.1086/191713.
- Hasegawa, T. I., & Herbst, E. 1993, "New gas-grain chemical models of
  quiescent dense interstellar clouds: the effects of H2 tunnelling reactions
  and cosmic ray induced desorption", MNRAS, 261, 83. DOI:
  https://doi.org/10.1093/mnras/261.1.83.
- Bisschop, S. E., Fraser, H. J., Oberg, K. I., van Dishoeck, E. F.,
  & Schlemmer, S. 2006, "Desorption rates and sticking coefficients for CO and
  N2 interstellar ices", A&A, 449, 1297. DOI:
  https://doi.org/10.1051/0004-6361:20054051.
- Oberg, K. I., Fuchs, G. W., Awad, Z., Fraser, H. J., Schlemmer, S.,
  van Dishoeck, E. F., & Linnartz, H. 2007, "Photodesorption of CO Ice",
  ApJ, 662, L23. DOI: https://doi.org/10.1086/519281.
- Hollenbach, D., Kaufman, M. J., Bergin, E. A., & Melnick, G. J. 2009,
  "Water, O2, and Ice in Molecular Clouds", ApJ, 690, 1497. DOI:
  https://doi.org/10.1088/0004-637X/690/2/1497.
- Kramer, C., Aalto, S., Simon, R., Kaufman, M. J., Hollenbach, D. J.,
  Bergin, E., & Melnick, G. J. 2008, "Tracking Water, O2 and Ice in Molecular
  Clouds: PDRs Models with Photodesorption and Grain Chemistry", EAS
  Publications Series, 31, 43. DOI:
  https://doi.org/10.1051/eas:0831009.

3D angular directions:

- Gorski, K. M., Hivon, E., Banday, A. J., Wandelt, B. D., Hansen, F. K.,
  Reinecke, M., & Bartelmann, M. 2005, "HEALPix: A Framework for
  High-Resolution Discretization and Fast Analysis of Data Distributed on the
  Sphere", ApJ, 622, 759. DOI: https://doi.org/10.1086/427976.

ODE solver:

- Hindmarsh, A. C., Brown, P. N., Grant, K. E., Lee, S. L., Serban, R.,
  Shumaker, D. E., & Woodward, C. S. 2005, "SUNDIALS: Suite of nonlinear and
  differential/algebraic equation solvers", ACM TOMS, 31, 363. DOI:
  https://doi.org/10.1145/1089014.1089020.

Radiative-transfer context:

- Dullemond, C. P. 2012, "RADMC-3D: A multi-purpose radiative transfer tool",
  Astrophysics Source Code Library, ascl:1202.015. URL:
  https://ascl.net/1202.015.
