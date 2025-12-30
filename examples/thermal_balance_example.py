"""
Example: Gas thermal balance calculation

Demonstrates the modular thermal balance solver with:
- Photoelectric heating (PAH scale)
- Cosmic ray heating
- C II fine-structure cooling
- Gas-dust collisional exchange
- Carbon ionization closure
"""

from diskbridge.chemistry.thermal import run_thermal


def run_thermal_example(rad):
    """Standalone thermal balance.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model with dust temperature and UV field
    """
    
    result = run_thermal(
        rad,
        model="thermal_balance_v1",
        config={
            'zeta_cr': 1e-17,
            'pah_scale': 1.0,
            'X_C_tot': 1.4e-4,
            'n_iter': 3,
            'tol': 0.01,
        },
        write=True,
    )
    
    print("\nThermal Results:")
    print(f"  Tgas range: {result.tgas.to('K').magnitude.min():.1f} - "
          f"{result.tgas.to('K').magnitude.max():.1f} K")
    print(f"  Converged: {result.meta['converged']}")
    print(f"  Iterations: {result.meta['n_iter']}")
    
    if 'nCplus' in result.fields:
        print(f"\n  C+ density: {result.fields['nCplus'].to('cm^-3').magnitude.max():.2e} cm^-3")
    if 'ne' in result.fields:
        print(f"  Electron density: {result.fields['ne'].to('cm^-3').magnitude.max():.2e} cm^-3")
    
    return result


def run_coupled_thermochemistry_example(rad, disc_mask):
    """Coupled thermochemistry iteration.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model
    disc_mask : ndarray
        Boolean disc mask
    """
    from diskbridge.chemistry import run_thermochemistry
    from diskbridge._units import Quantity
    
    chem_result, therm_result = run_thermochemistry(
        rad,
        chemistry_model="co_two_phase_time",
        chemistry_config={
            "Xco_tot": 1e-4,
            "nside": 4,
            "b_kms": 0.3,
        },
        thermal_model="thermal_balance_v1",
        thermal_config={
            "zeta_cr": 1e-17,
            "pah_scale": 0.5,
            "n_iter": 3,
        },
        n_iter=2,
        write=True,
    )
    
    print("\nCoupled Thermochemistry Results:")
    print(f"  CO gas: {chem_result.number_densities['co_gas'].to('cm^-3').magnitude.max():.2e} cm^-3")
    print(f"  Tgas: {therm_result.tgas.to('K').magnitude.min():.1f} - "
          f"{therm_result.tgas.to('K').magnitude.max():.1f} K")
    
    return chem_result, therm_result


if __name__ == "__main__":
    print(__doc__)
    print("\nUsage:")
    print("  from diskbridge.radmc3d.model import RadModel")
    print("  ")
    print("  # Standalone thermal")
    print("  result = run_thermal_example(rad)")
    print("  ")
    print("  # Coupled thermochemistry")
    print("  chem, therm = run_coupled_thermochemistry_example(rad, disc_mask)")
