from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diskbridge.radmc3d.model import RadModel

import diskbridge
from diskbridge.chemistry.types import ChemistryResult
from diskbridge.chemistry.driver import compute_abundance_pinte


def run(rad: 'RadModel', config: dict) -> ChemistryResult:
    """Run Pinte-style abundance switches chemistry.
    
    Parameters
    ----------
    rad : RadModel
        RADMC-3D model wrapper
    config : dict
        Configuration with keys:
        - molecule: str (default: 'co')
        - X0: float (default: from diskbridge.params.abundance)
        - photodissociation: bool (default: from diskbridge.params)
        - freezeout: bool (default: from diskbridge.params)
        - photodesorption: bool (default: from diskbridge.params)
        - smooth_log_chi_nH_dex: float (default: 0.0)
        - smooth_Tfrz_K: float (default: 0.0)
        
    Returns
    -------
    ChemistryResult
        Result with abundances and number_densities
        
    Notes
    -----
    Chemistry constants (T_FRZ, EPS_FRZ, LOG_CHI_OVER_NH_PDISS, etc.) are
    loaded from config.toml. Override via diskbridge.load_config(path).
    """
    molecule = config.get('molecule', 'co')
    X0 = config.get('X0', float(diskbridge.params.abundance))
    photodissociation = config.get('photodissociation', diskbridge.params.photodissociation)
    freezeout = config.get('freezeout', diskbridge.params.freezeout)
    photodesorption = config.get('photodesorption', diskbridge.params.photodesorption)
    smooth_log_chi_nH_dex = config.get('smooth_log_chi_nH_dex', 0.0)
    smooth_Tfrz_K = config.get('smooth_Tfrz_K', 0.0)
    
    T = rad.ensure_temperature()
    nH = rad.ensure_nH()
    
    needs_chi = bool(photodissociation or photodesorption)
    chi = rad.ensure_chi() if needs_chi else None
    
    X, n_mol = compute_abundance_pinte(
        molecule=str(molecule).lower(),
        T=T,
        nH=nH,
        chi=chi,
        chi_eff=None,
        X0=float(X0),
        photodissociation=bool(photodissociation),
        freezeout=bool(freezeout),
        photodesorption=bool(photodesorption),
        smooth_log_chi_nH_dex=float(smooth_log_chi_nH_dex),
        smooth_Tfrz_K=float(smooth_Tfrz_K),
    )
    
    mol_lower = str(molecule).lower()
    
    return ChemistryResult(
        abundances={mol_lower: X},
        number_densities={mol_lower: n_mol},
        fields={},
        meta={'model': 'pinte_switches'},
    )
