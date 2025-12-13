"""
Visser+2009 CO shielding functions with auto-download.

- Downloads Leiden CO shielding archive if not present.
- Extracts shield.A.B.C-D-E.dat files.
- Parses a selected file into shielding grids for isotopologues.
- Provides interpolated theta(Nco, Nh2, b) factors.

Leiden source + naming convention:
  shield.A.B.C-D-E.dat
  A: b (km/s), B: Tex (K), C/D/E: isotope ratios
  Archive URL from Leiden CO photodissociation page.

References:
  Visser+2009 A&A 503, 323
  Leiden CO shielding archive: CO_shielding_functions.zip
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import re
import zipfile
import urllib.request

import numpy as np
from scipy.interpolate import RegularGridInterpolator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_VISSER_DATA_DIR = REPO_ROOT / "data" / "visser"


# Leiden archive URL (public)
LEIDEN_CO_ZIP = (
    "https://home.strw.leidenuniv.nl/~ewine/photo/data/"
    "CO_photodissociation/CO_shielding_functions.zip"
)

ArrayLike = Union[float, np.ndarray]


def ensure_visser_tables(
    data_dir: str | Path,
    url: str = LEIDEN_CO_ZIP,
    force: bool = False,
) -> Path:
    """
    Ensure Visser shielding tables exist locally.

    Parameters
    ----------
    data_dir : str or Path
        Target directory to store/extract tables.
    url : str
        Download URL for the Leiden shielding archive.
    force : bool
        Redownload/reextract even if files already exist.

    Returns
    -------
    Path
        Path to extracted data directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Quick check: any shield.*.dat already there?
    existing = list(data_dir.glob("shield.*.dat"))
    if existing and not force:
        return data_dir

    zip_path = data_dir / "CO_shielding_functions.zip"
    if force and zip_path.exists():
        zip_path.unlink()

    if not zip_path.exists():
        print(f"[Visser] Downloading shielding archive to {zip_path} ...")
        with urllib.request.urlopen(url) as resp:
            zip_bytes = resp.read()
        zip_path.write_bytes(zip_bytes)

    print(f"[Visser] Extracting {zip_path} to {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return data_dir


@dataclass(frozen=True)
class ShieldingGrid2D:
    logNco_grid: np.ndarray        # (nNco,)
    logNh2_grid: np.ndarray        # (nNh2,)
    theta_grid: np.ndarray         # (nNco, nNh2)
    fill_value: float = 1.0
    bounds_error: bool = False

    def __post_init__(self):
        lgNco = np.asarray(self.logNco_grid, float)
        lgNh2 = np.asarray(self.logNh2_grid, float)
        th = np.asarray(self.theta_grid, float)

        if th.shape != (lgNco.size, lgNh2.size):
            raise ValueError("theta_grid shape mismatch with axis grids.")

        if RegularGridInterpolator is None:
            raise ImportError("scipy required for shielding interpolation.")

        interp = RegularGridInterpolator(
            (lgNco, lgNh2),
            th,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )
        object.__setattr__(self, "_interp", interp)

    def theta(self, logNco: ArrayLike, logNh2: ArrayLike) -> np.ndarray:
        logNco = np.asarray(logNco, float)
        logNh2 = np.asarray(logNh2, float)
        a, b = np.broadcast_arrays(logNco, logNh2)
        pts = np.stack([a.ravel(), b.ravel()], axis=-1)
        out = self._interp(pts).reshape(a.shape)
        return np.clip(out, 0.0, 1.0)


class VisserShielding:
    """
    Loader/interpolator for Visser+09 CO shielding functions.

    Typical usage
    -------------
    vis = VisserShielding(data_dir=".../visser_tables")
    theta = vis.theta("co", Nco, Nh2, b_kms=0.3)
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        filename: Optional[str] = None,
        b_kms: float = 0.3,
        auto_download: bool = True,
    ):
        if data_dir is None:
            self.data_dir = DEFAULT_VISSER_DATA_DIR
        else:
            self.data_dir = Path(data_dir)

        if auto_download:
            ensure_visser_tables(self.data_dir)

        if filename is None:
            filename = self._select_nearest_b_file(b_kms)

        self.filepath = self.data_dir / filename
        if not self.filepath.exists():
            raise FileNotFoundError(f"Shielding file not found: {self.filepath}")

        self.b_kms = self._parse_co_b_from_file(self.filepath)
        self._grids: Dict[str, ShieldingGrid2D] = self._parse_visser_file(self.filepath)

    # --------- public API ----------

    def available_files(self) -> list[str]:
        return sorted([p.name for p in self.data_dir.glob("shield.*.dat")])

    def theta(
        self,
        isotop: str,
        Nco: ArrayLike,
        Nh2: ArrayLike,
        b_kms: Optional[float] = None,
        log_floor: Tuple[float, float] = (8.0, 10.0),
    ) -> np.ndarray:
        """
        Evaluate shielding theta for one isotopologue.

        Parameters
        ----------
        isotop : str
            "co", "13co", "c18o", "c17o" (case-insensitive).
        Nco, Nh2 : scalar or ndarray
            Column densities [cm^-2].
        b_kms : float, optional
            If given and differs from loaded file, user should load another file.
        log_floor : (logNco_min, logNh2_min)
            Floors to avoid log10(0).

        Returns
        -------
        ndarray
            theta values.
        """
        iso = isotop.lower()
        if iso not in self._grids:
            raise KeyError(f"Isotopologue {iso} not in {self.filepath.name}. "
                           f"Available: {list(self._grids)}")

        if b_kms is not None and abs(b_kms - self.b_kms) > 1e-6:
            raise ValueError(
                f"This VisserShielding instance was loaded for b={self.b_kms} km/s. "
                f"Load another file for b={b_kms}."
            )

        Nco = np.asarray(Nco, float)
        Nh2 = np.asarray(Nh2, float)
        logNco = np.log10(np.maximum(Nco, 10.0**log_floor[0]))
        logNh2 = np.log10(np.maximum(Nh2, 10.0**log_floor[1]))

        return self._grids[iso].theta(logNco, logNh2)

    # --------- filename helpers ----------

    @staticmethod
    def _parse_b_from_name(name: str) -> float:
        # shield.A.B.C-D-E.dat where A=b(km/s)
        m = re.match(r"shield\.([0-9.]+)\.", name)
        if not m:
            return np.nan
        return float(m.group(1))

    @staticmethod
    def _parse_co_b_from_file(path: Path) -> float:
        text = path.read_text()
        for line in text.splitlines():
            if "b(CO,H2,H)" in line:
                m = re.search(r"=\s*([0-9.]+)", line)
                if m:
                    return float(m.group(1))
                break
        return np.nan

    def _select_nearest_b_file(self, b_kms: float) -> str:
        files = self.available_files()
        if not files:
            raise FileNotFoundError(f"No shield.*.dat files found in {self.data_dir}")

        bvals_list: list[float] = []
        for fname in files:
            path = self.data_dir / fname
            try:
                val = self._parse_co_b_from_file(path)
            except Exception:
                val = np.nan
            bvals_list.append(val)

        bvals = np.array(bvals_list, float)
        j = np.nanargmin(np.abs(bvals - b_kms))
        return files[j]

    # --------- parser ----------

    def _parse_visser_file(self, path: Path) -> Dict[str, ShieldingGrid2D]:
        """
        Parse a Visser CO shielding file from Leiden database.

        Actual Leiden format:
          - Header lines: 'n[N(12CO)] = <nNco>', 'n[N(H2)] = <nNh2>'
          - 'N(12CO)' label, then nNco column density values (one per line)
          - 'N(H2)' label, then nNh2 column density values (one per line)
          - Isotopologue blocks: label (e.g. '12C16O'), then theta matrix
            (nNco rows x nNh2 cols, values may span multiple lines)
        """
        text = path.read_text()
        lines = text.splitlines()

        # Parse header to get grid sizes
        nNco = None
        nNh2 = None
        for ln in lines:
            if "n[N(12CO)]" in ln and "=" in ln:
                nNco = int(ln.split("=")[1].strip())
            elif "n[N(H2)]" in ln and "=" in ln:
                nNh2 = int(ln.split("=")[1].strip())

        if nNco is None or nNh2 is None:
            raise ValueError(
                f"Could not parse grid dimensions from {path}. "
                f"Expected 'n[N(12CO)] = ...' and 'n[N(H2)] = ...' lines."
            )

        # Find N(12CO) and N(H2) grid sections
        Nco_values = []
        Nh2_values = []
        
        in_nco_section = False
        in_nh2_section = False
        
        for ln in lines:
            stripped = ln.strip()
            
            # Detect section starts
            if stripped == "N(12CO)":
                in_nco_section = True
                in_nh2_section = False
                continue
            elif stripped == "N(H2)":
                in_nco_section = False
                in_nh2_section = True
                continue
            
            # Check if this is an isotopologue label (ends the N(H2) section)
            if in_nh2_section and len(Nh2_values) >= nNh2:
                in_nh2_section = False
            
            # Parse values
            if in_nco_section and len(Nco_values) < nNco:
                try:
                    val = float(stripped)
                    Nco_values.append(val)
                except ValueError:
                    pass
            elif in_nh2_section and len(Nh2_values) < nNh2:
                try:
                    val = float(stripped)
                    Nh2_values.append(val)
                except ValueError:
                    pass

        if len(Nco_values) != nNco or len(Nh2_values) != nNh2:
            raise ValueError(
                f"Grid size mismatch in {path}: expected {nNco} N(CO) and {nNh2} N(H2) "
                f"values, got {len(Nco_values)} and {len(Nh2_values)}"
            )

        # Convert to log10 (Leiden files give actual column densities)
        logNco = np.log10(np.maximum(np.array(Nco_values), 1e-99))
        logNh2 = np.log10(np.maximum(np.array(Nh2_values), 1e-99))

        # Parse isotopologue blocks
        # Isotopologue labels: 12C16O, 13C16O, 12C18O, 12C17O
        isotope_labels = ["12C16O", "13C16O", "12C18O", "12C17O"]
        grids: Dict[str, ShieldingGrid2D] = {}

        for iso_label in isotope_labels:
            # Find the line with this isotopologue
            iso_idx = None
            for i, ln in enumerate(lines):
                if ln.strip() == iso_label:
                    iso_idx = i
                    break
            
            if iso_idx is None:
                continue  # This isotopologue not in file
            
            # Collect all numerical values after the label until we have nNco * nNh2
            all_values = []
            for ln in lines[iso_idx + 1:]:
                stripped = ln.strip()
                # Stop if we hit another isotopologue label or section
                if stripped in isotope_labels or stripped in ["N(12CO)", "N(H2)"]:
                    break
                # Parse all floats on this line
                parts = stripped.split()
                for p in parts:
                    try:
                        all_values.append(float(p))
                    except ValueError:
                        pass
                # Stop if we have enough
                if len(all_values) >= nNco * nNh2:
                    break
            
            if len(all_values) < nNco * nNh2:
                raise ValueError(
                    f"Not enough theta values for {iso_label} in {path}: "
                    f"expected {nNco * nNh2}, got {len(all_values)}"
                )
            
            # Reshape into (nNco, nNh2) matrix
            theta_grid = np.array(all_values[:nNco * nNh2]).reshape(nNco, nNh2)
            
            # Normalize label
            label_norm = (
                iso_label.lower()
                .replace("12c16o", "co")
                .replace("13c16o", "13co")
                .replace("12c18o", "c18o")
                .replace("12c17o", "c17o")
            )
            
            grids[label_norm] = ShieldingGrid2D(logNco, logNh2, theta_grid)

        if not grids:
            raise ValueError(f"No isotopologue blocks parsed from {path}")

        return grids
