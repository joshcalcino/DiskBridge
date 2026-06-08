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

# db-keywords: shielding, co-shielding, chemistry, mesh, coordinates, io, interpolation, files
# db-role: helper
# db-scope: package
# db-purpose: Visser+2009 CO shielding functions with auto-download.

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

N_SHIELD_MIN = 1.0e10


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
    fill_value: Optional[float] = None
    bounds_error: bool = False

    def __post_init__(self):
        lgNco = np.asarray(self.logNco_grid, float)
        lgNh2 = np.asarray(self.logNh2_grid, float)
        th = np.asarray(self.theta_grid, float)

        if th.shape != (lgNco.size, lgNh2.size):
            raise ValueError("theta_grid shape mismatch with axis grids.")

        th = np.clip(th, 1.0e-300, 1.0)
        log_th = np.log(th)

        interp = RegularGridInterpolator(
            (lgNco, lgNh2),
            log_th,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )
        object.__setattr__(self, "_interp", interp)

    def theta(self, logNco: ArrayLike, logNh2: ArrayLike) -> np.ndarray:
        logNco = np.asarray(logNco, float)
        logNh2 = np.asarray(logNh2, float)
        a, b = np.broadcast_arrays(logNco, logNh2)
        pts = np.stack([a.ravel(), b.ravel()], axis=-1)
        out_log = self._interp(pts).reshape(a.shape)
        out = np.exp(out_log)
        return np.clip(out, 0.0, 1.0)


class VisserShielding:
    """Loader and interpolator for the Visser CO shielding tables.

    Downloads and parses the Leiden CO shielding archive and evaluates the
    interpolated shielding factor for an isotopologue as a function of the CO
    and H2 column densities at a given Doppler parameter.

    Examples
    --------
    >>> vis = VisserShielding(data_dir=".../visser_tables")
    >>> theta = vis.theta("co", Nco, Nh2, b_kms=0.3)

    References
    ----------
    Visser et al. 2009, A&A 503, 323; see the shielding guide.
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
        self._grid_cache: Dict[str, Dict[str, ShieldingGrid2D]] = {
            self.filepath.name: self._grids
        }

    # --------- public API ----------

    def available_files(self) -> list[str]:
        return sorted([p.name for p in self.data_dir.glob("shield.*.dat")])

    def theta(
        self,
        isotop: str,
        Nco: ArrayLike,
        Nh2: ArrayLike,
        b_kms: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate the CO shielding factor for one isotopologue.

        Interpolates the loaded Visser table in ``log10(Nco)`` and
        ``log10(Nh2)``. Cells below the table floor in both columns are treated
        as unshielded (theta = 1).

        Parameters
        ----------
        isotop : str
            One of "co", "13co", "c18o", "c17o" (case-insensitive).
        Nco, Nh2 : float or ndarray
            CO and H2 column densities [cm^-2]; broadcast against each other.
        b_kms : float, optional
            Doppler parameter [km/s]. Must match the loaded table; load a
            different file for another value.

        Returns
        -------
        ndarray
            Shielding factor in [0, 1], broadcast to the common input shape.

        Raises
        ------
        KeyError
            If ``isotop`` is not present in the loaded table.
        ValueError
            If ``b_kms`` differs from the loaded table's Doppler parameter.

        References
        ----------
        Visser et al. 2009, A&A 503, 323; see the shielding guide.
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

        Nco_b, Nh2_b = np.broadcast_arrays(Nco, Nh2)
        out = np.ones_like(Nco_b, dtype=float)

        unshielded = (Nco_b < N_SHIELD_MIN) & (Nh2_b < N_SHIELD_MIN)
        need_interp = ~unshielded
        if np.any(need_interp):
            logNco = np.log10(np.maximum(Nco_b[need_interp], N_SHIELD_MIN))
            logNh2 = np.log10(np.maximum(Nh2_b[need_interp], N_SHIELD_MIN))
            out[need_interp] = self._grids[iso].theta(logNco, logNh2)

        return np.clip(out, 0.0, 1.0)

    def theta_interpolated_b(
        self,
        isotop: str,
        Nco: ArrayLike,
        Nh2: ArrayLike,
        b_kms: ArrayLike,
    ) -> np.ndarray:
        """Evaluate shielding with interpolation between bracketing b tables.

        Interpolation is linear in ``b`` and logarithmic in ``theta`` after the
        normal in-table interpolation over ``log10(Nco)`` and ``log10(Nh2)``.
        Files are restricted to the same table family as the loaded file
        (same excitation temperature and isotope-ratio suffix).
        """
        iso = isotop.lower()
        bvals, filenames = self._available_b_family()
        if bvals.size == 0:
            return self.theta(iso, Nco, Nh2, b_kms=self.b_kms)

        Nco_arr = np.asarray(Nco, float)
        Nh2_arr = np.asarray(Nh2, float)
        b_arr = np.asarray(b_kms, float)
        Nco_b, Nh2_b, b_b = np.broadcast_arrays(Nco_arr, Nh2_arr, b_arr)

        out = np.ones_like(Nco_b, dtype=float)
        unshielded = (Nco_b < N_SHIELD_MIN) & (Nh2_b < N_SHIELD_MIN)
        need_interp = ~unshielded
        if not np.any(need_interp):
            return out

        Nco_work = Nco_b[need_interp]
        Nh2_work = Nh2_b[need_interp]
        b_work = np.nan_to_num(b_b[need_interp], nan=float(self.b_kms))
        b_work = np.clip(b_work, float(bvals[0]), float(bvals[-1]))

        hi_idx = np.searchsorted(bvals, b_work, side="right")
        lo_idx = np.maximum(hi_idx - 1, 0)
        hi_idx = np.minimum(hi_idx, bvals.size - 1)

        theta_work = np.empty_like(b_work, dtype=float)
        for lo, hi in sorted(set(zip(lo_idx.tolist(), hi_idx.tolist()))):
            sel = (lo_idx == lo) & (hi_idx == hi)
            th_lo = self._theta_from_file(
                filenames[lo],
                iso,
                Nco_work[sel],
                Nh2_work[sel],
            )
            if lo == hi or bvals[lo] == bvals[hi]:
                theta_work[sel] = th_lo
                continue

            th_hi = self._theta_from_file(
                filenames[hi],
                iso,
                Nco_work[sel],
                Nh2_work[sel],
            )
            frac = (b_work[sel] - bvals[lo]) / (bvals[hi] - bvals[lo])
            log_th = (
                (1.0 - frac) * np.log(np.clip(th_lo, 1.0e-300, 1.0))
                + frac * np.log(np.clip(th_hi, 1.0e-300, 1.0))
            )
            theta_work[sel] = np.exp(log_th)

        out[need_interp] = np.clip(theta_work, 0.0, 1.0)
        return np.clip(out, 0.0, 1.0)

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

    @staticmethod
    def _table_family(name: str) -> str:
        m = re.match(r"shield\.[^.]+\.(.+)$", name)
        return m.group(1) if m else ""

    def _available_b_family(self) -> tuple[np.ndarray, list[str]]:
        family = self._table_family(self.filepath.name)
        pairs: list[tuple[float, str]] = []
        for fname in self.available_files():
            if self._table_family(fname) != family:
                continue
            path = self.data_dir / fname
            try:
                b_val = self._parse_co_b_from_file(path)
            except Exception:
                b_val = np.nan
            if np.isfinite(b_val):
                pairs.append((float(b_val), fname))
        pairs.sort(key=lambda item: item[0])
        unique: list[tuple[float, str]] = []
        for b_val, fname in pairs:
            if unique and abs(unique[-1][0] - b_val) <= 1.0e-12:
                continue
            unique.append((b_val, fname))
        if not unique:
            return np.array([], dtype=float), []
        return (
            np.array([item[0] for item in unique], dtype=float),
            [item[1] for item in unique],
        )

    def _grids_for_file(self, filename: str) -> Dict[str, ShieldingGrid2D]:
        if filename not in self._grid_cache:
            self._grid_cache[filename] = self._parse_visser_file(self.data_dir / filename)
        return self._grid_cache[filename]

    def _theta_from_file(
        self,
        filename: str,
        isotop: str,
        Nco: np.ndarray,
        Nh2: np.ndarray,
    ) -> np.ndarray:
        grids = self._grids_for_file(filename)
        if isotop not in grids:
            raise KeyError(
                f"Isotopologue {isotop} not in {filename}. "
                f"Available: {list(grids)}"
            )
        Nco = np.asarray(Nco, float)
        Nh2 = np.asarray(Nh2, float)
        Nco_b, Nh2_b = np.broadcast_arrays(Nco, Nh2)
        out = np.ones_like(Nco_b, dtype=float)
        unshielded = (Nco_b < N_SHIELD_MIN) & (Nh2_b < N_SHIELD_MIN)
        need_interp = ~unshielded
        if np.any(need_interp):
            logNco = np.log10(np.maximum(Nco_b[need_interp], N_SHIELD_MIN))
            logNh2 = np.log10(np.maximum(Nh2_b[need_interp], N_SHIELD_MIN))
            out[need_interp] = grids[isotop].theta(logNco, logNh2)
        return np.clip(out, 0.0, 1.0)

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
