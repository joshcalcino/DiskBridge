"""Physical dust-opacity settings and consistent material resolution for optool."""

# db-keywords: dust, opacity, config, units, serialization
# db-role: canonical
# db-scope: package
# db-purpose: One dust material and shape specification shared by dust and RT.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType

import numpy as np

from diskbridge._units import Quantity
from .dust_species import SPECIES_REGISTRY, resolve_species_grain_density, species_optconst_path


def resolve_optool_executable(executable: str | Path | None = None) -> Path:
    """Locate an explicit optool executable, OPTOOL_EXECUTABLE, or PATH entry.

    Parameters
    ----------
    executable : str or pathlib.Path, optional
        Explicit standalone executable path or command name.

    Returns
    -------
    pathlib.Path
        Absolute executable path. No installation is performed.
    """
    selected = str(executable or os.environ.get("OPTOOL_EXECUTABLE") or "optool")
    found = shutil.which(os.path.expanduser(selected))
    if found is None:
        raise FileNotFoundError(
            f"Optool executable not found: {selected}. Set OPTOOL_EXECUTABLE "
            "or provide executable= with the standalone executable path."
        )
    return Path(found).resolve()


def _run_optool(command, *, cwd: Path, threads: int = 1, log_path: Path | None = None):
    environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE")
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
    if log_path is not None:
        log_path.write_text(result.stdout + result.stderr)
    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Optool failed with exit status {result.returncode}: {details}")
    return result


def _file_density(path: Path) -> float | None:
    with path.open() as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith(("#", "!", "*")):
                continue
            fields = text.split()
            if len(fields) == 2:
                count, density = map(float, fields)
                if count.is_integer() and count >= 2 and np.isfinite(density) and density > 0:
                    return density
            return None
    raise ValueError(f"Empty optical-constants file: {path}")


@dataclass(frozen=True)
class DustOpacity:
    """Define composition, shape and sampling for one existing dust population.

    Parameters
    ----------
    material : str, pathlib.Path, or mapping of str to float
        Registered DiskBridge species, optool material key, external optical-
        constants path, or a mixture with material mass fractions.
    method : {'mie', 'dhs'}, optional
        Compact spheres or a distribution of hollow spheres; defaults to Mie.
    fmax : float, optional
        Maximum DHS hollow fraction. Defaults to 0 for Mie and 0.8 for DHS.
        This shape parameter is independent of bulk porosity.
    porosity : float, optional
        Bulk vacuum volume fraction, by default 0.
    density : pint.Quantity, optional
        Intrinsic density for a single external material lacking metadata.
        Registered species retain their registered intrinsic density.
    size_width : float, optional
        Smoothing standard deviation in natural log radius, by default 0.05.
        Zero uses one exact representative radius.
    size_samples : int, optional
        Smoothing samples over plus/minus three widths, by default 20.
    n_angles : int, optional
        Even number of optool angular intervals for matrix output, by default
        180. RADMC output contains n_angles + 1 endpoints. Native scalar
        output uses optool's default 180 angular intervals.
    chop_angle : float, optional
        Forward-scattering cap in degrees in optool's convention, by default 1.
    """

    material: str | Path | Mapping[str, float]
    method: str = "mie"
    fmax: float | None = None
    porosity: float = 0.0
    density: Quantity | None = None
    size_width: float = 0.05
    size_samples: int = 20
    n_angles: int = 180
    chop_angle: float = 1.0

    def __post_init__(self):
        if self.method not in {"mie", "dhs"}:
            raise ValueError("method must be 'mie' or 'dhs'")
        fmax = (0.0 if self.method == "mie" else 0.8) if self.fmax is None else float(self.fmax)
        if not np.isfinite(fmax) or not 0 <= fmax < 1 or (self.method == "mie" and fmax != 0):
            raise ValueError("fmax must be in [0, 1), and must be zero for Mie")
        object.__setattr__(self, "fmax", fmax)
        for name, upper in (("porosity", 1.0), ("chop_angle", 180.0)):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value < upper:
                raise ValueError(f"{name} must be finite and in [0, {upper})")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.size_width) or self.size_width < 0:
            raise ValueError("size_width must be finite and nonnegative")
        if (isinstance(self.size_samples, bool) or int(self.size_samples) != self.size_samples
                or self.size_samples < (2 if self.size_width else 1)):
            raise ValueError("size_samples must be a positive integer, at least two for smoothing")
        if (isinstance(self.n_angles, bool) or int(self.n_angles) != self.n_angles
                or self.n_angles < 4 or self.n_angles % 2):
            raise ValueError("n_angles must be an even integer of at least four")
        object.__setattr__(self, "size_width", float(self.size_width))
        object.__setattr__(self, "size_samples", int(self.size_samples))
        object.__setattr__(self, "n_angles", int(self.n_angles))
        if isinstance(self.material, Mapping):
            if not self.material:
                raise ValueError("material mixture must not be empty")
            values = {str(key): float(value) for key, value in self.material.items()}
            if any(not np.isfinite(value) or value <= 0 for value in values.values()):
                raise ValueError("material mass fractions must be positive and finite")
            total = sum(values.values())
            if not np.isfinite(total):
                raise ValueError("total material mass fraction must be finite")
            object.__setattr__(self, "material", MappingProxyType({
                key: values[key] / total for key in sorted(values)
            }))
        elif not isinstance(self.material, (str, Path)) or not str(self.material).strip():
            raise ValueError("material must be a species, optical-constants path, or mass-fraction mapping")
        elif isinstance(self.material, Path):
            object.__setattr__(self, "material", self.material.expanduser().resolve())
        if self.density is not None:
            if not isinstance(self.density, Quantity):
                raise TypeError("density must be a Pint quantity")
            density = self.density.to("g/cm^3")
            if np.ndim(density.magnitude) or not np.isfinite(density.magnitude) or density.magnitude <= 0:
                raise ValueError("density must be a positive finite scalar")
            object.__setattr__(self, "density", density)

    def __reduce__(self):
        return (type(self).from_dict, (self.to_dict(),))

    @property
    def species_name(self) -> str:
        """Return a stable safe basename identifying the complete specification."""
        label = "mixture" if isinstance(self.material, Mapping) else Path(str(self.material)).stem
        label = (re.sub(r"[^A-Za-z0-9_]", "_", label).strip("_") or "dust")[:28]
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{label}_{digest}"

    def to_dict(self) -> dict:
        """Serialize physical and numerical settings.

        Returns
        -------
        dict
            JSON-compatible settings, with optional density stored as a
            value/unit record in g/cm3.
        """
        return dict(material=dict(self.material) if isinstance(self.material, Mapping) else str(self.material),
                    method=self.method, fmax=self.fmax, porosity=self.porosity,
                    density=None if self.density is None else {
                        "value": float(self.density.to("g/cm^3").magnitude), "unit": "g/cm^3"},
                    size_width=float(self.size_width), size_samples=int(self.size_samples),
                    n_angles=int(self.n_angles), chop_angle=self.chop_angle)

    @classmethod
    def from_dict(cls, settings: Mapping) -> "DustOpacity":
        """Restore a serialized opacity specification.

        Parameters
        ----------
        settings : mapping
            Current settings serialized by :meth:`to_dict`.

        Returns
        -------
        DustOpacity
            Immutable material, shape and sampling specification.
        """
        values = dict(settings)
        density = values.get("density")
        if density is not None:
            values["density"] = Quantity(density["value"], density["unit"])
        return cls(**values)

    def _material_arguments(self, opacity_dir=None):
        entries = dict(self.material) if isinstance(self.material, Mapping) else {self.material: 1.0}
        directory = Path(opacity_dir) if opacity_dir is not None else Path(__file__).resolve().parents[3] / "data/opac"
        arguments, densities, files = [], [], []
        for key, fraction in entries.items():
            text = str(key)
            registered = text in SPECIES_REGISTRY
            file_material = registered or isinstance(key, Path) or "/" in text or "\\" in text or text.endswith(".lnk")
            if file_material:
                path = (species_optconst_path(text, directory) if registered else Path(text)).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Optical-constants file not found: {path}")
                intrinsic = (float(resolve_species_grain_density(text, self.density).to("g/cm^3").magnitude)
                             if registered else _file_density(path))
                if self.density is not None and not registered:
                    if len(entries) != 1:
                        raise ValueError("density override requires one external material")
                    supplied = float(self.density.to("g/cm^3").magnitude)
                    if intrinsic is not None and not np.isclose(intrinsic, supplied, rtol=1e-12):
                        raise ValueError("density differs from the registered or file material density")
                    intrinsic = supplied
                if intrinsic is None:
                    raise ValueError(f"External optical constants require density metadata or density=: {path}")
                arguments += ["-c", str(path), repr(fraction), repr(intrinsic)]
                densities.append((fraction, intrinsic))
                files.append(path)
            else:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+.-]*", text):
                    raise ValueError(f"Invalid optool material key: {text}")
                if self.density is not None:
                    raise ValueError("density override is supported for a single external material")
                arguments += ["-c", text, repr(fraction)]
        bulk = None if len(densities) != len(entries) else (1 - self.porosity) / sum(f / rho for f, rho in densities)
        return arguments, bulk, files

    def _blend_material(self, arguments, wavelength_file: Path, directory: Path, executable: Path,
                        *, porosity=None, require_density=True):
        local_arguments = list(arguments)
        for index, value in enumerate(arguments):
            if index and arguments[index - 1] == "-c" and Path(value).is_absolute():
                local = directory / f"material_{index:02d}.lnk"
                shutil.copyfile(value, local)
                local_arguments[index] = local.name
        porosity = self.porosity if porosity is None else porosity
        command = [str(executable), "-err", *local_arguments, "-p", str(porosity),
                   "-a", "0.1", "-l", wavelength_file.name, "-b", "-o", ".", "-q"]
        result = _run_optool(command, cwd=directory)
        blend = directory / "optool_mix.lnk"
        if not blend.is_file():
            raise RuntimeError(f"Optool did not produce the blended material: {result.stdout}{result.stderr}")
        density = _file_density(blend)
        if density is None and require_density:
            raise RuntimeError("Optool produced a blended material without a positive density")
        return blend, density, result

    def resolve_material(self, opacity_dir=None, executable=None) -> tuple[list[str], Quantity]:
        """Resolve material arguments and a consistent bulk grain density.

        Parameters
        ----------
        opacity_dir : str or pathlib.Path, optional
            Optical-constants directory for registered DiskBridge species.
        executable : str or pathlib.Path, optional
            Standalone optool executable for built-in compositions.

        Returns
        -------
        arguments : list of str
            Material-only optool arguments, without a shell command.
        bulk_density : pint.Quantity
            Density in g/cm3, including porosity. Built-in compositions use
            optool's emitted compact-material density multiplied by one minus
            porosity. The opacity writer uses this same density for its
            actual grain-mass normalization.
        """
        arguments, bulk, _ = self._material_arguments(opacity_dir)
        if bulk is None:
            binary = resolve_optool_executable(executable)
            with tempfile.TemporaryDirectory(prefix="diskbridge_material_") as temporary:
                directory = Path(temporary)
                wavelengths = directory / "wavelengths.inp"
                wavelengths.write_text("2\n1\n2\n")
                _, compact, _ = self._blend_material(arguments, wavelengths, directory, binary,
                                                      porosity=0.0)
                bulk = compact * (1 - self.porosity)
        return arguments, Quantity(bulk, "g/cm^3")
