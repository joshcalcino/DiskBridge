"""Generate native RADMC-3D opacity files with the standalone optool executable."""

# db-keywords: dust, opacity, radmc3d, units, io, serialization
# db-role: canonical
# db-scope: package
# db-purpose: File-native optool execution for existing dust bins and wavelengths.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from diskbridge._units import Quantity
from diskbridge.model.dust_opacity import DustOpacity, _run_optool, resolve_optool_executable


def _density_material_file(source: Path, destination: Path, density: float) -> Path:
    """Set only a private optical-input density header, preserving every n,k row."""
    lines = source.read_text().splitlines()
    first = next(i for i, line in enumerate(lines)
                 if line.strip() and not line.lstrip().startswith(("#", "!", "*")))
    # A native porous blend may round its density to zero. Its two-column
    # count/density line is still a header and is replaced with model density.
    has_header = len(lines[first].split()) == 2
    data_start = first + int(has_header)
    count = sum(bool(line.strip()) and not line.lstrip().startswith(("#", "!", "*"))
                for line in lines[data_start:])
    content = lines[:first] + [f"{count} {density:.17g}"] + lines[data_start:]
    destination.write_text("\n".join(content) + "\n")
    return destination


def write_dust_opacity(
    opacity: DustOpacity,
    grain_size: Quantity,
    wavelengths: Quantity,
    output_dir: str | Path,
    *,
    species_name: str,
    scattering_mode: int,
    threads: int = 1,
    executable: str | Path | None = None,
    opacity_dir: str | Path | None = None,
    bulk_density: Quantity | None = None,
) -> Path:
    """Write one existing grain bin's native optool RADMC opacity table.

    Parameters
    ----------
    opacity : diskbridge.DustOpacity
        Physical material, shape and numerical sampling specification.
    grain_size : pint.Quantity
        Positive representative radius, convertible to microns.
    wavelengths : pint.Quantity
        Positive strictly increasing wavelength vector with at least two values.
    output_dir : str or pathlib.Path
        Directory receiving the native table, log and provenance JSON.
    species_name : str
        Safe RADMC species basename, including any bin index.
    scattering_mode : int
        RADMC mode from 0 through 5; modes at least 3 write full matrix tables.
    threads : int, optional
        Positive OpenMP thread count, by default 1.
    executable : str or pathlib.Path, optional
        Standalone optool executable; otherwise OPTOOL_EXECUTABLE or PATH.
    opacity_dir : str or pathlib.Path, optional
        Optical-constants directory for registered species.
    bulk_density : pint.Quantity, optional
        Authoritative stored grain density, including porosity. The model's
        writer supplies its bin density so opacity, settling and chemistry
        retain one grain-mass normalization across saved-model reuse.

    Returns
    -------
    pathlib.Path
        Native dustkappa_<species>.inp or dustkapscatmat_<species>.inp path.

    Notes
    -----
    Per-mass Gaussian opacity weights map to grain-number weights proportional
    to weight/radius**3. Optool owns scattering, angular conversion and native
    table serialization. Scalar output uses optool's 180-angle default;
    DustOpacity.n_angles controls full-matrix output only.
    """
    if not isinstance(opacity, DustOpacity):
        raise TypeError("opacity must be a DustOpacity specification")
    if not isinstance(grain_size, Quantity) or not isinstance(wavelengths, Quantity):
        raise TypeError("grain_size and wavelengths must be Pint quantities")
    radius = np.asarray(grain_size.to("micron").magnitude, dtype=float)
    waves = np.asarray(wavelengths.to("micron").magnitude, dtype=float)
    if radius.ndim or not np.isfinite(radius) or radius <= 0:
        raise ValueError("grain_size must be a positive finite scalar")
    if (waves.ndim != 1 or waves.size < 2 or not np.all(np.isfinite(waves))
            or np.any(waves <= 0) or np.any(np.diff(waves) <= 0)):
        raise ValueError("wavelengths must be a positive increasing vector with at least two values")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", species_name):
        raise ValueError("species_name must be a safe RADMC basename")
    if len(species_name) > 50:
        raise ValueError("species_name exceeds optool's 50-character label limit")
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("threads must be a positive integer")
    if isinstance(scattering_mode, bool) or int(scattering_mode) != scattering_mode or not 0 <= scattering_mode <= 5:
        raise ValueError("scattering_mode must be an integer from zero to five")
    if bulk_density is not None:
        if not isinstance(bulk_density, Quantity):
            raise TypeError("bulk_density must be a Pint quantity")
        bulk = np.asarray(bulk_density.to("g/cm^3").magnitude, dtype=float)
        if bulk.ndim or not np.isfinite(bulk) or bulk <= 0:
            raise ValueError("bulk_density must be a positive finite scalar")
        bulk_density = float(bulk)
    binary = resolve_optool_executable(executable)
    material, resolved_density, material_files = opacity._material_arguments(opacity_dir)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "dustkapscatmat" if scattering_mode >= 3 else "dustkappa"
    output = directory / f"{prefix}_{species_name}.inp"
    log_path = directory / f"optool_{species_name}.log"
    if opacity.size_width:
        radii = np.exp(np.linspace(np.log(float(radius)) - 3 * opacity.size_width,
                                  np.log(float(radius)) + 3 * opacity.size_width,
                                  int(opacity.size_samples)))
        weights = np.exp(-0.5 * (np.log(radii / float(radius)) / opacity.size_width)**2)
        weights /= weights.sum()
    else:
        radii, weights = np.array([float(radius)]), np.array([1.0])
    number_weights = weights / radii**3
    number_weights /= number_weights.sum()
    with tempfile.TemporaryDirectory(prefix="diskbridge_opacity_", dir=directory) as temporary:
        work = Path(temporary)
        wave_file, size_file = work / "wavelengths.inp", work / "sizes.inp"
        np.savetxt(wave_file, waves, fmt="%.17g", header=str(len(waves)), comments="")
        np.savetxt(size_file, np.column_stack((radii, number_weights)), fmt="%.17g",
                   header=str(len(radii)), comments="")
        porosity, blend_log = opacity.porosity, ""
        needs_blend = resolved_density is None or len(material_files) != 1
        if needs_blend:
            if bulk_density is None:
                bulk_density = (resolved_density if resolved_density is not None else
                                float(opacity.resolve_material(opacity_dir, binary)[1].to("g/cm^3").magnitude))
            blend, _, blending = opacity._blend_material(material, wave_file, work, binary,
                                                          require_density=False)
            corrected = _density_material_file(blend, work / "resolved_material.lnk", bulk_density)
            material = ["-c", corrected.name, "1", repr(bulk_density)]
            porosity = 0.0  # The blended index and density already include vacuum.
            blend_log = blending.stdout + blending.stderr
        else:
            if bulk_density is None:
                bulk_density = resolved_density
            intrinsic = bulk_density / (1 - opacity.porosity)
            corrected = _density_material_file(material_files[0], work / "resolved_material.lnk", intrinsic)
            material = ["-c", corrected.name, "1", repr(intrinsic)]
        command = [str(binary), "-err", *material, "-p", str(porosity)]
        command += ["-mie"] if opacity.method == "mie" else ["-dhs", str(opacity.fmax)]
        command += ["-a", size_file.name, "-l", wave_file.name, "-chop", str(opacity.chop_angle),
                    "-radmc", species_name, "-o", ".", "-q"]
        if scattering_mode >= 3:
            command += ["-s", str(opacity.n_angles)]
        try:
            result = _run_optool(command, cwd=work, threads=int(threads), log_path=log_path)
        finally:
            if blend_log:
                log_path.write_text(blend_log + (log_path.read_text() if log_path.exists() else ""))
        native = work / output.name
        if not native.is_file() or not native.stat().st_size:
            raise RuntimeError(f"Optool did not produce {output.name}; see {log_path}: {result.stdout}{result.stderr}")
        with native.open() as stream:
            data_lines = (line.strip() for line in stream if line.strip() and not line.lstrip().startswith(("#", "!", "*")))
            try:
                header = [int(next(data_lines)) for _ in range(3 if scattering_mode >= 3 else 2)]
            except (ValueError, StopIteration) as error:
                raise RuntimeError(f"Optool produced an incomplete native opacity table: {native}") from error
        expected = [1, len(waves), opacity.n_angles + 1] if scattering_mode >= 3 else [3, len(waves)]
        if header != expected:
            raise RuntimeError(f"Unexpected optool native opacity header {header}; expected {expected}")
        version_result = _run_optool([str(binary), "-version"], cwd=work)
        version = (version_result.stdout or version_result.stderr).strip()
        if not version:
            raise RuntimeError("Optool did not report its version")
        native.replace(output)
    provenance = dict(opacity=opacity.to_dict(), grain_radius_um=float(radius),
                      bulk_density_g_cm3=bulk_density, wavelengths_um=waves.tolist(),
                      smoothing_radii_um=radii.tolist(), per_mass_weights=weights.tolist(),
                      command=command, threads=int(threads),
                      executable=dict(path=str(binary), version=version,
                                      sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),
                      optical_constants=[dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                         for path in material_files])
    (directory / f"optool_{species_name}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    opposite = "dustkappa" if scattering_mode >= 3 else "dustkapscatmat"
    (directory / f"{opposite}_{species_name}.inp").unlink(missing_ok=True)
    return output
