# db-keywords: gow17, radmc3d, field, io, serialization, paths
# db-role: canonical
# db-scope: package
# db-purpose: Package module for gow17, radmc3d, field, io.

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


LAMDA_URLS = {
    "co": "https://home.strw.leidenuniv.nl/~moldata/datafiles/co.dat",
    "catom": "https://home.strw.leidenuniv.nl/~moldata/datafiles/catom.dat",
    "hco+": "https://home.strw.leidenuniv.nl/~moldata/datafiles/hco%2B.dat",
}

LAMDA_COLLIDER_ID_TO_NAME = {
    1: "h2",
    2: "p-h2",
    3: "o-h2",
    4: "e",
    5: "h",
    6: "he",
    7: "h+",
}

GOW17_LAMDA_COLLIDERS = {
    "co": ["p-h2", "o-h2"],
    "catom": ["h", "p-h2", "o-h2", "e"],
    "hco+": ["p-h2", "o-h2"],
}


def _is_collision_block_header(line: str) -> bool:
    text = line.strip().upper()
    return text.startswith("!COLLISIONS BETWEEN") or text.startswith("!PARTNER ")


def gow17_lamda_colliders(species: str) -> list[str]:
    key = str(species).lower().strip()
    try:
        return list(GOW17_LAMDA_COLLIDERS[key])
    except KeyError as exc:
        raise ValueError(
            f"No strict GOW17/LAMDA collider policy exists for {species!r}. "
            f"Supported line species are {sorted(GOW17_LAMDA_COLLIDERS)}."
        ) from exc


def read_lamda_collision_order(path: str | Path) -> list[str]:
    path = Path(path)
    lines = path.read_text().splitlines()
    order: list[str] = []
    for i, line in enumerate(lines):
        if not _is_collision_block_header(line):
            continue
        j = i + 1
        while j < len(lines):
            text = lines[j].strip()
            if text and not text.startswith("!"):
                collider_id = int(text.split()[0])
                try:
                    order.append(LAMDA_COLLIDER_ID_TO_NAME[collider_id])
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown LAMDA collider id {collider_id} in {path}"
                    ) from exc
                break
            j += 1
    return order


def assert_lamda_collision_order(path: str | Path, expected: list[str]) -> None:
    actual = read_lamda_collision_order(path)
    if actual != expected:
        raise ValueError(
            f"Collision partner order mismatch in {Path(path).name}.\n"
            f"Expected strict GOW17/LAMDA order: {expected}\n"
            f"Found file order: {actual}\n\n"
            "RADMC-3D maps collider density fields to collision-rate tables by order. "
            "This run is refused."
        )


def _collision_blocks_by_name(path: str | Path) -> tuple[list[str], dict[str, list[str]], list[str]]:
    path = Path(path)
    lines = path.read_text().splitlines()
    block_starts = [
        i
        for i, line in enumerate(lines)
        if _is_collision_block_header(line)
    ]
    if not block_starts:
        return lines, {}, []

    trailer_start = len(lines)
    for i in range(block_starts[-1] + 1, len(lines)):
        text = lines[i].strip().upper()
        if text.startswith("! NOTES") or text.startswith("!NOTES"):
            trailer_start = i
            break

    header = lines[: block_starts[0]]
    trailer = lines[trailer_start:] if trailer_start < len(lines) else []
    blocks: dict[str, list[str]] = {}
    for n, start in enumerate(block_starts):
        stop = block_starts[n + 1] if n + 1 < len(block_starts) else trailer_start
        block = lines[start:stop]
        collider_name = None
        for raw in block[1:]:
            text = raw.strip()
            if text and not text.startswith("!"):
                collider_id = int(text.split()[0])
                collider_name = LAMDA_COLLIDER_ID_TO_NAME[collider_id]
                break
        if collider_name is None:
            raise ValueError(f"Could not identify collision block in {path}")
        blocks[collider_name] = block
    return header, blocks, trailer


def write_reordered_lamda_file(
    source: str | Path,
    destination: str | Path,
    colliders: list[str],
) -> Path:
    """Write a molecule file with collision blocks matching ``colliders``."""

    source = Path(source)
    destination = Path(destination)
    header, blocks, trailer = _collision_blocks_by_name(source)
    missing = [collider for collider in colliders if collider not in blocks]
    if missing:
        raise ValueError(f"{source} is missing requested collision block(s): {missing}")

    out = list(header)
    for i, line in enumerate(out):
        if line.strip().upper().startswith("!NUMBER OF COLL PARTNERS"):
            if i + 1 >= len(out):
                raise ValueError(f"Malformed molecule file header in {source}")
            out[i + 1] = str(len(colliders))
            break
    else:
        raise ValueError(f"Missing !NUMBER OF COLL PARTNERS in {source}")

    for collider in colliders:
        out.extend(blocks[collider])
    out.extend(trailer)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out) + "\n")
    assert_lamda_collision_order(destination, colliders)
    return destination


def ensure_official_lamda_file(species: str, moldata_dir: str | Path) -> Path:
    """Download the current canonical LAMDA file for ``species``.

    The online file is fetched on every call and atomically replaces the local
    copy only after a successful download. Network failures are propagated;
    an existing local file is never used as a fallback.

    Parameters
    ----------
    species : str
        Supported LAMDA species identifier.
    moldata_dir : path-like
        Directory in which the refreshed canonical file is stored.

    Returns
    -------
    pathlib.Path
        Refreshed local file path.
    """

    species = str(species).lower().strip()
    moldata_dir = Path(moldata_dir)
    moldata_dir.mkdir(parents=True, exist_ok=True)
    if species not in LAMDA_URLS:
        raise ValueError(f"No official LAMDA URL configured for {species!r}")
    path = moldata_dir / f"{species}.dat"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{species}.",
        suffix=".download",
        dir=moldata_dir,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        if shutil.which("curl") is None:
            raise RuntimeError(
                "Current online LAMDA staging requires the 'curl' executable"
            )
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--output",
                str(temporary_path),
                LAMDA_URLS[species],
            ],
            check=True,
        )
        if temporary_path.stat().st_size == 0:
            raise ValueError(f"LAMDA returned an empty file for {species!r}")
        available_colliders = read_lamda_collision_order(temporary_path)
        missing_colliders = [
            collider
            for collider in gow17_lamda_colliders(species)
            if collider not in available_colliders
        ]
        if missing_colliders:
            raise ValueError(
                f"Downloaded LAMDA file for {species!r} is missing required "
                f"collision block(s): {missing_colliders}"
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def install_validated_molecule_file(
    species: str,
    moldata_dir: str | Path,
    inputs_dir: str | Path,
) -> Path:
    species = str(species).lower().strip()
    source = ensure_official_lamda_file(species, moldata_dir)
    destination = Path(inputs_dir) / f"molecule_{species}.inp"
    expected = gow17_lamda_colliders(species)
    actual = read_lamda_collision_order(source)
    if actual == expected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    else:
        write_reordered_lamda_file(source, destination, expected)
    assert_lamda_collision_order(destination, gow17_lamda_colliders(species))
    return destination
