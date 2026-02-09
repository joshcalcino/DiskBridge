from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, find_packages, setup


def _read_pyproject() -> dict:
    root = Path(__file__).resolve().parent
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(str(pyproject_path))

    import tomllib

    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _project_metadata(pyproject: dict) -> dict:
    project = pyproject.get("project", {})
    return {
        "name": project.get("name", "diskbridge"),
        "version": project.get("version", "0.0.0"),
        "description": project.get("description", ""),
        "python_requires": project.get("requires-python", ">=3.9"),
        "install_requires": project.get("dependencies", []),
    }


def _gow17_extension() -> Extension:
    root = Path(__file__).resolve().parent

    sundials_prefix = root / "dependencies" / "sundials" / "install"
    sundials_include = sundials_prefix / "include"
    sundials_lib = sundials_prefix / "lib"

    include_dirs = [
        str(root / "external" / "pdr"),
        str(sundials_include),
        pybind11.get_include(),
        np.get_include(),
    ]

    library_dirs = [str(sundials_lib)]

    libraries = [
        "sundials_cvode",
        "sundials_sunlinsoldense",
        "sundials_sunmatrixdense",
        "sundials_nvecserial",
        "sundials_core",
    ]

    extra_compile_args = ["-O3", "-std=c++17", "-fopenmp"]

    extra_link_args: list[str] = ["-fopenmp"]
    runtime_library_dirs: list[str] = []
    if os.name != "nt":
        runtime_library_dirs = [str(sundials_lib)]

    sources = [
        "src/diskbridge/_gow17.cpp",
        "external/pdr/cvodeDense.cpp",
        "external/pdr/gow17.cpp",
        "external/pdr/interp.cpp",
        "external/pdr/ode.cpp",
        "external/pdr/radfield.cpp",
        "external/pdr/shielding.cpp",
        "external/pdr/slab.cpp",
        "external/pdr/sundial.cpp",
        "external/pdr/thermo.cpp",
    ]

    return Extension(
        name="diskbridge._gow17",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        runtime_library_dirs=runtime_library_dirs,
    )


def main() -> None:
    pyproject = _read_pyproject()
    meta = _project_metadata(pyproject)

    setup(
        name=meta["name"],
        version=meta["version"],
        description=meta["description"],
        python_requires=meta["python_requires"],
        install_requires=meta["install_requires"],
        package_dir={"": "src"},
        packages=find_packages("src"),
        package_data={"diskbridge": ["data/*", "templates/*"]},
        ext_modules=[_gow17_extension()],
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
