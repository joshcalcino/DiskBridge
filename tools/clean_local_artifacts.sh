#!/bin/sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  sh tools/clean_local_artifacts.sh [--dry-run]

What it removes by default:
  - Python bytecode and cache directories
  - macOS metadata files
  - pytest cache
  - mypy and ruff caches
  - setuptools/build outputs
  - copied binary extension artifacts
  - native SUNDIALS install/build artifacts that should be rebuilt locally

Safe mode does NOT remove:
  - example run outputs
  - validation outputs
  - user-created working directories
  - downloaded/source dependencies

Options:
  --dry-run   Print what would be removed
  -h, --help  Show this help message
EOF
}

remove_path() {
  target=$1
  if [ ! -e "$target" ]; then
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    printf 'would remove %s\n' "$target"
    return 0
  fi

  rm -rf -- "$target"
  printf 'removed %s\n' "$target"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

printf 'repo root: %s\n' "$ROOT_DIR"
if [ "$DRY_RUN" -eq 1 ]; then
  printf 'dry-run: yes\n'
fi

SAFE_PATHS="
.pytest_cache
.mypy_cache
.ruff_cache
build
dist
src/diskbridge.egg-info
dependencies/sundials/install
dependencies/sundials/build
"

printf '%s\n' "$SAFE_PATHS" | while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

find . -type d -name '__pycache__' -print | while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

find . \( -name '.DS_Store' -o -name '._*' \) -print | while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

find src -maxdepth 2 -type f \( -name '_gow17*.so' -o -name '*.so' -o -name '*.dylib' -o -name '*.pyd' \) -print |
while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

find dependencies/sundials -type f \( -name '*.so' -o -name '*.dylib' -o -name '*.a' -o -name '*.o' \) -print 2>/dev/null |
while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

find dependencies/sundials -type f \( -name 'CMakeCache.txt' -o -name 'cmake_install.cmake' \) -print 2>/dev/null |
while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

find dependencies/sundials -type d -name 'CMakeFiles' -print 2>/dev/null |
while IFS= read -r target; do
  [ -n "$target" ] || continue
  remove_path "$target"
done

printf 'done\n'
