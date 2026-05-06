# Reproducibility

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Validate

```bash
python3 scripts/validate_package.py
```

Expected result:

```text
Release-package validation passed.
```

## Reproduce Tables

```bash
python3 scripts/reproduce_tables.py
```

Outputs:

- `reproduced/tables/*.csv`
- `reproduced/tables/table_checksums.csv`

## Reproduce Figure Assets

```bash
python3 scripts/reproduce_figures.py
```

Outputs:

- `reproduced/figures/main/*`
- `reproduced/figures/supplementary/*`
- `reproduced/figures/figure_asset_checksums.csv`

## Refresh Manifest

```bash
python3 scripts/build_release_manifest.py
```

Output:

- `docs/release_manifest.csv`

## Expected Working Directory

All commands above assume the current working directory is the repository root.
