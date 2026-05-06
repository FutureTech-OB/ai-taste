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

## Reproduce Management Statistics

```bash
python3 scripts/reproducibility/compute_s19_pairwise_sft_kappa.py
python3 scripts/reproducibility/compute_s20_sft_consensus_per_class.py
python3 scripts/reproducibility/compute_s21_ai_human_complementarity.py
python3 scripts/reproducibility/compute_s22_mcnemar_compendium.py
python3 scripts/reproducibility/compute_s23_headroom_captured.py
python3 scripts/reproducibility/smoke_test_management_stats.py
```

Expected smoke-test result:

```text
Smoke test passed: all package reference numbers reproduced from S19-S23.
```

## Refresh Manifest

```bash
python3 scripts/build_release_manifest.py
```

Output:

- `docs/release_manifest.csv`

## Expected Working Directory

All commands above assume the current working directory is the repository root.
