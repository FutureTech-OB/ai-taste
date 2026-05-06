# Scripts

Run all commands from the repository root.

## Validate

```bash
python3 scripts/validate_package.py
```

Checks required files, figure coverage, provenance coverage, supplementary table coverage, JSON syntax, row counts, pairwise data coverage, and path hygiene.

## Reproduce Tables

```bash
python3 scripts/reproduce_tables.py
```

Copies the canonical machine-readable supplementary tables to `reproduced/tables/` and writes checksums.

## Reproduce Figure Assets

```bash
python3 scripts/reproduce_figures.py
```

Copies the released main and supplementary figure assets to `reproduced/figures/` and writes checksums.

## Build Manifest

```bash
python3 scripts/build_release_manifest.py
```

Regenerates `docs/release_manifest.csv`.

## Reproduce v6.3 Statistics

The five statistics added in the v6.3 update are reproduced by scripts in
`scripts/reproducibility/`. See
[`scripts/reproducibility/README.md`](reproducibility/README.md) for the
mapping from each script to its Supplementary Table or Section.

Quick run:

```bash
python3 scripts/reproducibility/compute_s19_pairwise_sft_kappa.py
python3 scripts/reproducibility/compute_s20_sft_consensus_per_class.py
python3 scripts/reproducibility/compute_s21_ai_human_complementarity.py
python3 scripts/reproducibility/compute_s22_mcnemar_compendium.py
python3 scripts/reproducibility/compute_s23_headroom_captured.py
python3 scripts/reproducibility/smoke_test_v6_3.py
```
