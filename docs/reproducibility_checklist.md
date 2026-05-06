# Reproducibility Checklist

External-user verification checklist:

- [x] Create a clean Python environment.
- [x] Install dependencies from `requirements.txt`.
- [x] Run `python3 scripts/validate_package.py`.
- [x] Confirm all seven cross-field benchmarks have 200 records and non-null `rq_with_context`.
- [x] Run `python3 scripts/reproduce_tables.py`.
- [x] Confirm `reproduced/tables/table_checksums.csv` is created.
- [x] Run `python3 scripts/reproduce_figures.py`.
- [x] Confirm `reproduced/figures/figure_asset_checksums.csv` is created.
- [x] Run `python3 scripts/build_release_manifest.py`.
- [x] Confirm `docs/release_manifest.csv` is current.
- [x] Search for machine-specific paths.
- [x] Confirm repository landing-page metadata are current.

External users can rerun the same checklist from a clean clone. The generated `reproduced/` directory is intentionally not included in the release tree; the reproduction scripts recreate it.
