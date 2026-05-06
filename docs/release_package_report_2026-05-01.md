# Release Package Report

Date: 2026-05-01

This report records the preparation of a journal-neutral reproducibility package for external use.

## Final Folder Structure

```text
README.md
LICENSE
requirements.txt
manuscript/
figures/
  main/
  supplementary/
  provenance/
data/
  management_deep_probe/
    benchmark/
    predictions/
    human_ratings/
    pairwise/
    statistics/
    support_tables/
  cross_field_standardized/
    benchmarks/
    predictions/
    statistics/
    journal_tiers/
  supplementary_tables/
  figure_support/
scripts/
docs/
```

## Files Kept And Why

- Manuscript markdown and references: source text for the submitted paper.
- Main and supplementary figure assets: final visual outputs reported in the paper.
- Figure provenance files: panel-level support for all main figures and supplementary figures.
- Management deep-probe data: 120-item benchmark, management prediction records, de-identified human ratings, pairwise outcomes, statistics, and support tables needed for management-specific analyses.
- Cross-field standardized data: seven 200-item benchmarks, canonical prediction records for economics, business and finance, communication, political science, psychology multidisciplinary, public administration, and sociology, plus the all-eight-field GPT-5.2 historical comparator records and May 2026 GPT-5.5 audit records.
- Supplementary Tables ST1-ST24 and ST2b: machine-readable table release with `TABLE_INDEX.csv`.
- Validation and reproduction scripts: commands for clean-clone package checks, table reproduction, figure asset reproduction, and manifest generation.
- Release manifest: file-by-file list of public package files, type, size, and supported output. The manifest intentionally excludes `docs/release_manifest.csv` itself to keep repeated manifest regeneration stable.

## Files Removed Or Excluded And Why

- Obsolete figure-generation scaffolds and template scripts: not used by the reported results and likely to confuse external users.
- Redundant prompt-variant prediction diagnostics: not cited by the paper.
- Duplicate source prediction files: merged into canonical release prediction surfaces where possible.
- Provider-local checkpoint identifiers: replaced by stable release aliases.
- Raw survey exports, request logs, training logs, model checkpoints, and cache files: not required to reproduce reported outputs and not appropriate for release.
- Earlier field-distance analysis files: not reported in the paper.
- Journal-tier decision-history files: replaced with clean lookup tables and per-field mapping files.
- Old path layout using separate management/economics and six-field directories: replaced with `management_deep_probe/` and `cross_field_standardized/`.

## Dataset Field Notes

- `published_year` for the six additional cross-field benchmarks: intentionally null for 1,200 rows because those values were not present in the available source files.
- `core_rq_short` for the six additional cross-field benchmarks: intentionally null for 1,200 rows because those shorter summaries were not present in the available source files.
- The management benchmark predates the standardized seven-field schema and does not include `published_year` or `core_rq_short`; its released `rq_with_context` field is the evaluation input used for the reported management analyses.

## Validation Commands

Commands run from the package root:

```bash
python3 -m py_compile scripts/*.py
python3 scripts/build_release_manifest.py
python3 scripts/validate_package.py
python3 scripts/reproduce_tables.py
python3 scripts/reproduce_figures.py
```

Results:

- `python3 -m py_compile scripts/*.py`: passed.
- `python3 scripts/build_release_manifest.py`: passed; wrote `docs/release_manifest.csv` with 214 files, excluding the manifest file itself.
- `python3 scripts/validate_package.py`: passed; checked 142 required files, 18 directories, JSON syntax, row counts, seven-field benchmark schema, and path hygiene.
- `python3 scripts/reproduce_tables.py`: passed; reproduced 22 table files to `reproduced/tables/` and wrote checksums.
- `python3 scripts/reproduce_figures.py`: passed; reproduced 32 figure assets to `reproduced/figures/` and wrote checksums.
- The generated `reproduced/` directory was removed after validation so the release tree contains only source package files. Rerun the commands above to recreate it.

Search checks:

- No matches for local or template paths: macOS home paths, Windows home paths, template-package paths, release-workspace paths, local training-output paths, or legacy prediction paths.
- No matches for provider-local fine-tune identifiers or API-secret markers.
- No package-label matches for old target-journal wording in public documentation, scripts, data documentation, or figure provenance.

May 2026 GPT-5.5 audit update:

- Supplementary Figure 11 assets and provenance were added to the public figure release.
- GPT-5.5 chat/log-probability, GPT-5.5 High item summary, and GPT-5.5 High run-level records were added under `data/cross_field_standardized/predictions/`.
- GPT-5.5 selective-prediction, calibration-band, high-reasoning field-summary, and paired item-level chat-versus-high-reasoning audit statistics were added under `data/cross_field_standardized/statistics/`.
- `ST2b_current_gpt55_all_field_audit.csv` was added to `data/supplementary_tables/`, and the figure-data index now includes SF11.

Manuscript/reference update:

- The public manuscript source, Supplementary Information source, reference markdown, and BibTeX files were synchronized after a selective reference expansion.
- The reference list now includes targeted support for LLM review limitations, automated AI research, grant-review expertise/scoring variability, chain-of-thought limitations, novelty/frontier evaluation bias, and SI-only instruction-tuning background.
- `docs/release_manifest.csv` was regenerated after the final manuscript, reference, figure, and documentation fixes and contains 214 files; `scripts/validate_package.py` passed with 142 required files checked.

v6.3 final-check update:

- The public manuscript source and Supplementary Information source were synchronized from the current final-check markdown sources.
- The management prompt-sensitivity prediction records referenced by the SI were added under the public `data/management_deep_probe/predictions/` surface and the package SI path references were remapped to those public files.
- Machine-readable CSV exports for ST21-ST24 were added under `data/supplementary_tables/`.
- The existing S19-S23 v6.3 stat JSONs and `scripts/reproducibility/` smoke scripts were retained as the executable support layer for ST21-ST24 and the SM4 headroom note.
- Main Figures 1-5 and Supplementary Figures SF1-SF11 were checked against the canonical current figure paths and were already byte-identical.

## Residual Reproducibility Risks

- The release reproduces machine-readable tables and the final figure asset bundle. The original plotting scripts are not retained as default release scripts because they depended on unreleased source directories.
- Live model inference and fine-tuning are not part of deterministic reproduction because provider-side models and hosted weights are not redistributed.
- Some table values are released as curated machine-readable tables rather than recomputed from every raw provider response, because the package intentionally uses minimized prediction records.
