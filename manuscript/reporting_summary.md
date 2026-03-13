# Reporting summary

## Study design

- Held-out benchmark of 120 article-derived research pitches in organizational psychology and management, balanced across four journal-outcome tiers (`30` pitches per tier; chance baseline `25%`).
- Source articles underlying the benchmark were published after June 30, 2025. The benchmark is fully disjoint from the supervised fine-tuning slices used in the package.
- Main supervised fine-tuning slice: `4,479` processed research-pitch / journal-outcome pairs. Temporal old-trace slice: `3,368` processed pairs, used only for the descriptive persistence check.
- Main evaluator comparison spans 26 AI configurations plus two human panels under one frozen four-tier rubric.

## Human panels and exclusions

- Expert panel: `48` gatekeepers, `384` total ratings, `8` benchmark pitches per rater.
- Junior panel: `174` doctoral or postdoctoral raters, `2,530` total ratings after excluding responses completed in under one minute per pitch.
- Total human ratings retained for analysis: `2,914`.
- Human tier labels were mapped deterministically onto the shared four-tier output space used for the AI evaluations.

## Ethics

- The human study was approved by institutional review board review (`Project No. THU-04-2026-0034`).
- Released human-rating files are de-identified.

## Statistics and uncertainty

- Primary endpoint: four-class exact-match accuracy on the `120`-pitch benchmark.
- Secondary endpoints: macro-F1, per-tier precision / recall / F1, confusion matrices, calibration metrics, selective-prediction accuracy, and inter-rater agreement.
- Paired evaluator comparisons use McNemar tests on paired correctness vectors. Frontier-average comparison is reported with the released exact one-sided binomial test. Frontier-cohort heterogeneity uses Cochran's `Q`.
- Pairwise-comparison significance for Fig. 5 / Extended Data Fig. 2 uses two-sided exact McNemar/binomial tests on discordant pairs and is reported as raw unadjusted `P` values in the shipped package tables and statistics files.
- The compressed-input transfer analysis in Extended Data Fig. 7 compares one-sentence idea statements with full idea summaries on the same 120 held-out articles; one-sided exact binomial tests versus the `25%` chance baseline are used there only as compact above-chance diagnostics.
- Confidence intervals and uncertainty bands follow the released table / figure contracts: bootstrap resampling (`10,000` draws) for macro-F1 and resampling-based summaries, plus binomial or Wilson intervals where those are the released figure-table definitions.
- Multiple-testing corrections are applied only where explicitly stated in the manuscript or tables.

## Data, code, and release boundary

- Released package contents include the manuscript markdown sources, this reporting summary, figure assets and notes, benchmark records, model predictions, pairwise outputs, de-identified human ratings, released tables (`T01-T21`), released statistics (`S01-S15`), the training-data placeholder documentation, and the validator / reproduction scripts.
- Public entry points for rerunning the released analysis are:
  - `python3 scripts/validate_package.py`
  - `PYTHON_BIN=python3 bash scripts/reproduce_tables.sh`
  - `PYTHON_BIN=python3 bash scripts/reproduce_figures.sh`
- The public table pipeline rebuilds `T01-T21` and `S01-S12` from package-local inputs. The public figure pipeline also regenerates `S15_CoreRQShortTransferStats.json` from the bundled one-sentence transfer prediction file.
- `S13_Figure6NumbersAudit.json` and `S14_ED2PairwiseRawPValues.json` are shipped audit artifacts that are validated as packaged and consumed by the released figure pipeline.
- The package excludes direct participant identifiers, raw survey exports, model checkpoints, unreleased preprocessing intermediates, and the full internal training corpus; the supervised fine-tuning data are represented here by a release-boundary placeholder and documentation only.
