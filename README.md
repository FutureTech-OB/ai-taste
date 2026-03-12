# ai-taste

This repository releases the paper, figures, tables, de-identified evaluation data, and reproducibility code for the study:

**"Fine-tuned AI learns tacit scientific judgment from institutional traces."**

The study tests whether supervised fine-tuning can teach language models to evaluate research idea quality. The release includes a 120-article benchmark, released training-data article pools, model predictions, de-identified human ratings, released figures and tables, and the scripts used to regenerate the public results from repository-local files.

## Contents

| Path | Contents |
|------|----------|
| `manuscript/` | Release paper PDF |
| `figures/` | Main, extended data, and supplementary figures plus figure notes |
| `data/benchmark/` | Benchmark article records |
| `data/predictions/` | Model prediction files |
| `data/pairwise/` | Pairwise comparison outputs |
| `data/human_ratings/` | De-identified human ratings and stable anonymous reproducibility inputs |
| `data/tables/` | Released tables and figure-data files |
| `data/statistics/` | Released statistics used by the analysis pipeline |
| `data/train_data/` | Released training-data article pools for recent-trace and old-trace SFT preparation |
| `scripts/` | Validation and reproduction scripts |

## Setup

Requirements:

- Python 3.10+

Install dependencies from the repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Reproducing the release

From the repository root:

```bash
./.venv/bin/python scripts/validate_package.py
PYTHON_BIN=./.venv/bin/python bash scripts/reproduce_tables.sh
PYTHON_BIN=./.venv/bin/python bash scripts/reproduce_figures.sh
```

These commands:

- validate the package structure
- rebuild the released public tables and core statistics
- regenerate the released figures

Regenerated outputs are written under `reproduced/` as local build output.

## Scope

This release is intended to let outside readers inspect the benchmark and evaluation outputs, trace released figures and tables to repository-local inputs, and rerun the public analysis pipeline.

Key release indices:

- `data/tables/TABLE_INDEX.csv`
- `data/tables/FIGURE_DATA_INDEX.csv`

The package rebuilds:

- `data/tables/T01` through `data/tables/T21`
- `data/statistics/S01` through `data/statistics/S14`

`data/statistics/S13_Figure6NumbersAudit.json` and `data/statistics/S14_ED2PairwiseRawPValues.json` are included as shipped audit artifacts used by Figures 6 and 5 / Extended Data Figure 2.

## Data note

Human-rating files are de-identified. The public release excludes direct participant identifiers, raw survey exports, model checkpoints, and unreleased preprocessing intermediates beyond the shipped `data/train_data` article pools.

## Citation and license

Citation and license files will be added with the public release.
