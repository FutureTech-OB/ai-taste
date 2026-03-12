# Data

This folder contains the released inputs and numerical outputs needed for the public reproduction path.

| Directory | Contents |
|-----------|----------|
| `benchmark/` | Benchmark article records |
| `predictions/` | Model prediction files used in the paper |
| `pairwise/` | Pairwise comparison outputs and summary metrics |
| `human_ratings/` | De-identified human ratings and reproducibility inputs |
| `tables/` | Released tables and figure-data files |
| `statistics/` | Released summary statistics |
| `train_data/` | Released training-data article pools for recent-trace and old-trace SFT preparation |

For the public rebuild:

- `scripts/reproduce_tables.sh` rebuilds `T01` through `T21` and `S01` through `S14`
- `scripts/reproduce_figures.sh` regenerates the released figures from repository-local inputs

Two index files are the main entry points for readers:

- `tables/TABLE_INDEX.csv`
- `tables/FIGURE_DATA_INDEX.csv`

Human data in this release are de-identified. Raw survey exports, model checkpoints, and unreleased preprocessing intermediates are not included.
