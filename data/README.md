# Data

This folder contains the released inputs and numerical outputs needed for the public reproduction path.

| Directory | Contents |
|-----------|----------|
| `benchmark/` | Benchmark article records |
| `predictions/` | Model prediction files used in the paper and the economics extension figures |
| `pairwise/` | Pairwise comparison outputs and summary metrics |
| `human_ratings/` | De-identified human ratings and reproducibility inputs |
| `tables/` | Released tables and figure-data files |
| `statistics/` | Released summary statistics |

For the public rebuild:

- `scripts/reproduce_tables.sh` rebuilds `T01` through `T21` and `S01` through `S12`
- `scripts/reproduce_figures.sh` regenerates released `Figure2` through `Figure7`, `ExtendedDataFigure1` through `ExtendedDataFigure8`, `SupplementaryFigure1` through `SupplementaryFigure7`, plus `S15` through `S18` from repository-local inputs

`Figure1` is retained as a shipped frozen canonical asset rather than regenerated in the public figure pipeline.

Two index files are the main entry points for readers:

- `tables/TABLE_INDEX.csv`
- `tables/FIGURE_DATA_INDEX.csv`

Raw prediction JSONL files and some table `Model Key` columns intentionally retain stable canonical keys such as `CYqJRxId`, `ckpt-step-304`, `ckppt-380`, and `ckppt-228`. The public human-readable model-name mapping for those keys is documented in `data/predictions/README.md` and `data/predictions/sft_checkpoint_key_mapping.csv`.

Human data in this release are de-identified. Raw survey exports, model checkpoints, and unreleased preprocessing intermediates are not included.

Training code, training-data preparation notes, and open-weight checkpoint downloads are maintained separately at [FutureTech-OB/AI-Taste-Training](https://github.com/FutureTech-OB/AI-Taste-Training).
