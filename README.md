# ai-taste

This repository releases the public reproducibility package for the study:

**Paper title:** Fine-tuned language models learn tacit scientific judgment from institutional traces

The study tests whether supervised fine-tuning can teach language models to evaluate research idea quality. This snapshot includes the 120-article management benchmark, the 200-article economics extension benchmark, model predictions, de-identified human ratings, released figures and tables, documentation describing the withheld training corpus, and the scripts used to regenerate the public results from repository-local files. The final approved paper PDF and synced manuscript sources are not yet included in `manuscript/`.

## Contents

| Path | Contents |
|------|----------|
| `manuscript/` | Placeholder directory for the final public paper assets; not required for the current rerun path |
| `figures/` | Main, extended data, and supplementary figures plus figure notes |
| `data/benchmark/` | Management benchmark records plus the released economics extension benchmark |
| `data/predictions/` | Model prediction files, including the released economics extension and pooled management+economics prediction sets |
| `data/pairwise/` | Pairwise comparison outputs |
| `data/human_ratings/` | De-identified human ratings and stable anonymous reproducibility inputs |
| `data/tables/` | Released tables and figure-data files |
| `data/statistics/` | Released statistics used by the analysis pipeline |
| `data/train_data/` | Training-data placeholder (full data released upon publication) |
| `scripts/` | Validation and reproduction scripts |

## Setup

Requirements:

- Python 3.10+

Install dependencies from the repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The validated public rerun path below does not require `pandoc` or any external document-conversion toolchain. A repo-local `.venv/` is supported and ignored by git.

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
- regenerate the released figures and the ED7 / economics-extension statistics artifacts

Regenerated outputs are written under `reproduced/` as ignored local build output; `reproduced/` is not part of the shipped release surface.

## Scope

This release is intended to let outside readers inspect the benchmark and evaluation outputs, trace released figures and tables to repository-local inputs, and rerun the public analysis pipeline. In this snapshot, `manuscript/` remains placeholder-only; the public code/data rerun path does not depend on any manuscript asset.

Key release indices:

- `data/tables/TABLE_INDEX.csv`
- `data/tables/FIGURE_DATA_INDEX.csv`

The package rebuilds:

- `data/tables/T01` through `data/tables/T21`
- `data/statistics/S01` through `data/statistics/S12`
- `data/statistics/S15_CoreRQShortTransferStats.json` during the figure rebuild
- `data/statistics/S16_EconomicsExtensionStats.json` during the figure rebuild
- `data/statistics/S17_PooledFieldExtensionStats.json` during the figure rebuild
- `data/statistics/S18_CrossFieldTransferStats.json` during the figure rebuild

`data/statistics/S13_Figure6NumbersAudit.json` and `data/statistics/S14_ED2PairwiseRawPValues.json` are included as shipped audit artifacts and are validated as packaged. Figure 6 is rebuilt directly from the released prediction and human-rating files; the pairwise panels in Figure 5 / Extended Data Figure 2 use `S14_ED2PairwiseRawPValues.json` together with the released pairwise outputs. The bundled short-input transfer source file is `data/predictions/core_rq_short_transfer_predictions.jsonl`, which stores the one-sentence idea statement, the matched full idea summary, and the released short-input model outputs used for Extended Data Fig. 7. The economics extension figures use `data/benchmark/economics_benchmark_articles_200.jsonl`, `data/predictions/economics_predictions.jsonl`, and `data/predictions/pooled_management_economics_predictions.jsonl`.

## Data note

Human-rating files are de-identified. The public release excludes direct participant identifiers, raw survey exports, and model checkpoints. Training data will be released upon publication.

The `manuscript/` directory is intentionally placeholder-only in this release snapshot. The final approved paper PDF and any synced manuscript sources can be added later without affecting the repo-local table/figure reproduction path documented above.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
