# Reproducibility Scripts

This folder contains the Python sources used by the public reproduction scripts.

Main components:

- `generate_main_figure2.py`, `generate_main_figures.py`, `generate_main_figure6.py`
- `generate_extended_and_supplementary_figures.py` (ED1-ED7, Supplementary Figures 1-6, ST4/ST7 figure assets)
- `recompute_public_tables_and_stats.py`
- `recompute_core_tables_and_stats.py`
- `recompute_public_support_tables.py`
- `vendor_analysis/analysis/`
- `figure_style_policy.py`

From the repository root, the public entry points are:

```bash
PYTHON_BIN=python3 bash scripts/reproduce_tables.sh
PYTHON_BIN=python3 bash scripts/reproduce_figures.sh
```

Generated figure outputs are written under `reproduced/` as local build output. Released tables are refreshed in place under `data/tables/`. The public table pipeline rebuilds `data/statistics/S01-S12` in place, and the figure pipeline also regenerates `data/statistics/S15_CoreRQShortTransferStats.json` for Extended Data Fig. 7.

`data/statistics/S13_Figure6NumbersAudit.json` and `data/statistics/S14_ED2PairwiseRawPValues.json` are shipped audit artifacts consumed by the released figure pipeline and are validated as packaged rather than regenerated. The bundled short-input transfer source file is `data/predictions/core_rq_short_transfer_predictions.jsonl`, which packages the one-sentence idea statement, the matched full idea summary, and the ED7 released predictions in one public JSONL.
