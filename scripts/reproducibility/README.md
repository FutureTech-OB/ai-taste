# Reproducibility Scripts

This folder contains the Python sources used by the public reproduction scripts.

Main components:

- `generate_main_figure2.py`, `generate_main_figures.py`, `generate_main_figure6.py`
- `generate_extended_and_supplementary_figures.py`
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

Generated figure outputs are written under `reproduced/` as local build output. Released tables and statistics are refreshed in place under `data/tables/` and `data/statistics/`.
