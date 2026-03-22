# Reproducibility Scripts

This folder contains the Python sources used by the public reproduction scripts.

Public entry points:

- `Figure1` is shipped as a frozen canonical asset and is not rebuilt by the public figure pipeline
- `build_frontier_diagnostics_figure.py` (Figure 2)
- `build_core_result_figures.py` (Figures 3-5)
- `build_consensus_figure.py` (Figure 6)
- `build_extended_and_supplementary_figures.py` (ED1-ED7, Supplementary Figures 1-6)
- `build_economics_extension_figures.py` (Figure 7, ED8, Supplementary Figure 7)
- `build_release_tables_and_stats.py`

Internal helper modules:

- `_generate_frontier_diagnostics_figure.py`
- `_generate_core_result_figures.py`
- `_generate_consensus_figure.py`
- `_generate_extended_and_supplementary_figures.py`
- `_recompute_core_tables_and_stats.py`
- `_recompute_public_support_tables.py`
- `_recompute_public_tables_and_stats.py`
- `vendor_analysis/analysis/`
- `figure_style_policy.py`

External readers should use the repository-root shell scripts or the `build_*` wrappers above rather than calling the `_`-prefixed helpers directly.

From the repository root, the public entry points are:

```bash
PYTHON_BIN=python3 bash scripts/reproduce_tables.sh
PYTHON_BIN=python3 bash scripts/reproduce_figures.sh
```

Generated figure outputs are written under `reproduced/` as local build output. Released tables are refreshed in place under `data/tables/`. The public table pipeline rebuilds `data/statistics/S01-S12` in place, and the figure pipeline also regenerates `data/statistics/S15_CoreRQShortTransferStats.json`, `data/statistics/S16_EconomicsExtensionStats.json`, `data/statistics/S17_PooledFieldExtensionStats.json`, and `data/statistics/S18_CrossFieldTransferStats.json`.

`data/statistics/S13_Figure6NumbersAudit.json` and `data/statistics/S14_ED2PairwiseRawPValues.json` are shipped audit artifacts that are validated as packaged rather than regenerated. Figure 6 is rebuilt from the released prediction and human-rating files; the pairwise panels read `S14_ED2PairwiseRawPValues.json` together with the released pairwise outputs. The bundled short-input transfer source file is `data/predictions/core_rq_short_transfer_predictions.jsonl`, which packages the one-sentence idea statement, the matched full idea summary, and the ED7 released predictions in one public JSONL. The economics extension rebuild additionally regenerates `S16_EconomicsExtensionStats.json`, `S17_PooledFieldExtensionStats.json`, and `S18_CrossFieldTransferStats.json` from the released economics prediction files.
