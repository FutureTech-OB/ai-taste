# Management Deep-Probe Data

This block contains the management-specific analyses built around the 120-item benchmark.
It is scoped to released benchmark, prediction, rater, pairwise, statistic, and
support-table files.

| Path | Purpose |
|---|---|
| `benchmark/` | Released 120-item management benchmark |
| `predictions/` | Management prediction records for frontier, SFT, base, RL, temporal, and transfer analyses |
| `human_ratings/` | De-identified expert and junior-researcher rating summaries plus reproducibility files |
| `pairwise/` | Pairwise ranking outcomes and metrics used in mechanism analyses |
| `statistics/` | JSON summary statistics used by management figures and tables |
| `support_tables/` | Management support tables and `TABLE_INDEX.csv` |

The management block is separate from the cross-field block because it includes additional deep-probe analyses that are not available for the other seven fields.
