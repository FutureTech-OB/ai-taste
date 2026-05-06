# Data Dictionary

## Management Deep-Probe Data

`data/management_deep_probe/benchmark/management_benchmark_120.jsonl` contains the 120 management benchmark records. Management prediction files are under `data/management_deep_probe/predictions/`.

Common management fields:

| Field | Meaning |
|---|---|
| `title` | Article title |
| `journal` | Source journal |
| `domain` | Management topic/domain label, where available |
| `level` or `rank` | Unified four-tier label |
| `rq_with_context` | Released research-idea text used for evaluation |
| `prediction` | Resolved evaluator prediction |
| `logp` | First-token log-probability dictionary, where available |
| `vote_counts` | Multi-sample vote-count dictionary, where available |
| `correct` | Whether the prediction matches the unified ground-truth label |

## Cross-Field Standardized Data

Cross-field benchmarks are under `data/cross_field_standardized/benchmarks/`. Each of the seven fields has 200 records and 50 records per tier.

Common benchmark fields:

| Field | Meaning |
|---|---|
| `benchmark_id` | Release-stable benchmark identifier |
| `field` | Normalized field key |
| `field_display` | Human-readable field name |
| `title` | Article title |
| `journal` | Source journal |
| `published_year` | Publication year, where available |
| `rank` | Unified four-tier label |
| `rq_with_context` | Released research-idea text used for evaluation |
| `core_rq_short` | Shorter core-question text, where available |
| `source_baseid` | Source row identifier, where available |

`data/cross_field_standardized/predictions/seven_field_core_predictions.jsonl` is the canonical standardized prediction file for the seven non-management fields.

Common prediction fields:

| Field | Meaning |
|---|---|
| `benchmark_id` | Join key to the benchmark record |
| `field` | Normalized field key |
| `model_key` | Release-stable evaluator key |
| `model_display` | Human-readable evaluator name |
| `model_role` | Evaluator family, such as SFT, base, or frontier |
| `gold` | Ground-truth unified label |
| `prediction` | Resolved evaluator prediction |
| `top2_prediction` | Second-ranked prediction, where available |
| `logp` | Log-probability dictionary, where available |
| `vote_counts` | Multi-sample vote-count dictionary, where available |
| `correct` | Whether `prediction` matches `gold` |

## Human Rating Files

Human rating files under `data/management_deep_probe/human_ratings/` contain de-identified article-level summaries and release-stable rater pseudonyms. They exclude names, email addresses, institutions, IP addresses, and survey-system identifiers.

## Supplementary Tables

`data/supplementary_tables/TABLE_INDEX.csv` maps Supplementary Table identifiers ST1-ST24 and ST2b to CSV files.

## Figure Provenance

`data/figure_support/FIGURE_DATA_INDEX.csv` maps each main and supplementary figure to supporting package files. `figures/provenance/` contains caption sidecars and panel-specific data/provenance files.
