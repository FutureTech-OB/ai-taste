# Data

This directory contains the released data surface for reproducing the paper tables, figure data, and final figures.

## Directory Guide

| Path | Purpose |
|---|---|
| `management_deep_probe/` | Management-specific 120-item benchmark, model predictions, human ratings, pairwise comparisons, statistics, and support tables |
| `cross_field_standardized/` | Standardized seven-field benchmark records plus GPT-5.2 historical chat/log-probability comparator records and GPT-5.5 all-field comparison records |
| `supplementary_tables/` | Machine-readable Supplementary Tables ST1-ST24 and ST2b |
| `figure_support/` | Figure-to-data support index |

## Field Blocks

The management block is intentionally separate because it contains the deep-probe analyses: human comparators, pairwise ranking tasks, mechanism tests, transfer checks, and management-only support statistics.

The cross-field block standardizes the seven non-management fields used in the paper: economics, business and finance, communication, political science, psychology multidisciplinary, public administration, and sociology. Each field has a 200-item benchmark with 50 items per tier. The same block also holds GPT-5.2 historical chat/log-probability comparator files and GPT-5.5 all-field comparison files used for Figure 4b, Supplementary Table ST2b, and Supplementary Figure 11.

Schema coverage differs by source block. All released benchmark records include the evaluation text used by the paper (`rq_with_context`) and the unified tier label. Economics also includes populated `published_year` and `core_rq_short`; the six additional non-management fields retain those optional columns as intentional null placeholders because the available source files did not include those values. The management benchmark predates the standardized seven-field schema and does not include those two optional fields.

## Minimization

The release uses derived and minimized files where those files are sufficient to reproduce the reported tables and figures. It excludes raw survey exports, model training logs, provider request logs, model checkpoints, cache directories, and exploratory data not used by the paper.

## Label Space

All released prediction analyses use the unified four-tier label space:

`exceptional`, `strong`, `fair`, `limited`

Source curation labels and human-facing survey labels are mapped to this unified label space in `supplementary_tables/ST18_label_normalization.csv`.
