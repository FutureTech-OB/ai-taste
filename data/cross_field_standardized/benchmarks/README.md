# Cross-Field Benchmarks

This directory contains one 200-item benchmark file for each of the seven non-management fields:
These benchmark files are released evaluation surfaces with a shared schema.

| File | Field |
|---|---|
| `economics_benchmark_200.jsonl` | Economics |
| `business_finance_benchmark_200.jsonl` | Business and finance |
| `communication_benchmark_200.jsonl` | Communication |
| `political_science_benchmark_200.jsonl` | Political science |
| `psychology_multidisciplinary_benchmark_200.jsonl` | Psychology, multidisciplinary |
| `public_administration_benchmark_200.jsonl` | Public administration |
| `sociology_benchmark_200.jsonl` | Sociology |

Each file contains 50 `exceptional`, 50 `strong`, 50 `fair`, and 50 `limited` records.

Common fields:

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

Current source coverage: economics includes `published_year` and `core_rq_short`; the six additional fields include `rq_with_context` but have null `published_year` and null `core_rq_short`.
