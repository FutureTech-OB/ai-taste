# Management Benchmark

`management_benchmark_120.jsonl` contains the 120 management benchmark records used for the management deep-probe analyses.

Common fields include:

| Field | Meaning |
|---|---|
| `title` | Article title |
| `journal` | Source journal |
| `domain` | Management topic/domain label, where available |
| `level` or `rank` | Unified four-tier label |
| `rq_with_context` | Released research-idea text used for evaluation |

The management benchmark predates the standardized seven-field schema and does not include `published_year` or `core_rq_short` fields. The released `rq_with_context` text is the evaluation input used for the reported management analyses.
