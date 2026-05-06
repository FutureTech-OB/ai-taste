# Cross-Field Standardized Data

This block contains the standardized seven-field analyses for the non-management fields, the historical GPT-5.2 chat/log-probability comparator, and GPT-5.5 all-field audit records used for Figure 4b, Supplementary Table ST2b, and Supplementary Figure 11.

| Path | Purpose |
|---|---|
| `benchmarks/` | Seven field-specific 200-item benchmark files |
| `predictions/` | Standardized cross-field prediction records, historical GPT-5.2 chat/log-probability comparator records, and GPT-5.5 audit records |
| `statistics/` | Cross-field calibration, ensemble, transfer, benchmark summary, and current-model audit statistics |
| `journal_tiers/` | Journal-to-tier lookup files for the six additional fields |

The seven standardized non-management fields are economics, business and finance, communication, political science, psychology multidisciplinary, public administration, and sociology. The GPT-5.2 chat/log-probability comparator and GPT-5.5 audit files additionally include management so that the current-model comparison aligns with the eight-field benchmark set.

Economics is included in this standardized block rather than treated as a separate benchmark family. The six additional fields use the same row schema and prediction format; `published_year` and `core_rq_short` are set to null where those values were not present in the available source files.
