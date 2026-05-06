# Cross-Field Prediction Records

This directory contains standardized prediction records for the seven non-management fields plus all-eight-field GPT-5.2/GPT-5.5 current-model comparison records.
It contains released model-output records only; live inference calls and
provider logs are outside the package.

| File | Use |
|---|---|
| `seven_field_core_predictions.jsonl` | Canonical cross-field prediction file across seven fields, eight evaluator tracks, and 1,400 benchmark items |
| `economics_predictions.jsonl` | Released economics prediction records preserved as a field-specific source surface |
| `management_sft_transfer_to_seven_fields.jsonl` | Management-SFT transfer predictions evaluated across the seven non-management fields |
| `gpt52_chat_logp_all_fields.jsonl` | GPT-5.2 chat/log-probability comparator records for Figure 4b across all eight field benchmarks |
| `gpt55_chat_logp_all_fields.jsonl` | GPT-5.5 chat/log-probability records for Figure 4b, SI Table ST2b, and Supplementary Figure 11 across all eight field benchmarks |
| `gpt55_high_reasoning_item_summary.jsonl` | GPT-5.5 High item-level eight-run summaries used for SI Table ST2b and Supplementary Figure 11 |
| `gpt55_high_reasoning_runs.jsonl` | GPT-5.5 High per-run records, eight valid runs for each of the 1,520 benchmark items |

The canonical file contains 11,200 records: 7 fields x 8 evaluator tracks x 200 benchmark items.
