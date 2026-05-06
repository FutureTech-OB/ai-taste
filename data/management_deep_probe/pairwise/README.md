# Pairwise Evaluation Data

This directory contains the pairwise ranking outcomes used in Figure 5.
It includes aggregate metrics and per-pair outcomes needed to reproduce the
released pairwise comparisons.

## Model Folders

| Folder | Evaluator |
|---|---|
| `sft_gpt4_1/` | GPT-4.1 SFT |
| `frontier_gemini3_1_pro/` | Gemini 3.1 Pro |
| `frontier_gpt5_2_high/` | GPT-5.2 High |
| `frontier_grok4_1_fast/` | Grok 4.1 Fast |
| `baseline_gpt4_1/` | GPT-4.1 base |

Each folder contains:

- `metrics.json`: aggregate pairwise metrics.
- `pair_results.jsonl`: 300 per-pair trial outcomes with a public `pair_id` join key.

Exact paired comparison support statistics are in `data/management_deep_probe/statistics/S14_PairwiseRawPValues.json`.
