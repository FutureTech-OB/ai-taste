# Pairwise Evaluation Data (Extended Data Figure 2)

This folder contains the canonical pairwise outcome files used for Extended Data Figure 2.

## Included model folders
- `sft_gpt4_1/`
- `frontier_gemini3_1_pro/`
- `frontier_gpt5_2_high/`
- `baseline_gpt4_1/`

Each folder contains:
- `metrics.json`: aggregate pairwise metrics (weighted accuracy, per-distance metrics, pair-type metrics).
- `pair_results.jsonl`: per-pair trial outcomes with a public `pair_id` join key.

## Scope
- The shared release ships the 4-model subset used by Fig. 5 / Extended Data Fig. 2:
  - SFT GPT-4.1
  - Gemini 3.1 Pro
  - GPT-5.2 High
  - GPT-4.1 baseline
- Raw paired exact McNemar outputs for this plotted comparator set are released in `data/statistics/S14_ED2PairwiseRawPValues.json`.
