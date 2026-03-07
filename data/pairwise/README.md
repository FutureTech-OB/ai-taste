# Pairwise Evaluation Data (Extended Data Figure 2)

This folder contains the canonical pairwise outcome files used for Extended Data Figure 2.

## Included model folders
- `sft_gpt4_1/`
- `frontier_gemini3_1_pro/`
- `frontier_grok4_1_fast/`
- `frontier_gpt5_2_high/`
- `baseline_gpt4_1/`

Each folder contains:
- `metrics.json`: aggregate pairwise metrics (for example weighted accuracy, per-distance metrics).
- `pair_results.jsonl`: per-pair trial outcomes.

## Scope
- Extended Data Figure 2 consumes a 4-model subset: SFT GPT-4.1, GPT-5.2 High, GPT-4.1 baseline, and Gemini 3.1 Pro.
- `frontier_grok4_1_fast/` is retained as a legacy comparator archive from earlier package snapshots.
