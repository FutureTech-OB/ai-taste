# Prediction Files Overview

## Files
- `frontier_10models_8runs.jsonl`:
  - Canonical conservative frontier cohort for submission analyses.
  - Contains 11 frontier models with 8-run protocol structure.
- `sft_predictions.jsonl`:
  - Contains four SFT model tracks and comparator tracks used in analysis.
  - Four SFT base-model IDs: `CYqJRxId`, `ckpt-step-304`, `ckppt-380`, `ckppt-228`.
- `sft_temporal_old_predictions.jsonl`:
  - Minimal old-trace SFT release file used for Extended Data Fig. 6.
  - Contains only the two matched old-trace log-probability tracks needed for the public temporal-persistence comparison.
- `chat_predictions.jsonl`:
  - Chat/log-probability evaluator predictions.
- `rl_predictions.jsonl`:
  - RL checkpoint prediction outputs used for RL-vs-SFT comparison context.
- `gemini_3_1_pro_standalone.jsonl`:
  - Standalone Gemini 3.1 Pro source file used when building the canonical frontier file.
- `prompt_variants/`:
  - Prompt-sensitivity prediction files (`expert`, `simple`, `journal`).

## Public Schema Note
- Vote-based prediction files in this release keep only the public analysis fields:
  - `mode`, `prediction`, `prediction_majority`, `avg_accuracy`, `vote_valid_n`, `vote_counts`, `vote_is_tie`, `vote_predictions`
- SFT files keep only the released log-probability surfaces and ensemble metadata needed by the shipped rebuild scripts.

## Gemini Coverage Note
- The conservative 11-model frontier cohort includes **Gemini 2.5 Pro** and **Gemini 3.1 Pro**.
- Gemini 3.1 Pro source predictions are provided in `gemini_3_1_pro_standalone.jsonl` and merged into the canonical frontier file during table/figure recomputation.
- Gemini prompt-variant diagnostics are represented in prompt-variant files where applicable.
