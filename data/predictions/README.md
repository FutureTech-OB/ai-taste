# Prediction Files Overview

## Files
- `frontier_10models_8runs.jsonl`:
  - Canonical conservative frontier cohort for submission analyses.
  - Contains 11 frontier models with 8-run protocol structure.
- `sft_predictions.jsonl`:
  - Contains four SFT model tracks and comparator tracks used in analysis.
  - Four SFT base-model IDs: `CYqJRxId`, `ckpt-step-304`, `ckppt-380`, `ckppt-228`.
- `chat_predictions.jsonl`:
  - Chat/log-probability evaluator predictions.
- `rl_predictions.jsonl`:
  - RL checkpoint prediction outputs used for RL-vs-SFT comparison context.
- `gemini_3_1_pro_standalone.jsonl`:
  - Standalone Gemini 3.1 Pro source file used when building the canonical frontier file.
- `prompt_variants/`:
  - Prompt-sensitivity prediction files (`expert`, `simple`, `journal`).

## Gemini Coverage Note
- The conservative 11-model frontier cohort includes **Gemini 2.5 Pro** and **Gemini 3.1 Pro**.
- Gemini 3.1 Pro source predictions are provided in `gemini_3_1_pro_standalone.jsonl` and merged into the canonical frontier file during table/figure recomputation.
- Gemini prompt-variant diagnostics are represented in prompt-variant files where applicable.
