# Prediction Files Overview

## Files
- `frontier_10models_8runs.jsonl`:
  - Canonical conservative frontier cohort for the released analyses.
  - Contains 11 frontier models with 8-run protocol structure.
- `sft_predictions.jsonl`:
  - Contains four SFT model tracks and comparator tracks used in analysis.
  - Four SFT base-model IDs: `CYqJRxId`, `ckpt-step-304`, `ckppt-380`, `ckppt-228`.
  - Discrete SFT predictions are derived from `logp` with the canonical label order `exceptional > strong > fair > limited`; exact `logp` ties are therefore broken by that fixed order, not by raw JSON key order.
  - This is important for public reproducibility: naive `argmax(logp)` counting can disagree with the released tables for tied rows, especially for `ckpt-step-304`, while the released tables/figures/manuscript follow the canonical tie rule.
- `sft_temporal_old_predictions.jsonl`:
  - Minimal old-trace SFT release file used for Extended Data Fig. 6.
  - Contains only the two matched old-trace log-probability tracks needed for the public temporal-persistence comparison.
- `core_rq_short_transfer_predictions.jsonl`:
  - Public compressed-input transfer file used for Extended Data Fig. 7.
  - Packages the same held-out 120 articles as one-sentence idea statements together with the matched full idea summaries for side-by-side traceability.
  - Contains the four GPT-family base/SFT prediction tracks used for the harder format-transfer check; no model was trained on this one-sentence format.
- `economics_predictions.jsonl`:
  - Public economics extension prediction file used for Figure 7 and Supplementary Figure 7.
  - Contains the released 200-item economics benchmark predictions for the architecture-matched base/SFT comparisons, the management-trained GPT-4.1 transfer checkpoint, and the released frontier-reference comparators.
- `pooled_management_economics_predictions.jsonl`:
  - Public pooled-field prediction file used for Extended Data Fig. 8.
  - Contains the released management+economics pooled-checkpoint outputs on the combined 320-item held-out surface.
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
- Economics extension files use stable public model keys rather than provider-local checkpoint paths.

## Gemini Coverage Note
- The conservative 11-model frontier cohort includes **Gemini 2.5 Pro** and **Gemini 3.1 Pro**.
- Gemini 3.1 Pro source predictions are provided in `gemini_3_1_pro_standalone.jsonl` and merged into the canonical frontier file during table/figure recomputation.
- Gemini prompt-variant diagnostics are represented in prompt-variant files where applicable.
