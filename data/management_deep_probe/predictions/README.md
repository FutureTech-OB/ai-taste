# Management Prediction Records

This directory contains prediction records for analyses on the 120-item management benchmark.
It contains released prediction outputs only; provider request logs and
training logs are not included.

| File | Use |
|---|---|
| `frontier_10models_8runs.jsonl` | Frontier-model multi-run management predictions |
| `frontier_thinking_11models_singleshot.jsonl` | Single-shot eleven-model frontier reasoning cohort used for paired McNemar and headroom support |
| `frontier_prompt_sensitivity_expert_12models_8runs.jsonl` | Expert-rubric prompt-sensitivity records used for the management frontier headline and prompt-selection analysis |
| `frontier_prompt_sensitivity_simple_8models_8runs.jsonl` | Simplified-rubric prompt-sensitivity records |
| `frontier_prompt_sensitivity_journal_7models_8runs.jsonl` | Journal-anchored prompt-sensitivity records |
| `sft_predictions.jsonl` | Main supervised fine-tuned model predictions |
| `chat_predictions.jsonl` | Chat/base comparison predictions |
| `rl_predictions.jsonl` | Reinforcement-learning mechanism-probe evaluation outputs; includes eight stochastic votes, majority/tie metadata, and per-pitch accuracy fields, not training logs |
| `sft_temporal_old_predictions.jsonl` | Temporal robustness predictions |
| `core_rq_short_transfer_predictions.jsonl` | Core-question transfer variant predictions |

Prediction records use the unified four-tier label space: `exceptional`, `strong`, `fair`, and `limited`.
