# Extended Data Tables

- `ExtendedDataTable1_BaseModelControls.csv`
  - Architecture-matched base-model controls before supervised fine-tuning, including accuracy, macro-F1, confidence intervals, headroom skill, per-tier accuracy, and prediction counts.
- `ExtendedDataTable2_SFTvsRLComparison.csv`
  - Comparison of supervised fine-tuning and reinforcement-learning summaries for the Qwen architectures and for the pooled aggregate comparison.

Notes:
- Aggregate RL performance is reported as run-pooled avg8 accuracy and non-tied majority-vote accuracy.
- Architecture-specific RL values are not separately reported where the available RL outputs are pooled across architectures.
