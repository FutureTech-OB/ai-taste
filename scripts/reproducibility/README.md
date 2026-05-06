# v6.3 Reproduction Scripts

These scripts reproduce the five Supplementary statistics added in the v6.3
update. Each script reads only files inside `data/management_deep_probe/`
and writes a single JSON output under
`data/management_deep_probe/statistics/`. Outputs are byte-stable across
machines (sorted dict keys, fixed float precision).

| Script | Output | Defends |
|---|---|---|
| `compute_s19_pairwise_sft_kappa.py` | `S19_PairwiseSFTKappa.json` | ST21 — pairwise Cohen's kappa across the four management SFT models, plus Fleiss kappa contrast for expert and student raters. |
| `compute_s20_sft_consensus_per_class.py` | `S20_SFTConsensusPerClass.json` | ST22 — coverage and per-class accuracy on the SFT 4/4 unanimous consensus subset. |
| `compute_s21_ai_human_complementarity.py` | `S21_AIHumanComplementarity.json` | ST23 — AI–human complementarity 2x2 contingency and oracle ceiling for the SFT 2-model ensemble against expert and student plurality. |
| `compute_s22_mcnemar_compendium.py` | `S22_McNemarCompendium.json` | ST24 — pairwise McNemar tests of the SFT 2-model ensemble against four comparators (frontier-mean plurality, best frontier, expert majority, junior majority). |
| `compute_s23_headroom_captured.py` | `S23_HeadroomCaptured.json` | SM4 — headroom-captured statistic and SFT-over-comparator multipliers used in the main-text headroom interpretation. |

## Usage

Run each script from the package root. Each prints a one-line confirmation
matching the v6.3 reference numbers:

```bash
python3 scripts/reproducibility/compute_s19_pairwise_sft_kappa.py
python3 scripts/reproducibility/compute_s20_sft_consensus_per_class.py
python3 scripts/reproducibility/compute_s21_ai_human_complementarity.py
python3 scripts/reproducibility/compute_s22_mcnemar_compendium.py
python3 scripts/reproducibility/compute_s23_headroom_captured.py
```

To verify all outputs against the v6.3 reference numbers:

```bash
python3 scripts/reproducibility/smoke_test_v6_3.py
```

## Conventions

- **Label space (4-class):** exceptional, strong, fair, limited.
- **Argmax tie-break** for SFT logp -> single label: alphabetical AI-label-space
  ordering (`exceptional < fair < limited < strong`). Implemented via
  `sorted()` to avoid the dict-insertion-order trap.
- **Plurality vote** for human and frontier panels: `tie_policy='exclude'`
  (drop tied articles from the denominator).
- **Frontier mean** in the McNemar table is the per-article plurality across
  the eleven thinking-model `prediction` fields (alphabetical exclusion on
  ties). This is the editorially-useful pairing for a paired test; the
  alternative scalar mean-of-individual-accuracies is reported separately
  when needed.
- **Best frontier** = `google/gemini-3.1-pro-preview`.
- **Primary SFT ensemble pair** = `gpt-4.1-nano-ob` + `qwen3-30b-ob`,
  exposed under the package key `best_2_model_combo` in
  `sft_predictions.jsonl`.

## Dependencies

Standard library only. No `numpy` or `scipy` is required: the McNemar
chi-square survival function is computed via `math.erfc`.
