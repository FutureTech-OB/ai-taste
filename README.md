# Reproducibility Package

This repository contains the external-use reproducibility package for:

**LLMs learn scientific taste from institutional traces across the social sciences**

The package is organized so a researcher can validate the released data files, reproduce the machine-readable supplementary tables, and reproduce the released figure asset bundle from a clean clone.

## Contents

| Path | Contents |
|---|---|
| `figures/main/` | Final main figure assets |
| `figures/supplementary/` | Final supplementary figure assets |
| `figures/provenance/` | Figure captions, panel-level provenance files, and support statistics |
| `data/management_deep_probe/` | Management 120-item benchmark, prediction records, human ratings, pairwise comparisons, statistics, and support tables |
| `data/cross_field_standardized/` | Seven-field benchmark and prediction records plus GPT-5.2 historical chat/log-probability comparator records and GPT-5.5 all-field audit records |
| `data/supplementary_tables/` | Machine-readable Supplementary Tables ST1-ST24 and ST2b |
| `data/figure_support/` | Figure-to-data support index |
| `scripts/` | Validation, table reproduction, figure asset reproduction, manifest generation, and statistics-reproduction scripts |
| `docs/` | Data dictionary, reproducibility instructions, checklist, and manifest |

## Availability Links

- Reproducibility package: [FutureTech-OB/ai-taste](https://github.com/FutureTech-OB/ai-taste)
- Training code, training-data preparation and release notes, and open-weight checkpoints: [FutureTech-OB/AI-Taste-Training](https://github.com/FutureTech-OB/AI-Taste-Training)

Use the reproducibility package as the main public repository for released data, validation scripts, table outputs, and figure assets. Use the training repository only for training resources and checkpoints.

## Environment

Requirements:

- Python 3.10 or newer

Recommended setup:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

The validation and manifest scripts use only the Python standard library. The requirements file includes the scientific Python stack needed for downstream analysis and figure regeneration extensions.

## Validate The Package

From the repository root:

```bash
python3 scripts/validate_package.py
```

The validator checks required files, row counts, JSON syntax, figure/provenance coverage, supplementary table coverage, pairwise data coverage, seven-field benchmark schema, and path hygiene.

## Reproduce Released Outputs

Reproduce machine-readable supplementary tables:

```bash
python3 scripts/reproduce_tables.py
```

Reproduce the released figure asset bundle and checksums:

```bash
python3 scripts/reproduce_figures.py
```

Refresh the file manifest:

```bash
python3 scripts/build_release_manifest.py
```

Generated outputs are written under `reproduced/`, which is ignored by version control.

Reproduce the five management statistics that back Supplementary Tables ST21-ST24
and the headroom note:

```bash
python3 scripts/reproducibility/compute_s19_pairwise_sft_kappa.py
python3 scripts/reproducibility/compute_s20_sft_consensus_per_class.py
python3 scripts/reproducibility/compute_s21_ai_human_complementarity.py
python3 scripts/reproducibility/compute_s22_mcnemar_compendium.py
python3 scripts/reproducibility/compute_s23_headroom_captured.py
python3 scripts/reproducibility/smoke_test_management_stats.py
```

Each script writes a JSON file under
`data/management_deep_probe/statistics/` (S19-S23) with byte-stable
formatting (sorted keys, fixed precision). The smoke test verifies all
five outputs against the package reference numbers.

## Scope Notes

The package includes the minimized data files needed to verify the reported tables, figure data, and final figure assets, including the GPT-5.5 all-field audit used by Figure 4b, Supplementary Table ST2b, and Supplementary Figure 11. Live model inference, model training runs, provider-hosted model weights, and unreleased model checkpoints are not redistributed. Provider model names, access windows, and prediction records are documented in `data/supplementary_tables/ST19_model_inventory.csv`.

For paper-facing availability text, see `docs/availability.md`.

## License

Data are released under CC BY 4.0. Code is released under the MIT License.
