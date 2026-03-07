#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TABLE_DIR="$ROOT_DIR/data/tables"

count=$(find "$TABLE_DIR" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')
echo "Found $count table CSV files in data/tables"

"$PYTHON_BIN" "$ROOT_DIR/scripts/reproducibility/recompute_public_tables_and_stats.py"

"$PYTHON_BIN" - <<'PY' "$TABLE_DIR"
import csv
import sys
from pathlib import Path

table_dir = Path(sys.argv[1])

for p in sorted(table_dir.glob('*.csv')):
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        rows = sum(1 for _ in f) - 1
    print(f"{p.name}: {max(rows, 0)} rows")

# Focused CI contract checks used by manuscript figures.
def read_rows(path: Path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

errors = []

# T04: accuracy + macro-F1 CIs should be present for key evaluator rows
need_rows = {
    'Expert Majority Vote (excl. ties)',
    'Student Majority Vote (full, excl. ties)',
    'Expert Individual Average',
    'Student Individual Average',
    'Flagship Average (11 models)',
    'SFT 2-Model Ensemble',
}
t04 = read_rows(table_dir / 'T04_AIvsHumanSummary.csv')
by_eval = {r.get('Evaluator', ''): r for r in t04}
for ev in sorted(need_rows):
    if ev not in by_eval:
        errors.append(f"T04 missing evaluator row: {ev}")
        continue
    row = by_eval[ev]
    for c in ['95% CI Lower', '95% CI Upper', 'Macro F1 95% CI Lower', 'Macro F1 95% CI Upper']:
        if str(row.get(c, '')).strip() == '':
            errors.append(f"T04 blank CI value for {ev} / {c}")

for name in ['T11_MonteCarloMatchedPanel.csv', 'T19_StudentReliabilityByPanelSize.csv', 'T20_ExpertReliabilityByPanelSize.csv']:
    rows = read_rows(table_dir / name)
    if not rows:
        errors.append(f"{name} has no rows")
        continue
    cols = set(rows[0].keys())
    if name == 'T11_MonteCarloMatchedPanel.csv':
        for c in ['ci_lower', 'ci_upper']:
            if c not in cols:
                errors.append(f"{name} missing column: {c}")
    else:
        for c in ['accuracy_ci_lower', 'accuracy_ci_upper']:
            if c not in cols:
                errors.append(f"{name} missing column: {c}")

if errors:
    print('CI/table contract: FAILED')
    for e in errors:
        print(f'- {e}')
    raise SystemExit(1)

print('CI/table contract: PASS')
PY

echo "Table check completed"
