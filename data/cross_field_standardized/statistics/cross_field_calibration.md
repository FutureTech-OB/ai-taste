# Cross-Field Calibration and Selective Prediction Analysis

This released summary documents calibration and selective-prediction statistics
using package-relative data files.

**Date:** 2026-04-29
**Fields:** 8 (management, economics, + 6 new social science fields)
**Models:** 3 SFT families (Qwen3-30B, Qwen3-4B, GPT-4.1-nano)

## 1. Confidence Calibration per Model per Field

Confidence = exp(max log-probability). Gap = mean confidence for correct - incorrect predictions.

### Qwen3-30B SFT

| Field | N | Acc | Conf(Correct) | Conf(Incorrect) | Gap | p-value | ECE |
|-------|---|-----|--------------|----------------|-----|---------|-----|
| Mgmt | 120 | 58.3% | 0.755 | 0.657 | +0.098 | 0.002 | 0.146 |
| Econ | 200 | 69.5% | 0.900 | 0.766 | +0.135 | <.001 | 0.178 |
| BusFin | 200 | 53.5% | 0.802 | 0.694 | +0.108 | <.001 | 0.219 |
| Comm | 200 | 67.5% | 0.786 | 0.664 | +0.121 | <.001 | 0.099 |
| PolSci | 200 | 58.5% | 0.765 | 0.673 | +0.092 | <.001 | 0.147 |
| Psych | 200 | 85.5% | 0.929 | 0.698 | +0.231 | <.001 | 0.045 |
| PubAdmin | 200 | 55.0% | 0.700 | 0.565 | +0.134 | <.001 | 0.141 |
| Sociol | 200 | 65.5% | 0.782 | 0.620 | +0.161 | <.001 | 0.086 |

### Qwen3-4B SFT

| Field | N | Acc | Conf(Correct) | Conf(Incorrect) | Gap | p-value | ECE |
|-------|---|-----|--------------|----------------|-----|---------|-----|
| Mgmt | 120 | 59.2% | 0.909 | 0.899 | +0.010 | 0.014 | 0.358 |
| Econ | 200 | 64.0% | 0.777 | 0.594 | +0.183 | <.001 | 0.106 |
| BusFin | 200 | 53.5% | 0.703 | 0.577 | +0.126 | <.001 | 0.110 |
| Comm | 200 | 61.0% | 0.690 | 0.563 | +0.127 | <.001 | 0.051 |
| PolSci | 200 | 50.5% | 0.639 | 0.577 | +0.062 | 0.003 | 0.127 |
| Psych | 200 | 76.0% | 0.876 | 0.623 | +0.253 | <.001 | 0.065 |
| PubAdmin | 200 | 51.0% | 0.678 | 0.547 | +0.130 | <.001 | 0.115 |
| Sociol | 200 | 51.0% | 0.769 | 0.605 | +0.165 | <.001 | 0.180 |

### GPT-4.1-nano SFT

| Field | N | Acc | Conf(Correct) | Conf(Incorrect) | Gap | p-value | ECE |
|-------|---|-----|--------------|----------------|-----|---------|-----|
| Mgmt | 120 | 57.5% | 0.578 | 0.484 | +0.094 | 0.002 | 0.096 |
| Econ | 200 | 68.5% | 0.804 | 0.629 | +0.175 | <.001 | 0.084 |
| BusFin | 200 | 55.5% | 0.700 | 0.586 | +0.114 | <.001 | 0.103 |
| Comm | 200 | 60.5% | 0.683 | 0.586 | +0.097 | <.001 | 0.142 |
| PolSci | 200 | 54.0% | 0.615 | 0.521 | +0.094 | <.001 | 0.078 |
| Psych | 200 | 81.5% | 0.868 | 0.640 | +0.228 | <.001 | 0.045 |
| PubAdmin | 200 | 53.5% | 0.617 | 0.490 | +0.127 | <.001 | 0.092 |
| Sociol | 200 | 54.0% | 0.703 | 0.500 | +0.204 | <.001 | 0.098 |

## 2. Selective Prediction: Accuracy at Top-K% Confidence

Predictions sorted by confidence (descending). Accuracy computed at each coverage level.

### Qwen3-30B SFT

| Field | Top-10% | Top-15% | Top-20% | Top-25% | Top-33% | Top-50% | Top-100% | 80% Threshold | Perfect @K% |
|-------|------|------|------|------|------|------|------|---------------|-------------|
| Mgmt | 100.0% | 88.9% | 75.0% | 73.3% | 71.8% | 70.0% | 58.3% | 15% | 10% |
| Econ | 100.0% | 100.0% | 97.5% | 98.0% | 98.5% | 89.0% | 69.5% | 70% | 15% |
| BusFin | 95.0% | 90.0% | 92.5% | 80.0% | 74.2% | 67.0% | 53.5% | 25% | 2.5% |
| Comm | 100.0% | 96.7% | 92.5% | 92.0% | 83.3% | 79.0% | 67.5% | 55% | 10% |
| PolSci | 80.0% | 73.3% | 67.5% | 72.0% | 74.2% | 69.0% | 58.5% | 10% | 5% |
| Psych | 100.0% | 100.0% | 97.5% | 98.0% | 97.0% | 98.0% | 85.5% | 100% | 15% |
| PubAdmin | 95.0% | 93.3% | 90.0% | 86.0% | 81.8% | 69.0% | 55.0% | 35% | 5% |
| Sociol | 95.0% | 96.7% | 95.0% | 94.0% | 87.9% | 83.0% | 65.5% | 50% | 5% |

### Qwen3-4B SFT

| Field | Top-10% | Top-15% | Top-20% | Top-25% | Top-33% | Top-50% | Top-100% | 80% Threshold | Perfect @K% |
|-------|------|------|------|------|------|------|------|---------------|-------------|
| Mgmt | 91.7% | 83.3% | 75.0% | 80.0% | 84.6% | 68.3% | 59.2% | 35% | 5% |
| Econ | 95.0% | 96.7% | 97.5% | 96.0% | 92.4% | 84.0% | 64.0% | 55% | 5% |
| BusFin | 85.0% | 83.3% | 85.0% | 82.0% | 74.2% | 68.0% | 53.5% | 25% | 3.0% |
| Comm | 95.0% | 93.3% | 95.0% | 88.0% | 81.8% | 72.0% | 61.0% | 35% | 5% |
| PolSci | 75.0% | 70.0% | 65.0% | 58.0% | 59.1% | 58.0% | 50.5% | N/A | 1.5% |
| Psych | 100.0% | 100.0% | 95.0% | 96.0% | 97.0% | 96.0% | 76.0% | 90% | 15% |
| PubAdmin | 90.0% | 83.3% | 77.5% | 72.0% | 68.2% | 63.0% | 51.0% | 15% | 1.0% |
| Sociol | 100.0% | 100.0% | 92.5% | 88.0% | 81.8% | 67.0% | 51.0% | 35% | 15% |

### GPT-4.1-nano SFT

| Field | Top-10% | Top-15% | Top-20% | Top-25% | Top-33% | Top-50% | Top-100% | 80% Threshold | Perfect @K% |
|-------|------|------|------|------|------|------|------|---------------|-------------|
| Mgmt | 100.0% | 94.4% | 79.2% | 76.7% | 69.2% | 66.7% | 57.5% | 15% | 10% |
| Econ | 100.0% | 100.0% | 100.0% | 96.0% | 93.9% | 85.0% | 68.5% | 65% | 20% |
| BusFin | 90.0% | 83.3% | 87.5% | 82.0% | 74.2% | 66.0% | 55.5% | 30% | 5% |
| Comm | 100.0% | 90.0% | 82.5% | 84.0% | 78.8% | 67.0% | 60.5% | 30% | 10% |
| PolSci | 90.0% | 86.7% | 77.5% | 72.0% | 71.2% | 65.0% | 54.0% | 15% | 3.0% |
| Psych | 100.0% | 100.0% | 100.0% | 100.0% | 98.5% | 96.0% | 81.5% | 100% | 30% |
| PubAdmin | 95.0% | 80.0% | 77.5% | 74.0% | 71.2% | 63.0% | 53.5% | 15% | 5% |
| Sociol | 100.0% | 96.7% | 97.5% | 94.0% | 84.8% | 74.0% | 54.0% | 40% | 10% |

## 3. Cross-Field Comparison (Averaged Across 3 Models)

### 3a. Confidence Gap Ranking (Correct - Incorrect)

Larger gap = model is better at distinguishing correct from incorrect predictions.

| Rank | Field | Mean Gap | Mean Acc | Mean ECE | Sig Models |
|------|-------|---------|---------|---------|------------|
| 1 | Psych | +0.237 | 81.0% | 0.052 | 3/3 |
| 2 | Sociol | +0.176 | 56.8% | 0.121 | 3/3 |
| 3 | Econ | +0.164 | 67.3% | 0.122 | 3/3 |
| 4 | PubAdmin | +0.130 | 53.2% | 0.116 | 3/3 |
| 5 | BusFin | +0.116 | 54.2% | 0.144 | 3/3 |
| 6 | Comm | +0.115 | 63.0% | 0.097 | 3/3 |
| 7 | PolSci | +0.083 | 54.3% | 0.117 | 3/3 |
| 8 | Mgmt | +0.067 | 58.3% | 0.200 | 3/3 |

### 3b. Selective Prediction Quality Ranking

Ranked by accuracy achieved at top-25% confidence (averaged across 3 models).

| Rank | Field | Top-10% | Top-25% | Top-50% | Mean 80% Threshold |
|------|-------|---------|---------|---------|-------------------|
| 1 | Psych | 100.0% | 98.0% | 96.7% | 97% |
| 2 | Econ | 98.3% | 96.7% | 86.0% | 63% |
| 3 | Sociol | 98.3% | 92.0% | 74.7% | 42% |
| 4 | Comm | 98.3% | 88.0% | 72.7% | 40% |
| 5 | BusFin | 90.0% | 81.3% | 67.0% | 27% |
| 6 | PubAdmin | 93.3% | 77.3% | 65.0% | 22% |
| 7 | Mgmt | 97.2% | 76.7% | 68.3% | 22% |
| 8 | PolSci | 81.7% | 67.3% | 64.0% | 12% |

### 3c. Does Calibration Quality Correlate with Overall Accuracy?

- **Accuracy vs ECE**: Spearman r = -0.262, p = 0.531
  - Negative correlation: higher accuracy fields tend to have lower ECE (better calibration)
- **Accuracy vs Confidence Gap**: Spearman r = 0.333, p = 0.420
  - Positive correlation: higher accuracy fields also have larger confidence gaps (better discrimination)

## 4. Key Findings

1. **Best calibrated field**: Psych (mean confidence gap = +0.237, ECE = 0.052)
2. **Worst calibrated field**: Mgmt (mean confidence gap = +0.067, ECE = 0.200)

3. **Fields achieving 80%+ via selective prediction** (at any coverage level, any model):
   - Mgmt (Qwen3-30B SFT): 80% accuracy at top-15%
   - Mgmt (Qwen3-4B SFT): 80% accuracy at top-35%
   - Mgmt (GPT-4.1-nano SFT): 80% accuracy at top-15%
   - Econ (Qwen3-30B SFT): 80% accuracy at top-70%
   - Econ (Qwen3-4B SFT): 80% accuracy at top-55%
   - Econ (GPT-4.1-nano SFT): 80% accuracy at top-65%
   - BusFin (Qwen3-30B SFT): 80% accuracy at top-25%
   - BusFin (Qwen3-4B SFT): 80% accuracy at top-25%
   - BusFin (GPT-4.1-nano SFT): 80% accuracy at top-30%
   - Comm (Qwen3-30B SFT): 80% accuracy at top-55%
   - Comm (Qwen3-4B SFT): 80% accuracy at top-35%
   - Comm (GPT-4.1-nano SFT): 80% accuracy at top-30%
   - PolSci (Qwen3-30B SFT): 80% accuracy at top-10%
   - PolSci (GPT-4.1-nano SFT): 80% accuracy at top-15%
   - Psych (Qwen3-30B SFT): 80% accuracy at top-100%
   - Psych (Qwen3-4B SFT): 80% accuracy at top-90%
   - Psych (GPT-4.1-nano SFT): 80% accuracy at top-100%
   - PubAdmin (Qwen3-30B SFT): 80% accuracy at top-35%
   - PubAdmin (Qwen3-4B SFT): 80% accuracy at top-15%
   - PubAdmin (GPT-4.1-nano SFT): 80% accuracy at top-15%
   - Sociol (Qwen3-30B SFT): 80% accuracy at top-50%
   - Sociol (Qwen3-4B SFT): 80% accuracy at top-35%
   - Sociol (GPT-4.1-nano SFT): 80% accuracy at top-40%

4. **Fields achieving 100% accuracy** at some top-K%:
   - Mgmt (Qwen3-30B SFT): 100% at top-10%
   - Mgmt (Qwen3-4B SFT): 100% at top-5%
   - Mgmt (GPT-4.1-nano SFT): 100% at top-10%
   - Econ (Qwen3-30B SFT): 100% at top-15%
   - Econ (Qwen3-4B SFT): 100% at top-5%
   - Econ (GPT-4.1-nano SFT): 100% at top-20%
   - BusFin (Qwen3-30B SFT): 100% at top-2.5%
   - BusFin (Qwen3-4B SFT): 100% at top-3.0%
   - BusFin (GPT-4.1-nano SFT): 100% at top-5%
   - Comm (Qwen3-30B SFT): 100% at top-10%
   - Comm (Qwen3-4B SFT): 100% at top-5%
   - Comm (GPT-4.1-nano SFT): 100% at top-10%
   - PolSci (Qwen3-30B SFT): 100% at top-5%
   - PolSci (Qwen3-4B SFT): 100% at top-1.5%
   - PolSci (GPT-4.1-nano SFT): 100% at top-3.0%
   - Psych (Qwen3-30B SFT): 100% at top-15%
   - Psych (Qwen3-4B SFT): 100% at top-15%
   - Psych (GPT-4.1-nano SFT): 100% at top-30%
   - PubAdmin (Qwen3-30B SFT): 100% at top-5%
   - PubAdmin (Qwen3-4B SFT): 100% at top-1.0%
   - PubAdmin (GPT-4.1-nano SFT): 100% at top-5%
   - Sociol (Qwen3-30B SFT): 100% at top-5%
   - Sociol (Qwen3-4B SFT): 100% at top-15%
   - Sociol (GPT-4.1-nano SFT): 100% at top-10%

## 5. Per-Model Summary

### Qwen3-30B SFT

- Mean accuracy across 8 fields: 64.2%
- Mean confidence gap: +0.135
- Mean ECE: 0.133

### Qwen3-4B SFT

- Mean accuracy across 8 fields: 58.3%
- Mean confidence gap: +0.132
- Mean ECE: 0.139

### GPT-4.1-nano SFT

- Mean accuracy across 8 fields: 60.6%
- Mean confidence gap: +0.141
- Mean ECE: 0.092
