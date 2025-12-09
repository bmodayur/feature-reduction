# Analysis Workflow: Scripts for Reproducing Results

This document provides step-by-step bash commands for running the analysis pipeline using standalone scripts. For API-level workflows using Python/Jupyter, see [WORKFLOWS.md](WORKFLOWS.md).

## Overview

The analysis pipeline follows three main stages:

```
RFE → ROC → BAM
 │      │     └─ Sample size planning
 │      └─ Screener metrics (Se, Sp, AUC)
 └─ Feature selection (59 → 20 features)
```

**Current Best Model**: 20-feature with inverse transform
- AUC: 0.902
- Sensitivity: 92.1%
- Specificity: 77.5%
- Optimal threshold: -0.246

---

## Prerequisites

Ensure your environment is set up:

```bash
# Option A: Poetry
poetry shell

# Option B: Conda
conda activate early-markers

# Option C: venv
source venv/bin/activate
```

Set PYTHONPATH if running from project root:

```bash
export PYTHONPATH=.
```

---

## Step 1: Feature Selection (RFE)

**Script**: `scripts/run_rfe_with_age_forced.py`

Runs Enhanced Adaptive RFE to select optimal features, forcing `age_in_weeks` as the 20th feature.

```bash
python scripts/run_rfe_with_age_forced.py
```

**Runtime**: ~30-60 minutes (50 trials × 15 CV folds)

**Outputs**:
- `data/pkl/bd_final20_forced_age.pkl` - Serialized BayesianData object
- `data/json/features_final20_forced_age.json` - Selected features list
- Console output showing selected features and importances

**Expected Output**:
```
Loading data...
Running Enhanced Adaptive RFE...
Trial 1/50 complete
...
RFE Complete: 19 features selected

Top features by importance:
1. Ankle_medianx (0.089)
2. Knee_IQR_acc_angle (0.076)
...

Adding age_in_weeks as 20th feature
Final 20 features saved to: data/json/features_final20_forced_age.json
```

**Documentation**: [RFE_METHODOLOGY.md](RFE_METHODOLOGY.md)

---

## Step 2: ROC Analysis

**Script**: `scripts/compute_roc_after_rfe.py`

Computes Bayesian Surprise scores and ROC metrics using the selected features. Applies the inverse transform to velocity/acceleration features.

```bash
python scripts/compute_roc_after_rfe.py
```

**Runtime**: ~2-5 minutes

**Outputs**:
- `data/pkl/bd_final20_forced_age.pkl` - Updated with surprise scores and metrics
- `data/img/roc_final20_forced_age.png` - ROC curve visualization
- `results/metrics_final20_forced_age.xlsx` - Excel report with detailed metrics

**Expected Output**:
```
Loading BayesianData from: data/pkl/bd_final20_forced_age.pkl
Computing Bayesian Surprise with 20 features...
  Training on 116 normative infants
  Testing on 340 samples (140 at-risk, 200 normal)

Applying inverse transform to 11 velocity/acceleration features...

ROC Analysis:
  AUC: 0.902
  Optimal threshold (Youden's J): -0.246
  Sensitivity: 92.1%
  Specificity: 77.5%
  PPV: 74.1%
  NPV: 93.4%

Z-score distributions:
  Normal (risk=0): μ=-0.10, σ=1.34
  At-risk (risk=1): μ=-2.49, σ=2.98
  Cohen's d: -1.10 (large effect)

Results saved to: results/metrics_final20_forced_age.xlsx
```

**Documentation**: [FEATURE_TRANSFORMATION_METHODOLOGY.md](FEATURE_TRANSFORMATION_METHODOLOGY.md)

---

## Step 3: Sample Size Planning (BAM)

**Script**: `scripts/run_bam_20feature_nov21.py`

Uses Bayesian Assurance Method to estimate required sample size for a validation study.

```bash
python scripts/run_bam_20feature_nov21.py
```

**Runtime**: ~1-2 minutes

**Outputs**:
- Console output with sample size estimates
- Optional: `results/bam_estimates.json`

**Expected Output**:
```
BAM Sample Size Estimation
==========================
Pilot data:
  Sensitivity: 92.1% (129/140)
  Specificity: 77.5% (155/200)

Target:
  CI width: 0.20 (±10%)
  Assurance: 80%
  Confidence level: 95%

Results:
  Required total N: 198
  Positive cases needed: ~40
  Negative cases needed: ~158
  Achieved assurance: 82.5%
```

**Documentation**: [BAM_for_20FeatureSet.md](BAM_for_20FeatureSet.md)

---

## Quick Reference: All Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `run_rfe_with_age_forced.py` | Feature selection with forced age | 30-60 min |
| `compute_roc_after_rfe.py` | ROC metrics with transform | 2-5 min |
| `run_bam_20feature_nov21.py` | Sample size estimation | 1-2 min |
| `visualize_transform_3panel.py` | Transformation figure | <1 min |

---

## Running the Full Pipeline

To run all steps sequentially:

```bash
# Set environment
export PYTHONPATH=.

# Step 1: Feature selection (longest step)
python scripts/run_rfe_with_age_forced.py

# Step 2: ROC analysis
python scripts/compute_roc_after_rfe.py

# Step 3: Sample size planning
python scripts/run_bam_20feature_nov21.py

# Optional: Generate transformation figure
python scripts/visualize_transform_3panel.py
```

---

## Troubleshooting

### ModuleNotFoundError: early_markers

Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=.
# Or run with:
PYTHONPATH=. python scripts/script_name.py
```

### FileNotFoundError: features_merged*.pkl

The merged features file should be in `data/pkl/`. Check available files:
```bash
ls data/pkl/features_merged*.pkl
```

### Memory errors during RFE

Reduce parallelism:
```python
# In run_rfe_with_age_forced.py, reduce n_jobs
n_jobs = 4  # Instead of default 12
```

### Different results from documented values

Check random seed in `early_markers/cribsy/common/constants.py`:
```python
RAND_STATE = 20250313  # Should match
```

---

## See Also

- [WORKFLOWS.md](WORKFLOWS.md) - Python API workflows
- [RFE_METHODOLOGY.md](RFE_METHODOLOGY.md) - Feature selection details
- [FEATURE_TRANSFORMATION_METHODOLOGY.md](FEATURE_TRANSFORMATION_METHODOLOGY.md) - Inverse transform rationale
- [BAM_for_20FeatureSet.md](BAM_for_20FeatureSet.md) - Sample size methodology
