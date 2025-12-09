# Architecture Documentation

This document explains the architecture, methodology, and design patterns used in the early-markers codebase.

## Table of Contents

1. [Overview](#overview)
2. [Bayesian Surprise Methodology](#bayesian-surprise-methodology)
3. [System Architecture](#system-architecture)
4. [Data Flow](#data-flow)
5. [Class Relationships](#class-relationships)
6. [State Management](#state-management)

## Overview

The early-markers project implements a Bayesian statistical framework for identifying developmental anomalies in infant movement data. The system uses:

- **Bayesian surprise**: Z-scores computed from negative log-likelihoods of movement features
- **Feature selection**: Enhanced Adaptive RFE for robust feature identification
- **Sample size estimation**: Bayesian Assurance Method (BAM) for study planning
- **Synthetic data generation**: For validation and power analysis

## Bayesian Surprise Methodology

### Concept

Bayesian surprise quantifies how unexpected an infant's movement patterns are compared to a normative reference population. Higher surprise scores indicate greater deviation from typical development.

### Mathematical Formulation

For each infant `i` and feature `f`:

1. **Negative log-likelihood per feature**:
   ```
   -log P(x_if | N(μ_ref, σ²_ref)) = 
       0.5 * log(2π * σ²_ref) + (x_if - μ_ref)² / (2 * σ²_ref)
   ```

2. **Summed surprise across features**:
   ```
   S_i = Σ_f [-log P(x_if | N(μ_ref, σ²_ref))]
   ```

3. **Standardized z-score**:
   ```
   z_i = (S_i - μ_train) / σ_train
   ```
   
4. **P-value**:
   ```
   p_i = 2 * SF(|z_i|)  [two-tailed survival function]
   ```

### Why This Approach?

- **Probabilistic interpretation**: Each feature contributes based on its likelihood
- **Multi-dimensional**: Combines information from all movement features
- **Standardized**: Z-scores allow comparison across different feature sets
- **Validated**: Based on Bayesian information theory principles

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Raw Data                                 │
│  (.pkl files: real data, .ipc files: synthetic data)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                BayesianData                                  │
│  - Loads and transforms data                                 │
│  - Manages train/test split                                  │
│  - Orchestrates analysis pipeline                            │
└───────────┬────────────────────────────┬────────────────────┘
            │                            │
            ▼                            ▼
┌─────────────────────┐       ┌────────────────────────────┐
│  Feature Selection  │       │  Bayesian Surprise         │
│  - EnhancedRFE      │       │  - Compute statistics      │
│  - Standard RFE     │       │  - Calculate z-scores      │
│  - AdaptiveStep RFE │       │  - Generate surprise DF    │
└──────────┬──────────┘       └────────────┬───────────────┘
           │                               │
           └──────────────┬────────────────┘
                          ▼
              ┌─────────────────────────┐
              │   ROC Analysis          │
              │   - Compute metrics     │
              │   - Generate curves     │
              │   - Find optimal thresh │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │   Output Artifacts      │
              │   - Excel reports       │
              │   - PNG plots           │
              │   - CSV/PKL files       │
              └─────────────────────────┘
```

### Module Structure

```
early_markers/
├── cribsy/                    # Core analysis module
│   ├── __init__.py           # Random seed initialization
│   ├── bam.py                # Legacy BAM sample size functions
│   ├── thresholds.py         # Pre-computed thresholds
│   ├── common/
│   │   ├── bayes.py          # BayesianData: Main orchestration class
│   │   ├── adaptive_rfe.py   # EnhancedAdaptiveRFE: Feature selection
│   │   ├── bam_unified.py    # BAMEstimator: Unified sample size estimation
│   │   ├── bam_export.py     # BAM results export utilities
│   │   ├── constants.py      # Configuration constants
│   │   ├── enums.py          # Type definitions (MetricType, RfeType, ErrorType)
│   │   ├── metrics_n.py      # Sample size estimation formulas
│   │   ├── rfe.py            # Base RFE implementation
│   │   └── data.py           # Data loading utilities (legacy)
│   └── notebooks/            # Jupyter analysis notebooks
│       ├── 01_rfe_real.ipynb          # Real data feature selection (~22 min)
│       ├── 01_rfe_real_fast.ipynb     # Fast RFE variant (~5-10 min)
│       ├── 02_synthetic_sdv.ipynb     # Synthetic data generation with SDV
│       ├── 03_rfe_synthetic.ipynb     # Synthetic data validation
│       ├── 04_bam_sample_size.ipynb   # BAM sample size estimation
│       ├── 05_test_retest_sample_size.ipynb  # ICC test-retest analysis
│       └── 06_tost_sample_size.ipynb  # TOST equivalence testing
└── common/
    └── xlsx.py               # Excel formatting utilities
```

### Notebook Pipeline Roles

**Primary Analysis Workflow:**
1. **01_rfe_real.ipynb**: Feature selection on real data
   - Loads `features_merged.pkl` (1,250 infants × 56 features)
   - Runs Enhanced Adaptive RFE (50 trials, 200 estimators)
   - Iterates until MIN_K features remain
   - Outputs: `bd_real.pkl`, Excel reports
   - Runtime: ~20-30 minutes

2. **03_rfe_synthetic.ipynb**: Validation with synthetic data
   - Loads `.ipc` files from 02_synthetic_sdv.ipynb
   - Runs same RFE pipeline on synthetic data
   - Compares feature stability (60-80% overlap expected)
   - Outputs: `bd_train{N}_test{M}.pkl`
   - Runtime: ~1.5 minutes per iteration

**Supporting Workflows:**

3. **01_rfe_real_fast.ipynb**: Fast development iteration
   - Same as 01_rfe_real but with 10 trials, 100 estimators
   - For rapid prototyping and testing
   - Runtime: ~5-10 minutes

4. **02_synthetic_sdv.ipynb**: Generate synthetic data
   - Uses SDV (Synthetic Data Vault) library
   - Creates statistically similar datasets
   - Outputs: `.ipc` files for use in 03_rfe_synthetic
   - Runtime: ~2-5 minutes

**Sample Size Planning:**

5. **04_bam_sample_size.ipynb**: BAM methodology
   - Bayesian Assurance Method for sensitivity/specificity
   - Uses `BAMEstimator` from `bam_unified.py`
   - Plans studies with target precision (e.g., ±0.10 CI width)

6. **05_test_retest_sample_size.ipynb**: Reliability analysis
   - ICC-based test-retest sample size estimation
   - Plans reliability studies

7. **06_tost_sample_size.ipynb**: Equivalence testing
   - TOST (Two One-Sided Tests) for proportions
   - Plans equivalence/non-inferiority studies

## Data Flow

### Complete Pipeline

```
1. RAW DATA LOADING
   ├─→ .pkl files (Pandas): Real infant movement data
   └─→ .ipc files (Polars): Synthetic data for validation

2. DATA TRANSFORMATION
   ├─→ Risk encoding: risk_raw → binary (0=normal, 1=at-risk)
   ├─→ Category assignment:
   │    - category=1: Training (normative + sampled at-risk)
   │    - category=2: Testing (held-out at-risk)
   └─→ Feature engineering: part + feature_name → "BodyPart_Metric"

3. TRAIN/TEST SPLIT
   ├─→ base_train: All category=1 infants
   └─→ base_test: All category=2 infants

4. REFERENCE STATISTICS
   └─→ For each feature in training data:
        - mean_ref: Population mean
        - sd_ref: Population standard deviation
        - var_ref: Population variance

5. FEATURE SELECTION (Optional)
   ├─→ EnhancedAdaptiveRFE: Parallel trials with noise injection
   ├─→ Standard RFE: Single-run feature elimination
   └─→ Manual selection: Specify feature list

6. SURPRISE CALCULATION
   ├─→ Training surprise: Compute z-scores for training infants
   │    (Used to establish distribution parameters)
   └─→ Testing surprise: Compute z-scores for test infants
        (Used for model evaluation)

7. ROC ANALYSIS
   ├─→ Vary threshold over z-score range
   ├─→ Compute confusion matrix at each threshold
   ├─→ Calculate metrics: sensitivity, specificity, PPV, NPV, F1, accuracy
   └─→ Find optimal threshold (maximum Youden's J)

8. OUTPUT GENERATION
   ├─→ Excel workbook: Detailed metrics for each trial
   ├─→ PNG plots: ROC curves with AUC
   └─→ PKL/CSV files: Intermediate results for caching
```

### Data Format Transformations

**Long Format (Internal)**:
```
infant | category | risk | feature          | value
-------|----------|------|------------------|-------
inf_01 | 1        | 0    | Ankle_IQRvelx    | 0.234
inf_01 | 1        | 0    | Wrist_medianx    | 1.567
inf_02 | 2        | 1    | Ankle_IQRvelx    | 0.891
```

**Wide Format (For ML)**:
```
infant | category | risk | Ankle_IQRvelx | Wrist_medianx | ...
-------|----------|------|---------------|---------------|-----
inf_01 | 1        | 0    | 0.234         | 1.567         | ...
inf_02 | 2        | 1    | 0.891         | 2.103         | ...
```

## Class Relationships

### Core Classes

```
┌─────────────────────────────────────────────────────────────┐
│                      BayesianData                            │
│  Main API class - orchestrates entire analysis pipeline     │
│                                                              │
│  Internal State:                                            │
│    _base: Complete dataset (long format)                    │
│    _base_train: Training subset                             │
│    _base_test: Testing subset                               │
│    _frames: Dict[str, BayesianFrames]                       │
│    _rfes: Dict[str, BayesianRfeResult]                      │
│    _surprise: Dict[str, BayesianSurprise]                   │
│    _metrics: Dict[str, BayesianRocResult]                   │
└─────────┬───────────────────────────────────────────────────┘
          │
          │ creates and manages
          │
          ├──→ ┌──────────────────────────────────────┐
          │    │    BayesianFrames (dataclass)        │
          │    │  Container for related DataFrames    │
          │    │  - train: Training data              │
          │    │  - test: Testing data                │
          │    │  - stats: Reference statistics       │
          │    │  - train_surprise: Train z-scores    │
          │    │  - test_surprise: Test z-scores      │
          │    └──────────────────────────────────────┘
          │
          ├──→ ┌──────────────────────────────────────┐
          │    │   BayesianRfeResult (dataclass)      │
          │    │  RFE output container                │
          │    │  - name: RFE identifier              │
          │    │  - k: Number of features selected    │
          │    │  - features: List of feature names   │
          │    └──────────────────────────────────────┘
          │
          ├──→ ┌──────────────────────────────────────┐
          │    │   BayesianSurprise (dataclass)       │
          │    │  Surprise distribution parameters    │
          │    │  - model_name: Model identifier      │
          │    │  - k: Number of features             │
          │    │  - mean_neg_log_p: Training mean     │
          │    │  - sd_neg_log_p: Training SD         │
          │    └──────────────────────────────────────┘
          │
          └──→ ┌──────────────────────────────────────┐
               │   BayesianRocResult (dataclass)      │
               │  ROC analysis output                 │
               │  - model_name: Model identifier      │
               │  - features: Feature list            │
               │  - auc: Area under ROC curve         │
               │  - youdens_j: Optimal Youden's J     │
               │  - threshold_j: Optimal threshold    │
               │  - metrics: Full metrics DataFrame   │
               │  - plot_file: Path to ROC plot       │
               └──────────────────────────────────────┘
```

### Feature Selection Classes

```
┌───────────────────────────────────────────────────────┐
│         EnhancedAdaptiveRFE                           │
│  Advanced feature selection with:                     │
│  - Noise injection for stability testing              │
│  - Adaptive step sizing                               │
│  - Statistical significance (binomial test)           │
│  - Hyperparameter tuning (GridSearchCV)              │
└────────────────┬──────────────────────────────────────┘
                 │
                 │ uses
                 │
                 └──→ ┌───────────────────────────────┐
                      │   AdaptiveStepRFE             │
                      │  Dynamic step size calculator │
                      │  - Cross-validation based     │
                      │  - Performance-driven         │
                      └───────────────────────────────┘
```

## State Management

### BayesianData State Pattern

The `BayesianData` class uses a dictionary-based state management pattern:

1. **Model Naming Convention**: `{prefix}_k_{num_features}`
   - Example: `"trial_1_k_50"` = Trial 1 with 50 features

2. **State Dictionaries**:
   ```python
   _frames[model_name] → BayesianFrames
   _rfes[rfe_name] → BayesianRfeResult  
   _surprise[model_name] → BayesianSurprise
   _metrics[model_name] → BayesianRocResult
   ```

3. **Workflow Pattern**:
   ```python
   bd = BayesianData(base_file="data.pkl")
   
   # State 1: Base data loaded
   # _base, _base_train, _base_test populated
   
   # State 2: Features selected
   bd.run_rfe_on_base(rfe_name="rfe_1")
   # _rfes["rfe_1"] populated
   
   # State 3: Surprise computed
   bd.run_surprise_with_rfe(model_prefix="model", rfe_name="rfe_1")
   # _frames["model_k_50"] populated
   # _surprise["model_k_50"] populated
   
   # State 4: Metrics calculated
   bd.run_metrics_from_surprise(metrics_name="trial_1", model_name="model_k_50")
   # _metrics["model_k_50"] populated
   ```

4. **Property Accessors** (read-only):
   ```python
   bd.base                    # Complete dataset
   bd.base_train              # Training subset
   bd.base_test               # Testing subset
   bd.features(model_name)    # Feature list for model
   bd.metrics_df(model_name)  # Metrics DataFrame
   ```

### Thread Safety

**Not Thread-Safe**: The `BayesianData` class maintains mutable internal state. Do not share instances across threads without external synchronization.

### Memory Considerations

- Large datasets are kept in memory (Polars DataFrames)
- Each model creates additional DataFrame copies
- Use `augment=False` unless specifically needed
- Consider sampling with `train_n` and `test_n` for memory constraints

## Performance Optimization

### Computational Costs

1. **Data Loading**: O(n × f) where n=infants, f=features
2. **RFE**: O(trials × CV_folds × estimators)
   - Default: 50 × 15 × 200 = 150,000 tree fits
3. **Surprise Calculation**: O(n × f) - vectorized in Polars
4. **ROC Analysis**: O(n × thresholds) - typically linear

### Optimization Strategies

- Use `n_jobs` for parallel Random Forest training
- Cache RFE results with pickle files
- Use synthetic data sampling for development/testing
- Profile with small feature sets first

## References

- Bayesian information theory
- ROC curve analysis (Fawcett, 2006)
- Recursive Feature Elimination (Guyon et al., 2002)
- Wilson confidence intervals (Wilson, 1927)
