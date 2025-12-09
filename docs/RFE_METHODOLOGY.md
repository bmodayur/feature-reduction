# Enhanced Adaptive RFE Methodology

**Created**: 2025-12-03
**Purpose**: Document the feature selection methodology used in the Early Markers project

---

## Executive Summary

The Early Markers project uses an **Enhanced Adaptive Recursive Feature Elimination (RFE)** approach to select the most informative movement features for infant developmental screening. This document explains the methodology, rationale, and implementation details.

---

## 1. Why Feature Reduction?

### 1.1 Clinical Validation Constraints

The primary driver for feature reduction is **sample size feasibility** for clinical validation studies:

| Feature Count | Estimated Sample Size (N) | Feasibility |
|---------------|---------------------------|-------------|
| 59 (all) | ~600+ infants | Impractical |
| 38 features | ~400 infants | Challenging |
| 20 features | ~100-150 infants | Tractable |
| 15 features | ~100 infants | Optimal |

**Key Insight**: Each additional feature increases the dimensionality of the model, requiring more samples to achieve stable performance estimates. The Bayesian Assurance Method (BAM) calculations (Dec 2025) show that the 20-feature model with transformation requires approximately **N ≈ 100 infants** for clinical validation with:
- Target HDI width: 0.15 (±7.5% precision)
- Target assurance: 0.80 (80% confidence - achieved: 100%)
- Joint sensitivity/specificity constraints
- Pilot metrics: 92.1% sensitivity, 77.5% specificity, AUC 0.902

**Note**: The favorable sample size estimate is driven by the strong pilot performance. For stricter precision (±5%), approximately N=100 would achieve only ~83% assurance, suggesting a larger sample may be prudent.

### 1.2 Overfitting Prevention

With limited infant data (~146 subjects), using all 59 features risks:
- **Overfitting**: Model learns noise rather than signal
- **Instability**: Small changes in data cause large changes in predictions
- **Poor generalization**: High training performance but low test performance

### 1.3 Clinical Interpretability

Fewer features means:
- Easier clinical interpretation
- More robust across different recording conditions
- Simpler deployment in clinical settings

---

## 2. The Enhanced Adaptive RFE Method

### 2.1 Overview

The Enhanced Adaptive RFE combines five key innovations:

```
┌─────────────────────────────────────────────────────────────┐
│                  Enhanced Adaptive RFE                       │
├─────────────────────────────────────────────────────────────┤
│  1. Noise Injection      - Test selection stability          │
│  2. Adaptive Step Sizing - Dynamic elimination rate          │
│  3. Hyperparameter Tuning - Optimal RF configuration         │
│  4. Statistical Testing   - Binomial consensus (α=0.05)      │
│  5. Parallel Execution    - 50 independent trials            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Algorithm Steps

#### Step 1: Noise Feature Injection
```
Add 10% random noise features to the original feature set
Purpose: Features that are truly important should be selected
         over noise features consistently
```

#### Step 2: Parallel RFE Trials (n=50)
```
For each trial (in parallel):
    1. Bootstrap sample the training data
    2. Run standard RFE with Random Forest
    3. Record which features were selected
    4. Track feature rankings
```

#### Step 3: Adaptive Step Sizing
```
At each RFE iteration:
    - Evaluate CV performance at min_step and max_step
    - If performance is stable: use larger step (faster)
    - If performance is sensitive: use smaller step (careful)
```

#### Step 4: Statistical Consensus
```
For each feature:
    - Count how many trials selected it
    - Apply binomial test: H0 = selection by chance (p=0.5)
    - Keep feature if p < α (default α=0.05)
```

#### Step 5: Final Feature Set
```
Return features that:
    1. Pass statistical significance test
    2. Are NOT noise features
    3. Meet target retention percentage
```

### 2.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RFE_N_TRIALS` | 50 | Number of parallel trials |
| `RFE_ALPHA` | 0.05 | Significance level for binomial test |
| `RFE_NOISE_RATIO` | 0.1 | Proportion of noise features added |
| `RFE_KEEP_PCT` | 0.9 | Target feature retention per iteration |
| `RKFOLD_SPLITS` | 15 | Cross-validation folds |
| `N_ESTIMATORS` | 300 | Random Forest trees |

---

## 3. Implementation

### 3.1 Code Location

```
early_markers/cribsy/common/adaptive_rfe.py
    - EnhancedAdaptiveRFE: Main class
    - AdaptiveStepRFE: Dynamic step calculator

early_markers/cribsy/common/bayes.py
    - BayesianData.run_adaptive_rfe(): High-level interface
```

### 3.2 Usage Example

```python
from early_markers.cribsy.common.bayes import BayesianData

# Initialize with data
bd = BayesianData(base_file="features_merged_20251121_091511.pkl")

# Run Enhanced Adaptive RFE
selected_features = bd.run_adaptive_rfe(
    model_prefix='my_analysis',
    features=bd.base_features,  # Start with all 59 features
    tot_k=len(bd.base_features)
)

print(f"Selected {len(selected_features)} features")
# Typically returns 30-50 features after statistical filtering
```

### 3.3 Post-RFE Feature Reduction

RFE typically returns 30-50 statistically significant features. To get to a target count (e.g., 20), we apply **Random Forest importance ranking**:

```python
from sklearn.ensemble import RandomForestRegressor

# Train RF on RFE-selected features
rf = RandomForestRegressor(n_estimators=300, random_state=RAND_STATE)
rf.fit(X_train, y_train)

# Rank by importance
feature_importance = list(zip(selected_features, rf.feature_importances_))
feature_importance.sort(key=lambda x: x[1], reverse=True)

# Take top 19 + force age_in_weeks
final_features = [f for f, imp in feature_importance[:19]] + ['age_in_weeks']
```

---

## 4. Forcing Age Inclusion

### 4.1 Rationale

The `age_in_weeks` feature is **forced** into the final feature set because:

1. **Age-Adjusted Modeling**: Infant movement patterns change dramatically with age (1-12 months). Including age allows a single model across all ages.

2. **Sample Size Reduction**: Without age, we'd need separate models per age group:
   - 1-4 months: N₁ infants
   - 5-8 months: N₂ infants
   - 9-12 months: N₃ infants
   - Total: N₁ + N₂ + N₃ ≈ 600+ infants

   With age-adjusted: Single model, N ≈ 200 infants (2-3× reduction)

3. **Clinical Reality**: Age is always known and easily obtained.

### 4.2 Implementation

```python
# Run RFE on 58 movement features (excluding age)
features_no_age = [f for f in bd.base_features if f != 'age_in_weeks']
selected = bd.run_adaptive_rfe(features=features_no_age)

# Take top 19 by importance + force age
final_features = top_19_by_importance + ['age_in_weeks']  # = 20 features
```

---

## 5. Comparison with Standard RFE

| Aspect | Standard RFE | Enhanced Adaptive RFE |
|--------|--------------|----------------------|
| Stability | Single run, variable results | 50 trials, consensus-based |
| Noise Robustness | May select spurious features | Noise injection validates stability |
| Step Size | Fixed | Adaptive based on CV performance |
| Statistical Rigor | None | Binomial test (p < 0.05) |
| Runtime | ~5 minutes | ~30-60 minutes |
| Reproducibility | Low | High (statistical filtering) |

---

## 6. Validation

### 6.1 Feature Stability Across Runs

When Enhanced Adaptive RFE is run multiple times:
- **Core features** (6-8) appear in >90% of runs
- **Secondary features** (10-12) appear in 60-80% of runs
- **Marginal features** (remaining) vary between runs

### 6.2 Cross-Dataset Generalization

The Nov 21, 2025 RFE run demonstrated cross-dataset generalization:
- Features selected on `features_merged.pkl` (old dataset)
- Evaluated on `features_merged_20251121_091511.pkl` (new dataset)
- Result: AUC 0.902 (with transformation)

This suggests the method selects genuinely informative features, not dataset-specific artifacts.

### 6.3 Stable Core Features

Across multiple RFE runs, these features consistently appear:

| Feature | Body Part | Type | Stability |
|---------|-----------|------|-----------|
| `Hip_stdev_angle` | Hip | Position | Very High |
| `Ankle_IQRaccy` | Ankle | Acceleration | Very High |
| `Ankle_IQRx` | Ankle | Position | High |
| `Ankle_meanent` | Ankle | Entropy | High |
| `Ankle_medianvelx` | Ankle | Velocity | High |
| `age_in_weeks` | - | Demographic | Forced |

---

## 7. References

### 7.1 Academic References

1. Guyon, I., et al. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1-3), 389-422.

2. Xu, P., et al. (2014). Predictor augmentation for feature selection. *ICML Workshop on Feature Extraction*.

### 7.2 Project Documentation

- `early_markers/cribsy/common/adaptive_rfe.py` - Implementation with detailed docstrings
- `docs/WORKFLOWS.md` - Usage examples
- `docs/ARCHITECTURE.md` - System overview
- `docs/FEATURE_SET_PROVENANCE.md` - Feature set tracking

---

## 8. Appendix: Configuration Constants

From `early_markers/cribsy/common/constants.py`:

```python
# RFE Configuration
RFE_N_TRIALS = 50          # Parallel trials for consensus
RFE_ALPHA = 0.05           # Significance level
RFE_NOISE_RATIO = 0.1      # 10% noise features
RFE_KEEP_PCT = 0.9         # 90% retention per iteration
RFE_MIN_STEP = 1           # Minimum features to eliminate
RFE_MAX_STEP = 5           # Maximum features to eliminate
RFE_TOLERANCE = 0.01       # CV performance tolerance

# Random Forest Configuration
N_ESTIMATORS = 300         # Trees in forest
RAND_STATE = 20250313      # Reproducibility seed

# Cross-Validation
RKFOLD_SPLITS = 15         # K-fold splits
RKFOLD_REPEATS = 1         # Repetitions
```

---

## 9. BAM Sample Size Results (Dec 2025)

### 9.1 Primary Analysis

Based on the 20-feature Nov 21 RFE model evaluated with 10-fold CV and 1/(1+|x|) transformation:

| Metric | Value |
|--------|-------|
| Sensitivity | 92.1% (129/140 at-risk) |
| Specificity | 77.5% (155/200 normal) |
| AUC | 0.902 |

**BAM Result**: N = 100 infants (achieved assurance: 100%)

### 9.2 Sensitivity Analysis

| Target Width | Target Assurance | Required N | Achieved |
|--------------|------------------|------------|----------|
| ±7.5% (0.15) | 80% | 100 | 100% |
| ±7.5% (0.15) | 90% | 100 | 100% |
| ±5.0% (0.10) | 80% | 100 | 83% |
| ±10% (0.20) | 80% | 100 | 100% |

### 9.3 Sample Breakdown (20% Prevalence)

With expected prevalence of ~20%:
- At-risk infants: ~20
- Normal infants: ~80
- Total: 100

### 9.4 Interpretation

The favorable N=100 estimate reflects:
1. **Strong pilot performance**: 92.1% sensitivity and 77.5% specificity are excellent for a screening tool
2. **Large pilot sample**: 340 effective observations via 10-fold CV provides good prior information
3. **Transformation benefit**: The 1/(1+|x|) transformation improved specificity from ~61% to 77.5%

**Recommendation**: Given the uncertainty in pilot estimates and importance of clinical validation, a conservative target of N=150-200 may be prudent to ensure robust performance estimation.

**Script**: `scripts/run_bam_20feature_nov21.py`
**Results**: `data/json/bam_sample_size_20feature_nov21.json`

---

**Last Updated**: 2025-12-04
