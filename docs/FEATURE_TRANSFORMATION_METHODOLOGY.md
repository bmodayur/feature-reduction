# Feature Transformation Methodology for Bayesian Surprise Model

**Date**: November 26, 2025
**Version**: 1.1
**Author**: Generated with Claude Code
**Random Seed**: 20250313 (for reproducible k-fold CV)

## Executive Summary

This document describes a feature transformation technique that significantly improves the diagnostic performance of the Bayesian surprise model for detecting at-risk infants. By applying an inverse transformation to velocity and acceleration features, we achieve:

- **+16.5% improvement in specificity** (61.0% → 77.5%)
- **Maintained sensitivity** at 92.1% (vs 92.9% baseline)
- **+0.16 improvement in Youden's J** (0.54 → 0.70)
- **+5.3% improvement in AUC** (0.85 → 0.90)

---

## 1. Background and Motivation

### 1.1 The Bayesian Surprise Formula

The Bayesian surprise model computes the negative log-probability of each feature value under a Gaussian reference distribution:

```
-log P(x) = 0.5 × log(2πσ²) + (x - μ)² / (2σ²)
```

The **(x - μ)²** term means that deviations in **both directions** from the mean contribute to surprise. An infant with velocity far below OR far above the normative mean will generate high surprise for that feature.

So why does the transformation help, if the formula already captures both directions?

### 1.2 The Real Problem: Asymmetric Feature Scales

The issue is not the formula—it's the **asymmetric scale** of kinematic features like velocity and acceleration.

Consider velocity with normative mean μ = 1.0 units:

| Condition | Raw Value | Deviation | Squared Deviation |
|-----------|-----------|-----------|-------------------|
| **Hypokinesia** (severely reduced) | 0.2 | -0.8 | **0.64** |
| **Hyperkinesia** (severely elevated) | 5.0 | +4.0 | **16.0** |

Despite both representing severe abnormality, hyperkinesia produces **25× more surprise** than hypokinesia!

This asymmetry arises because:
- **Below-mean range**: 0 → μ (finite, only ~1.0 units available)
- **Above-mean range**: μ → ∞ (infinite, unbounded upward)

The normative standard deviation σ is inflated by occasional high values in typical infants, making hypokinetic patterns appear "less surprising" relative to the reference distribution.

### 1.3 Clinical Reality: Both Extremes Are Pathological

Developmental motor disorders manifest as both reduced and excessive movement:

| Condition | Movement Pattern | Clinical Significance |
|-----------|------------------|----------------------|
| **Hypokinesia** | Reduced amplitude, slow movement | Hypotonia, weakness, lethargy |
| **Hyperkinesia** | Excessive, jerky movements | Spasticity, dystonia, tremor |

A screening model should be **equally sensitive** to both patterns. But with asymmetric raw scales, the model is biased toward detecting hyperkinesia while potentially missing hypokinesia.

### 1.4 The Solution: Symmetric Scale via Transformation

The transformation f(x) = 1/(1+|x|) creates a **symmetric scale** where equal clinical severity produces equal surprise contribution:

```
Raw values:        f(0.2) = 0.83    f(1.0) = 0.50    f(5.0) = 0.17

Transformed deviations from normative mean (≈0.5):
  Hypokinesia:  |0.83 - 0.50| = 0.33
  Hyperkinesia: |0.17 - 0.50| = 0.33
```

Now both extremes are **equidistant** from the transformed normative mean, producing comparable surprise contributions.

### 1.5 Why This Specific Transformation?

The inverse transformation f(x) = 1/(1+|x|) was chosen because:

1. **Symmetric sensitivity**: Equal absolute deviations above and below the transformed mean
2. **Bounded output**: Values always fall in (0, 1], preventing outliers from dominating σ
3. **Monotonic compression**: Very high values compress toward 0, very low values expand toward 1
4. **Continuous and smooth**: No discontinuities that could create artifacts
5. **Intuitive interpretation**: Transformed value ≈ "inverse of movement magnitude"

```
Raw feature scale (asymmetric):

    Hypo           Normal           Hyper
     ↓               ↓                ↓
  |--+---------------|----------------+------→ ∞
  0.0               1.0              5.0
     ← 0.8 units →   ← 4.0 units →

Transformed scale (symmetric):

    Hyper          Normal           Hypo
     ↓               ↓                ↓
  |--+---------------+---------------+--|
  0.0              0.5              1.0
     ← 0.33 units →  ← 0.33 units →
```

---

## 2. Transformation Methodology

### 2.1 The Inverse Transformation

We apply the following transformation to velocity and acceleration features:

```
x_transformed = 1 / (1 + |x|)
```

Where `x` is the original feature value.

### 2.2 Mathematical Properties

| Original Value (x) | Transformed Value | Interpretation |
|-------------------|-------------------|----------------|
| 0 | 1.0 | No movement (concerning) |
| 0.5 | 0.667 | Low movement |
| 1.0 | 0.5 | Moderate movement |
| 2.0 | 0.333 | High movement |
| 5.0 | 0.167 | Very high movement (concerning) |
| ∞ | 0.0 | Extreme movement |

**Key properties**:
- **Bounded output**: Always produces values in (0, 1]
- **Monotonically decreasing**: Higher input → lower output
- **Symmetric for magnitude**: `f(x) = f(-x)` due to absolute value
- **Compresses extremes**: Both very low and very high values become distinguishable from "normal" moderate values

### 2.3 Visual Representation

```
Original Scale:
    Hypokinesia          Normal           Dyskinesia
    (low values)        (moderate)       (high values)
        |                   |                   |
        v                   v                   v
   ─────┼───────────────────┼───────────────────┼─────
       0.0                 1.0                 2.0+

Transformed Scale:
    Hypokinesia          Normal           Dyskinesia
    (high transformed)   (moderate)      (low transformed)
        |                   |                   |
        v                   v                   v
   ─────┼───────────────────┼───────────────────┼─────
       1.0                 0.5                 0.0+

After transformation, BOTH abnormal patterns (hypokinesia AND dyskinesia)
result in LOWER surprise values, enabling one-tailed thresholding to
capture both.
```

---

## 3. Features Transformed

### 3.1 Velocity Features (7 features)

| Feature | Body Part | Description |
|---------|-----------|-------------|
| `Ankle_IQRvelx` | Ankle | IQR of x-velocity |
| `Ankle_medianvelx` | Ankle | Median x-velocity |
| `Ankle_medianvely` | Ankle | Median y-velocity |
| `Ankle_IQRvely` | Ankle | IQR of y-velocity |
| `Elbow_IQR_vel_angle` | Elbow | IQR of angular velocity |
| `Elbow_median_vel_angle` | Elbow | Median angular velocity |
| `Hip_IQR_vel_angle` | Hip | IQR of angular velocity |

### 3.2 Acceleration Features (4 features)

| Feature | Body Part | Description |
|---------|-----------|-------------|
| `Ankle_IQRaccy` | Ankle | IQR of y-acceleration |
| `Elbow_IQR_acc_angle` | Elbow | IQR of angular acceleration |
| `Shoulder_IQR_acc_angle` | Shoulder | IQR of angular acceleration |
| `Wrist_IQRaccx` | Wrist | IQR of x-acceleration |

### 3.3 Features NOT Transformed (9 features)

These features are used without transformation:

| Feature | Body Part | Reason |
|---------|-----------|--------|
| `Ankle_IQRx` | Ankle | Position variability (not rate-based) |
| `Ankle_lrCorr_x` | Ankle | Correlation (already bounded) |
| `Ankle_meanent` | Ankle | Entropy (already bounded) |
| `Hip_stdev_angle` | Hip | Angular variability (not rate-based) |
| `Hip_entropy_angle` | Hip | Entropy (already bounded) |
| `Shoulder_mean_angle` | Shoulder | Mean angle (not rate-based) |
| `Wrist_mediany` | Wrist | Position (not rate-based) |
| `Wrist_IQRx` | Wrist | Position variability (not rate-based) |
| `age_in_weeks` | - | Demographic (not movement) |

---

## 4. Experimental Validation

### 4.1 Methodology

We tested 11 different transformation approaches using 10-fold cross-validation:

1. **original_one_tailed** - Baseline (no transformation, z < threshold)
2. **original_two_tailed** - Baseline with |z| > threshold
3. **abs_deviation** - |Value - mean| transformation
4. **squared_deviation** - (Value - mean)² transformation
5. **abs_zscore** - |z| per feature before summing
6. **abs_zscore_two_tailed** - |z| per feature with |z| > threshold
7. **squared_zscore** - z² per feature before summing
8. **inverse_vel_acc** - 1/(1+|x|) for velocity/acceleration features
9. **log_magnitude** - log(1 + |Value|) transformation
10. **rank_based** - Percentile distance from median
11. **rank_based_two_tailed** - Rank-based with |z| > threshold

### 4.2 Results Summary

**20-Feature (NEW) Model - Reproducible Results (seed=20250313):**

| Condition | AUC | Sensitivity | Specificity | Youden's J | Threshold |
|-----------|-----|-------------|-------------|------------|-----------|
| **With Transform** | **0.902** | **92.1%** | **77.5%** | **0.696** | -0.230 |
| Baseline (No Transform) | 0.849 | 92.9% | 61.0% | 0.539 | - |
| **Improvement** | +0.053 | -0.7% | +16.5% | +0.158 | - |

**All Feature Sets Comparison (with inverse transform):**

| Model | AUC | Sensitivity | Specificity | Youden's J |
|-------|-----|-------------|-------------|------------|
| **20-Feature (NEW)** | **0.902** | **92.1%** | **77.5%** | **0.696** |
| 15-Feature (NEW) | 0.860 | 85.7% | 70.0% | 0.557 |
| 19-Feature (NEW) | 0.822 | 78.6% | 66.5% | 0.451 |
| 20-Feature (LEGACY) | 0.886 | 85.7% | 76.5% | 0.622 |
| 18-Feature (LEGACY) | 0.885 | 84.3% | 72.0% | 0.563 |
| 15-Feature (LEGACY) | 0.887 | 78.6% | 79.0% | 0.576 |

### 4.3 Key Findings

1. **Two-tailed approaches consistently underperform** - All two-tailed methods ranked in the bottom half, confirming that at-risk infants cluster in one direction.

2. **Inverse transformation provides best overall improvement** - 16.5% specificity gain (61% → 77.5%) while maintaining ~92% sensitivity.

3. **Z-score separation increases with inverse transform** - Mean z-score difference between groups increases, indicating better discrimination.

4. **20-Feature (NEW) is the best model** - Highest AUC (0.902) and Youden's J (0.696) with transformation applied.

---

## 5. Implementation

### 5.1 Configuration

In `scripts/compute_roc_after_rfe.py`:

```python
# Reproducibility
RANDOM_SEED = 20250313  # For reproducible k-fold CV results

# Enable/disable transformation
APPLY_INVERSE_TRANSFORM = True  # Set to False for baseline
RUN_COMPARISON_MODE = True      # Set to True to compare WITH vs WITHOUT transform

# Features to transform
VELOCITY_FEATURES = [
    'Ankle_IQRvelx', 'Ankle_medianvelx', 'Ankle_medianvely', 'Ankle_IQRvely',
    'Elbow_IQR_vel_angle', 'Elbow_median_vel_angle', 'Hip_IQR_vel_angle'
]
ACCELERATION_FEATURES = [
    'Ankle_IQRaccy', 'Elbow_IQR_acc_angle', 'Shoulder_IQR_acc_angle', 'Wrist_IQRaccx'
]
TRANSFORM_FEATURES = VELOCITY_FEATURES + ACCELERATION_FEATURES
```

### 5.2 Transformation Function

```python
def apply_inverse_transform(features_df, features_to_transform):
    """
    Apply inverse transformation: 1/(1+|x|) to specified features.
    """
    df = features_df.copy()
    mask = df['feature'].isin(features_to_transform)
    df.loc[mask, 'Value'] = 1 / (1 + np.abs(df.loc[mask, 'Value']))
    return df
```

### 5.3 When Transformation is Applied

The transformation is applied in the k-fold CV loop:

1. Load raw features from pickle file
2. Create train/test split
3. **Apply inverse transformation** (if enabled)
4. Compute reference statistics from training set
5. Calculate Bayesian surprise
6. Normalize z-scores
7. Compute ROC metrics

---

## 6. Clinical Interpretation

### 6.1 Why Does This Work?

The inverse transformation works because:

1. **Captures both movement abnormalities**: Hypokinesia (low values → high transformed) and dyskinesia (high values → low transformed) both result in transformed values that differ from typical moderate movement.

2. **Compresses extreme values**: Very high accelerations (jerky movements) are compressed toward zero, preventing them from dominating the surprise calculation.

3. **Preserves relative ordering within normal range**: Moderate movements still maintain differentiation for normative distribution modeling.

### 6.2 Clinical Implications

| Metric | Baseline | With Transform | Clinical Impact |
|--------|----------|----------------|-----------------|
| Sensitivity | 92.9% | 92.1% | Essentially maintained |
| Specificity | 61.0% | 77.5% | **27% fewer false positives** |
| PPV | 62.5% | 74.1% | Higher confidence in positive results |
| NPV | 92.4% | 93.4% | Maintained negative predictive value |
| AUC | 0.849 | 0.902 | **+5.3% improvement** |

**False positive reduction**: For every 100 normal infants screened:
- Baseline: 39 false positives
- With transform: 22.5 false positives
- **16-17 fewer unnecessary referrals per 100 normal infants**

---

## 7. Recommendations

### 7.1 For Clinical Validation

1. **Use the inverse transformation** as the default for the 20-feature NEW model
2. Set `APPLY_INVERSE_TRANSFORM = True` in configuration
3. Report both baseline and transformed metrics for comparison

### 7.2 For Future Development

1. **Feature-specific transformations**: Consider different transformations for different feature types
2. **Adaptive thresholds**: The optimal threshold shifts from +0.17 (baseline) to -0.25 (transformed)
3. **Age-stratified analysis**: Verify transformation benefits hold across age groups

### 7.3 For Reproducibility

1. Document transformation parameters in all publications
2. Include both raw and transformed feature values in datasets
3. Version control the transformation configuration

---

## 8. Files and References

### 8.1 Implementation Files

| File | Description |
|------|-------------|
| `scripts/compute_roc_after_rfe.py` | Main ROC analysis with transformation |
| `scripts/explore_feature_transformations.py` | Transformation comparison experiments |
| `transformation_comparison_results.csv` | Full experimental results |

### 8.2 Related Documentation

| Document | Description |
|----------|-------------|
| `CLAUDE.md` | Project context and methodology overview |
| `docs/WORKFLOWS.md` | Step-by-step pipeline documentation |
| `docs/ITERATION_FORCED_AGE_RFE.md` | Age-adjusted feature selection |

---

## Appendix A: Complete Transformation Comparison Results

**Reproducible Results (RANDOM_SEED = 20250313, 10-fold CV)**

### A.1 20-Feature (NEW) Model - Baseline vs Transformed

```
Metric          |  Baseline  | Transformed |  Change
---------------------------------------------------------
AUC             |    0.8485  |     0.9018  |  +0.0532
Sensitivity     |    0.9286  |     0.9214  |  -0.0071
Specificity     |    0.6100  |     0.7750  |  +0.1650
Youden's J      |    0.5386  |     0.6964  |  +0.1579
Threshold       |        -   |    -0.2303  |      -
```

### A.2 All Models with Inverse Transform Applied

```
Model               |   AUC  | Sensitivity | Specificity | Youden J | Threshold
---------------------------------------------------------------------------------
20-Feature (NEW)    | 0.9018 |     0.9214  |      0.7750 |   0.6964 |   -0.2303
15-Feature (LEGACY) | 0.8866 |     0.7857  |      0.7900 |   0.5757 |   -0.2716
20-Feature (LEGACY) | 0.8861 |     0.8571  |      0.7650 |   0.6221 |   -0.3955
15-Feature (NEW)    | 0.8596 |     0.8571  |      0.7000 |   0.5571 |   -0.1597
18-Feature (LEGACY) | 0.8849 |     0.8429  |      0.7200 |   0.5629 |    0.0133
19-Feature (NEW)    | 0.8216 |     0.7857  |      0.6650 |   0.4507 |   -0.0794
```

### A.3 Improvement from Transformation (All Models)

```
Model               | AUC Change | Se Change | Sp Change | Youden J Change
---------------------------------------------------------------------------
20-Feature (NEW)    |   +0.0532  |   -0.0071 |   +0.1650 |        +0.1579
19-Feature (NEW)    |   +0.0305  |   +0.0500 |   +0.0450 |        +0.0950
15-Feature (NEW)    |   +0.0207  |   +0.0714 |   +0.0150 |        +0.0864
20-Feature (LEGACY) |   +0.0167  |    0.0000 |   +0.0750 |        +0.0750
18-Feature (LEGACY) |   +0.0274  |   -0.0143 |   +0.0400 |        +0.0257
15-Feature (LEGACY) |   +0.0266  |   -0.0643 |   +0.0900 |        +0.0257
```

---

### A.4 Mean separation between Typical and At-risk groups using the 20-Feature set

Cohen's d measures the standardized difference between two group means in standard deviation units.

  Interpretation

  | Cohen's d | Interpretation | Your Results        |
  |-----------|----------------|---------------------|
  | 0.2       | Small effect   |                     |
  | 0.5       | Medium effect  |                     |
  | 0.8       | Large effect   |                     |
  | 0.74      | Medium-large   | Baseline features   |
  | 1.02      | Large          | Inverse transformed |

  What These Values Mean

  d = 0.74 (baseline): The typical and at-risk groups differ by 0.74 standard deviations. There's substantial
  overlap between distributions, but meaningful separation.

  d = 1.02 (transformed): The groups differ by ~1 full standard deviation. This is a large effect - the average
  at-risk infant's z-score exceeds about 84% of typical infants' scores.

  Practical Significance

  For a screening tool:

  | Cohen's d | % Non-overlap | Probability of Correct Ranking |
  |-----------|---------------|--------------------------------|
  | 0.74      | ~43%          | ~70%                           |
  | 1.02      | ~55%          | ~76%                           |

  The transformation increased non-overlap by ~12 percentage points, meaning better discrimination between groups.

  Relationship to Your AUC

  The improvement from d=0.74 → d=1.02 (+38%) aligns with the AUC improvement from 0.847 → 0.902 (+6.5%). Cohen's d
   captures mean separation, while AUC captures overall discriminability across all thresholds - both confirm the
  transformation substantially improved screening performance.


## Appendix B: Code for Reproducing Results

```python
# Run transformation comparison experiments
conda run -n conda-caffe python scripts/explore_feature_transformations.py

# Run main analysis with transformation enabled
# In scripts/compute_roc_after_rfe.py, ensure:
APPLY_INVERSE_TRANSFORM = True

# Then run:
conda run -n conda-caffe python scripts/compute_roc_after_rfe.py
```

---

*Document generated: November 26, 2025*
*Updated with reproducible results using RANDOM_SEED = 20250313*
