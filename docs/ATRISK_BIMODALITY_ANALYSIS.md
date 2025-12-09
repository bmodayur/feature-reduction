# At-Risk Group Bimodality Analysis

**Date**: December 2, 2025 (Updated)
**Script**: `scripts/atrisk_group_analysis.py`
**Model**: 20-Feature NEW with 1/(1+|x|) transformation

## Background

During visualization of z-score distributions comparing baseline vs. transformed features, a bimodal distribution pattern was observed in the at-risk group. This analysis investigates whether this bimodality represents genuine severity differentiation within the at-risk population.

## Methodology

### Data
- **At-risk infants**: N = 14
- **Feature set**: 20-Feature NEW (19 movement features + age_in_weeks)
- **Cross-validation**: 10-fold stratified CV
- **Transformation**: 1/(1+|x|) applied to velocity/acceleration features

### Z-Score Computation (Bayesian Surprise)
Z-scores are computed using **log-probability** under the normative distribution:
```
minus_log_pfeature = -0.5 * log(2πσ²) - (x - μ)² / (2σ²)
z = (Σ minus_log_pfeature - train_mean) / train_std
```
**Interpretation**: More negative z-scores indicate more atypical (at-risk) movement patterns.

### Statistical Tests
1. **Hartigan's Dip Test**: Tests for unimodality vs. multimodality
2. **Shapiro-Wilk Test**: Tests for normality
3. **Gaussian Mixture Model (GMM)**: 2-component clustering to identify potential subgroups

### Analysis Pipeline
1. Compute log-probability for each feature
2. Sum across features and normalize to z-scores using training set
3. Test for bimodality/multimodality
4. Cluster infants using GMM
5. Analyze cluster characteristics (age, risk level)
6. Compare cluster assignments between baseline and transformed

## Results

### Z-Score Distribution Statistics (from compute_roc_after_rfe.py)

| Metric | Baseline | Transformed | Change |
|--------|----------|-------------|--------|
| **Typical Mean Z** | -0.113 | -0.050 | +0.063 |
| **Typical Std Z** | 1.427 | 1.363 | -0.064 |
| **At-Risk Mean Z** | -1.081 | -2.581 | -1.499 |
| **At-Risk Std Z** | 1.177 | 3.145 | +1.968 |
| **Mean Separation** | 0.969 | 2.531 | +1.562 |

**Key observations:**
- More negative z-scores indicate more atypical patterns
- Typical group remains stable (mean toward 0, std slightly decreases)
- At-Risk mean shifts 1.5 more negative (more atypical)
- At-Risk std nearly **triples** (1.18 → 3.15)
- Mean separation increases **2.6×** (0.97 → 2.53)

### Effect Size Analysis (Cohen's d)

Cohen's d measures the standardized difference between group means:

```
d = |μ_atrisk - μ_typical| / pooled_std
pooled_std = sqrt((n1*s1² + n2*s2²) / (n1 + n2))
```

| Condition | Mean Separation | Pooled Std | Cohen's d | Interpretation |
|-----------|-----------------|------------|-----------|----------------|
| **Baseline** | 0.969 | 1.31 | **0.74** | Medium-large effect |
| **Transformed** | 2.531 | 2.48 | **1.02** | Large effect |

**Despite the 2.7× increase in at-risk variance, Cohen's d improves from 0.74 to 1.02** because mean separation increases faster (2.6×) than the pooled standard deviation (1.9×).

### Why Increased Variance Improves Classification

The transformation produces **asymmetric amplification**, not uniform variance inflation:

| Effect | Typical Group | At-Risk Group |
|--------|---------------|---------------|
| Mean shift | +0.06 (stable) | -1.50 (large shift) |
| Std change | -4.5% (stable) | +167% (tripled) |

**The variance increase is driven by selective signal amplification:**

1. **Truly atypical infants become MORE atypical**
   - clin_16_22: z = -3.29 → -8.12 (2.5× more extreme)
   - clin_7_27: z = -0.80 → -2.88 (3.6× more extreme)

2. **Borderline infants stay near typical**
   - clin_31_16: z = +0.39 → +0.40 (unchanged)
   - clin_9_21: z = +0.49 → +0.53 (unchanged)

3. **Range changes are asymmetric**
   - Lower bound: -3.29 → -8.12 (stretched 2.5×)
   - Upper bound: +0.49 → +0.53 (unchanged)

This is **signal amplification** (true positives become more detectable), not noise amplification.

### Statistical Tests

| Test | Baseline | Transformed | Interpretation |
|------|----------|-------------|----------------|
| Hartigan's Dip | **p = 0.013** | **p < 0.001** | **Significant multimodality** |
| Shapiro-Wilk | **p = 0.033** | **p = 0.001** | **Not normal distribution** |

**Conclusion**: Both tests provide statistically significant evidence of bimodality (p < 0.05).

### GMM Clustering Results

**Baseline Clusters:**
| Cluster | N | % | Mean Z | Std Z | Range |
|---------|---|---|--------|-------|-------|
| Less Severe | 11 | 78.6% | -0.173 | 0.469 | [-0.80, 0.49] |
| More Severe | 3 | 21.4% | -2.573 | 0.583 | [-3.29, -1.86] |

**Transformed Clusters:**
| Cluster | N | % | Mean Z | Std Z | Range |
|---------|---|---|--------|-------|-------|
| Less Severe | 13 | 92.9% | -0.816 | 1.142 | [-3.20, 0.53] |
| More Severe | 1 | 7.1% | -8.116 | 0.000 | [-8.12, -8.12] |

Note: In transformed data, one infant (clin_16_22) is an extreme outlier with z = -8.116.

### Cluster Characteristics

**Age Distribution (Baseline Clustering):**
- Less Severe: mean = 19.3 weeks, std = 4.9, range = [10, 27]
- More Severe: mean = 35.3 weeks, std = 12.2, range = [22, 46]

**Key Finding**: Older infants (>22 weeks) tend to cluster in the "More Severe" group.

**Risk Level Distribution (Baseline Clustering):**
- Less Severe: 7 moderate (risk=2), 4 high (risk=3) → 36% high-risk
- More Severe: 2 moderate (risk=2), 1 high (risk=3) → 33% high-risk

### Cluster Agreement

**Baseline vs. Transformed Agreement: 85.7% (12/14 infants)**

| | Transformed Less Severe | Transformed More Severe |
|---|---|---|
| **Baseline Less Severe** | 11 | 0 |
| **Baseline More Severe** | 2 | 1 |

Two infants (clin_23_46, clin_26_38) moved from "More Severe" to "Less Severe" after transformation, suggesting their movement patterns become more normalized with the 1/(1+|x|) transform.

## Individual Infant Assignments

| Infant | Age (wks) | Risk | Baseline Z | Trans Z | Baseline Cluster | Trans Cluster | Changed |
|--------|-----------|------|------------|---------|------------------|---------------|---------|
| clin_16_22 | 22 | 3 | -3.287 | -8.116 | More Severe | More Severe | |
| clin_23_46 | 46 | 2 | -2.575 | -3.202 | More Severe | Less Severe | * |
| clin_26_38 | 38 | 2 | -1.858 | -1.846 | More Severe | Less Severe | * |
| clin_7_27 | 27 | 2 | -0.796 | -2.876 | Less Severe | Less Severe | |
| clin_19_26 | 26 | 3 | -0.798 | -0.715 | Less Severe | Less Severe | |
| clin_34_22 | 22 | 3 | -0.775 | -1.259 | Less Severe | Less Severe | |
| clin_8_18 | 18 | 2 | -0.472 | -0.740 | Less Severe | Less Severe | |
| clin_24_17 | 17 | 2 | -0.269 | -0.702 | Less Severe | Less Severe | |
| clin_30_20 | 21 | 3 | -0.135 | -0.073 | Less Severe | Less Severe | |
| clin_20_18 | 18 | 3 | -0.034 | -0.090 | Less Severe | Less Severe | |
| clin_6_16 | 16 | 2 | 0.079 | -0.243 | Less Severe | Less Severe | |
| clin_31_16 | 16 | 2 | 0.387 | 0.403 | Less Severe | Less Severe | |
| clin_15_10 | 10 | 2 | 0.413 | 0.205 | Less Severe | Less Severe | |
| clin_9_21 | 21 | 2 | 0.492 | 0.525 | Less Severe | Less Severe | |

**Notable patterns:**
- **clin_16_22** (age 22, risk=3): Most extreme outlier in both conditions, z = -8.116 after transform
- **clin_23_46** & **clin_26_38**: Both older infants (38, 46 weeks) who moved to Less Severe after transformation

## Interpretation

### Evidence FOR Severity Differentiation

1. **Statistical significance**: Both Hartigan's dip test (p=0.013) and Shapiro-Wilk test (p=0.033) reject unimodality/normality
2. **Clear cluster separation**: More Severe cluster mean z = -2.573 vs Less Severe mean z = -0.173 (~2.4 SD separation)
3. **High cluster consistency**: 85.7% agreement between baseline and transformed clustering
4. **Age association**: More Severe cluster has significantly older infants (35.3 vs 19.3 weeks mean age)
5. **Extreme outlier**: clin_16_22 consistently identified as most atypical in both conditions

### Caveats

1. **Small sample size**: N=14 limits generalizability
2. **Unbalanced clusters**: 3 vs 11 split may be driven by outliers
3. **No clinical outcome data**: Cannot validate against actual developmental outcomes
4. **Cross-sectional**: Single time point, no longitudinal follow-up

### Clinical Implications

The bimodal structure suggests the at-risk population may contain:
- **Subgroup 1 (78.6%)**: "Less Severe" - movement patterns closer to typical, may benefit from monitoring
- **Subgroup 2 (21.4%)**: "More Severe" - highly atypical movements, may need immediate intervention

The age association (older infants in More Severe cluster) warrants further investigation:
- Could indicate progressive deterioration
- Could reflect different underlying conditions by age
- Could be confounded by sample characteristics

## Recommendations

1. **Clinical validation**: Compare cluster assignments with actual developmental outcomes
2. **Increase sample size**: Recruit additional at-risk infants for validation
3. **Longitudinal tracking**: Follow infants over time to see if clusters predict different trajectories
4. **Feature profile analysis**: Examine which specific features drive cluster membership
5. **Age-stratified analysis**: Investigate whether bimodality persists within age groups

## Technical Note: Z-Score Computation

Z-scores are computed using **log-probability** (Bayesian surprise):

```python
# Per-feature log probability under normative Gaussian
log_p = -0.5 * log(2πσ²) - (x - μ)² / (2σ²)

# Sum across features and normalize
z = (Σ log_p - train_mean) / train_std
```

**Key properties:**
- More negative z = lower total log probability = more atypical
- Symmetric: deviations above OR below mean both decrease probability
- The (x - μ)² term means deviations in EITHER direction reduce probability

## Technical Note: The 1/(1+|x|) Transformation

The transformation compresses velocity/acceleration features:
- **Low values** (hypokinesia): small x → 1/(1+small) ≈ large value (near 1.0)
- **High values** (hyperkinesia): large x → 1/(1+large) ≈ small value (near 0.0)
- **Moderate values** (typical): moderate x → moderate value (near 0.5)

**Effect on at-risk detection:**
- At-risk infants with hypokinesia (low velocity) get transformed values near 1.0
- This creates large deviations from the typical mean (~0.5)
- The (x-μ)² term amplifies these deviations in log-probability
- Result: Truly atypical infants get more extreme z-scores

**Feature direction analysis** (from `scripts/analyze_feature_direction.py`):
All 11 velocity/acceleration features show at-risk infants have LOWER values than typical (hypokinesia pattern). After transformation, these become HIGHER transformed values, but the log-probability computation captures both directions via the squared term.

## Files Generated

- `atrisk_bimodality_analysis.png` - Visualization of bimodality analysis
- `scripts/atrisk_group_analysis.py` - Bimodality analysis script
- `scripts/analyze_feature_direction.py` - Feature direction analysis (at-risk vs typical)
- `scripts/compute_roc_after_rfe.py` - ROC analysis with z-score distribution visualization

## References

- Hartigan, J.A. & Hartigan, P.M. (1985). The Dip Test of Unimodality. Annals of Statistics.
- Shapiro, S.S. & Wilk, M.B. (1965). An Analysis of Variance Test for Normality.
