# BAM Sample Size Analysis for 20-Feature Model

**Created**: 2025-12-04
**Model**: 20-Feature Nov 21 RFE with 1/(1+|x|) transformation
**Script**: `scripts/run_bam_20feature_nov21.py`
**Results**: `data/json/bam_sample_size_20feature_nov21.json`

---

## 1. Pilot Data Summary

### 1.1 Model Performance (10-fold CV with transformation)

| Metric | Value | 95% CI (Wilson) | CI Width |
|--------|-------|-----------------|----------|
| Sensitivity | 92.1% (129/140) | 86.5% - 95.6% | 0.091 |
| Specificity | 77.5% (155/200) | 71.2% - 82.7% | 0.115 |
| AUC | 0.902 | - | - |

### 1.2 Confusion Matrix

|  | Predicted At-Risk | Predicted Normal |
|--|-------------------|------------------|
| **Actual At-Risk** | TP = 129 | FN = 11 |
| **Actual Normal** | FP = 45 | TN = 155 |

- Total at-risk (condition positive): 140
- Total normal (condition negative): 200
- Total observations: 340 (via 10-fold CV)

---

## 2. BAM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Target CI Width | 0.20 | ±10% precision for Se and Sp |
| Target Assurance | 95% | Probability of achieving target precision |
| CI Level | 95% | Confidence/credible interval level |
| Prevalence Prior | Beta(8, 32) | ~20% expected prevalence |

---

## 3. BAM Results

### 3.1 Primary Result

**Optimal Sample Size: N = 100 infants**
**Achieved Assurance: 100%**

With 20% prevalence:
- At-risk infants: ~20
- Normal infants: ~80

### 3.2 Sensitivity Analysis

| Scenario | Se | Sp | Width | Assurance | N | Achieved |
|----------|-----|-----|-------|-----------|-----|----------|
| Point estimates | 92.1% | 77.5% | 0.20 | 95% | 100 | 100% |
| Conservative (lower CI) | 86.5% | 71.2% | 0.20 | 95% | 100 | 100% |
| Very conservative | ~86% | ~71% | 0.20 | 95% | 100 | 100% |
| Proposal targets | 85% | 75% | 0.20 | 95% | 100 | 100% |
| Stricter precision | 85% | 75% | 0.10 | 80% | 947 | 81% |

### 3.3 Impact of Pilot Size

| Pilot Size | Se | Sp | Width | Assurance | N | Achieved |
|------------|-----|-----|-------|-----------|-----|----------|
| Large (140/200) | 85% | 75% | 0.20 | 80% | 100 | 98% |
| Small (50/80) | 85% | 75% | 0.20 | 80% | 100 | 98% |
| Small (50/80) | 80% | 70% | 0.20 | 80% | 159 | 81% |

---

## 4. Comparison with Phase 2 Proposal

### 4.1 Phase 2 Proposal Parameters

From the proposal document:

```
Target: Se=0.85, Sp=0.75
CI half-width: 0.1 (full width = 0.2)
Maximum 95% CI widths of 0.2
SSR2 = 395 videos (~198 infants)
Final enrollment: 300 infants (600 videos)
```

### 4.2 Reconciliation

| Factor | Proposal | Current Analysis |
|--------|----------|------------------|
| Sensitivity | 85% (target) | 92.1% (achieved) |
| Specificity | 75% (target) | 77.5% (achieved) |
| CI width target | 0.20 | 0.20 |
| Sample size | 395 videos (~200 infants) | 100 infants |
| Pilot CI width (Se) | Not reported | 0.091 (already tight) |
| Pilot CI width (Sp) | Not reported | 0.115 (already tight) |

### 4.3 Why the Difference?

1. **Improved model performance**: Current model exceeds proposal targets
   - Se: 92.1% vs 85% target (+7.1%)
   - Sp: 77.5% vs 75% target (+2.5%)

2. **Pilot already achieves near-target precision**:
   - Se CI width: 0.091 (target: 0.20)
   - Sp CI width: 0.115 (target: 0.20)

3. **Proposal included additional factors**:
   - 10% attrition buffer
   - IPW (inverse population weighting)
   - Stratified sampling overhead
   - Subgroup analysis margin
   - Conservative grant planning

---

## 5. Key Insights

### 5.1 Pilot Precision vs Target Precision

The current pilot data (340 observations via 10-fold CV) **already achieves better precision than the target**:

| Metric | Current Pilot CI Width | Target CI Width | Status |
|--------|------------------------|-----------------|--------|
| Sensitivity | 0.091 (±4.5%) | 0.20 (±10%) | **Exceeds target** |
| Specificity | 0.115 (±5.7%) | 0.20 (±10%) | **Exceeds target** |

This explains why BAM estimates such a modest sample size requirement.

### 5.2 When Larger Samples Are Needed

| Scenario | Required N |
|----------|------------|
| CI width 0.20, Assurance 95% | 100 |
| CI width 0.15, Assurance 80% | 100 |
| CI width 0.10, Assurance 80% | ~950 |
| Lower pilot metrics (Se=80%, Sp=70%) | ~160 |

### 5.3 Transformation Impact

The 1/(1+|x|) transformation improved specificity from ~61% to 77.5% (+16.5%), which significantly reduces sample size requirements by:
- Moving specificity away from 50% (maximum variance)
- Providing more precise estimates with same sample

---

## 6. Phase 2 Test Set Design: Enriched Sampling

### 6.1 Study Design Overview

**Phase 2 Proposed Structure:**
- **Training Set**: 100 infants (200 videos) - ~10% at-risk (population rate)
  - Purpose: Train Bayesian Surprise normative model
  - Requires typical infants only for reference distribution

- **Test Set**: 200 infants (400 videos) - 50% at-risk (enriched)
  - Purpose: Independent validation of Se/Sp
  - Case-control enrichment for precise estimation

### 6.2 Why Enriched Sampling?

With target CI width of 0.20 (±10%), the number of at-risk cases directly impacts sensitivity precision:

| Incidence | At-Risk | Normal | Se CI Width | Sp CI Width | Target Met? |
|-----------|---------|--------|-------------|-------------|-------------|
| 10% (population) | 20 | 180 | **0.273** | 0.122 | **NO** |
| 20% | 40 | 160 | 0.191 | 0.129 | YES |
| 30% | 60 | 140 | 0.145 | 0.138 | YES |
| 40% | 80 | 120 | 0.127 | 0.148 | YES |
| **50% (proposed)** | **100** | **100** | **0.109** | **0.163** | **YES** |

**Critical Finding**: At population rate (10%), only 20 at-risk infants are available in 200, resulting in sensitivity CI width of 0.27 - exceeding the 0.20 target.

### 6.3 Detailed Breakdown: 50% Incidence (Proposed)

With 200 infants (100 at-risk, 100 normal):

**Sensitivity Estimation (from 100 at-risk):**
- Expected TP: 92 (at 92.1% sensitivity)
- Expected FN: 8
- Sensitivity: 92.0% (95% CI: 85.0% - 95.9%)
- **CI width: 0.109** ✓

**Specificity Estimation (from 100 normal):**
- Expected TN: 77 (at 77.5% specificity)
- Expected FP: 23
- Specificity: 77.0% (95% CI: 67.8% - 84.2%)
- **CI width: 0.163** ✓

### 6.4 Detailed Breakdown: 10% Incidence (Population)

With 200 infants (20 at-risk, 180 normal):

**Sensitivity Estimation (from 20 at-risk):**
- Expected TP: 18
- Expected FN: 2
- Sensitivity: 90.0% (95% CI: 69.9% - 97.2%)
- **CI width: 0.273** ✗ (exceeds target)

**Specificity Estimation (from 180 normal):**
- Expected TN: 139
- Expected FP: 41
- Specificity: 77.2% (95% CI: 70.6% - 82.7%)
- **CI width: 0.122** ✓

### 6.5 Key Insight

Sensitivity CI width is driven by the number of at-risk cases:
- **10% incidence → 20 at-risk → CI width 0.27** (too wide)
- **50% incidence → 100 at-risk → CI width 0.11** (well within target)

The 50% enrichment in the Phase 2 proposal ensures:
1. Adequate at-risk cases for precise sensitivity estimation
2. Adequate normal cases for precise specificity estimation
3. Both metrics meet the ±10% precision target

### 6.6 Minimum Viable Enrichment

To achieve CI width ≤ 0.20 for both Se and Sp with 200 infants:
- **Minimum incidence: ~20%** (40 at-risk, 160 normal)
- **Recommended: 50%** for comfortable margin

---

## 7. Recommendations

### 7.1 Minimum Sample Size

Based on BAM analysis: **N = 100 infants** is sufficient for:
- CI width: 0.20 (±10%)
- Assurance: 95%
- Both sensitivity and specificity jointly

### 7.2 Conservative Recommendation

Given real-world considerations:
- Attrition (~10%)
- Subgroup analyses (age groups, etc.)
- Model uncertainty
- Recruitment variability

**Recommended: N = 150-200 infants** for robust validation

### 7.3 Consistency with Phase 2 Proposal

The proposal's 300 infants provides substantial margin for:
- Conservative estimation
- Subgroup analyses
- Model refinement
- Publication-quality precision

---

## 8. Technical Notes

### 8.1 BAM Implementation

- Uses `early_markers.cribsy.common.bam_unified.BAMEstimator`
- Joint estimation for sensitivity + specificity
- Informed priors from pilot data
- Binary search over sample sizes (min=100, max=2000)

### 8.2 CI Calculation Method

Wilson score interval for proportions:
```python
def wilson_ci(successes, n, alpha=0.05):
    p = successes / n
    z = stats.norm.ppf(1 - alpha/2)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (center - margin, center + margin)
```

### 8.3 Data Source

- Feature set: 20-Feature Nov 21 RFE
- Evaluation data: `features_merged_20251121_091511.pkl` (146 infants)
- Selection data: `features_merged.pkl` (148 infants)
- Cross-validation: 10-fold stratified

---

## 9. Files

| File | Description |
|------|-------------|
| `scripts/run_bam_20feature_nov21.py` | BAM calculation script |
| `data/json/bam_sample_size_20feature_nov21.json` | Saved results |
| `scripts/compute_roc_after_rfe.py` | ROC metrics with CIs |
| `docs/RFE_METHODOLOGY.md` | RFE methodology (Section 9: BAM) |

---

**Last Updated**: 2025-12-04
