# Bidirectional Atypicality Detection

**Date**: December 3, 2025
**Scripts**:
- `scripts/visualize_transformation.py`
- `scripts/test_hyperkinesia_detection.py`
- `scripts/test_linear_only_transform.py`

## Dataset

The model is trained and validated on movement data from **146 infants**:

| Source | N | Risk Status | Description |
|--------|---|-------------|-------------|
| YouTube | 85 | Typical (risk=0) | Normative infant videos curated from YouTube |
| Chambers Study | 19 | Mixed | Clinical study infants |
| Early Markers Study | 42 | Mixed | Clinical study infants with developmental follow-up |

**Final composition**:
- Typical infants (risk ≤ 1): 132
- At-risk infants (risk > 1): 14

Movement features are extracted from pose estimation (joint positions over time) and include velocity, acceleration, variability, and entropy measures across body parts.

## Overview

This document validates that the Bayesian Surprise model with 1/(1+|x|) transformation can detect **both hypokinesia (reduced movement) and hyperkinesia (excessive movement)**. Real at-risk infants in our dataset show hypokinesia, but the model's mathematical properties ensure bidirectional sensitivity.

## The 1/(1+|x|) Transformation

### Formula and Properties

```
f(x) = 1 / (1 + |x|)
```

- Maps low values (hypokinesia) → high transformed values (near 1.0)
- Maps high values (hyperkinesia) → low transformed values (near 0)
- Typical values → moderate transformed values (~0.5-0.8)

### Features Transformed

11 velocity/acceleration features from the 20-Feature NEW set:

| Type | Features |
|------|----------|
| Linear velocity (4) | Ankle_IQRvelx, Ankle_medianvelx, Ankle_medianvely, Ankle_IQRvely |
| Linear acceleration (2) | Ankle_IQRaccy, Wrist_IQRaccx |
| Angular velocity (3) | Elbow_IQR_vel_angle, Elbow_median_vel_angle, Hip_IQR_vel_angle |
| Angular acceleration (2) | Elbow_IQR_acc_angle, Shoulder_IQR_acc_angle |

## Feature Scale Analysis

The 11 transformed features span 3 orders of magnitude:

| Feature Type | Typical Mean | Transformed Value | Sensitivity |
|--------------|--------------|-------------------|-------------|
| Linear velocity | 0.1-0.3 | 0.77-0.91 | HIGH (0.69) |
| Linear acceleration | 1.0-1.1 | 0.47-0.48 | Medium (0.23) |
| Angular velocity | ~38 | 0.026 | Low (6.6×10⁻⁴) |
| Angular acceleration | 164-179 | 0.006 | Very Low (3.4×10⁻⁵) |

**Key insight**: Linear velocity features have ~20,000× higher transformation sensitivity than angular acceleration features.

## Transform Subset Comparison

Despite lower sensitivity, transforming ALL vel/acc features yields best results:

| Transform Set | N | AUC | Youden's J | Z-Separation |
|---------------|---|------|------------|--------------|
| **all_vel_acc** | 11 | **0.867** | **0.656** | **2.53** |
| angular_only | 5 | 0.844 | 0.621 | 2.38 |
| linear_vel_acc | 6 | 0.822 | 0.580 | 1.34 |
| linear_vel_only | 4 | 0.811 | 0.554 | 1.12 |
| none (baseline) | 0 | 0.798 | 0.537 | 1.01 |

Angular features contribute MORE to classification than linear features, suggesting larger baseline separation between at-risk and typical for angular movement patterns.

## Hyperkinesia Detection Validation

### Motivation

Real at-risk infants show **hypokinesia** (reduced movement). We validated bidirectional detection using synthetic hyperkinetic infants.

### Synthetic Infant Generation

Created 3 types of synthetic hyperkinetic infants (N=10 each, severity=1.5 std above typical):

1. **Lower body hyperkinesia**: Elevated ankle/hip features
2. **Upper body hyperkinesia**: Elevated wrist/elbow/shoulder features
3. **Global hyperkinesia**: All body parts elevated

### Results (10-fold CV)

| Group | Mean Z | Detection Rate | AUC vs Typical |
|-------|--------|----------------|----------------|
| **Real At-Risk (hypokinesia)** | -2.69 | 90.7% | 0.868 |
| Synthetic Lower Body | -189.5 | **100%** | 0.920 |
| Synthetic Upper Body | -0.73 | **88%** | 0.833 |
| Synthetic Global | -1.62 | **100%** | 0.920 |

### Key Findings

1. **Lower body hyperkinesia**: Extremely detectable (Mean Z = -189) due to high ankle velocity sensitivity
2. **Upper body hyperkinesia**: Also detected (88%) despite lower angular feature sensitivity
3. **Global hyperkinesia**: 100% detection rate
4. **The model is bidirectionally sensitive** - detects both reduced AND excessive movement

## Why Bidirectional Detection Works

### Mathematical Mechanism

The Bayesian Surprise model computes log-probability:

```
log_p = -0.5 × log(2πσ²) - (x - μ)² / (2σ²)
```

The **(x - μ)²** term is **symmetric** - deviations in EITHER direction decrease probability.

### Detection Pathway

| Pattern | Original Value | Transformed Value | Deviation from Typical | Result |
|---------|---------------|-------------------|----------------------|--------|
| Hypokinesia | LOW | HIGH (near 1.0) | Large positive | DETECTED |
| Typical | Moderate | Moderate (~0.5-0.8) | Small | Normal |
| Hyperkinesia | HIGH | LOW (near 0) | Large negative | DETECTED |

The transformation maps BOTH abnormal patterns to regions far from the typical distribution. The symmetric squared deviation catches both.

## Effect Size Analysis

The transformation improves Cohen's d despite increasing at-risk variance:

| Condition | Mean Separation | At-Risk Std | Cohen's d |
|-----------|-----------------|-------------|-----------|
| Baseline | 0.969 | 1.177 | 0.74 (medium-large) |
| Transformed | 2.531 | 3.145 | 1.02 (large) |

The variance increase reflects **asymmetric amplification**: truly atypical infants become MORE atypical (signal), not uniform noise.

## Clinical Implications

1. **Current population**: Primarily hypokinesia pattern, well detected (AUC=0.87, Sens=90.7%)

2. **Future hyperkinetic cases**: Model should detect dyskinesia, spasticity, or excessive movement without modification

3. **Mixed presentations**: Infants with complex patterns (some features hypo, some hyper) would be flagged due to overall deviation

4. **Validation opportunity**: As more real infant data is acquired, compare detection across different movement phenotypes

## Summary

The 1/(1+|x|) transformation combined with Bayesian Surprise log-probability provides **bidirectional atypicality detection**:

- Real hypokinetic at-risk infants: **90.7% detected**
- Synthetic hyperkinetic infants: **88-100% detected**
- The model doesn't distinguish direction - it detects ANY deviation from typical movement

This validates the approach for diverse clinical presentations as the dataset grows.

## Files

- `transformation_analysis.png` - Visualization of transformation properties
- `scripts/test_hyperkinesia_detection.py` - Synthetic validation script
- `scripts/test_linear_only_transform.py` - Transform subset comparison
- `scripts/visualize_transformation.py` - Generates analysis figure

## References

- Bayesian Surprise methodology: `CLAUDE.md`, `docs/WORKFLOWS.md`
- Feature transformation details: `docs/FEATURE_TRANSFORMATION_METHODOLOGY.md`
- At-risk bimodality: `docs/ATRISK_BIMODALITY_ANALYSIS.md`
