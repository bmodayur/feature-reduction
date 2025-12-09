# Feature Set Provenance Tracking

**Created**: 2025-12-03
**Purpose**: Track the origin, data sources, and generation methods for all feature sets

---

## Critical Data Files

| File | Location | Infants | Date Created | Notes |
|------|----------|---------|--------------|-------|
| `features_merged.pkl` | `data/pkl/` | 148 | Oct 16, 2025 | **DEPRECATED** - Default in BayesianData() |
| `features_merged_20251121_091511.pkl` | `data/pkl/` | 146 | Nov 21, 2025 | **CURRENT** - Use for all new analysis |

**WARNING**: Only 104 infants are common between these files. Always verify which file is being used.

---

## Feature Sets Inventory

### 1. NEW RFE 20-Feature Set (December 2025)

**Generation Date**: 2025-12-03
**Script**: `scripts/run_rfe_with_age_forced.py`
**Data File**: `features_merged_20251121_091511.pkl` ✓
**Output Files**:
- `data/json/final_20_features_forced_age.json`
- `data/pkl/bd_final20_forced_age.pkl`

**Performance** (single train/test split):
- AUC: 0.830
- Sensitivity: 85.7%
- Specificity: 72.5%
- Youden's J: 0.5821

**Performance** (10-fold CV, WITH inverse transform):
- AUC: 0.8710
- Sensitivity: 90.0%
- Specificity: 74.5%
- Youden's J: 0.6450
- Evaluation script: `scripts/compute_roc_rfe_dec2025.py`

**Performance** (10-fold CV, baseline NO transform):
- AUC: 0.7987
- Sensitivity: 84.3%
- Specificity: 65.5%
- Youden's J: 0.4979

**Features** (ranked by importance):
```
 1. Hip_stdev_angle        (0.1943)
 2. Hip_mean_angle         (0.0700)
 3. Elbow_IQR_acc_angle    (0.0610)
 4. Hip_entropy_angle      (0.0559)
 5. Ankle_IQRaccy          (0.0332)
 6. Ankle_IQRaccx          (0.0283)
 7. Shoulder_IQR_acc_angle (0.0282)
 8. Ankle_medianvelx       (0.0266)
 9. Shoulder_lrCorr_angle  (0.0251)
10. Elbow_IQR_vel_angle    (0.0216)
11. Ankle_meanent          (0.0215)
12. Wrist_IQRaccx          (0.0191)
13. Ankle_mediany          (0.0176)
14. Ankle_IQRx             (0.0173)
15. Shoulder_stdev_angle   (0.0166)
16. Wrist_IQRvelx          (0.0161)
17. Knee_lrCorr_angle      (0.0158)
18. Ankle_IQRvely          (0.0147)
19. Shoulder_mean_angle    (0.0146)
20. age_in_weeks           [FORCED]
```

**Provenance**: VERIFIED - Generated with traceable script and data file

---

### 2. compute_roc_after_rfe.py 20-Feature Set (Nov 21 RFE)

**Generation Date**: 2025-11-21 at 10:35 AM
**Script**: `scripts/run_rfe_with_age_forced.py`
**Data File for SELECTION**: `features_merged.pkl` (OLD file, 148 infants)
**Data File for EVALUATION**: `features_merged_20251121_091511.pkl` (NEW file, 146 infants)
**Log File**: `rfe_20_feature_20251121.log` ← KEY EVIDENCE
**Output Files**: Features copied to `compute_roc_after_rfe.py` lines 213-234

**Performance** (10-fold CV with 1/(1+|x|) transformation):
- AUC: 0.9018
- Sensitivity: 92.1%
- Specificity: 77.5%
- Youden's J: 0.6964

**Features** (ranked by RF importance from Nov 21 RFE):
```
 1. Elbow_IQR_acc_angle    (0.1017)
 2. Hip_stdev_angle        (0.0593)
 3. Shoulder_mean_angle    (0.0470)
 4. Ankle_IQRvelx          (0.0422)
 5. Ankle_IQRx             (0.0401)
 6. Wrist_mediany          (0.0353)
 7. Ankle_IQRaccy          (0.0334)
 8. Ankle_medianvelx       (0.0325)
 9. Hip_entropy_angle      (0.0320)
10. Ankle_lrCorr_x         (0.0266)
11. Wrist_IQRx             (0.0266)
12. Ankle_meanent          (0.0265)
13. Elbow_IQR_vel_angle    (0.0257)
14. Shoulder_IQR_acc_angle (0.0208)
15. Hip_IQR_vel_angle      (0.0207)
16. Ankle_medianvely       (0.0204)
17. Ankle_IQRvely          (0.0202)
18. Elbow_median_vel_angle (0.0196)
19. Wrist_IQRaccx          (0.0190)
20. age_in_weeks           [FORCED]
```

**Provenance**: ✓ VERIFIED via `rfe_20_feature_20251121.log`

**Key Insight**: Features were SELECTED on old data (`features_merged.pkl`) but
EVALUATED on new data (`features_merged_20251121_091511.pkl`). This cross-dataset
generalization may explain the strong performance (AUC 0.902).

---

### 3. ITERATION_DOC 20-Feature Set (docs/ITERATION_FORCED_AGE_RFE.md)

**Generation Date**: Unknown (documented prior to Dec 2025)
**Script**: Referenced `run_rfe_with_age_forced.py`
**Data File**: Unknown - possibly `features_merged.pkl` (old file)
**Output Files**: `data/pkl/bd_final20_forced_age.pkl` (may have been overwritten)

**Performance**:
- AUC: 0.825
- Sensitivity: 78.6%
- Specificity: 80.0%

**Features**:
```
Ankle_IQRaccx, Ankle_IQRaccy, Ankle_IQRvelx, Ankle_IQRvely, Ankle_IQRx,
Ankle_meanent, Ankle_medianvelx, Ankle_mediany, Ankle_stdent,
Elbow_IQR_acc_angle,
Hip_mean_angle, Hip_stdev_angle,
Knee_IQR_acc_angle, Knee_IQR_vel_angle, Knee_lrCorr_angle,
Shoulder_mean_angle, Shoulder_stdev_angle,
Wrist_IQRx, Wrist_mediany,
age_in_weeks
```

**Provenance**: UNCLEAR - May have used deprecated data file

---

## Feature Set Comparison Matrix

### Overlap Analysis

| Comparison | Common Features | Percentage |
|------------|-----------------|------------|
| NEW RFE vs compute_roc | 13/20 | 65% |
| NEW RFE vs ITERATION_DOC | 10/20 | 50% |
| compute_roc vs ITERATION_DOC | 10/20 | 50% |
| **ALL THREE** | **6/20** | **30%** |

### Universally Stable Features (in all 3 sets)
1. `Ankle_IQRaccy`
2. `Ankle_IQRx`
3. `Ankle_meanent`
4. `Ankle_medianvelx`
5. `Hip_stdev_angle`
6. `age_in_weeks`

### Features by Body Part Across Sets

| Body Part | NEW RFE | compute_roc | ITERATION_DOC |
|-----------|---------|-------------|---------------|
| Ankle | 8 | 8 | 9 |
| Hip | 3 | 3 | 2 |
| Shoulder | 4 | 2 | 2 |
| Elbow | 2 | 3 | 1 |
| Wrist | 2 | 3 | 2 |
| Knee | 1 | 0 | 3 |
| Age | 1 | 1 | 1 |

---

## Key Differences Explained

### Why Different Feature Sets?

1. **Different Data Files**:
   - Nov 21 RFE (compute_roc): Selected on `features_merged.pkl` (OLD)
   - Dec 3 RFE (NEW RFE): Selected on `features_merged_20251121_091511.pkl` (NEW)
   - Only 104 infants overlap between files

2. **RFE Stochasticity**:
   - Enhanced Adaptive RFE uses 50 random trials
   - Different random states produce different feature rankings
   - RF importance ranking varies between runs

3. **Cross-Dataset Generalization** (Key Finding):
   - Nov 21 RFE features were SELECTED on old data
   - But EVALUATED on new data in compute_roc_after_rfe.py
   - This may explain superior AUC (0.902) - features that generalize across datasets

### Performance Differences

| Set | AUC | Evaluation Method | Transform | Selection Data |
|-----|-----|-------------------|-----------|----------------|
| Dec 3 RFE | 0.830 | Single split | None | NEW |
| Dec 3 RFE | 0.799 | 10-fold CV | None | NEW |
| Dec 3 RFE | 0.871 | 10-fold CV | 1/(1+\|x\|) | NEW |
| **Nov 21 RFE** | **0.902** | **10-fold CV** | **1/(1+\|x\|)** | **OLD** |
| ITERATION_DOC | 0.825 | Single split | None | Unknown |

**Key Finding**: The Nov 21 RFE features (now verified via log file) achieve the best AUC (0.902).
The ~3% performance gap vs Dec 3 RFE may be due to:
1. Cross-dataset generalization (selected on OLD, evaluated on NEW)
2. RFE stochasticity producing different feature rankings
3. Different RF importance orderings between runs

**Note**: The higher AUC for compute_roc may be due to:
1. 10-fold CV (more robust estimate)
2. 1/(1+|x|) transformation on 11 velocity/acceleration features
3. Different feature set selection

---

## Recommendations

### Immediate Actions

1. ~~Run NEW RFE features through compute_roc with transformation~~ ✓ DONE (Dec 3)
2. ~~Document the origin of compute_roc features~~ ✓ DONE - Found in `rfe_20_feature_20251121.log`
3. **Update ITERATION_FORCED_AGE_RFE.md** with correct provenance information

### Which Feature Set to Use?

| Priority | Recommendation |
|----------|----------------|
| **Best Performance** | Nov 21 RFE features (AUC 0.902) - use `compute_roc_after_rfe.py` |
| **Matched Selection/Eval Data** | Dec 3 RFE features (AUC 0.871) - use `compute_roc_rfe_dec2025.py` |

**Note**: Both feature sets now have verified provenance. The Nov 21 features perform better
but were selected on a different (older) dataset than used for evaluation.

### Going Forward

1. **Always record**:
   - Exact data file used (with timestamp)
   - Script that generated features
   - Date of generation
   - Random seed if applicable

2. **Use consistent naming**:
   - Include data file date in output names (e.g., `bd_20feat_20251121.pkl`)

3. **Archive intermediate results**:
   - Don't overwrite previous model files
   - Keep versioned copies

---

## Scripts Reference

| Script | Purpose | Data File Used |
|--------|---------|----------------|
| `run_rfe_with_age_forced.py` | Generate 20-feature set with forced age | Configurable (line 29) |
| `compute_roc_after_rfe.py` | Evaluate feature sets with 10-fold CV | `features_merged_20251121_091511.pkl` |
| `compute_roc_rfe_dec2025.py` | Evaluate Dec 3 RFE features with 10-fold CV | `features_merged_20251121_091511.pkl` |
| `monitor_progress.py` | Monitor RFE progress | N/A |

## Key Evidence Files

| File | Purpose | Contents |
|------|---------|----------|
| `rfe_20_feature_20251121.log` | Nov 21 RFE run log | Exact features and importance scores |
| `data/json/final_20_features_forced_age.json` | Dec 3 RFE output | Feature list and metrics |

---

**Last Updated**: 2025-12-03
**Perry Mason Investigation**: Complete - Nov 21 RFE origin verified via log file
