# Data Formats

**Data specifications and format reference for early-markers**

This document describes all data formats used in the early-markers Bayesian surprise analysis pipeline.

## Table of Contents

1. [Input Data Formats](#input-data-formats)
2. [Internal Data Formats](#internal-data-formats)
3. [Output Data Formats](#output-data-formats)
4. [Feature Specifications](#feature-specifications)
5. [Metadata Columns](#metadata-columns)
6. [Data Transformations](#data-transformations)

---

## Input Data Formats

### Raw Data Format (Pickle - `.pkl`)

**File**: `features_merged.pkl` (Pandas DataFrame)

**Location**: `/Volumes/secure/code/early-markers/early_markers/legacy/EmmacpMetrics.zip`

**Columns**:
- `infant` (str): Unique infant identifier (e.g., 'clin_01_3')
- `category` (int): Original category (0 or 1, transformed later)
- `risk` (int): Original risk score (transformed to binary)
- `age_in_weeks` (float): Infant age in weeks
- `part` (str): Body part name (e.g., 'Ankle', 'Wrist', 'Hip')
- `feature_name` (str): Feature type (e.g., 'position_entropy', 'velocity_mean')
- `Value` (float): Feature measurement value

**Example**:
```
   infant  category  risk  age_in_weeks   part         feature_name    Value
0  clin_01_3    1      1      12.5      Ankle_L   position_entropy   0.1234
1  clin_01_3    1      1      12.5      Wrist_R   velocity_mean      0.5678
2  clin_02_1    0      0      14.2      Hip_L     acceleration_std   0.9012
```

**Size**: ~70,000 rows (1,250 infants × 56 features)

---

### Synthetic Data Format (IPC - `.ipc`)

**File**: `*.ipc` (Apache Arrow IPC / Feather format)

**Location**: `/Volumes/secure/data/early_markers/cribsy/ipc/`

**Technology**: 
- Apache Arrow IPC format (Feather v2)
- Binary columnar storage format
- Zero-copy reads with Polars
- Optimized for large datasets

**Columns**: Same as processed long format (see below)
- `infant` (str): Infant identifier
- `category` (int): 1=training, 2=testing
- `risk` (int): 0=normal, 1=at-risk
- `feature` (str): Combined feature name (e.g., 'Ankle_L_position_entropy')
- `value` (float): Feature measurement

**Purpose**: 
- Fast loading with Polars (10-100x faster than pickle)
- Pre-processed long format for immediate use
- Used exclusively for synthetic/augmented datasets
- Enables efficient filtering and aggregation

**Advantages over Pickle**:
- Type-safe: Schema is embedded in file
- Language-agnostic: Can be read by Python, R, Julia, etc.
- Memory-efficient: No deserialization overhead
- Faster: Zero-copy reads, no Python object creation

**Example Files**:
- `synth_sdv_1000_long.ipc`: 1,000 synthetic infants generated with SDV
- `synth_gretel_5000_long.ipc`: 5,000 synthetic infants from Gretel.ai

**Generation**: Created by synthetic data generation notebooks (02_synthetic_sdv.ipynb)

**Loading with Polars**:
```python
import polars as pl
from early_markers.cribsy.common.constants import IPC_DIR

# Read synthetic data
df = pl.read_ipc(IPC_DIR / "synth_sdv_1000_long.ipc")

# Filter and pivot in one pass (fast!)
df_wide = df.filter(
    pl.col("category") == 1
).pivot(
    index=["infant", "category", "risk"],
    on="feature",
    values="value"
)
```

**Note**: Real data still uses `.pkl` format for historical reasons. Consider migrating to `.ipc` for production.

---

## Internal Data Formats

### Long Format (Base Data)

**DataFrame**: `_base`, `_base_train`, `_base_test`

**Format**: Polars DataFrame

**Columns**:
- `infant` (str): Unique identifier
- `category` (int): 1=training, 2=testing
- `risk` (int): 0=normal, 1=at-risk (binary)
- `feature` (str): Feature name (format: "{part}_{feature_type}")
- `value` (float): Measurement value

**Example**:
```
┌───────────┬──────────┬──────┬─────────────────────────────┬────────┐
│ infant    │ category │ risk │ feature                     │ value  │
│ str       │ i64      │ i64  │ str                         │ f64    │
╞═══════════╪══════════╪══════╪═════════════════════════════╪════════╡
│ clin_01_3 │ 1        │ 1    │ Ankle_L_position_entropy    │ 0.1234 │
│ clin_01_3 │ 1        │ 1    │ Wrist_R_velocity_mean       │ 0.5678 │
│ clin_02_1 │ 1        │ 0    │ Ankle_L_position_entropy    │ 0.0987 │
└───────────┴──────────┴──────┴─────────────────────────────┴────────┘
```

**Shape**: (N_infants × N_features, 5)
- Training: ~56,000 rows (1,000 infants × 56 features)
- Testing: ~14,000 rows (250 infants × 56 features)

---

### Wide Format (Model Data)

**DataFrames**: Returned by `base_wide`, `base_train_wide`, `base_test_wide` properties

**Format**: Polars DataFrame (pivoted from long format)

**Columns**:
- `infant` (str): Unique identifier
- `category` (int): 1 or 2
- `risk` (int): 0 or 1
- Feature columns (56 features, each as f64)
  - `Ankle_L_position_entropy`
  - `Ankle_L_velocity_mean`
  - ... (53 more features)
  - `age_in_weeks`

**Example**:
```
┌───────────┬──────────┬──────┬─────────────┬─────────────┬─────┬──────────────┐
│ infant    │ category │ risk │ Ankle_L_... │ Wrist_R_... │ ... │ age_in_weeks │
│ str       │ i64      │ i64  │ f64         │ f64         │     │ f64          │
╞═══════════╪══════════╪══════╪═════════════╪═════════════╪═════╪══════════════╡
│ clin_01_3 │ 1        │ 1    │ 0.1234      │ 0.5678      │ ... │ 12.5         │
│ clin_02_1 │ 1        │ 0    │ 0.0987      │ 0.3456      │ ... │ 14.2         │
└───────────┴──────────┴──────┴─────────────┴─────────────┴─────┴──────────────┘
```

**Shape**: (N_infants, 59)
- 3 metadata columns + 56 feature columns

**Use Cases**:
- RFE input (requires wide format)
- Statistical analysis
- Correlation matrices

---

### Reference Statistics Format

**DataFrame**: `BayesianFrames.stats`

**Format**: Polars DataFrame

**Columns**:
- `feature` (str): Feature name
- `mean_ref` (f64): Mean from normative training data (risk=0)
- `sd_ref` (f64): Standard deviation (population, ddof=0)
- `var_ref` (f64): Variance (population, ddof=0)

**Example**:
```
┌─────────────────────────────┬──────────┬─────────┬──────────┐
│ feature                     │ mean_ref │ sd_ref  │ var_ref  │
│ str                         │ f64      │ f64     │ f64      │
╞═════════════════════════════╪══════════╪═════════╪══════════╡
│ Ankle_L_position_entropy    │ 0.1234   │ 0.0456  │ 0.0021   │
│ Wrist_R_velocity_mean       │ 0.5678   │ 0.1234  │ 0.0152   │
│ Hip_L_acceleration_std      │ 0.9012   │ 0.2345  │ 0.0550   │
└─────────────────────────────┴──────────┴─────────┴──────────┘
```

**Computation**: 
- Computed from training data where `risk=0` (normative infants only)
- Serves as reference distribution for Bayesian surprise

---

### Surprise Scores Format

**DataFrames**: `BayesianFrames.train_surprise`, `BayesianFrames.test_surprise`

**Format**: Polars DataFrame

**Columns**:
- `infant` (str): Unique identifier
- `risk` (int): 0=normal, 1=at-risk
- `minus_log_pfeature` (f64): Sum of -log P across features (raw surprise)
- `z` (f64): Standardized surprise z-score
- `p` (f64): Two-tailed p-value (rounded to 3 decimals)

**Example**:
```
┌───────────┬──────┬───────────────────┬────────┬───────┐
│ infant    │ risk │ minus_log_pfeature│ z      │ p     │
│ str       │ i64  │ f64               │ f64    │ f64   │
╞═══════════╪══════╪═══════════════════╪════════╪═══════╡
│ clin_01_3 │ 1    │ 68.5              │ 1.85   │ 0.064 │
│ clin_02_1 │ 0    │ 52.3              │ -0.23  │ 0.818 │
│ clin_03_2 │ 1    │ 75.2              │ 2.63   │ 0.009 │
└───────────┴──────┴───────────────────┴────────┴───────┘
```

**Interpretation**:
- Higher `z` values indicate more surprising (deviant) patterns
- `p < 0.05` suggests significant deviation from normative distribution
- Normative infants (risk=0) should cluster around z=0

---

### RFE Results Format

**Class**: `BayesianRfeResult`

**Fields**:
- `name` (str): RFE identifier (e.g., 'rfe_90pct')
- `k` (int): Number of features selected
- `features` (list[str]): Selected feature names

**Example**:
```python
BayesianRfeResult(
    name='rfe_trial_1',
    k=45,
    features=[
        'Ankle_L_position_entropy',
        'Wrist_R_velocity_mean',
        'Hip_L_acceleration_std',
        # ... 42 more features
    ]
)
```

**Storage**: Stored in `BayesianData._rfes` dictionary

---

## Output Data Formats

### ROC Metrics Format

**DataFrame**: `BayesianRocResult.metrics`

**Format**: Polars DataFrame

**Columns** (20 total):

**Threshold & Raw Metrics**:
- `threshold` (f64): Surprise z-score threshold
- `roc_thresh` (f64): From sklearn ROC curve
- `tpr` (f64): True positive rate (sensitivity)
- `fpr` (f64): False positive rate
- `sens` (f64): Sensitivity (same as TPR)
- `spec` (f64): Specificity
- `ppv` (f64): Positive predictive value
- `npv` (f64): Negative predictive value
- `acc` (f64): Accuracy
- `f1` (f64): F1 score
- `j` (f64): Youden's J (sens + spec - 1)

**Confidence Intervals** (tuples):
- `sens_ci` (tuple): (lower, upper) for sensitivity
- `spec_ci` (tuple): (lower, upper) for specificity
- `tpr_ci` (tuple): (lower, upper) for TPR
- `fpr_ci` (tuple): (lower, upper) for FPR
- `ppv_ci` (tuple): (lower, upper) for PPV
- `npv_ci` (tuple): (lower, upper) for NPV
- `acc_ci` (tuple): (lower, upper) for accuracy
- `f1_ci` (tuple): (lower, upper) for F1

**Formatted CIs** (strings):
- `sens_w_ci` (str): e.g., "0.824 [0.780, 0.870]"
- `spec_w_ci` (str): e.g., "0.850 [0.810, 0.890]"
- ... (similar for all metrics)

**Sample Size Estimates**:
- `rough_n_min` (int): Minimum recommended total N
- `rough_n_max` (int): Maximum recommended total N

**Example**:
```
┌───────────┬──────┬──────┬──────┬─────────────────┬─────────────────┐
│ threshold │ sens │ spec │ j    │ sens_ci         │ spec_ci         │
│ f64       │ f64  │ f64  │ f64  │ tuple           │ tuple           │
╞═══════════╪══════╪══════╪══════╪═════════════════╪═════════════════╡
│ 1.85      │0.824 │0.850 │0.674 │(0.780, 0.870)   │(0.810, 0.890)   │
│ 2.00      │0.789 │0.880 │0.669 │(0.745, 0.835)   │(0.845, 0.915)   │
└───────────┴──────┴──────┴──────┴─────────────────┴─────────────────┘
```

---

### Confusion Matrix Primitives

**DataFrame**: `BayesianRocResult.primitives`

**Format**: Polars DataFrame

**Columns**:
- `thresh` (f64): Surprise z-score threshold
- `roc_thresh` (f64): From sklearn ROC curve
- `tp` (int): True positives
- `tn` (int): True negatives
- `fp` (int): False positives
- `fn` (int): False negatives
- `n` (int): Total samples (tp+tn+fp+fn)

**Example**:
```
┌───────────┬─────┬─────┬─────┬─────┬─────┐
│ thresh    │ tp  │ tn  │ fp  │ fn  │ n   │
│ f64       │ i64 │ i64 │ i64 │ i64 │ i64 │
╞═══════════╪═════╪═════╪═════╪═════╪═════╡
│ 1.85      │ 103 │ 170 │ 30  │ 22  │ 325 │
│ 2.00      │ 98  │ 176 │ 24  │ 27  │ 325 │
└───────────┴─────┴─────┴─────┴─────┴─────┘
```

---

### Excel Report Format

**Files**: `{tag}_report.xlsx`

**Location**: `/Volumes/secure/data/early_markers/cribsy/xlsx/`

**Worksheets**:

1. **Summary Sheet**:
   - Top N models by AUC
   - Top N models by Youden's J
   - Hyperlinks to detailed sheets
   - Conditional formatting (green for best)

2. **Model Detail Sheets** (one per model):
   - Model name and feature count
   - AUC and Youden's J
   - Metrics at optimal threshold
   - Full threshold sweep table
   - Confidence intervals
   - Sample size estimates

**Formatting**:
- Headers: Bold, colored background
- Numbers: Rounded to appropriate precision
- Percentages: 1 decimal place
- Conditional formatting: Highlights optimal values

---

## Feature Specifications

### Feature Naming Convention

**Format**: `{BodyPart}_{Side}_{FeatureType}`

**Components**:
- **BodyPart**: Ankle, Wrist, Knee, Elbow, Hip, Shoulder, Ear, Eye
- **Side**: L (left) or R (right)
- **FeatureType**: Metric type (see below)

### Feature Types

**Position Features** (8 body parts × 2 sides = 16):
- `*_position_entropy`: Entropy of position distribution
- `*_position_mean`: Mean position
- `*_position_std`: Standard deviation of position

**Velocity Features** (8 body parts × 2 sides = 16):
- `*_velocity_entropy`: Entropy of velocity distribution
- `*_velocity_mean`: Mean velocity magnitude
- `*_velocity_std`: Standard deviation of velocity

**Acceleration Features** (8 body parts × 2 sides = 16):
- `*_acceleration_entropy`: Entropy of acceleration distribution
- `*_acceleration_mean`: Mean acceleration magnitude
- `*_acceleration_std`: Standard deviation of acceleration

**Correlation Features** (selected pairs):
- `*_correlation_xy`: Correlation between x and y coordinates
- `*_correlation_vx_vy`: Correlation between velocity components

**Special Features**:
- `age_in_weeks`: Infant age (added as 57th feature)

### Complete Feature List

Total: **56 features** (defined in `constants.FEATURES`)

**Ankle Features** (6):
- Ankle_L_position_entropy
- Ankle_L_velocity_mean
- Ankle_L_acceleration_std
- Ankle_R_position_entropy
- Ankle_R_velocity_mean
- Ankle_R_acceleration_std

**Wrist Features** (6):
- Wrist_L_position_entropy
- Wrist_L_velocity_mean
- Wrist_L_acceleration_std
- Wrist_R_position_entropy
- Wrist_R_velocity_mean
- Wrist_R_acceleration_std

**Other Body Parts**: Hip, Knee, Elbow, Shoulder, Ear, Eye (similar pattern)

**Plus**: age_in_weeks

---

## Metadata Columns

### Infant Identifiers

**Format**: `{prefix}_{number}_{session}`

**Examples**:
- `clin_01_3`: Clinical study, participant 01, session 3
- `ctrl_42_1`: Control group, participant 42, session 1

**Properties**:
- Unique within dataset
- Consistent across all features for same infant
- Used as primary key for joins

### Category Encoding

**Values**:
- `1`: Training data
  - All normative infants (risk=0)
  - Sampled at-risk infants (random selection)
  - Used for computing reference statistics
  - Used for fitting RFE models

- `2`: Testing data
  - Held-out at-risk infants
  - N_NORM_TO_TEST normative infants (e.g., 20)
  - Used for computing surprise on unseen data
  - Used for ROC analysis

**Transformation**: Applied in `_set_base_dataframes()`

### Risk Encoding

**Original** (`risk_raw`):
- Values: 0, 1, 2, 3, ... (ordinal scale)
- Meaning: Higher = greater risk

**Binary** (`risk`):
- `0`: Normal/normative (risk_raw ≤ 1)
- `1`: At-risk (risk_raw > 1)

**Purpose**: Binary classification for ROC analysis

---

## Data Transformations

### Transformation 1: Raw → Long Format

**Input**: Pandas DataFrame from pickle
**Output**: Polars DataFrame in long format

**Steps**:
1. Filter invalid data (part='umber', infant='clin_100_6')
2. Rename columns: `risk → risk_raw`, `Value → value`
3. Transform risk: `risk = (risk_raw > 1) ? 1 : 0`
4. Transform category: `category = (category==0 || risk==0) ? 1 : 2`
5. Create feature column: `feature = part + "_" + feature_name`
6. Add age_in_weeks as separate feature
7. Sample N_NORM_TO_TEST normative for test set

**Code**: `BayesianData._set_base_dataframes()`

---

### Transformation 2: Long → Wide Format

**Input**: Long format DataFrame
**Output**: Wide format DataFrame

**Operation**: Pivot on `feature` column

**Polars Code**:
```python
wide_df = long_df.pivot(
    index=['infant', 'category', 'risk'],
    on=['feature'],
    values='value'
).with_columns(
    pl.col(pl.Float64).round(4)  # Round to 4 decimals
)
```

**Use**: Required for RFE, correlation analysis, direct feature access

---

### Transformation 3: Compute Reference Statistics

**Input**: Training data (long format, category=1, risk=0)
**Output**: Statistics DataFrame

**Operation**: Group by feature, compute statistics

**Polars Code**:
```python
stats = train_df.filter(
    pl.col('risk') == 0
).group_by('feature').agg(
    pl.col('value').mean().alias('mean_ref'),
    pl.col('value').std(ddof=0).alias('sd_ref'),
    pl.col('value').var(ddof=0).alias('var_ref')
)
```

**Note**: Uses population statistics (ddof=0) for consistency

---

### Transformation 4: Compute Per-Feature Surprise

**Input**: Data joined with reference statistics
**Output**: Data with `minus_log_pfeature` column

**Formula**:
```
minus_log_p = -log P(x | N(μ, σ²))
            = 0.5 × log(2πσ²) + (x - μ)² / (2σ²)
```

**Polars Code**:
```python
df_with_surprise = df.join(stats, on='feature').with_columns(
    minus_log_pfeature = (
        -1 * (0.5 * np.log(2 * np.pi * pl.col('var_ref'))
            + ((pl.col('value') - pl.col('mean_ref'))**2)
            / (2 * pl.col('var_ref')))
    )
)
```

**Interpretation**: Higher values = more surprising for that feature

---

### Transformation 5: Aggregate to Surprise Scores

**Input**: Per-feature surprise scores
**Output**: Per-infant surprise scores

**Operation**: Sum across features, standardize

**Steps**:
1. Group by infant: `S_i = Σ(minus_log_pfeature)`
2. Compute normalization from training: `μ = mean(S_train)`, `σ = std(S_train)`
3. Standardize: `z_i = (S_i - μ) / σ`
4. Compute p-value: `p_i = 2 × SF(|z_i|)` where SF is survival function

**Code**: `BayesianData._set_surprise_data()`

---

### Transformation 6: Compute ROC Metrics

**Input**: Surprise scores with risk labels
**Output**: Metrics at multiple thresholds

**Process**:
1. Extract unique z-score thresholds
2. For each threshold:
   - Apply: `y_pred = (z < threshold) ? 1 : 0`
   - Compute confusion matrix (TP, TN, FP, FN)
   - Calculate metrics: sens, spec, PPV, NPV, acc, F1, J
   - Compute Wilson score 95% CIs
3. Find optimal threshold (max Youden's J)

**Code**: `BayesianData.compute_roc_metrics()`

---

## Data Validation

### Validation Checks

**On Load**:
- ✓ File exists and is readable
- ✓ Expected columns present
- ✓ No duplicate (infant, feature) pairs in long format
- ✓ Feature names match expected list (56 features)

**After Transformation**:
- ✓ Training/test split is disjoint
- ✓ All infants have all features (no missing data)
- ✓ Risk labels are binary (0 or 1)
- ✓ Category labels are valid (1 or 2)

**After Surprise Computation**:
- ✓ Normative infants have z ≈ 0 (mean close to 0)
- ✓ At-risk infants have higher mean z
- ✓ No infinite or NaN values in surprise scores

### Data Quality Filters

**Applied in `_set_base_dataframes()`**:
- Remove: `part == 'umber'` (known bad sensor)
- Remove: `infant == 'clin_100_6'` (known data quality issue)

---

## File Paths

### Input Paths
```python
RAW_DATA = Path("/Volumes/secure/code/early-markers/early_markers/emmacp_metrics/features_merged.pkl")
IPC_DIR = Path("/Volumes/secure/data/early_markers/cribsy/ipc")
```

### Output Paths
```python
PKL_DIR = Path("/Volumes/secure/data/early_markers/cribsy/pkl")
XLSX_DIR = Path("/Volumes/secure/data/early_markers/cribsy/xlsx")
PLOT_DIR = Path("/Volumes/secure/data/early_markers/cribsy/png")
CSV_DIR = Path("/Volumes/secure/data/early_markers/cribsy/csv")
JSON_DIR = Path("/Volumes/secure/data/early_markers/cribsy/json")
HTML_DIR = Path("/Volumes/secure/data/early_markers/cribsy/html")
```

**Note**: Paths are hardcoded in `constants.py` and machine-specific

---

## Data Examples

### Complete Record Example (Long Format)

```python
{
    'infant': 'clin_01_3',
    'category': 1,
    'risk': 1,
    'feature': 'Ankle_L_position_entropy',
    'value': 0.1234
}
```

### Complete Record Example (Wide Format)

```python
{
    'infant': 'clin_01_3',
    'category': 1,
    'risk': 1,
    'Ankle_L_position_entropy': 0.1234,
    'Ankle_L_velocity_mean': 0.5678,
    # ... 54 more features ...
    'age_in_weeks': 12.5
}
```

### Complete Surprise Record

```python
{
    'infant': 'clin_01_3',
    'risk': 1,
    'minus_log_pfeature': 68.5,
    'z': 1.85,
    'p': 0.064
}
```

---

## See Also

- [API.md](API.md) - API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [WORKFLOWS.md](WORKFLOWS.md) - Common patterns
- [CONFIGURATION.md](CONFIGURATION.md) - Parameters
- `constants.py` - Data paths and feature lists
