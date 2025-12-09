# Workflows

**Common analysis patterns and workflows for  Bayesian surprise analysis**

This document provides step-by-step workflows for common analysis tasks in the early-markers project.

## Table of Contents

1. [Notebook Workflows](#notebook-workflows)
2. [Basic Analysis Workflow](#basic-analysis-workflow)
3. [Feature Selection Workflow](#feature-selection-workflow)
4. [Model Comparison Workflow](#model-comparison-workflow)
5. [Synthetic Data Validation](#synthetic-data-validation)
6. [BAM Sample Size Planning](#bam-sample-size-planning)
7. [Excel Report Generation](#excel-report-generation)
8. [ROC Analysis and Threshold Selection](#roc-analysis-and-threshold-selection)

---

## Notebook Workflows

**Goal**: Execute end-to-end analysis using Jupyter notebooks.

### Primary Analysis: Real Data Feature Selection

**Notebook**: `01_rfe_real.ipynb`

**Purpose**: Complete feature selection and validation on real infant movement data.

**Steps**:
1. Start Jupyter: `poetry run jupyter notebook`
2. Open `early_markers/cribsy/notebooks/01_rfe_real.ipynb`
3. Run all cells (Kernel → Restart & Run All)
4. Wait ~20-30 minutes for completion

**Outputs**:
- `bd_real.pkl`: Serialized BayesianData with all results
- `real_*.xlsx`: Excel reports with metrics
- Selected features printed in final cell

**Runtime**: ~20-30 minutes (50 RFE trials × 15 CV folds × 200 trees)

### Fast Development Iteration

**Notebook**: `01_rfe_real_fast.ipynb`

**Purpose**: Rapid prototyping with reduced computational requirements.

**Differences from standard**:
- 10 trials instead of 50 (5x faster)
- 100 estimators instead of 200 (2x faster)  
- Combined: 10x speedup

**Runtime**: ~5-10 minutes

**Use cases**:
- Testing code changes
- Experimenting with parameters
- Quick validation

**Note**: Results may differ slightly from production run. Always re-run with standard parameters for final analysis.

### Synthetic Data Workflow

**Step 1: Generate Synthetic Data**

**Notebook**: `02_synthetic_sdv.ipynb`

**Purpose**: Create statistically similar synthetic datasets using SDV library.

**Process**:
1. Load real data as reference
2. Configure SDV synthesizer (GaussianCopula or CTGAN)
3. Fit synthesizer to real data distributions
4. Generate N synthetic infants
5. Export to `.ipc` format (Apache Arrow)

**Outputs**:
- `synth_sdv_{N}_long.ipc`: Synthetic data in long format
- `sdv_metadata.json`: Schema and data types
- Quality reports (Data Validity, Structure scores)

**Runtime**: ~2-5 minutes

**Step 2: Validate with Synthetic Data**

**Notebook**: `03_rfe_synthetic.ipynb`

**Purpose**: Verify feature selection stability on synthetic data.

**Process**:
1. Load synthetic `.ipc` file
2. Sample TRAIN_N and TEST_N infants
3. Run identical RFE pipeline as 01_rfe_real
4. Compare selected features with real data results

**Expected Results**:
- 60-80% feature overlap (good)
- Similar AUC values (±0.05)
- Faster convergence due to cleaner synthetic data

**Runtime**: ~1.5 minutes per RFE iteration

### BAM Sample Size Planning Workflow

**Notebook**: `04_bam_sample_size.ipynb`

**Purpose**: Estimate required sample size for future validation studies.

**Scenario**: You have pilot data and want to plan a definitive study.

**Process**:
```python
from early_markers.cribsy.common.bam_unified import BAMEstimator

# Initialize estimator
estimator = BAMEstimator(seed=20250313, verbose=True)

# Pilot study results
pilot_se = (18, 20)  # 18 TPs out of 20 positive cases
pilot_sp = (35, 40)  # 35 TNs out of 40 negative cases

# Estimate sample size for ±0.10 precision
result = estimator.estimate_joint(
    pilot_se=pilot_se,
    pilot_sp=pilot_sp,
    target_width=0.10,      # Want CI width ≤ 0.10
    ci=0.95,                # 95% credible interval
    target_assurance=0.80,  # 80% probability of achieving target
    prevalence_prior=(8, 32)  # ~20% prevalence
)

print(f"Required total N: {result.n}")
print(f"Positive cases: {result.n_pos}")
print(f"Negative cases: {result.n_neg}")
```

**Outputs**:
- Sample size estimate (total N)
- Breakdown by case status
- Achieved assurance probability

**Use Cases**:
- Planning validation studies
- Grant applications
- Power analysis

---

## Basic Analysis Workflow

**Goal**: Run a complete Bayesian surprise analysis from data loading to metrics.

### Step 1: Initialize BayesianData

```python
from early_markers.cribsy.common.bayes import BayesianData

# Load real data (default)
bd = BayesianData()

# Or load specific file
bd = BayesianData(base_file="features_merged.pkl")

# Or load synthetic data
bd = BayesianData(base_file="synthetic_features.ipc")
```

### Step 2: Inspect Data

```python
# Check data dimensions
print(f"Total samples: {len(bd.base)}")
print(f"Training samples: {bd.base_train.select('infant').unique().shape[0]}")
print(f"Test samples: {bd.base_test.select('infant').unique().shape[0]}")

# View available features
print(f"Features: {len(bd.base_features)}")
print(bd.base_features[:5])  # First 5 features
```

### Step 3: Run Enhanced Adaptive RFE (Recommended)

```python
# This will take 30-60 minutes with default parameters
# Note: Current notebooks use legacy method name run_adaptive_rfe()
# Target API: run_rfe_on_base()
features = bd.run_adaptive_rfe(
    model_prefix='analysis_1',
    features=bd.base_features,
    tot_k=len(bd.base_features)
)

print(f"Selected {len(features)} features")

# Alternative: Use target API (if migrated)
# features = bd.run_rfe_on_base(
#     rfe_name='rfe_analysis_1',
#     features_to_keep_pct=0.9
# )
```

### Step 4: Compute Surprise Scores

```python
# Compute Bayesian surprise using selected features
bd.run_surprise_with_features(
    model_prefix='analysis_1',
    features=features,
    overwrite=True
)
```

### Step 5: Compute ROC Metrics

```python
# Get model name
model_name = f'analysis_1_k_{len(features)}'

# Compute comprehensive metrics
# Note: Legacy method name, target API: run_metrics_from_surprise()
bd.compute_roc_metrics(
    model_prefix='analysis_1',
    feature_size=len(features)
)
```

### Step 5: Examine Results

```python
# Get metrics result
result = bd.metrics(model_name)

print(f"AUC: {result.auc:.3f}")
print(f"Youden's J: {result.youdens_j:.3f}")
print(f"Optimal threshold: {result.threshold_j:.2f}")

# Access detailed metrics
metrics_df = result.metrics
optimal = metrics_df.filter(pl.col('j') == result.youdens_j)

print(f"Sensitivity: {optimal['sens'][0]:.3f}")
print(f"Specificity: {optimal['spec'][0]:.3f}")
print(f"PPV: {optimal['ppv'][0]:.3f}")
print(f"NPV: {optimal['npv'][0]:.3f}")
```

### Step 6: Save State (Optional)

```python
# Save for later
import pickle
with open('analysis_state.pkl', 'wb') as f:
    pickle.dump(bd, f)

# Load later
with open('analysis_state.pkl', 'rb') as f:
    bd = pickle.load(f)
```

**Expected Time**: 35-65 minutes (mostly RFE)  
**Output**: ROC metrics, selected features, surprise scores

---

## Feature Selection Workflow

**Goal**: Compare different feature selection methods and parameters.

### Method 1: Enhanced Adaptive RFE (Recommended)

```python
bd = BayesianData()

# Run with default parameters
features = bd.run_adaptive_rfe(
    model_prefix='adaptive_default',
    features=bd.base_features,
    tot_k=len(bd.base_features)
)

print(f"Selected {len(features)} significant features")
```

**Advantages**:
- Statistical significance testing (binomial test)
- Noise injection for robustness
- Parallel trials (50 by default)
- Cross-validation (15 folds)

**Time**: 30-60 minutes

### Method 2: Simple RFE (Legacy, Faster)

```python
# Much faster but less robust
features = bd.run_multi_trial_rfe(
    model_prefix='simple',
    feature_size=45  # Target number
)

print(f"Selected {len(features)} features")
```

**Advantages**:
- Fast (minutes instead of hours)
- Deterministic results
- Simple to understand

**Time**: 2-5 minutes

### Method 3: Manual Feature Selection

```python
# Use domain knowledge to select features
manual_features = [
    'Ankle_L_position_entropy',
    'Wrist_R_velocity_mean',
    'Hip_L_acceleration_std',
    # ... add 42 more features
]

# Compute surprise with manual selection
bd.run_surprise_with_features('manual', manual_features)
```

### Comparing Methods

```python
# Run all three
adaptive_features = bd.run_adaptive_rfe('adaptive', bd.base_features, len(bd.base_features))
simple_features = bd.run_multi_trial_rfe('simple', feature_size=45)
bd.run_surprise_with_features('manual', manual_features)

# Compute metrics for all
for prefix, features in [('adaptive', adaptive_features), 
                          ('simple', simple_features),
                          ('manual', manual_features)]:
    bd.compute_roc_metrics(prefix, len(features))
    result = bd.metrics(f'{prefix}_k_{len(features)}')
    print(f"{prefix}: AUC={result.auc:.3f}, J={result.youdens_j:.3f}")
```

**Output**: Comparison of AUC and Youden's J across methods

---

## Model Comparison Workflow

**Goal**: Test multiple models with different feature counts or subsets.

### Workflow: Feature Count Sweep

```python
bd = BayesianData()

# Test different target feature counts
target_counts = [30, 40, 50, 60]
results = {}

for k in target_counts:
    print(f"\nTesting with {k} features...")
    
    # Run RFE
    features = bd.run_multi_trial_rfe(f'sweep_{k}', feature_size=k)
    
    # Compute metrics
    bd.compute_roc_metrics(f'sweep_{k}', feature_size=k)
    
    # Store results
    model_name = f'sweep_{k}_k_{k}'
    result = bd.metrics(model_name)
    results[k] = {
        'auc': result.auc,
        'youdens_j': result.youdens_j,
        'sensitivity': result.metrics.filter(pl.col('j') == result.youdens_j)['sens'][0],
        'specificity': result.metrics.filter(pl.col('j') == result.youdens_j)['spec'][0]
    }

# Compare results
import pandas as pd
comparison = pd.DataFrame(results).T
print("\nComparison:")
print(comparison)

# Find best model
best_k = comparison['auc'].idxmax()
print(f"\nBest model: {best_k} features (AUC={comparison.loc[best_k, 'auc']:.3f})")
```

### Workflow: Body Part Analysis

```python
# Test models focused on specific body parts
body_parts = ['Ankle', 'Wrist', 'Hip', 'Shoulder']

for part in body_parts:
    # Filter features for this body part
    part_features = [f for f in bd.base_features if part in f]
    print(f"\n{part}: {len(part_features)} features")
    
    # Run analysis
    bd.run_surprise_with_features(f'part_{part.lower()}', part_features)
    bd.compute_roc_metrics(f'part_{part.lower()}', len(part_features))
    
    # Report
    model_name = f'part_{part.lower()}_k_{len(part_features)}'
    result = bd.metrics(model_name)
    print(f"AUC: {result.auc:.3f}, J: {result.youdens_j:.3f}")
```

**Output**: Performance comparison across body parts

---

## Synthetic Data Validation

**Goal**: Validate analysis pipeline with synthetic data where ground truth is known.

### Step 1: Generate Synthetic Data

```python
# Assume synthetic data already generated with known properties
bd_synth = BayesianData(base_file="synthetic_known.ipc")

print(f"Synthetic samples: {bd_synth.base.select('infant').unique().shape[0]}")
```

### Step 2: Run Analysis Pipeline

```python
# Run same pipeline as real data
features = bd_synth.run_adaptive_rfe(
    model_prefix='synthetic_test',
    features=bd_synth.base_features,
    tot_k=len(bd_synth.base_features)
)

bd_synth.compute_roc_metrics('synthetic_test', len(features))
```

### Step 3: Compare to Expected Results

```python
result = bd_synth.metrics(f'synthetic_test_k_{len(features)}')

# Known ground truth for synthetic data
expected_auc = 0.85
expected_features = 42

print(f"Expected AUC: {expected_auc:.3f}")
print(f"Actual AUC: {result.auc:.3f}")
print(f"Difference: {abs(result.auc - expected_auc):.3f}")

print(f"\nExpected features: {expected_features}")
print(f"Actual features: {len(features)}")
print(f"Difference: {abs(len(features) - expected_features)}")
```

### Step 4: Validate Sample Size Estimates

```python
# Compare actual vs predicted sample sizes
print(f"Actual test N: {result.test_n}")
print(f"Estimated min N: {result.rough_n_min}")
print(f"Estimated max N: {result.rough_n_max}")

if result.test_n >= result.rough_n_min:
    print("✓ Sample size adequate")
else:
    print("✗ Sample size below minimum estimate")
```

**Output**: Validation that pipeline works correctly on synthetic data

---

## Sample Size Planning

**Goal**: Determine minimum sample size needed for target precision.

### Workflow 1: BAM (Bayesian Assurance Method)

```python
from early_markers.cribsy.bam import informed_bam_performance

# Target precision
target_half_width = 0.05  # ±0.05 for 95% CI

# Pilot estimates
pilot_sensitivity = 0.85
pilot_specificity = 0.90
prevalence = 0.15

# Calculate required N
result = informed_bam_performance(
    pilot_se=pilot_sensitivity,
    pilot_sp=pilot_specificity,
    prevalence_prior=(15, 85),  # Beta prior
    hdi_width=target_half_width
)

print(f"Required N for sensitivity: {result['n_sens_required']}")
print(f"Required N for specificity: {result['n_spec_required']}")
print(f"Maximum N needed: {max(result['n_sens_required'], result['n_spec_required'])}")
```

### Workflow 2: Frequentist Sample Size

```python
from early_markers.cribsy.common.metrics_n import RocMetricSampleSize
from early_markers.cribsy.common.enums import MetricType

calc = RocMetricSampleSize()

# Estimate N for sensitivity
n_sens = calc.estimate_n(
    metric_type=MetricType.SENSITIVITY,
    sens=0.85,
    prev=0.15,
    ci=0.10  # CI width (±0.05)
)

print(f"Required total N: {n_sens:.0f}")
print(f"Required positive cases: {int(n_sens * 0.15)}")
print(f"Required negative cases: {int(n_sens * 0.85)}")
```

### Workflow 3: Empirical Validation

```python
# Use existing analysis to check precision
bd = BayesianData()
# ... run analysis ...

result = bd.metrics('model_k_45')
optimal = result.metrics.filter(pl.col('j') == result.youdens_j)

# Extract CI width
sens = optimal['sens'][0]
sens_ci = optimal['sens_ci'][0]
sens_lb, sens_ub = sens_ci

actual_half_width = max(sens - sens_lb, sens_ub - sens)

print(f"Sensitivity: {sens:.3f}")
print(f"CI: [{sens_lb:.3f}, {sens_ub:.3f}]")
print(f"Half-width: {actual_half_width:.3f}")

if actual_half_width <= 0.05:
    print("✓ Target precision achieved")
else:
    print(f"✗ Need larger sample (current half-width: {actual_half_width:.3f})")
```

**Output**: Sample size requirements and validation

---

## Excel Report Generation

**Goal**: Generate formatted Excel reports with metrics and visualizations.

### Workflow: Basic Report

```python
bd = BayesianData()
# ... run analysis ...

# Generate report for a specific metrics result
bd.write_excel_report(tag='analysis_1')
```

**Output**: Excel file with:
- Summary sheet with top models
- Individual sheets per model with detailed metrics
- Formatted tables with conditional formatting
- Confidence intervals

### Workflow: Custom Excel Export

```python
import xlsxwriter

# Get metrics
result = bd.metrics('model_k_45')
metrics_df = result.metrics

# Create custom workbook
wb = xlsxwriter.Workbook('custom_report.xlsx')
ws = wb.add_worksheet('Metrics')

# Write metrics with formatting
from early_markers.cribsy.common.xlsx import set_workbook_formats

formats = set_workbook_formats(wb)

# Write header
ws.write(0, 0, 'Threshold', formats['heading2'])
ws.write(0, 1, 'Sensitivity', formats['heading2'])
ws.write(0, 2, 'Specificity', formats['heading2'])

# Write data
for i, row in enumerate(metrics_df.to_dicts(), start=1):
    ws.write(i, 0, row['threshold'], formats['number_2dp'])
    ws.write(i, 1, row['sens'], formats['percent_1dp'])
    ws.write(i, 2, row['spec'], formats['percent_1dp'])

wb.close()
print("Custom report saved: custom_report.xlsx")
```

**Output**: Customized Excel report

---

## ROC Analysis and Threshold Selection

**Goal**: Analyze ROC curve and select optimal operating threshold.

### Workflow: Find Optimal Threshold

```python
bd = BayesianData()
# ... run analysis ...

result = bd.metrics('model_k_45')

# Youden's J optimal threshold (default)
print(f"Youden's J threshold: {result.threshold_j:.2f}")
print(f"Youden's J value: {result.youdens_j:.3f}")

# Find metrics at optimal threshold
optimal = result.metrics.filter(pl.col('j') == result.youdens_j)
print(f"Sensitivity: {optimal['sens'][0]:.3f}")
print(f"Specificity: {optimal['spec'][0]:.3f}")
```

### Workflow: Custom Threshold Selection

```python
# Find threshold for target sensitivity
target_sens = 0.90

# Filter to thresholds with sensitivity ≥ target
candidates = result.metrics.filter(pl.col('sens') >= target_sens)

# Among candidates, maximize specificity
best = candidates.sort('spec', descending=True).head(1)

print(f"Threshold: {best['threshold'][0]:.2f}")
print(f"Sensitivity: {best['sens'][0]:.3f}")
print(f"Specificity: {best['spec'][0]:.3f}")
print(f"PPV: {best['ppv'][0]:.3f}")
```

### Workflow: Threshold Sensitivity Analysis

```python
# Analyze performance across threshold range
thresholds_to_test = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

print("Threshold | Sens  | Spec  | PPV   | NPV")
print("-" * 50)

for thr in thresholds_to_test:
    row = result.metrics.filter(
        (pl.col('threshold') >= thr - 0.01) & 
        (pl.col('threshold') <= thr + 0.01)
    )
    
    if row.height > 0:
        r = row.head(1)
        print(f"{thr:8.2f}  | {r['sens'][0]:.3f} | {r['spec'][0]:.3f} | {r['ppv'][0]:.3f} | {r['npv'][0]:.3f}")
```

### Workflow: ROC Curve Visualization

```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Display saved ROC curve
img = imread(result.plot_file)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.title(f'ROC Curve - {result.model_name}')
plt.tight_layout()
plt.show()

# Or create custom plot
metrics_df = result.metrics
plt.figure(figsize=(8, 8))
plt.plot(metrics_df['fpr'], metrics_df['tpr'], 'b-', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.scatter([metrics_df.filter(pl.col('j') == result.youdens_j)['fpr'][0]],
           [metrics_df.filter(pl.col('j') == result.youdens_j)['tpr'][0]],
           c='red', s=100, label=f"Optimal (J={result.youdens_j:.3f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {result.auc:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Output**: ROC visualization and threshold analysis

---

## Tips and Best Practices

### Performance Optimization

1. **Use synthetic data for testing**
   - Much faster to iterate
   - Validate pipeline before running on real data

2. **Save intermediate states**
   ```python
   # After expensive RFE
   import pickle
   with open('after_rfe.pkl', 'wb') as f:
       pickle.dump(bd, f)
   ```

3. **Parallelize when possible**
   - RFE already uses parallelization (N_JOBS=12 by default)
   - Can run multiple analyses in parallel on different data

### Reproducibility

1. **Always check random seed**
   ```python
   from early_markers.cribsy.common.constants import RAND_STATE
   print(f"Random seed: {RAND_STATE}")
   ```

2. **Document analysis parameters**
   ```python
   analysis_log = {
       'date': '2025-01-20',
       'rfe_method': 'adaptive',
       'n_trials': 50,
       'alpha': 0.05,
       'features_selected': len(features),
       'auc': result.auc
   }
   ```

3. **Version control your notebooks**
   - Commit after each successful analysis
   - Tag important results

### Troubleshooting

**Problem**: RFE takes too long  
**Solution**: Use `run_multi_trial_rfe()` instead of `run_adaptive_rfe()`, or reduce `RFE_N_TRIALS`

**Problem**: Low AUC values  
**Solution**: Check for data quality issues, try different feature subsets, ensure proper train/test split

**Problem**: Wide confidence intervals  
**Solution**: Need larger sample size - use BAM or sample size calculator to estimate requirements

---

## See Also

- [API.md](API.md) - Complete API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [DATA_FORMATS.md](DATA_FORMATS.md) - Data specifications
- [CONFIGURATION.md](CONFIGURATION.md) - Parameter tuning
