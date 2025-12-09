# API Reference

This document provides a comprehensive reference for the public API of the early-markers library.

## Table of Contents

1. [BayesianData Class](#bayesiandata-class)
2. [EnhancedAdaptiveRFE Class](#enhancedadaptiverfe-class)
3. [BAM Functions](#bam-functions)
4. [Data Classes](#data-classes)

## BayesianData Class

The main API class for Bayesian surprise analysis.

### Import

```python
from early_markers.cribsy.common.bayes import BayesianData
```

### Constructor

```python
BayesianData(
    base_file: str | None = None,
    train_n: int | None = None,
    test_n: int | None = None,
    augment: bool = False
)
```

**Parameters:**
- `base_file`: Path to data file (`.pkl` for real data, `.ipc` for synthetic)
- `train_n`: Optional limit on training sample size
- `test_n`: Optional limit on testing sample size
- `augment`: Whether to augment real data with synthetic data

**Example:**
```python
# Load real data
bd = BayesianData(base_file="features_merged.pkl")

# Load synthetic data with sampling
bd_synthetic = BayesianData(
    base_file="synthetic_data.ipc",
    train_n=100,
    test_n=50
)
```

###Properties

```python
bd.base           # Complete dataset (long format)
bd.base_train     # Training subset
bd.base_test      # Testing subset
bd.base_wide      # Complete dataset (wide format)
bd.model_names    # List of model names
bd.metrics_names  # List of metrics names
```

### Core Workflow Methods

#### run_rfe_on_base()

Run Recursive Feature Elimination on base dataset.

```python
run_rfe_on_base(
    rfe_name: str,
    features_to_keep_pct: float = 0.9
) -> list[str]
```

**Parameters:**
- `rfe_name`: Identifier for this RFE run
- `features_to_keep_pct`: Proportion of features to retain (default: 0.9)

**Returns:** List of selected feature names

**Example:**
```python
bd = BayesianData(base_file="features_merged.pkl")

# Select features with Enhanced Adaptive RFE
features = bd.run_rfe_on_base(
    rfe_name="rfe_trial_1",
    features_to_keep_pct=0.9
)

print(f"Selected {len(features)} features")
```

#### run_surprise_with_rfe()

Compute Bayesian surprise using features from RFE.

```python
run_surprise_with_rfe(
    model_prefix: str,
    rfe_name: str,
    overwrite: bool = False
)
```

**Parameters:**
- `model_prefix`: Prefix for model naming
- `rfe_name`: Name of RFE run to use
- `overwrite`: Whether to overwrite existing model

**Example:**
```python
# Continue from previous example
bd.run_surprise_with_rfe(
    model_prefix="trial_1",
    rfe_name="rfe_trial_1"
)

# Model name will be "trial_1_k_50" if 50 features were selected
```

#### run_surprise_with_features()

Compute Bayesian surprise using specified features.

```python
run_surprise_with_features(
    model_prefix: str,
    features: list[str],
    overwrite: bool = False
)
```

**Parameters:**
- `model_prefix`: Prefix for model naming
- `features`: List of feature names to use
- `overwrite`: Whether to overwrite existing model

**Example:**
```python
# Use specific features
custom_features = [
    "Ankle_IQRvelx",
    "Wrist_medianx",
    "Knee_mean_angle"
]

bd.run_surprise_with_features(
    model_prefix="custom",
    features=custom_features
)
```

#### run_metrics_from_surprise()

Compute ROC metrics from surprise scores.

```python
run_metrics_from_surprise(
    metrics_name: str,
    model_name: str
)
```

**Parameters:**
- `metrics_name`: Identifier for metrics results
- `model_name`: Name of surprise model to evaluate

**Example:**
```python
# Continue from previous examples
features = bd.rfe_features("rfe_trial_1")
model_name = f"trial_1_k_{len(features)}"

bd.run_metrics_from_surprise(
    metrics_name="trial_1_metrics",
    model_name=model_name
)

# Access results
metrics_df = bd.metrics_df("trial_1_metrics")
print(metrics_df)
```

### Data Access Methods

#### features()

Get feature list for a model.

```python
features(model_name: str) -> list[str] | None
```

**Example:**
```python
feature_list = bd.features("trial_1_k_50")
```

#### metrics_df()

Get metrics DataFrame for a model.

```python
metrics_df(metrics_name: str) -> DataFrame | None
```

**Example:**
```python
metrics = bd.metrics_df("trial_1_metrics")
print(metrics.select(["threshold", "sens", "spec", "auc"]))
```

#### train_surprise() / test_surprise()

Get surprise DataFrames.

```python
train_surprise(model_name: str) -> DataFrame | None
test_surprise(model_name: str) -> DataFrame | None
```

**Example:**
```python
train_surp = bd.train_surprise("trial_1_k_50")
test_surp = bd.test_surprise("trial_1_k_50")

# Columns: infant, risk, minus_log_pfeature, z, p
```

### Report Generation

#### write_excel_report()

Generate comprehensive Excel report.

```python
write_excel_report(tag: str | None = None)
```

**Parameters:**
- `tag`: Optional tag for filename

**Example:**
```python
# Generates: cribsy_model_mytag_sample_size.xlsx
bd.write_excel_report(tag="mytag")
```

### Complete Workflow Example

```python
from early_markers.cribsy.common.bayes import BayesianData

# 1. Initialize
bd = BayesianData(base_file="features_merged.pkl")

# 2. Feature selection
features = bd.run_rfe_on_base(
    rfe_name="rfe_final",
    features_to_keep_pct=0.9
)
print(f"Selected {len(features)} features")

# 3. Compute surprise
bd.run_surprise_with_rfe(
    model_prefix="final",
    rfe_name="rfe_final"
)

# 4. Calculate metrics
model_name = f"final_k_{len(features)}"
bd.run_metrics_from_surprise(
    metrics_name="final_metrics",
    model_name=model_name
)

# 5. Access results
metrics_df = bd.metrics_df("final_metrics")
optimal = metrics_df.filter(
    pl.col("j") == pl.col("j").max()
).head(1)

print(f"Optimal threshold: {optimal.select('threshold').item()}")
print(f"Sensitivity: {optimal.select('sens').item():.3f}")
print(f"Specificity: {optimal.select('spec').item():.3f}")
print(f"AUC: {bd.metrics(model_name).auc:.3f}")

# 6. Generate report
bd.write_excel_report(tag="final")
```

## EnhancedAdaptiveRFE Class

Advanced feature selection with stability testing.

### Import

```python
from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE
```

### Constructor

```python
EnhancedAdaptiveRFE(
    n_trials: int = 50,
    alpha: float = 0.05,
    noise_ratio: float = 0.1
)
```

**Parameters:**
- `n_trials`: Number of parallel RFE trials
- `alpha`: Significance level for binomial test
- `noise_ratio`: Proportion of noise features to inject

**Example:**
```python
selector = EnhancedAdaptiveRFE(
    n_trials=50,
    alpha=0.05,
    noise_ratio=0.1
)
```

### Methods

#### fit()

Execute feature selection.

```python
fit(
    X: pd.DataFrame,
    y: pd.Series,
    features_to_keep_pct: float | None = 0.9
) -> EnhancedAdaptiveRFE
```

**Parameters:**
- `X`: Feature matrix (pandas DataFrame)
- `y`: Target variable (pandas Series)
- `features_to_keep_pct`: Proportion of features to retain

**Returns:** self (for method chaining)

**Example:**
```python
import pandas as pd

# Prepare data
X = df.pivot(on="feature", values="value")
y = df.groupby("infant").agg({"risk": "first"})["risk"]

# Fit selector
selector.fit(X, y, features_to_keep_pct=0.9)
```

#### get_significant_features()

Get statistically significant features.

```python
get_significant_features() -> list[str]
```

**Returns:** List of feature names that passed significance test

**Example:**
```python
significant_features = selector.get_significant_features()
print(f"Found {len(significant_features)} significant features")
```

### Complete Example

```python
from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE
import pandas as pd

# Load and prepare data
df_wide = df.pivot(on="feature", values="value")
X = df_wide.drop(["infant", "category", "risk"], axis=1)
y = df_wide["risk"]

# Configure and fit selector
selector = EnhancedAdaptiveRFE(
    n_trials=50,
    alpha=0.05,
    noise_ratio=0.1
)

selector.fit(X, y, features_to_keep_pct=0.9)

# Get results
features = selector.get_significant_features()
consensus = selector.consensus

# Examine consensus
for feature, info in consensus.items():
    if info["significant"]:
        print(f"{feature}: count={info['count']}, p={info['pvalue']:.4f}")
```

## BAM Functions

Bayesian Assurance Method for sample size estimation when planning new studies.

### Import

```python
from early_markers.cribsy.bam import (
    beta_hdi,
    bam_performance,
    informed_bam_performance
)
```

### beta_hdi()

Calculate Highest Density Interval for Beta distribution.

```python
beta_hdi(
    alpha: float,
    beta: float,
    ci: float = 0.95
) -> tuple[float, float]
```

**Parameters:**
- `alpha`: Beta distribution alpha parameter
- `beta`: Beta distribution beta parameter
- `ci`: Credible interval width (default: 0.95)

**Returns:** Tuple of (lower_bound, upper_bound)

**Example:**
```python
from early_markers.cribsy.bam import beta_hdi

# HDI for Beta(10, 5)
lower, upper = beta_hdi(alpha=10, beta=5, ci=0.95)
print(f"95% HDI: [{lower:.3f}, {upper:.3f}]")
```

### bam_performance()

Estimate sample size for single metric.

```python
bam_performance(
    pilot_data: np.ndarray,
    hdi_width: float = 0.15,
    ci: float = 0.95,
    target_assurance: float = 0.8,
    simulations: int = 2000,
    max_sample: int = 10000
) -> int
```

**Parameters:**
- `pilot_data`: Binary pilot data array
- `hdi_width`: Target HDI width
- `ci`: Credible interval level
- `target_assurance`: Desired assurance level
- `simulations`: Number of Monte Carlo simulations
- `max_sample`: Maximum sample size to consider

**Returns:** Estimated minimum sample size

**Example:**
```python
import numpy as np
from early_markers.cribsy.bam import bam_performance

# Pilot data with 15/20 successes
pilot = np.array([1]*15 + [0]*5)

# Estimate sample size for ±0.15 HDI width
n_required = bam_performance(
    pilot_data=pilot,
    hdi_width=0.15,
    ci=0.95,
    target_assurance=0.8
)

print(f"Required sample size: {n_required}")
```

### informed_bam_performance()

Estimate sample size for sensitivity and specificity.

```python
informed_bam_performance(
    pilot_se: tuple[int, int],
    pilot_sp: tuple[int, int],
    prevalence_prior: tuple[int, int] = (8, 32),
    hdi_width: float = 0.1,
    ci: float = 0.95,
    target_assurance: float = 0.8,
    simulations: int = 1000,
    max_sample: int = 5000
) -> int
```

**Parameters:**
- `pilot_se`: (true_positives, condition_positives)
- `pilot_sp`: (true_negatives, condition_negatives)
- `prevalence_prior`: Beta prior parameters for prevalence
- `hdi_width`: Target HDI width for both metrics
- `ci`: Credible interval level
- `target_assurance`: Desired assurance level
- `simulations`: Number of Monte Carlo simulations
- `max_sample`: Maximum sample size to consider

**Returns:** Estimated minimum sample size

**Example:**
```python
from early_markers.cribsy.bam import informed_bam_performance

# Pilot study results
pilot_se = (18, 20)  # 18 TPs out of 20 positives
pilot_sp = (35, 40)  # 35 TNs out of 40 negatives

n_required = informed_bam_performance(
    pilot_se=pilot_se,
    pilot_sp=pilot_sp,
    prevalence_prior=(8, 32),  # ~20% prevalence
    hdi_width=0.10,  # ±0.10 precision
    target_assurance=0.8
)

print(f"Required sample size: {n_required}")
```

### BAMEstimator (New Unified API)

**Module**: `early_markers.cribsy.common.bam_unified`

Unified Bayesian Assurance Method estimator supporting both single-metric and joint sensitivity/specificity estimation.

#### Import

```python
from early_markers.cribsy.common.bam_unified import BAMEstimator, BAMResult
```

#### Constructor

```python
BAMEstimator(
    seed: int | None = None,
    verbose: bool = False
)
```

**Parameters:**
- `seed`: Random seed for reproducibility (default: None)
- `verbose`: Enable progress logging (default: False)

**Example:**
```python
from early_markers.cribsy.common.bam_unified import BAMEstimator

estimator = BAMEstimator(seed=20250313, verbose=True)
```

#### estimate_single()

Estimate sample size for a single binary metric.

```python
estimate_single(
    pilot_data: tuple[int, int],
    target_width: float,
    ci: float = 0.95,
    target_assurance: float = 0.80,
    simulations: int = 2000,
    max_sample: int = 10000
) -> BAMResult
```

**Parameters:**
- `pilot_data`: (successes, total) from pilot study
- `target_width`: Desired HDI width
- `ci`: Credible interval level (default: 0.95)
- `target_assurance`: Target assurance probability (default: 0.80)
- `simulations`: Monte Carlo simulations (default: 2000)
- `max_sample`: Maximum N to search (default: 10000)

**Returns:** `BAMResult` with estimated sample size and diagnostics

**Example:**
```python
# Pilot study: 18 successes out of 20 trials
result = estimator.estimate_single(
    pilot_data=(18, 20),
    target_width=0.15,
    ci=0.95,
    target_assurance=0.80
)

print(f"Estimated N: {result.n}")
print(f"Achieved assurance: {result.achieved_assurance:.3f}")
print(f"HDI width at estimated N: {result.hdi_width:.3f}")
```

#### estimate_joint()

Estimate sample size for joint sensitivity and specificity.

```python
estimate_joint(
    pilot_se: tuple[int, int],
    pilot_sp: tuple[int, int],
    target_width: float,
    ci: float = 0.95,
    target_assurance: float = 0.80,
    prevalence_prior: tuple[int, int] = (8, 32),
    simulations: int = 1000,
    max_sample: int = 5000
) -> BAMResult
```

**Parameters:**
- `pilot_se`: (true_positives, condition_positives) from pilot
- `pilot_sp`: (true_negatives, condition_negatives) from pilot
- `target_width`: Desired HDI width for both metrics
- `ci`: Credible interval level (default: 0.95)
- `target_assurance`: Target assurance probability (default: 0.80)
- `prevalence_prior`: Beta prior for prevalence as (α, β) (default: (8, 32) = 20%)
- `simulations`: Monte Carlo simulations (default: 1000)
- `max_sample`: Maximum N to search (default: 5000)

**Returns:** `BAMResult` with estimated total sample size

**Example:**
```python
# Pilot study results
pilot_se = (18, 20)  # 90% sensitivity
pilot_sp = (35, 40)  # 87.5% specificity

# Estimate for ±0.10 precision on both metrics
result = estimator.estimate_joint(
    pilot_se=pilot_se,
    pilot_sp=pilot_sp,
    target_width=0.10,
    ci=0.95,
    target_assurance=0.80,
    prevalence_prior=(8, 32)  # 20% prevalence
)

print(f"Required total N: {result.n}")
print(f"Expected positives: {result.n_pos}")
print(f"Expected negatives: {result.n_neg}")
print(f"Achieved assurance: {result.achieved_assurance:.3f}")
```

#### BAMResult

Result object returned by BAMEstimator methods.

```python
@dataclass
class BAMResult:
    n: int                    # Estimated total sample size
    achieved_assurance: float # Achieved assurance probability
    hdi_width: float         # HDI width at estimated N
    target_width: float      # Target HDI width
    ci: float               # Credible interval level
    n_pos: int | None       # Expected positives (joint only)
    n_neg: int | None       # Expected negatives (joint only)
    metric_type: str        # "single" or "joint"
```

**Attributes:**
- `n`: Total sample size estimate
- `achieved_assurance`: Probability of achieving target width (should be ≥ target_assurance)
- `hdi_width`: Actual HDI width achieved at estimated N
- `target_width`: User-specified target width
- `ci`: Credible interval level used
- `n_pos`: Expected number of positive cases (joint estimation only)
- `n_neg`: Expected number of negative cases (joint estimation only)
- `metric_type`: Type of estimation performed

**Example:**
```python
result = estimator.estimate_joint(pilot_se=(18,20), pilot_sp=(35,40), target_width=0.10)

# Access results
print(f"Recommended N: {result.n}")
print(f"Assurance: {result.achieved_assurance:.1%}")  # Should be ≥80%
print(f"Width: {result.hdi_width:.3f}")  # Should be ≤0.10

# Check sample composition
if result.n_pos:
    print(f"Recruit {result.n_pos} positive cases and {result.n_neg} negative cases")
```

## Data Classes

### BayesianFrames

Container for related DataFrames.

```python
@dataclass
class BayesianFrames:
    train: DataFrame | None = None
    test: DataFrame | None = None
    stats: DataFrame | None = None
    train_surprise: DataFrame | None = None
    test_surprise: DataFrame | None = None
```

### BayesianRfeResult

RFE output container.

```python
@dataclass
class BayesianRfeResult:
    name: str
    k: int
    features: list[str]
```

### BayesianSurprise

Surprise distribution parameters.

```python
@dataclass
class BayesianSurprise:
    model_name: str
    k: int
    mean_neg_log_p: float
    sd_neg_log_p: float
```

### BayesianRocResult

ROC analysis output.

```python
@dataclass
class BayesianRocResult:
    model_name: str
    train_n: int
    test_n: int
    features: list[str]
    plot_file: str
    auc: float
    youdens_j: float
    roc_youdens_j: float
    threshold_j: float
    metrics: DataFrame
    primitives: DataFrame
    rough_n_min: int
    rough_n_max: int
```

### BayesianCI

Confidence interval analysis.

```python
@dataclass
class BayesianCI:
    metric: str
    value: float
    half_width: float
    train_n: int
    test_n: int
    lb: float
    ub: float
    lb_width: float  # auto-computed
    ub_width: float  # auto-computed
    lb_achieved: bool  # auto-computed
    ub_achieved: bool  # auto-computed
```

## Error Handling

### Common Exceptions

```python
# Missing base data
AttributeError: "A base DataFrame is not defined"

# Model not found
AttributeError: "Base DataFrames are not set"

# Overwrite protection
ValueError: "DataFrames for model: 'name' already exist"
```

### Best Practices

1. **Always check if data exists:**
   ```python
   if bd.model_names is not None:
       for name in bd.model_names:
           print(f"Model: {name}")
   ```

2. **Use overwrite parameter when needed:**
   ```python
   bd.run_surprise_with_features(
       model_prefix="test",
       features=features,
       overwrite=True  # Allow replacing existing model
   )
   ```

3. **Handle None returns:**
   ```python
   metrics = bd.metrics_df("model_name")
   if metrics is not None:
       # Process metrics
       pass
   ```

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- See [WORKFLOWS.md](WORKFLOWS.md) for common analysis patterns
- See [DATA_FORMATS.md](DATA_FORMATS.md) for data specifications
- See [CONFIGURATION.md](CONFIGURATION.md) for parameter tuning
