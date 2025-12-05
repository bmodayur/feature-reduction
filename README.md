# Early Markers - Bayesian Surprise Analysis for Infant Movement

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-complete-brightgreen.svg)](docs/)

**A research codebase for analyzing early infant movement markers using Bayesian statistical methods and machine learning techniques.**

---

## Overview

Early Markers implements a Bayesian surprise analysis framework to identify developmental anomalies in infant movement data. The system quantifies how unexpected an infant's movement patterns are compared to a normative reference population, enabling early detection of at-risk developmental trajectories.

### Key Features

- 🧠 **Bayesian Surprise Analysis**: Compute z-scores from movement feature distributions
- 🎯 **Advanced Feature Selection**: Enhanced Adaptive RFE with noise injection and consensus testing
- 📊 **Sample Size Estimation**: Bayesian Assurance Method (BAM) for study planning
- 🔬 **Synthetic Data Generation**: Validation and power analysis with SDV/Gretel
- 📈 **ROC Analysis**: Comprehensive metrics with confidence intervals
- 📑 **Automated Reporting**: Excel workbooks with formatted tables and plots

### Core Methodology

**Bayesian Surprise** quantifies deviation from normative patterns:

1. Compute negative log-likelihood for each feature: `-log P(x | N(μ_ref, σ²_ref))`
2. Sum across features: `S_i = Σ_f [-log P(x_if)]`
3. Standardize: `z_i = (S_i - μ_train) / σ_train`
4. Compute p-value: `p_i = 2 × SF(|z_i|)`

Higher z-scores indicate greater deviation from typical development.

---

## Quick Start

### Installation

**Requirements**: Python 3.12 (NOT 3.13)

```bash
# Clone repository
git clone <repository-url>
cd early-markers

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Basic Usage

```python
from early_markers.cribsy.common.bayes import BayesianData

# Initialize with data
bd = BayesianData(base_file="features_merged.pkl")

# Run feature selection
bd.run_rfe_on_base(
    rfe_name="rfe_trial_1",
    features_to_keep_pct=0.9
)

# Compute Bayesian surprise
bd.run_surprise_with_rfe(
    model_prefix="trial_1",
    rfe_name="rfe_trial_1"
)

# Calculate ROC metrics
bd.run_metrics_from_surprise(
    metrics_name="trial_1_metrics",
    model_name=f"trial_1_k_{bd.rfe_k('rfe_trial_1')}"
)

# Access results
metrics_df = bd.metrics_df("trial_1_metrics")
print(f"AUC: {metrics_df['auc'].iloc[0]:.3f}")
```

### Jupyter Notebooks

Primary workflow uses Jupyter notebooks:

```bash
poetry run jupyter notebook
```

Key notebooks in `early_markers/cribsy/notebooks/`:
- `01_rfe_real.ipynb` - Real data feature selection with iterative RFE
- `01_rfe_real_fast.ipynb` - Fast mode feature selection (10 trials)
- `02_synthetic_sdv.ipynb` - Synthetic data generation with SDV
- `03_rfe_synthetic.ipynb` - Feature selection on synthetic data
- `04_bam_sample_size.ipynb` - BAM sample size estimation
- `05_test_retest_sample_size.ipynb` - Test-retest reliability analysis
- `06_tost_sample_size.ipynb` - TOST equivalence testing

**Primary workflow**: Run notebooks 01 → 04 for complete analysis from raw data to sample size estimation.

---

## Two Pathways for Users

This project offers two pathways depending on your goals:

### 🔬 Pathway 1: Core API & Methodology

For users who want to **understand the methodology** or **run their own analyses**:

| Resource | Purpose |
|----------|---------|
| `early_markers/cribsy/notebooks/01-06` | Primary analysis notebooks |
| `early_markers/cribsy/common/` | Core Python modules |
| `docs/ARCHITECTURE.md` | System design |
| `docs/API.md` | API reference |
| `docs/WORKFLOWS.md` | Step-by-step guides |

**Start here**: Run notebooks 01 → 04 in `early_markers/cribsy/notebooks/`

### 📊 Pathway 2: Analysis Results & Scripts

For users who want to **understand the current model results** or **reproduce specific analyses**:

| Resource | Purpose |
|----------|---------|
| `scripts/` | Analysis scripts (RFE, ROC, BAM) |
| `docs/BAM_for_20FeatureSet.md` | Sample size analysis for 20-feature model |
| `docs/RFE_METHODOLOGY.md` | Feature selection methodology |
| `docs/FEATURE_TRANSFORMATION_METHODOLOGY.md` | 1/(1+\|x\|) transformation |
| `docs/FEATURE_SET_PROVENANCE.md` | Tracking feature set origins |

**Current Best Model**: 20-Feature Nov 21 RFE with transformation
- AUC: 0.902
- Sensitivity: 92.1%
- Specificity: 77.5%
- See: `scripts/compute_roc_after_rfe.py`

---

## Documentation

### 📚 Complete Documentation Suite

**Core Documentation** (Pathway 1):

| Document | Description |
|----------|-------------|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design and methodology |
| **[docs/API.md](docs/API.md)** | Complete API reference |
| **[docs/WORKFLOWS.md](docs/WORKFLOWS.md)** | Common analysis patterns |
| **[docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)** | Data specifications |
| **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** | Parameter tuning guide |

**Analysis Documentation** (Pathway 2):

| Document | Description |
|----------|-------------|
| **[docs/BAM_for_20FeatureSet.md](docs/BAM_for_20FeatureSet.md)** | Sample size analysis for current model |
| **[docs/RFE_METHODOLOGY.md](docs/RFE_METHODOLOGY.md)** | Enhanced Adaptive RFE methodology |
| **[docs/FEATURE_TRANSFORMATION_METHODOLOGY.md](docs/FEATURE_TRANSFORMATION_METHODOLOGY.md)** | 1/(1+\|x\|) transformation for bidirectional detection |
| **[docs/FEATURE_SET_PROVENANCE.md](docs/FEATURE_SET_PROVENANCE.md)** | Tracking feature set origins |
| **[docs/FEATURE_SETS_SUMMARY.md](docs/FEATURE_SETS_SUMMARY.md)** | Comparison of feature sets |

### 🎓 Getting Started Guide

**New users should read in this order:**

1. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Understand the Bayesian surprise methodology
2. **[DATA_FORMATS.md](docs/DATA_FORMATS.md)** - Learn about data structures
3. **[WORKFLOWS.md](docs/WORKFLOWS.md)** - Follow step-by-step analysis patterns
4. **[API.md](docs/API.md)** - Reference for all classes and methods
5. **[CONFIGURATION.md](docs/CONFIGURATION.md)** - Tune parameters for your use case

### 📖 Quick Links

- **[Basic Analysis Workflow](docs/WORKFLOWS.md#basic-analysis-workflow)** - 6-step guide
- **[Feature Selection Guide](docs/WORKFLOWS.md#feature-selection-workflow)** - RFE methods
- **[Configuration Profiles](docs/CONFIGURATION.md#configuration-profiles)** - Production/dev/reproducibility
- **[Data Transformations](docs/DATA_FORMATS.md#data-transformations)** - Pipeline details
- **[Troubleshooting](docs/WORKFLOWS.md#troubleshooting)** - Common issues

---

## Project Structure

```
early-markers/
├── README.md                          # This file
├── CLAUDE.md                          # Claude Code guidance
├── pyproject.toml                    # Poetry dependencies
├── requirements.txt                  # Pip dependencies
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # System design (Pathway 1)
│   ├── API.md                        # API reference (Pathway 1)
│   ├── WORKFLOWS.md                  # Analysis patterns (Pathway 1)
│   ├── DATA_FORMATS.md               # Data specifications (Pathway 1)
│   ├── CONFIGURATION.md              # Parameter tuning (Pathway 1)
│   ├── BAM_for_20FeatureSet.md       # Sample size analysis (Pathway 2)
│   ├── RFE_METHODOLOGY.md            # RFE methodology (Pathway 2)
│   ├── FEATURE_TRANSFORMATION_METHODOLOGY.md  # Transformation (Pathway 2)
│   ├── FEATURE_SET_PROVENANCE.md     # Feature set tracking (Pathway 2)
│   └── FEATURE_SETS_SUMMARY.md       # Feature set comparison (Pathway 2)
│
├── scripts/                          # Analysis scripts (Pathway 2)
│   ├── compute_roc_after_rfe.py      # ROC metrics for feature sets
│   ├── run_rfe_with_age_forced.py    # RFE with forced age inclusion
│   ├── run_bam_20feature_nov21.py    # BAM sample size calculation
│   └── *.py                          # Additional analysis scripts
├── early_markers/
│   ├── cribsy/                       # Core analysis module
│   │   ├── __init__.py              # Random seed init
│   │   ├── bam.py                   # Legacy BAM functions
│   │   ├── thresholds.py            # Pre-computed thresholds
│   │   ├── common/
│   │   │   ├── bayes.py             # Main orchestration (BayesianData)
│   │   │   ├── adaptive_rfe.py      # Enhanced feature selection
│   │   │   ├── bam_unified.py       # Unified BAM implementation
│   │   │   ├── bam_export.py        # BAM results export utilities
│   │   │   ├── rfe.py               # Base RFE
│   │   │   ├── constants.py         # Configuration
│   │   │   ├── enums.py             # Type definitions
│   │   │   ├── metrics_n.py         # Sample size formulas
│   │   │   └── data.py              # Legacy data loading
│   │   └── notebooks/               # Jupyter analysis notebooks
│   │       ├── 01_rfe_real.ipynb          # Real data RFE
│   │       ├── 01_rfe_real_fast.ipynb     # Fast RFE mode
│   │       ├── 02_synthetic_sdv.ipynb     # SDV synthetic data
│   │       ├── 03_rfe_synthetic.ipynb     # Synthetic RFE
│   │       ├── 04_bam_sample_size.ipynb   # BAM estimation
│   │       ├── 05_test_retest_sample_size.ipynb  # ICC analysis
│   │       ├── 06_tost_sample_size.ipynb  # TOST equivalence
│   │       └── hold/                      # Archive/legacy
│   ├── common/
│   │   └── xlsx.py                  # Excel formatting
│   └── emmacp_metrics/              # Raw data directory
└── working/                          # Working directory
```

---

## Core Components

### BayesianData Class

Main orchestration class for the analysis pipeline.

```python
from early_markers.cribsy.common.bayes import BayesianData

bd = BayesianData(
    base_file="features_merged.pkl",  # Data file
    synthetic=False,                   # Real vs synthetic data
    augment=False,                     # Add noise features
    train_n=None,                      # Sample size (None = all)
    test_n=None
)
```

**Key Methods**:
- `run_rfe_on_base()` - Feature selection
- `run_surprise_with_rfe()` - Compute Bayesian surprise
- `run_metrics_from_surprise()` - ROC analysis
- `metrics_df()` - Access results

### EnhancedAdaptiveRFE Class

Advanced feature selection with stability testing.

```python
from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE

selector = EnhancedAdaptiveRFE(
    n_trials=50,              # Parallel trials
    alpha=0.05,               # Significance level
    noise_ratio=0.1,          # Noise injection
    n_jobs=12                 # Parallelization
)

selector.fit(X, y, features_to_keep_pct=0.9)
selected = selector.get_significant_features()
```

### BAM Functions

Bayesian Assurance Method for sample size estimation when planning new studies.

```python
from early_markers.cribsy.common.bam_unified import BAMEstimator

# Initialize BAM estimator
estimator = BAMEstimator(seed=20250313, verbose=True)

# Estimate sample size for joint sensitivity + specificity
result = estimator.estimate_joint(
    pilot_se=(tp, total_positives),  # Pilot data tuple
    pilot_sp=(tn, total_negatives),  # Pilot data tuple
    target_width=0.15,               # Target CI width
    ci=0.95,                         # Confidence level
    target_assurance=0.80,           # Desired assurance
    prevalence_prior=(8, 32)         # Beta prior for prevalence
)

print(f"Recommended N: {result.n_total}")
print(f"Se CI width: {result.se_width:.3f}, Sp CI width: {result.sp_width:.3f}")
```

**Note**: See `04_bam_sample_size.ipynb` for complete BAM workflow and grid search examples.

---

## Data

### Movement Features (56 total)

**Body Parts** (8): Ankle, Wrist, Knee, Elbow, Hip, Shoulder, Ear, Eye (L/R)

**Feature Types**:
- Position: entropy, mean, std
- Velocity: entropy, mean, std
- Acceleration: entropy, mean, std
- Correlation: xy, vx_vy
- Age: `age_in_weeks`

**Example Features**:
- `Ankle_L_position_entropy`
- `Wrist_R_velocity_mean`
- `Hip_L_acceleration_std`

### Data Formats

**Long Format** (internal):
```
infant | category | risk | feature                | value
-------|----------|------|------------------------|-------
inf_01 | 1        | 0    | Ankle_L_position_entropy | 0.234
inf_01 | 1        | 0    | Wrist_R_velocity_mean    | 1.567
```

**Wide Format** (for ML):
```
infant | category | risk | Ankle_L_... | Wrist_R_... | ...
-------|----------|------|-------------|-------------|-----
inf_01 | 1        | 0    | 0.234       | 1.567       | ...
```

See **[DATA_FORMATS.md](docs/DATA_FORMATS.md)** for complete specifications.

---

## Configuration

### Key Parameters

All configuration in `early_markers/cribsy/common/constants.py`:

```python
# Reproducibility
RAND_STATE = 20250313        # Never change!

# RFE Settings
RFE_N_TRIALS = 50            # Parallel trials
RFE_ALPHA = 0.05             # Significance level
RFE_KEEP_PCT = 0.9           # Target 90% features
RFE_NOISE_RATIO = 0.1        # 10% noise injection

# Random Forest
N_ESTIMATORS = 200           # Trees per forest
N_JOBS = 8                   # Parallel jobs

# Cross-Validation
RKFOLD_REPEATS = 3           # Repeated k-fold
RKFOLD_SPLITS = 5            # K-fold splits (15 total)
```

### Configuration Profiles

**Production** (2-4 hours):
- `RFE_N_TRIALS=50`, `N_ESTIMATORS=200`
- Full reproducibility and stability

**Development** (15-30 minutes):
- `RFE_N_TRIALS=20`, `N_ESTIMATORS=100`
- 10-20× faster for iteration

**Reproducibility** (8-12 hours):
- `n_jobs=1` everywhere
- Bit-exact results across machines

See **[CONFIGURATION.md](docs/CONFIGURATION.md)** for complete tuning guide.

---

## Development

### Environment Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests (if available)
poetry run pytest

# Start Jupyter
poetry run jupyter notebook
```

### Path Configuration

**Machine-specific paths** in `constants.py`:

```python
BASE_DATA_DIR = Path("/Volumes/secure/data/early_markers/cribsy")
RAW_DATA_PATH = Path("/Volumes/secure/code/early-markers/early_markers/emmacp_metrics/features_merged.pkl")
```

**Update these for your machine!**

### Reproducibility

- Random seed set globally: `RAND_STATE = 20250313`
- All stochastic operations use this seed
- For exact reproducibility: set `n_jobs=1` in all estimators
- See **[CONFIGURATION.md](docs/CONFIGURATION.md)** for details

---

## Performance

### Computational Costs

| Operation | Complexity | Typical Time |
|-----------|------------|-------------|
| Data Loading | O(n × f) | < 1 minute |
| RFE | O(trials × folds × trees) | 1-2 hours |
| Surprise Calculation | O(n × f) | < 5 minutes |
| ROC Analysis | O(n × thresholds) | < 1 minute |

### Optimization Strategies

1. **Reduce RFE trials**: 50 → 20 (~2.5× speedup)
2. **Reduce RF trees**: 200 → 100 (~2× speedup)
3. **Increase step size**: `RFE_MAX_STEP` 5 → 10 (~1.5× speedup)
4. **Reduce CV folds**: (3,5) → (1,3) (~5× speedup)
5. **Simplify grid**: 27 → 9 combinations (~3× speedup)

**Combined**: 10-20× faster for development

See **[CONFIGURATION.md](docs/CONFIGURATION.md#performance-tuning)** for details.

---

## Output Artifacts

### Excel Reports

**Location**: `/Volumes/secure/data/early_markers/cribsy/xlsx/`

**Format**:
- Summary sheet with top models by AUC and Youden's J
- Detail sheets per model with full metrics
- Conditional formatting and hyperlinks
- Confidence intervals and sample size estimates

### Plots

**Location**: `/Volumes/secure/data/early_markers/cribsy/png/`

**Types**:
- ROC curves with AUC
- Feature importance heatmaps
- Stability plots
- Threshold sensitivity analysis

### Intermediate Data

**Pickle files** (`.pkl`): Cached intermediate results  
**IPC files** (`.ipc`): Synthetic datasets (Polars format)  
**CSV files** (`.csv`): Tabular exports  

---

## Common Workflows

### 1. Basic Analysis

```python
from early_markers.cribsy.common.bayes import BayesianData

bd = BayesianData(base_file="features_merged.pkl")
bd.run_rfe_on_base(rfe_name="rfe_1", features_to_keep_pct=0.9)
bd.run_surprise_with_rfe(model_prefix="model", rfe_name="rfe_1")
bd.run_metrics_from_surprise(
    metrics_name="trial_1",
    model_name=f"model_k_{bd.rfe_k('rfe_1')}"
)
metrics_df = bd.metrics_df("trial_1")
```

### 2. Model Comparison

```python
# Feature count sweep
for pct in [0.7, 0.8, 0.9, 0.95]:
    bd.run_rfe_on_base(rfe_name=f"rfe_{int(pct*100)}", features_to_keep_pct=pct)
    bd.run_surprise_with_rfe(model_prefix=f"sweep", rfe_name=f"rfe_{int(pct*100)}")
    # ... compute metrics

# Compare AUC across models
for name in bd.metrics_names:
    metrics = bd.metrics(name)
    print(f"{name}: AUC = {metrics.auc:.3f}")
```

### 3. Sample Size Planning

```python
from early_markers.cribsy.bam import informed_bam_performance

# Based on pilot results
results = informed_bam_performance(
    pilot_se=0.85,
    pilot_sp=0.80,
    prevalence_prior=(1, 1),
    hdi_width=0.15
)

print(f"Recommended sample size: {results['n_total']} (se: {results['n_se']}, sp: {results['n_sp']})")
```

See **[WORKFLOWS.md](docs/WORKFLOWS.md)** for complete examples.

---

## Testing

### Validation Strategy

**No formal test suite.** Validation performed through:

1. **Synthetic data generation**: Create datasets with known properties
2. **Jupyter notebooks**: Visual inspection and comparison
3. **Known thresholds**: Compare against pre-computed values in `thresholds.py`
4. **Cross-validation metrics**: Consistency checks across folds

### Running Validation

```bash
# Start Jupyter
poetry run jupyter notebook

# Run synthetic data notebook
# Navigate to: early_markers/cribsy/notebooks/synthetic.ipynb
# Execute all cells and verify metrics
```

---

## Troubleshooting

### Common Issues

**Issue**: Python version mismatch  
**Solution**: Must use Python 3.12 (NOT 3.13)  
```bash
pyenv install 3.12.7
pyenv local 3.12.7
poetry env use python3.12
```

**Issue**: Path not found errors  
**Solution**: Update machine-specific paths in `constants.py`  
```python
BASE_DATA_DIR = Path("/your/path/here")
```

**Issue**: Memory errors during RFE  
**Solution**: Reduce parallelization  
```python
RFE_N_JOBS = 4  # Instead of 12
N_JOBS = 4      # Instead of 8
```

**Issue**: Non-reproducible results  
**Solution**: Set n_jobs=1 for exact reproducibility  
```python
RFE_N_JOBS = 1
N_JOBS = 1
```

See **[WORKFLOWS.md](docs/WORKFLOWS.md#troubleshooting)** for more solutions.

---

## Contributing

### Code Style

- **Docstrings**: Google-style, PEP 257 compliant
- **Type hints**: Use throughout public APIs
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Imports**: Absolute imports, specific (no `import *`)

### Documentation

All code must include:
- Module-level docstrings explaining purpose
- Class docstrings with attributes and examples
- Method docstrings with Args, Returns, Raises, Examples
- Mathematical formulations where relevant

### Adding Features

1. Update relevant module in `early_markers/cribsy/common/`
2. Add comprehensive docstrings
3. Update `docs/API.md` with new methods
4. Add examples to `docs/WORKFLOWS.md` if applicable
5. Test with synthetic data validation

---

## References

### Academic

- **Bayesian Information Theory**: Kullback-Leibler divergence and surprise
- **ROC Analysis**: Fawcett, T. (2006). An introduction to ROC analysis
- **Recursive Feature Elimination**: Guyon et al. (2002)
- **Wilson Confidence Intervals**: Wilson, E.B. (1927)
- **Sample Size Estimation**: Flahault et al. (2005), Bujang & Adnan (2016)

### Technical

- **Polars**: Modern DataFrame library (https://pola.rs/)
- **scikit-learn**: Machine learning library
- **PyMC**: Bayesian inference
- **Poetry**: Dependency management

---

## License

Proprietary - All rights reserved.

---

## Contact

For questions or support, please contact the research team.

---

## Changelog

### 2025-12-04
- ✅ Added Two Pathways structure to README
- ✅ Created BAM sample size analysis for 20-Feature model (`docs/BAM_for_20FeatureSet.md`)
- ✅ Documented Phase 2 test set design with enriched sampling
- ✅ Added RFE methodology documentation (`docs/RFE_METHODOLOGY.md`)
- ✅ Feature set provenance tracking (`docs/FEATURE_SET_PROVENANCE.md`)
- ✅ Created analysis scripts in `scripts/` folder
- ✅ Current best model: 20-Feature Nov 21 RFE (AUC 0.902, Se 92.1%, Sp 77.5%)

### 2025-01-20
- ✅ Complete documentation suite created (6,360+ lines)
- ✅ Python files: 100% docstring coverage (3,170 lines)
- ✅ Markdown guides: 5 comprehensive documents (3,191 lines)
- ✅ All code verified with py_compile
- ✅ Cross-references validated and functional

### Earlier
- Initial codebase development
- Core Bayesian surprise implementation
- Enhanced Adaptive RFE implementation
- BAM sample size estimation
- Synthetic data generation integration

---

**Last Updated**: December 4, 2025
**Documentation Status**: ✅ Complete
**Python Version**: 3.12
**Current Best Model**: 20-Feature Nov 21 RFE (AUC 0.902)
**Maintainer**: EARLY MARKERS
