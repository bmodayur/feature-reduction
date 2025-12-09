# Configuration

**Parameter tuning and configuration guide for early-markers**

This document describes all configurable parameters in the early-markers Bayesian surprise analysis pipeline and provides guidance for tuning them.

## Table of Contents

1. [Overview](#overview)
2. [Reproducibility Settings](#reproducibility-settings)
3. [RFE Configuration](#rfe-configuration)
4. [Random Forest Configuration](#random-forest-configuration)
5. [Cross-Validation Configuration](#cross-validation-configuration)
6. [Data Split Configuration](#data-split-configuration)
7. [Path Configuration](#path-configuration)
8. [Performance Tuning](#performance-tuning)
9. [Parameter Sensitivity](#parameter-sensitivity)

---

## Overview

All configuration constants are defined in `early_markers/cribsy/common/constants.py`.

**Import**:
```python
from early_markers.cribsy.common.constants import (
    RAND_STATE, N_ESTIMATORS, RFE_N_TRIALS, RFE_ALPHA, ...
)
```

**Modification**: Edit `constants.py` directly and restart Python session.

**Validation**: Many parameters are validated at runtime with descriptive error messages.

---

## Reproducibility Settings

### `RAND_STATE`

**Value**: `20250313` (integer)

**Purpose**: Global random seed for all stochastic operations

**Used in**:
- NumPy: `np.random.seed(RAND_STATE)`
- Random Forest: `random_state=RAND_STATE`
- Cross-validation: `random_state=RAND_STATE`
- Train/test split: `random_state=RAND_STATE`
- Data sampling: `seed=RAND_STATE`

**Notes**:
- Set at module import in `__init__.py`
- Ensures bit-exact reproducibility across runs
- Critical for scientific reproducibility
- Value chosen arbitrarily (date-based: 2025-03-13)

**Recommendation**: **Never change** once analysis begins

**Example**:
```python
# Automatically applied when module loads
from early_markers.cribsy import BayesianData

# All subsequent operations are reproducible
bd = BayesianData(base_file="features_merged.pkl")
bd.run_rfe_on_base(rfe_name="rfe_1")  # Reproducible results
```

---

## RFE Configuration

### Core RFE Parameters

#### `RFE_N_TRIALS`

**Value**: `50` (integer)

**Purpose**: Number of parallel RFE trials in Enhanced Adaptive RFE

**Range**: 10-100
- **10-30**: Fast exploratory analysis, less stable
- **50**: Default, good balance (used in production)
- **100+**: Slow, marginal stability gains

**Memory Impact**: Linear (50 trials ≈ 4-8 GB RAM)

**Trade-offs**:
- ↑ Trials → ↑ Feature consensus stability
- ↑ Trials → ↑ Computation time
- ↑ Trials → ↑ Memory usage

**Recommendation**: 50 for production, 20 for development

**Example**:
```python
from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE

# Use default
selector = EnhancedAdaptiveRFE(n_trials=RFE_N_TRIALS)

# Override for fast iteration
selector = EnhancedAdaptiveRFE(n_trials=20)
```

---

#### `RFE_ALPHA`

**Value**: `0.05` (float)

**Purpose**: Significance level for binomial test in feature consensus

**Range**: 0.01-0.10
- **0.01**: Very conservative, fewer features selected
- **0.05**: Standard (95% confidence), balanced
- **0.10**: Liberal, more features selected

**Effect**:
- Features selected in > `(1 - alpha) × n_trials` trials pass threshold
- At α=0.05, n_trials=50: feature must appear in ≥48 trials

**Trade-offs**:
- ↓ Alpha → Fewer features → Higher precision, lower recall
- ↑ Alpha → More features → Lower precision, higher recall

**Recommendation**: 0.05 (standard practice)

**Example**:
```python
# Standard
selector = EnhancedAdaptiveRFE(alpha=0.05)

# Conservative (fewer features)
selector = EnhancedAdaptiveRFE(alpha=0.01)
```

---

#### `RFE_KEEP_PCT`

**Value**: `0.9` (float)

**Purpose**: Target percentage of features to retain (90%)

**Range**: 0.5-0.95
- **0.5**: Aggressive reduction (50% features)
- **0.9**: Conservative (90% features) - default
- **0.95**: Minimal reduction (95% features)

**Feature Count**:
- At 0.9: Keep ~50 of 56 features
- At 0.8: Keep ~45 of 56 features
- At 0.5: Keep ~28 of 56 features

**Trade-offs**:
- ↑ Keep % → More features → Better coverage, more noise
- ↓ Keep % → Fewer features → Less noise, potential information loss

**Recommendation**: 0.85-0.95 for movement data (high feature quality)

**Example**:
```python
# Default (90%)
bd.run_rfe_on_base(rfe_name="rfe_90", features_to_keep_pct=0.9)

# More aggressive (80%)
bd.run_rfe_on_base(rfe_name="rfe_80", features_to_keep_pct=0.8)
```

---

#### `RFE_NOISE_RATIO`

**Value**: `0.1` (float)

**Purpose**: Ratio of noise features to inject for stability testing

**Range**: 0.05-0.20
- **0.05**: Minimal noise (3 features for 56-feature set)
- **0.10**: Moderate (6 features) - default
- **0.20**: High noise (11 features)

**Effect**:
- Noise features = `floor(n_features × RFE_NOISE_RATIO)`
- Noise drawn from Gaussian N(0, 1)
- Features that rank below noise are considered unstable

**Trade-offs**:
- ↑ Ratio → More stringent test → Fewer features pass
- ↓ Ratio → Less stringent → More features pass

**Recommendation**: 0.10 (10% noise is standard)

**Example**:
```python
# Default noise
selector = EnhancedAdaptiveRFE(noise_ratio=0.1)

# No noise (faster, less robust)
selector = EnhancedAdaptiveRFE(noise_ratio=0.0)
```

---

### RFE Step Size Parameters

#### `RFE_MIN_STEP` and `RFE_MAX_STEP`

**Values**: 
- `RFE_MIN_STEP = 1` (integer)
- `RFE_MAX_STEP = 5` (integer)

**Purpose**: Adaptive step size range for feature elimination

**Behavior**:
- Start with `RFE_MAX_STEP` (eliminate 5 features at a time)
- If CV score drops: reduce step size
- Minimum step size = `RFE_MIN_STEP` (eliminate 1 feature at a time)

**Range**:
- **Min Step**: 1-3 (always use 1 for precision)
- **Max Step**: 3-10 (higher = faster but less precise)

**Trade-offs**:
- Larger max step → Faster RFE → Less precise feature selection
- Smaller max step → Slower RFE → More precise feature selection

**Recommendation**: (1, 5) for 50-60 features

**Example**:
```python
# Fast RFE (less precise)
selector = AdaptiveStepRFE(min_step=2, max_step=10)

# Precise RFE (slower)
selector = AdaptiveStepRFE(min_step=1, max_step=3)
```

---

#### `RFE_TOLERANCE`

**Value**: `0.01` (float)

**Purpose**: Minimum CV score improvement to continue elimination

**Range**: 0.001-0.05
- **0.001**: Very sensitive, more iterations
- **0.01**: Moderate sensitivity - default
- **0.05**: Less sensitive, fewer iterations

**Effect**:
- Stop eliminating if `CV_score_new - CV_score_old < RFE_TOLERANCE`

**Trade-offs**:
- ↓ Tolerance → More features eliminated → Longer runtime
- ↑ Tolerance → Fewer features eliminated → Faster runtime

**Recommendation**: 0.01 (1% improvement threshold)

---

### RFE Parallelization

#### `RFE_N_JOBS`

**Value**: `12` (integer)

**Purpose**: Number of parallel jobs for RFE trials

**Range**: 1 to CPU count
- **1**: No parallelization (fully reproducible)
- **4-8**: Moderate parallelization
- **12**: Default (high parallelization)
- **-1**: Use all CPUs

**Reproducibility Note**:
- Using `n_jobs > 1` may cause minor numerical differences
- For exact reproducibility, set to 1

**Memory Impact**: Linear (12 jobs ≈ 12× memory)

**Recommendation**: 
- **Development**: 4-8 (moderate)
- **Production**: 12 (fast)
- **Reproducibility**: 1 (slow but exact)

**Example**:
```python
# Fast (uses 12 cores)
selector = EnhancedAdaptiveRFE(n_jobs=12)

# Exact reproducibility (1 core)
selector = EnhancedAdaptiveRFE(n_jobs=1)
```

---

## Random Forest Configuration

### `N_ESTIMATORS`

**Value**: `200` (integer)

**Purpose**: Number of trees in Random Forest

**Range**: 50-500
- **50-100**: Fast, less stable
- **200**: Default, good balance
- **500**: Slow, marginal accuracy gains

**Trade-offs**:
- ↑ Trees → ↑ Accuracy (diminishing returns after 200)
- ↑ Trees → ↑ Runtime (linear)
- ↑ Trees → ↑ Memory (linear)

**Recommendation**: 200 for production

**Example**:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RAND_STATE
)
```

---

### `N_JOBS`

**Value**: `8` (integer)

**Purpose**: Number of parallel jobs for Random Forest training

**Range**: 1 to CPU count
- **1**: No parallelization (reproducible)
- **4-8**: Moderate parallelization
- **-1**: Use all CPUs

**Reproducibility Note**: 
- `n_jobs=1` guarantees exact reproducibility
- `n_jobs > 1` may cause minor differences even with `random_state`

**Recommendation**: 8 for speed, 1 for reproducibility

---

### `RFG_PARAM_GRID`

**Value**: Dictionary of hyperparameter ranges

```python
RFG_PARAM_GRID = {
    'max_depth': [5, 7, 9],
    'min_samples_leaf': [1, 2, 3],
    'max_features': [1, 'sqrt', 'log2'],
}
```

**Purpose**: GridSearchCV hyperparameter search space for RFE

**Components**:

1. **`max_depth`**: Maximum tree depth
   - `[5, 7, 9]`: Shallow to moderate trees
   - Prevents overfitting
   - Range: 3-15

2. **`min_samples_leaf`**: Minimum samples per leaf
   - `[1, 2, 3]`: Allow fine-grained splits
   - Higher values prevent overfitting
   - Range: 1-10

3. **`max_features`**: Features per split
   - `1`: Single feature (very restrictive)
   - `'sqrt'`: √n_features (standard for classification)
   - `'log2'`: log₂(n_features) (very restrictive)
   - Range: 1, 'sqrt', 'log2', or integers

**Grid Size**: 3 × 3 × 3 = **27 combinations**

**Tuning Strategy**:
- Start with default grid
- If overfitting: increase `min_samples_leaf`, decrease `max_depth`
- If underfitting: decrease `min_samples_leaf`, increase `max_depth`

**Example**:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=RAND_STATE)
grid = GridSearchCV(rf, RFG_PARAM_GRID, cv=3, n_jobs=8)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

**Expansion**:
```python
# More comprehensive grid (slower)
RFG_PARAM_GRID_EXTENDED = {
    'max_depth': [3, 5, 7, 9, 11],
    'min_samples_leaf': [1, 2, 3, 5, 10],
    'max_features': [1, 'sqrt', 'log2', 0.5],
    'min_samples_split': [2, 5, 10]
}
# Grid size: 5 × 5 × 4 × 3 = 300 combinations
```

---

## Cross-Validation Configuration

### Repeated K-Fold Parameters

#### `RKFOLD_REPEATS`

**Value**: `3` (integer)

**Purpose**: Number of times to repeat k-fold CV

**Range**: 1-10
- **1**: Standard k-fold (faster)
- **3**: Repeated k-fold (default)
- **5-10**: Highly stable estimates (slow)

**Effect**:
- Total folds = `RKFOLD_REPEATS × RKFOLD_SPLITS`
- At (3, 5): 15 total folds

**Recommendation**: 3 for production, 1 for development

---

#### `RKFOLD_SPLITS`

**Value**: `5` (integer)

**Purpose**: Number of folds in each k-fold repetition

**Range**: 3-10
- **3**: Fast, less stable
- **5**: Standard (default)
- **10**: Leave-one-out approximation (slow)

**Minimum Sample Size**: Must have ≥ `RKFOLD_SPLITS` samples per class

**Recommendation**: 5 (standard practice)

**Example**:
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rkf = RepeatedStratifiedKFold(
    n_splits=RKFOLD_SPLITS,
    n_repeats=RKFOLD_REPEATS,
    random_state=RAND_STATE
)
```

---

### RFE-Specific CV Parameters

#### `STEP_KFOLDS`

**Value**: `3` (integer)

**Purpose**: Number of CV folds for adaptive step size determination

**Range**: 2-5
- **2**: Very fast, less reliable
- **3**: Fast, good balance (default)
- **5**: Slow, more stable

**Used in**: `AdaptiveStepRFE._determine_step_size()`

**Recommendation**: 3 (good balance)

---

#### `RFG_KFOLDS`

**Value**: `3` (integer)

**Purpose**: Number of CV folds for GridSearchCV hyperparameter tuning

**Range**: 2-5
- **2**: Very fast
- **3**: Fast (default)
- **5**: Slow but stable

**Used in**: `EnhancedAdaptiveRFE._optimize_hyperparameters()`

**Grid Evaluations**: `len(param_grid) × RFG_KFOLDS`
- At 27 param combinations × 3 folds = 81 model fits

**Recommendation**: 3 (balance speed and reliability)

---

## Data Split Configuration

### `TEST_PCT`

**Value**: `None` (optional float)

**Purpose**: Percentage of data for test set (if specified)

**Range**: 0.1-0.3 (10%-30%)
- **0.1-0.2**: Small test set, more training data
- **0.2-0.3**: Larger test set, more reliable metrics

**Current Behavior**: 
- Not used (custom train/test logic in `_set_base_dataframes()`)
- Test set determined by:
  - All at-risk infants with `category=2`
  - `N_NORM_TO_TEST` random normative infants

**Future Use**: May be used for synthetic data splits

---

### `N_NORM_TO_TEST`

**Value**: `20` (integer)

**Purpose**: Number of normative infants to sample for test set

**Range**: 10-50
- **10-20**: Small normative test set
- **20-30**: Moderate (default)
- **50+**: Large, may deplete training normative data

**Purpose**: 
- Validate that normative infants score low on surprise
- Compute specificity in test set

**Recommendation**: 20 (10-15% of normative total)

**Example**:
```python
# In constants.py
N_NORM_TO_TEST = 20

# Applied in BayesianData._set_base_dataframes()
test_norm = norm_df.sample(n=N_NORM_TO_TEST, seed=RAND_STATE)
```

---

## Path Configuration

### Base Data Path

#### `BASE_DATA_DIR`

**Value**: `/Volumes/secure/data/early_markers/cribsy`

**Purpose**: Root directory for all data files

**Structure**:
```
/Volumes/secure/data/early_markers/cribsy/
├── pkl/        # Pickle files (intermediate data)
├── xlsx/       # Excel reports
├── png/        # Plots (ROC curves, heatmaps)
├── csv/        # CSV exports
├── json/       # JSON exports
├── html/       # HTML reports
└── ipc/        # IPC files (synthetic data)
```

**Configuration**:
```python
from pathlib import Path

BASE_DATA_DIR = Path("/Volumes/secure/data/early_markers/cribsy")

# Subdirectories
PKL_DIR = BASE_DATA_DIR / "pkl"
XLSX_DIR = BASE_DATA_DIR / "xlsx"
PLOT_DIR = BASE_DATA_DIR / "png"
CSV_DIR = BASE_DATA_DIR / "csv"
JSON_DIR = BASE_DATA_DIR / "json"
HTML_DIR = BASE_DATA_DIR / "html"
IPC_DIR = BASE_DATA_DIR / "ipc"
```

**Machine-Specific**: **Yes** - must be updated for each machine

---

### Input Data Paths

#### Raw Data

**Path**: `/Volumes/secure/code/early-markers/early_markers/emmacp_metrics/features_merged.pkl`

**File**: `features_merged.pkl` (Pandas DataFrame)

**Configuration**:
```python
RAW_DATA_PATH = Path("/Volumes/secure/code/early-markers/early_markers/emmacp_metrics/features_merged.pkl")
```

---

## Performance Tuning

### Speed Optimization

**Priority**: Tuning order from highest to lowest impact

1. **Reduce RFE trials** (`RFE_N_TRIALS`):
   - 50 → 20: ~2.5× speedup
   - Minimal impact on final feature selection

2. **Reduce Random Forest trees** (`N_ESTIMATORS`):
   - 200 → 100: ~2× speedup
   - Minor accuracy impact

3. **Increase RFE step size** (`RFE_MAX_STEP`):
   - 5 → 10: ~1.5× speedup
   - May miss optimal feature subset

4. **Reduce CV folds** (`RKFOLD_SPLITS`, `RKFOLD_REPEATS`):
   - (3, 5) → (1, 3): ~5× speedup
   - Less stable estimates

5. **Simplify GridSearch** (`RFG_PARAM_GRID`):
   - 27 combinations → 9: ~3× speedup
   - May miss optimal hyperparameters

**Fast Development Profile**:
```python
# In constants.py (for development)
RFE_N_TRIALS = 20          # Default: 50
N_ESTIMATORS = 100         # Default: 200
RFE_MAX_STEP = 10          # Default: 5
RKFOLD_SPLITS = 3          # Default: 5
RKFOLD_REPEATS = 1         # Default: 3
RFG_PARAM_GRID = {
    'max_depth': [7],
    'min_samples_leaf': [2],
    'max_features': ['sqrt'],
}
```

**Estimated speedup**: 10-20× faster for full pipeline

---

### Memory Optimization

**High Memory Scenarios**:
- Many RFE trials (`RFE_N_TRIALS > 50`)
- High parallelization (`RFE_N_JOBS > 12`)
- Large feature sets (> 100 features)

**Memory Reduction Strategies**:

1. **Reduce parallel jobs**:
   ```python
   RFE_N_JOBS = 4  # Instead of 12
   N_JOBS = 4      # Instead of 8
   ```

2. **Reduce RFE trials**:
   ```python
   RFE_N_TRIALS = 30  # Instead of 50
   ```

3. **Use sequential processing**:
   ```python
   RFE_N_JOBS = 1
   N_JOBS = 1
   ```

4. **Process smaller feature subsets**:
   ```python
   # Filter features before RFE
   features_subset = FEATURES[:30]  # First 30 features only
   ```

**Memory Profiling**:
```bash
# Monitor memory during RFE
poetry run python -m memory_profiler script.py
```

---

## Parameter Sensitivity

### High Sensitivity Parameters

**Small changes have large impact**:

1. **`RFE_ALPHA`** (0.01 vs 0.05):
   - Change: 5× increase
   - Impact: ±10-15 features selected
   - Recommendation: Use 0.05 consistently

2. **`RFE_KEEP_PCT`** (0.8 vs 0.9):
   - Change: 10% increase
   - Impact: ±6 features (from 45 to 51)
   - Recommendation: Use 0.85-0.95

3. **`RAND_STATE`** (different seeds):
   - Change: Any change
   - Impact: Different train/test split → different results
   - Recommendation: **Never change**

---

### Low Sensitivity Parameters

**Large changes have minor impact**:

1. **`N_ESTIMATORS`** (100 vs 200):
   - Change: 2× increase
   - Impact: <1% accuracy difference
   - Runtime: 2× longer

2. **`RFE_N_TRIALS`** (30 vs 50):
   - Change: 1.67× increase
   - Impact: ±1-2 features selected
   - Runtime: 1.67× longer

3. **`RKFOLD_REPEATS`** (1 vs 3):
   - Change: 3× increase
   - Impact: More stable CV estimates (±0.01-0.02 in metrics)
   - Runtime: 3× longer

---

### Interaction Effects

**Parameter combinations that interact**:

1. **`RFE_ALPHA` × `RFE_N_TRIALS`**:
   - Lower alpha requires more trials for stable consensus
   - Recommended combinations:
     - α=0.01 → n_trials=100
     - α=0.05 → n_trials=50
     - α=0.10 → n_trials=30

2. **`RFE_KEEP_PCT` × `RFE_MAX_STEP`**:
   - Higher keep % requires smaller steps for precision
   - Recommended combinations:
     - keep=0.9 → max_step=5
     - keep=0.8 → max_step=10
     - keep=0.5 → max_step=15

3. **`N_ESTIMATORS` × `N_JOBS`**:
   - More trees benefit more from parallelization
   - Recommended combinations:
     - n_estimators=50 → n_jobs=1-4
     - n_estimators=200 → n_jobs=8
     - n_estimators=500 → n_jobs=-1

---

## Configuration Validation

### Runtime Checks

**Automatic validation** in `constants.py`:
```python
# Validate paths exist
assert BASE_DATA_DIR.exists(), f"Data directory not found: {BASE_DATA_DIR}"

# Validate parameter ranges
assert 0 < RFE_KEEP_PCT < 1, "RFE_KEEP_PCT must be in (0, 1)"
assert 0 < RFE_ALPHA < 1, "RFE_ALPHA must be in (0, 1)"
assert RFE_MIN_STEP <= RFE_MAX_STEP, "RFE_MIN_STEP must be ≤ RFE_MAX_STEP"
```

### Manual Validation

**Check configuration**:
```python
from early_markers.cribsy.common import constants

print(f"Random seed: {constants.RAND_STATE}")
print(f"RFE trials: {constants.RFE_N_TRIALS}")
print(f"Data path: {constants.BASE_DATA_DIR}")
print(f"Grid size: {len(constants.RFG_PARAM_GRID['max_depth']) * 
                      len(constants.RFG_PARAM_GRID['min_samples_leaf']) * 
                      len(constants.RFG_PARAM_GRID['max_features'])}")
```

---

## Configuration Profiles

### Production Profile

**Use for final analyses and publications**:
```python
RAND_STATE = 20250313
RFE_N_TRIALS = 50
RFE_ALPHA = 0.05
RFE_KEEP_PCT = 0.9
RFE_NOISE_RATIO = 0.1
N_ESTIMATORS = 200
RKFOLD_REPEATS = 3
RKFOLD_SPLITS = 5
RFE_N_JOBS = 12
N_JOBS = 8
```

**Expected Runtime**: 2-4 hours for full pipeline

---

### Development Profile

**Use for rapid iteration**:
```python
RAND_STATE = 20250313  # Keep same
RFE_N_TRIALS = 20      # Reduced from 50
RFE_ALPHA = 0.05       # Keep same
RFE_KEEP_PCT = 0.9     # Keep same
RFE_NOISE_RATIO = 0.1  # Keep same
N_ESTIMATORS = 100     # Reduced from 200
RKFOLD_REPEATS = 1     # Reduced from 3
RKFOLD_SPLITS = 3      # Reduced from 5
RFE_N_JOBS = 4         # Reduced from 12
N_JOBS = 4             # Reduced from 8
```

**Expected Runtime**: 15-30 minutes for full pipeline

---

### Reproducibility Profile

**Use for exact reproducibility across machines**:
```python
RAND_STATE = 20250313
RFE_N_TRIALS = 50
RFE_ALPHA = 0.05
RFE_KEEP_PCT = 0.9
RFE_NOISE_RATIO = 0.1
N_ESTIMATORS = 200
RKFOLD_REPEATS = 3
RKFOLD_SPLITS = 5
RFE_N_JOBS = 1         # Sequential only
N_JOBS = 1             # Sequential only
```

**Expected Runtime**: 8-12 hours (very slow)

**Guarantees**: Bit-exact reproducibility

---

## See Also

- [API.md](API.md) - API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [WORKFLOWS.md](WORKFLOWS.md) - Common patterns
- [DATA_FORMATS.md](DATA_FORMATS.md) - Data specifications
- `constants.py` - Source of all configuration values
