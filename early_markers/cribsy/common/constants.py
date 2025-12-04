"""Configuration constants for the early-markers Bayesian surprise analysis.

This module contains all configuration constants used throughout the early-markers
codebase, including:

- Reproducibility settings (random seeds)
- Random Forest hyperparameters
- RFE (Recursive Feature Elimination) configuration
- Cross-validation settings
- Data paths and directory structure
- Feature definitions
- Display formatting mappings

Constants are organized into functional groups:

1. **Reproducibility**: RAND_STATE ensures consistent results across runs
2. **Random Forest**: N_JOBS, N_ESTIMATORS, RFG_PARAM_GRID control RF behavior
3. **RFE Settings**: RFE_* constants control feature selection
4. **Cross-Validation**: RKFOLD_*, STEP_KFOLDS control CV strategy
5. **Data Paths**: *_DIR constants define output locations
6. **Features**: FEATURES list defines 56 movement features
7. **Display**: *_MAP and *_COLS constants control Excel/table formatting

Note:
    Data paths are hardcoded to /Volumes/secure/data/early_markers/cribsy.
    Update these constants if running on a different machine.

Important:
    RAND_STATE must be set before any stochastic operations to ensure
    reproducibility. This is done automatically in the cribsy.__init__ module.

Example:
    >>> from early_markers.cribsy.common.constants import RAND_STATE, FEATURES
    >>> import numpy as np
    >>> np.random.seed(RAND_STATE)
    >>> print(f"Using {len(FEATURES)} features")
    Using 56 features
"""
from pathlib import Path

# =============================================================================
# Reproducibility and Data Split Configuration
# =============================================================================

RAND_STATE = 20250313
"""int: Global random seed for reproducibility.

This seed is used throughout the codebase to ensure consistent results across
runs. Set in cribsy.__init__ for numpy and scikit-learn operations.

Value: 20250313 (YYYYMMDD format from project inception)
"""

N_NORM_TO_TEST = 40
"""int: Number of normative infants to sample for testing.

Defines how many infants with risk=0 are randomly selected from the training
set to be included in the test set. This ensures the test set has representation
of both risk categories.
"""

TEST_PCT = 0.2
"""float: Expected proportion of data reserved for testing.

Used in rough sample size estimates. Typical value is 0.2 (20% test, 80% train).
"""

# =============================================================================
# Random Forest Configuration
# =============================================================================

N_JOBS = 8
"""int: Number of parallel jobs for Random Forest training.

Note: Use n_jobs=1 in RandomForestRegressor for exact reproducibility.
""" 

N_ESTIMATORS = 200
"""int: Default number of trees in Random Forest models."""

RFG_PARAM_GRID = {
    'max_depth': [5, 7, 9],
    'min_samples_leaf': [1, 2, 3],
    'max_features': [1, 'sqrt', 'log2'],
}
"""dict: Grid search parameter space for Random Forest hyperparameter tuning.

Used by EnhancedAdaptiveRFE.GridSearchCV to find optimal RF parameters.

Parameters:
    - max_depth: Maximum tree depth (5, 7, or 9 levels)
    - min_samples_leaf: Minimum samples required at leaf nodes
    - max_features: Number/method for selecting features at each split
"""

RFG_N_JOBS = None
"""int or None: Parallel jobs for GridSearchCV (None = use all cores)."""

RFG_N_ESTIMATORS = 200
"""int: Number of estimators for GridSearchCV Random Forest."""

RFG_KFOLDS = 3
"""int: Number of CV folds for GridSearchCV."""

STEP_KFOLDS = 3
"""int: Number of CV folds for adaptive step size determination."""

NOISE_RATIO = 0.1
"""float: Proportion of noise features to inject (10%).

Used to test feature selection stability by adding random features.
"""

# =============================================================================
# RFE (Recursive Feature Elimination) Configuration  
# =============================================================================

RFE_MIN_STEP = 1
"""int: Minimum number of features to eliminate per RFE step."""

RFE_MAX_STEP = 5
"""int: Maximum number of features to eliminate per RFE step."""

RFE_KEEP_PCT = 0.9
"""float: Target proportion of features to retain (90%).

EnhancedAdaptiveRFE aims to keep approximately this percentage of features
while maximizing stability and performance.
"""

RFE_TOLERANCE = 0.01
"""float: Performance tolerance for adaptive step adjustment (1%).

If CV performance drops by more than this amount, reduce step size.
"""

RFE_N_TRIALS = 50
"""int: Number of parallel RFE trials for consensus.

More trials increase stability but require more computation.
Default 50 trials × 15 CV folds = 750 model fits.
"""

RFE_N_JOBS = 12
"""int: Number of parallel jobs for RFE trials."""

RFE_ALPHA = 0.05
"""float: Significance level for binomial test (5%).

Features must appear significantly more often than chance (p < 0.05)
to be considered stable.
"""

RFE_NOISE_RATIO = 0.1
"""float: Proportion of noise features for stability testing.

Same as NOISE_RATIO but specifically for RFE operations.
"""

MIN_K = 10
"""int: Minimum number of features to select."""

# =============================================================================
# Cross-Validation Configuration
# =============================================================================

RKFOLD_REPEATS = 3
"""int: Number of repeats for RepeatedKFold CV."""

RKFOLD_SPLITS = 5
"""int: Number of folds for RepeatedKFold CV.

Total folds = RKFOLD_REPEATS × RKFOLD_SPLITS = 15 folds.
"""

# =============================================================================
# Age Categorization
# =============================================================================

AGE_BRACKETS = {
    0: "all",
    1: "lt6",
    2: "ge6",
}
"""dict: Age bracket identifiers for stratified analysis."""

AGE_LABELS = {
    0: "All",
    1: "< 6 Weeks",
    2: ">= 6 Weeks",
}
"""dict: Human-readable age bracket labels."""

# =============================================================================
# Data Paths and Directory Structure
# =============================================================================
# Note: These paths are machine-specific. Update if running elsewhere.

# Get the project root directory (3 levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

ROOT_DIR = _PROJECT_ROOT / "early_markers" / "legacy"
"""Path: Root directory for raw movement metrics data."""

SURPRISE_DIR = ROOT_DIR / "surprise_data/interim"
"""Path: Directory for interim surprise calculation data."""

DATA_DIR = _PROJECT_ROOT / "data"
"""Path: Base output directory for all analysis results.

All subdirectories (*_DIR) are created automatically if they don't exist.
"""
DATA_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = DATA_DIR / "png"
"""Path: Directory for plot outputs (ROC curves, etc.)."""

XLSX_DIR = DATA_DIR / "xlsx"
"""Path: Directory for Excel report outputs."""

IPC_DIR = DATA_DIR / "ipc"
"""Path: Directory for Apache Arrow IPC files (synthetic data)."""

JSON_DIR = DATA_DIR / "json"
"""Path: Directory for JSON outputs."""

HTML_DIR = DATA_DIR / "html"
"""Path: Directory for HTML outputs."""

PKL_DIR = DATA_DIR / "pkl"
"""Path: Directory for pickle files (cached DataFrames)."""

CSV_DIR = DATA_DIR / "csv"
"""Path: Directory for CSV exports."""

# Create all output directories
for dir_path in [PLOT_DIR, XLSX_DIR, IPC_DIR, JSON_DIR, HTML_DIR, PKL_DIR, CSV_DIR]:
    dir_path.mkdir(exist_ok=True)

RAW_DATA = PKL_DIR / "features_merged_20251121_091511.pkl"
"""Path: Default raw data file path for real infant movement data."""
# =============================================================================
# Feature Definitions
# =============================================================================

FEATURES = [
    "age_in_weeks",
    "Ankle_IQRaccx",
    "Ankle_IQRaccy",
    "Ankle_IQRvelx",
    "Ankle_IQRvely",
    "Ankle_IQRx",
    "Ankle_IQRy",
    "Ankle_lrCorr_x",
    "Ankle_meanent",
    "Ankle_medianvelx",
    "Ankle_medianvely",
    "Ankle_medianx",
    "Ankle_mediany",
    "Ear_lrCorr_x",
    "Elbow_IQR_acc_angle",
    "Elbow_IQR_vel_angle",
    "Elbow_entropy_angle",
    "Elbow_lrCorr_angle",
    "Elbow_lrCorr_x",
    "Elbow_mean_angle",
    "Elbow_median_vel_angle",
    "Elbow_stdev_angle",
    "Eye_lrCorr_x",
    "Hip_IQR_acc_angle",
    "Hip_IQR_vel_angle",
    "Hip_entropy_angle",
    "Hip_lrCorr_angle",
    # "Hip_lrCorr_x",
    "Hip_mean_angle",
    "Hip_median_vel_angle",
    "Hip_stdev_angle",
    "Knee_IQR_acc_angle",
    "Knee_IQR_vel_angle",
    "Knee_entropy_angle",
    "Knee_lrCorr_angle",
    "Knee_lrCorr_x",
    "Knee_mean_angle",
    "Knee_median_vel_angle",
    "Knee_stdev_angle",
    "Shoulder_IQR_acc_angle",
    "Shoulder_IQR_vel_angle",
    "Shoulder_entropy_angle",
    "Shoulder_lrCorr_angle",
    # "Shoulder_lrCorr_x",
    "Shoulder_mean_angle",
    "Shoulder_median_vel_angle",
    "Shoulder_stdev_angle",
    "Wrist_IQRaccx",
    "Wrist_IQRaccy",
    "Wrist_IQRvelx",
    "Wrist_IQRvely",
    "Wrist_IQRx",
    "Wrist_IQRy",
    "Wrist_lrCorr_x",
    "Wrist_meanent",
    "Wrist_medianvelx",
    "Wrist_medianvely",
    "Wrist_medianx",
    "Wrist_mediany",
]
"""list[str]: Complete list of 56 infant movement features.

Feature Naming Convention:
    BodyPart_MetricName
    
    Example: "Ankle_IQRvelx" = Ankle Interquartile Range velocity x-axis

Body Parts (8 total):
    - Ankle, Wrist: Extremity movement
    - Knee, Elbow, Hip, Shoulder: Joint angles
    - Ear, Eye: Head/facial movement

Metric Types:
    - Position: medianx, mediany (median position)
    - Velocity: medianvelx, medianvely, IQRvelx, IQRvely
    - Acceleration: IQRaccx, IQRaccy
    - Angles: mean_angle, median_vel_angle, IQR_vel_angle, IQR_acc_angle, stdev_angle
    - Entropy: meanent, entropy_angle (movement complexity)
    - Correlation: lrCorr_x, lrCorr_angle (left-right symmetry)
    - Dispersion: IQRx, IQRy (interquartile range)
    - Meta: age_in_weeks

Note:
    Some features are commented out (Hip_lrCorr_x, Shoulder_lrCorr_x) due to
    high correlation with other features.

Example:
    >>> len(FEATURES)
    56
    >>> [f for f in FEATURES if 'Ankle' in f]
    ['Ankle_IQRaccx', 'Ankle_IQRaccy', ...]
"""

# FEATURE_SET_38: list[tuple[str, str]] = [
#     ('Ankle', 'medianx'),
#     ('Wrist', 'medianx'),
#     ('Ankle', 'mediany'),
#     ('Wrist', 'mediany'),
#     ('Knee', 'mean_angle'),
#     ('Elbow', 'mean_angle'),
#     ('Ankle', 'IQRx'),
#     ('Wrist', 'IQRx'),
#     ('Ankle', 'IQRy'),
#     ('Wrist', 'IQRy'),
#     ('Knee', 'stdev_angle'),
#     ('Elbow', 'stdev_angle'),
#     ('Ankle', 'medianvelx'),
#     ('Wrist', 'medianvelx'),
#     ('Ankle', 'medianvely'),
#     ('Wrist', 'medianvely'),
#     ('Knee', 'median_vel_angle'),
#     ('Elbow', 'median_vel_angle'),
#     ('Ankle', 'IQRvelx'),
#     ('Wrist', 'IQRvelx'),
#     ('Ankle', 'IQRvely'),
#     ('Wrist', 'IQRvely'),
#     ('Knee', 'IQR_vel_angle'),
#     ('Elbow', 'IQR_vel_angle'),
#     ('Ankle', 'IQRaccx'),
#     ('Wrist', 'IQRaccx'),
#     ('Ankle', 'IQRaccy'),
#     ('Wrist', 'IQRaccy'),
#     ('Knee', 'IQR_acc_angle'),
#     ('Elbow', 'IQR_acc_angle'),
#     ('Ankle', 'meanent'),
#     ('Wrist', 'meanent'),
#     ('Knee', 'entropy_angle'),
#     ('Elbow', 'entropy_angle'),
#     ('Ankle', 'lrCorr_x'),
#     ('Wrist', 'lrCorr_x'),
#     ('Knee', 'lrCorr_angle'),
#     ('Elbow', 'lrCorr_angle'),
# ]

# =============================================================================
# Display and Formatting Mappings
# =============================================================================
# These dictionaries map internal column names to human-readable labels for
# Excel reports and table displays.

METRIC_MAP = {
    "tpr_w_ci": "TPR [95% CI]",
    "fpr_w_ci": "FPR [95% CI]",
    "sens_w_ci": "Sensitivity [95% CI]",
    "spec_w_ci": "Specificity [95% CI]",
    "ppv_w_ci": "PPV [95% CI]",
    "npv_w_ci": "NPV [95% CI]",
    "f1_w_ci": "F1 [95% CI]",
    "acc_w_ci": "Accuracy [95% CI]",
    "rough_n_min": "Min N",
    "rough_n_max": "Max N" ,
    "threshold": "Threshold",
}

SUMMARY_MAP = {
    "threshold": "Threshold",
    # "model_name": "Model",
    "metrics_name": "Trial",
    # "rfe_name": "RFE",
    "k": "Features",
    "auc": "AUC",
    "j": "Youden's J",
} | METRIC_MAP

DETAIL_MAP = SUMMARY_MAP | {
    'roc_thresh': "ROC Threshold",
    "tp": "TP",
    "tn": "TN",
    "fp": "FP",
    "fn": "FN",
    "n": "N",
    "tpr": "TPR",
    "fpr": "FPR",
    "sens": "Sensitivity",
    "spec": "Specificity",
    "ppv": "PPV",
    "npv": "NPV",
    "f1": "F1",
    "acc": "Accuracy",
    "tpr_ci": "TPR CI",
    "fpr_ci": "FPR CI",
    "sens_ci": "Sensitivity CI",
    "spec_ci": "Specificity CI",
    "ppv_ci": "PPV CI",
    "npv_ci": "NPV CI",
    "f1_ci": "F1 CI",
    "acc_ci": "Accuracy CI",
    "rough_n_min": "Min N",
    "rough_n_max": "Max N",
}

SUMMARY_COLS = [
    # "model_name": "Model",
    "metrics_name",
    # "rfe_name",
    "k",
    "auc",
    "tpr_w_ci",
    "fpr_w_ci",
    "sens_w_ci",
    "spec_w_ci",
    "ppv_w_ci",
    "npv_w_ci",
    "f1_w_ci",
    "acc_w_ci",
    "threshold",
    "j",
    "rough_n_min",
    "rough_n_max",
]

DETAIL_COLS = [
    # "model_name": "Model",
    "metrics_name",
    # "rfe_name",
    "tp",
    "tn",
    "fp",
    "fn",
    "n",
    'tpr',
    'fpr',
    'sens',
    'spec',
    'ppv',
    'npv',
    'acc',
    'f1',
    'sens_ci',
    'spec_ci',
    'tpr_ci',
    'fpr_ci',
    'ppv_ci',
    'npv_ci',
    'acc_ci',
    'f1_ci',
    "auc",
    'j',
    "k",
    'threshold',
    'roc_thresh',
    'rough_n_min',
    'rough_n_max',
]

MSS_SENS_COLS = [
    "sensitivity",
    "ci_width",
    "prevalence",
    "n",
    "n_pos",
    "n_neg",
    "conf_level",
]

MSS_SENS_MAP = {
    "sensitivity": "Sensitivity",
    "ci_width": "CI Width",
    "prevalence": "Prevalence",
    "n": "Total N",
    "n_pos": "N Positives",
    "n_neg": "N Negatives",
    "conf_level": "Confidence Level",
}

MSS_SPEC_COLS = [
    "specificity",
    "ci_width",
    "prevalence",
    "n",
    "n_pos",
    "n_neg",
    "conf_level",
]

MSS_SPEC_MAP = {
    "specificity": "Specificity",
    "ci_width": "CI Width",
    "prevalence": "Prevalence",
    "n": "Total N",
    "n_pos": "N Positives",
    "n_neg": "N Negatives",
    "conf_level": "Confidence Level",
}

MSS_PPV_COLS = [
    "ppv",
    # "sensitivity",
    "ci_width",
    "prevalence",
    "n",
    "n_pos",
    "n_neg",
    "conf_level"
]

MSS_PPV_MAP = {
    "ppv": "PPV",
    # "sensitivity": "Sensitivity",
    "ci_width": "CI Width",
    "prevalence": "Prevalence",
    "n": "Total N",
    "n_pos": "N Positives",
    "n_neg": "N Negatives",
    "conf_level": "Confidence Level",
}

MSS_NPV_COLS = [
    'npv',
    'sensitivity',
    'specificity',
    'ci_width',
    'prevalence',
    'n',
    'n_pos',
    'n_neg',
    'conf_level'
]

MSS_NPV_MAP = {
    "ppv": "PPV",
    # "sensitivity": "Sensitivity",
    # "specificity": "Specificity",
    "ci_width": "CI Width",
    "prevalence": "Prevalence",
    "n": "Total N",
    "n_pos": "N Positives",
    "n_neg": "N Negatives",
    "conf_level": "Confidence Level",
}

MSS_F1_COLS = [
    "f1",
    # "ppv",
    # "sensitivity",
    # "specificity",
    "ci_width",
    "prevalence",
    "n",
    "n_pos",
    "n_neg",
    "conf_level"
]

MSS_F1_MAP = {
    "f1": "F1",
    # "ppv": "PPV",
    # "sensitivity": "Sensitivity",
    # "specificity": "Specificity",
    "ci_width": "CI Width",
    "prevalence": "Prevalence",
    "n": "Total N",
    "n_pos": "N Positives",
    "n_neg": "N Negatives",
    "conf_level": "Confidence Level",
}
