"""Bayesian surprise analysis for infant movement data.

Core module providing the BayesianData class - the main API for computing
Bayesian surprise metrics from movement features. Implements complete pipeline
from raw data loading through feature selection, surprise calculation, and
ROC analysis.

Key Components:
    BayesianData: Main class managing the complete analysis pipeline
    BayesianFrames: Container for related DataFrames (train/test/stats/surprise)
    BayesianRfeResult: Result of feature selection process
    BayesianRocResult: Complete ROC analysis results
    BayesianSurprise: Model surprise statistics (mean, SD of -log P)
    BayesianCI: Confidence interval analysis for metrics

Workflow:
    1. Load base data (pkl or ipc format)
    2. Split into training (category=1) and testing (category=2)
    3. Run feature selection via Enhanced Adaptive RFE
    4. Compute Bayesian surprise:
       - Fit reference distribution on normative training data
       - Calculate -log P(x | N(μ, σ²)) for each feature
       - Sum across features: S = Σ(-log P)
       - Standardize: z = (S - μ_train) / σ_train
    5. Compute ROC metrics from surprise z-scores
    6. Generate Excel reports and plots

Data Format:
    Long format:
        - infant: Unique infant ID
        - category: 1=training, 2=testing
        - risk: Binary risk label (0=normal, 1=at-risk)
        - feature: Feature name (e.g., 'Ankle_L_position_entropy')
        - value: Observed feature value

    Wide format (after pivoting):
        - Rows: Infants
        - Columns: Features
        - Values: Feature measurements

File Paths:
    - Input: PKL_DIR or IPC_DIR
    - Plots: PLOT_DIR (PNG format)
    - Reports: XLSX_DIR (Excel with formatting)
    - Intermediate: PKL_DIR (pickled BayesianData states)

Example:
    >>> from early_markers.cribsy.common.bayes import BayesianData
    >>>
    >>> # Initialize with real data
    >>> bd = BayesianData(base_file="features_merged.pkl")
    >>>
    >>> # Run complete pipeline
    >>> bd.run_rfe_on_base("rfe_90pct", features_to_keep_pct=0.9)
    >>> bd.run_surprise_with_rfe("model_trial", "rfe_90pct")
    >>> bd.run_metrics_from_surprise("metrics_trial", f"model_trial_k_{bd.rfe_k('rfe_90pct')}")
    >>>
    >>> # Access results
    >>> metrics_df = bd.metrics_df("metrics_trial")
    >>> print(f"AUC: {metrics_df['auc'][0]:.3f}")
    >>> print(f"Sensitivity @ Youden's J: {metrics_df['sensitivity'][0]:.3f}")
    >>>
    >>> # Generate Excel report
    >>> bd.write_xlsx_results("metrics_trial", "report_trial.xlsx")

See Also:
    adaptive_rfe.py: Feature selection implementation
    constants.py: Configuration and paths
    enums.py: RfeType enumeration

References:
    - Bayesian surprise: Itti & Baldi (2009) Vision Research
    - ROC analysis: Fawcett (2006) Pattern Recognition Letters
    - Youden's J: Youden (1950) Cancer
"""
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl
from polars import DataFrame
import pandas as pd
from scipy.stats import norm
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc, confusion_matrix
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import KFold, RepeatedKFold
from loguru import logger

import matplotlib.pyplot as plt

from xlsxwriter import Workbook
from xlsxwriter.utility import xl_col_to_name

from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE
from early_markers.cribsy.common.xlsx import set_workbook_formats
from early_markers.cribsy.common.constants import (
    PLOT_DIR,
    XLSX_DIR,
    IPC_DIR,
    RAW_DATA,
    RAND_STATE,
    SUMMARY_MAP,
    SUMMARY_COLS,
    METRIC_MAP,
    DETAIL_COLS,
    DETAIL_MAP,
    N_NORM_TO_TEST,
    TEST_PCT,
    # AGE_BRACKETS,
    RFE_N_TRIALS,
    RFE_ALPHA,
    RFE_KEEP_PCT,
    N_JOBS,
    N_ESTIMATORS,
)
from early_markers.cribsy.common.enums import RfeType

# ☑️
# PEB 2025.04.01 22:46 => Refactor BayesianData to take a test filename and compute ROC metrics:
#  1. Load file and turn into a polars dataframe
#  2. Compute surprise on infants
#  3. Run ROC metrics on test surprise df and report.

# PEB 2025.04.01 22:49 => Refactor BayesianData to persist lower and upper CI half-widths, and find minimum sample sizes to achieve target half-widths
#  1. ☑️ CIs are persisted as tuples in metrics df for each model in bd._metrics
#  2. ☑️ Define a BayesianCI class:
#     - ☑️ metric, value, target_half_width
#     - ☑️ lb, ub, lb_width, ub_width
#     - ☑️ lb_achieved, ub_achieved
#  3. foreach metric, create a BayesianCI instance

# PEB 2025.04.01 23:06 => write pipeline that uses synthetic data to achieve target CI widths for Sens and Spec
#  1. find minimum sample sizes
#  2. validate with real test data

@dataclass
class BayesianFrames:
    """Container for related DataFrames in Bayesian surprise analysis.

    Groups together the various DataFrame transformations for a single model,
    from raw data through statistics to final surprise scores.

    Attributes:
        train (DataFrame | None): Training data in wide format (infants × features).
            Only includes category=1 data (normative + sampled at-risk).
        test (DataFrame | None): Testing data in wide format (infants × features).
            Only includes category=2 data (held-out at-risk cases).
        stats (DataFrame | None): Reference statistics (mean, SD) per feature,
            computed from normative training data (risk=0, category=1).
        train_surprise (DataFrame | None): Surprise scores for training infants.
            Columns: infant, minus_log_p, z, p_value, risk.
        test_surprise (DataFrame | None): Surprise scores for testing infants.
            Columns: infant, minus_log_p, z, p_value, risk.

    Example:
        >>> frames = BayesianFrames(
        ...     train=train_df,  # 80 infants × 50 features
        ...     test=test_df,    # 20 infants × 50 features
        ...     stats=stats_df,  # 50 features × 2 (mean, std)
        ... )
        >>> frames.stats['mean'].head()
        [0.123, 0.456, ...]  # Mean per feature

    Note:
        - All DataFrames are Polars format
        - train/test use same feature subset (selected via RFE)
        - stats only computed on risk=0 data (normative reference)
    """
    train: DataFrame | None = None
    test: DataFrame | None = None
    stats: DataFrame | None = None
    train_surprise: DataFrame | None = None
    test_surprise: DataFrame | None = None
    # data: DataFrame | None = None
    # test_surprise: DataFrame | None = None

@dataclass
class BayesianRfeResult:
    """Result of Recursive Feature Elimination (RFE) process.

    Stores the selected feature subset after Enhanced Adaptive RFE with
    statistical significance testing.

    Attributes:
        name (str): Unique identifier for this RFE run (e.g., 'rfe_90pct').
        k (int): Number of features selected (length of features list).
        features (list[str]): Names of significant features that passed
            binomial test (p < alpha). Order is not meaningful.

    Example:
        >>> rfe_result = BayesianRfeResult(
        ...     name="rfe_trial_1",
        ...     k=45,
        ...     features=["Ankle_L_position_entropy", "Wrist_R_velocity_mean", ...]
        ... )
        >>> print(f"{rfe_result.k} features selected in {rfe_result.name}")
        45 features selected in rfe_trial_1

    Note:
        - Features are selected via binomial test across 750 CV folds (50 trials × 15)
        - k is always equal to len(features) - redundant but convenient
        - name is used as dictionary key in BayesianData._rfes

    See Also:
        EnhancedAdaptiveRFE: Class that generates these results
        BayesianData.run_rfe_on_base(): Method that creates these results
    """
    name: str
    k: int
    features: list[str]

@dataclass
class BayesianRocResult:
    """Complete ROC analysis results for Bayesian surprise model.

    Contains ROC curve analysis, optimal threshold (Youden's J), confusion matrix
    metrics with confidence intervals, and visualization outputs.

    Attributes:
        model_name (str): Unique model identifier, format: "{prefix}_k_{num_features}".
        train_n (int): Number of training samples used.
        test_n (int): Number of test samples evaluated.
        features (list[str]): Feature subset used in this model.
        plot_file (str): Absolute path to saved ROC curve plot (PNG).
        auc (float): Area Under ROC Curve, range [0, 1].
        youdens_j (float): Youden's J statistic (sens + spec - 1), range [-1, 1].
            Maximum J identifies optimal threshold.
        roc_youdens_j (float): ROC curve value at Youden's J point (unused, kept for compatibility).
        threshold_j (float): Surprise z-score threshold at Youden's J maximum.
        metrics (DataFrame): Polars DataFrame with columns:
            - sensitivity, specificity, ppv, npv, accuracy, f1, auc
            - Each with 95% Wilson score confidence intervals as tuples
        primitives (DataFrame): Raw confusion matrix elements:
            - tp, tn, fp, fn (true/false positives/negatives)
        rough_n_min (int): Approximate minimum N from sample size estimation.
        rough_n_max (int): Approximate maximum N from sample size estimation.

    Example:
        >>> roc_result = bd._metrics["metrics_trial"]
        >>> print(f"AUC: {roc_result.auc:.3f}")
        AUC: 0.847
        >>> print(f"Optimal threshold (z-score): {roc_result.threshold_j:.2f}")
        Optimal threshold (z-score): 1.85
        >>> print(f"Sensitivity @ J: {roc_result.metrics['sensitivity'][0]:.3f}")
        Sensitivity @ J: 0.824

    Note:
        - Youden's J maximizes sensitivity + specificity simultaneously
        - threshold_j is a z-score (standardized surprise)
        - Confidence intervals use Wilson score method (better for proportions)
        - Plot saved automatically to PLOT_DIR

    See Also:
        BayesianData.run_metrics_from_surprise(): Generates these results
        BayesianCI: Extracts CI information from metrics DataFrame
    """
    model_name: str
    # rfe_name: str
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

@dataclass
class BayesianSurprise:
    """Normalization statistics for Bayesian surprise z-scores.

    Stores the training distribution parameters used to standardize surprise
    scores (sum of -log P across features) into z-scores.

    Attributes:
        model_name (str): Unique model identifier, format: "{prefix}_k_{num_features}".
        k (int): Number of features used in this model (same as in model_name).
        mean_neg_log_p (float): Mean of Σ(-log P) computed on training data.
            Used as μ in z-score formula: z = (S - μ) / σ.
        sd_neg_log_p (float): Standard deviation of Σ(-log P) on training data.
            Used as σ in z-score formula.

    Formula:
        For each infant i with feature values x_i = [x_i1, ..., x_ik]:
        1. Compute: S_i = Σ_j [-log P(x_ij | N(μ_j, σ_j²))]
        2. Standardize: z_i = (S_i - mean_neg_log_p) / sd_neg_log_p

    Example:
        >>> surprise_stats = BayesianSurprise(
        ...     model_name="model_trial_k_45",
        ...     k=45,
        ...     mean_neg_log_p=52.3,
        ...     sd_neg_log_p=8.7
        ... )
        >>> # Use for standardization
        >>> raw_surprise = 68.5  # Σ(-log P) for new infant
        >>> z_score = (raw_surprise - surprise_stats.mean_neg_log_p) / surprise_stats.sd_neg_log_p
        >>> z_score
        1.86...

    Note:
        - Statistics computed only on training data (category=1)
        - Applied to both training and testing data for standardization
        - Higher z-scores indicate more surprising (deviant) patterns
        - k is redundant with model_name but convenient for access

    See Also:
        BayesianData.run_surprise_with_rfe(): Computes these statistics
        BayesianFrames.train_surprise: Contains resulting z-scores
    """
    model_name: str
    k: int
    # features: list[str]
    # train: DataFrame
    # test: DataFrame
    mean_neg_log_p: float
    sd_neg_log_p: float

@dataclass
class BayesianCI:
    """Confidence interval analysis for diagnostic metrics.

    Analyzes whether observed confidence interval widths meet target precision
    requirements. Used for sample size adequacy assessment.

    Attributes:
        metric (str): Metric name (e.g., 'sensitivity', 'specificity').
        value (float): Point estimate of the metric, range [0, 1].
        half_width (float): Target maximum half-width for CI (e.g., 0.05 for ±0.05).
        train_n (int): Training sample size used.
        test_n (int): Test sample size used.
        lb (float): Lower bound of 95% Wilson score confidence interval.
        ub (float): Upper bound of 95% Wilson score confidence interval.
        lb_width (float): Distance from value to lb (computed). Should be ≤ half_width.
        ub_width (float): Distance from ub to value (computed). Should be ≤ half_width.
        lb_achieved (bool): Whether lower bound meets precision target (computed).
        ub_achieved (bool): Whether upper bound meets precision target (computed).

    Example:
        >>> # Target: Sensitivity 0.85 ± 0.05 (half_width=0.05)
        >>> ci = BayesianCI(
        ...     metric="sensitivity",
        ...     value=0.847,
        ...     lb=0.78,
        ...     ub=0.91,
        ...     half_width=0.05,
        ...     train_n=100,
        ...     test_n=50
        ... )
        >>> ci.lb_width  # 0.847 - 0.78 = 0.067
        0.067
        >>> ci.ub_width  # 0.91 - 0.847 = 0.063
        0.063
        >>> ci.lb_achieved  # 0.067 > 0.05
        False
        >>> ci.ub_achieved  # 0.063 > 0.05
        False
        >>> # Need larger sample size!

    Note:
        - Uses Wilson score intervals (better than Wald for proportions)
        - CIs can be asymmetric (lb_width ≠ ub_width)
        - Both bounds must achieve target for adequate precision
        - Computed fields set in __init__, not via dataclass

    See Also:
        BayesianData.get_cis_for_metrics(): Extracts CIs from metrics DataFrame
        bam.py: Bayesian approach for sample size planning
        metrics_n.py: Frequentist sample size estimation
    """
    metric: str
    value: float
    half_width: float
    train_n: int
    test_n: int
    lb: float
    ub: float
    lb_width: float = field(init=False)
    ub_width: float = field(init=False)
    lb_achieved: float = field(init=False)
    ub_achieved: float = field(init=False)

    def __init__(self, metric: str, value: float, lb: float, ub: float, half_width: float, train_n: int, test_n: int):
        self.metric = metric
        self.value = value
        self.half_width = half_width
        self.lb = lb
        self.ub = ub
        self.train_n = train_n
        self.test_n = test_n
        self.lb_width = self.value - self.lb
        self.ub_width = self.ub - self.value
        self.lb_achieved = self.lb_width <= self.half_width
        self.ub_achieved = self.ub_width <= self.half_width


@dataclass
class BayesianData:
    """Main API for Bayesian surprise analysis on infant movement data.

    Manages complete pipeline from data loading through feature selection,
    surprise calculation, and ROC analysis. Maintains state across multiple
    models and RFE runs.

    Workflow:
        1. Initialize with base data file (pkl or ipc format)
        2. Run RFE: `run_rfe_on_base(rfe_name, features_to_keep_pct=0.9)`
        3. Compute surprise: `run_surprise_with_rfe(model_prefix, rfe_name)`
        4. Compute metrics: `run_metrics_from_surprise(metrics_name, model_name)`
        5. Generate reports: `write_xlsx_results(metrics_name, filename)`

    State Management:
        Internal dictionaries store results keyed by names:
        - _rfes[rfe_name]: Feature selection results
        - _frames[model_name]: Data and statistics for each model
        - _surprise[model_name]: Normalization parameters
        - _metrics[metrics_name]: ROC analysis results

    Attributes:
        base_file (str | None): Path to base dataset. Extensions:
            - '.pkl': Real data (Pandas pickle)
            - '.ipc': Synthetic data (Polars IPC format)
            If None, defaults to 'features_merged.pkl'.
        base_model_name (str | None): Optional base model identifier (unused).
        base_features (list[str] | None): Optional feature list (unused).
        synthetic (bool): Auto-detected from file extension.
        augmented (bool): Whether data augmentation was applied.
        _base (DataFrame): Complete dataset in long format.
        _base_train (DataFrame): Training subset (category=1) in wide format.
        _base_test (DataFrame): Test subset (category=2) in wide format.
        _frames (dict[str, BayesianFrames]): Model data and statistics.
        _rfes (dict[str, BayesianRfeResult]): RFE results by name.
        _surprise (dict[str, BayesianSurprise]): Surprise normalization stats.
        _metrics (dict[str, BayesianRocResult]): ROC analysis results.

    Data Format:
        Long format (_base):
            Columns: infant, category, risk, feature, value

        Wide format (_base_train, _base_test):
            Rows: infants
            Columns: features
            Values: measurements

    Model Naming:
        Models created via run_surprise_with_rfe() use format:
        "{model_prefix}_k_{num_features}"

        Example: "trial_1_k_45" indicates 45 features

    Example:
        >>> from early_markers.cribsy.common.bayes import BayesianData
        >>>
        >>> # Initialize with real data
        >>> bd = BayesianData(base_file="features_merged.pkl")
        >>> print(f"Loaded {len(bd.base)} samples")
        Loaded 1250 samples
        >>>
        >>> # Run complete pipeline
        >>> bd.run_rfe_on_base("rfe_90", features_to_keep_pct=0.9)
        >>> print(f"Selected {bd.rfe_k('rfe_90')} features")
        Selected 50 features
        >>>
        >>> bd.run_surprise_with_rfe("model_trial", "rfe_90")
        >>> model_name = f"model_trial_k_{bd.rfe_k('rfe_90')}"
        >>> print(f"Created model: {model_name}")
        Created model: model_trial_k_50
        >>>
        >>> bd.run_metrics_from_surprise("metrics_trial", model_name)
        >>> metrics = bd.metrics_df("metrics_trial")
        >>> print(f"AUC: {metrics['auc'][0]:.3f}")
        AUC: 0.847
        >>>
        >>> # Generate Excel report
        >>> bd.write_xlsx_results("metrics_trial", "report.xlsx")
        >>>
        >>> # Access intermediate results
        >>> features_used = bd.rfe_features("rfe_90")
        >>> surprise_stats = bd._surprise[model_name]
        >>> print(f"Mean surprise: {surprise_stats.mean_neg_log_p:.2f}")
        Mean surprise: 52.34

    Advanced Usage:
        >>> # Run multiple RFE configurations
        >>> for pct in [0.8, 0.9, 1.0]:
        ...     rfe_name = f"rfe_{int(pct*100)}"
        ...     bd.run_rfe_on_base(rfe_name, features_to_keep_pct=pct)
        ...     k = bd.rfe_k(rfe_name)
        ...     print(f"{rfe_name}: {k} features")
        rfe_80: 45 features
        rfe_90: 50 features
        rfe_100: 56 features
        >>>
        >>> # Compare models
        >>> for rfe_name in ["rfe_80", "rfe_90", "rfe_100"]:
        ...     bd.run_surprise_with_rfe(f"model_{rfe_name}", rfe_name)
        ...     model = f"model_{rfe_name}_k_{bd.rfe_k(rfe_name)}"
        ...     bd.run_metrics_from_surprise(f"metrics_{rfe_name}", model)
        ...     auc = bd.metrics_df(f"metrics_{rfe_name}")['auc'][0]
        ...     print(f"{rfe_name}: AUC={auc:.3f}")
        rfe_80: AUC=0.834
        rfe_90: AUC=0.847
        rfe_100: AUC=0.852

    Persistence:
        >>> # Save state
        >>> bd.to_pickle("analysis_state.pkl")
        >>>
        >>> # Load state
        >>> bd_loaded = BayesianData.from_pickle("analysis_state.pkl")
        >>> bd_loaded.metrics_names
        ['metrics_rfe_80', 'metrics_rfe_90', 'metrics_rfe_100']

    Note:
        - All DataFrames use Polars format internally
        - Random seed (RAND_STATE) ensures reproducibility
        - RFE runs can take 30-60 minutes (parallelized)
        - Excel reports include formatted tables and confidence intervals

    See Also:
        EnhancedAdaptiveRFE: Feature selection implementation
        BayesianFrames: Data container structure
        BayesianRocResult: ROC analysis output
        constants.py: Configuration and paths

    References:
        - Bayesian surprise: Itti & Baldi (2009) Vision Research
        - Youden's J: Youden (1950) Cancer
        - Wilson score intervals: Wilson (1927) JASA
    """
    base_file: str | None
    base_model_name: str | None = None
    base_features: list[str] | None = None
    synthetic: bool | None = field(default=None, init=False)
    # frames = dict[int, dict[str, BayesianFrames]]  # [age_bracket][model_name]
    _base: DataFrame = field(init=False)
    _base_train: DataFrame = field(default=None, init=False)
    _base_test: DataFrame = field(default=None, init=False)
    _frames: dict[str, BayesianFrames] | None = field(default=None, init=False)  # [age_bracket][model_name]
    _rfes: dict[str, BayesianRfeResult] | None = field(default=None, init=False)  # [age_bracket][rfe_name]
    _surprise: dict[str, BayesianSurprise] | None = field(default=None, init=False)  # [age_bracket][model_name]
    _metrics: dict[str, BayesianRocResult] | None = field(default=None, init=False)  # [age_bracket][model_name]

    def __init__(self, base_file: str | None = None, train_n: int | None = None, test_n: int | None = None, augment: bool = False):
        self.augmented: bool = augment
        if base_file is None:
            self.base_file = "features_merged.pkl"
            self.synthetic = False
        elif base_file.lower().endswith(".pkl"):
            self.base_file = base_file
            self.synthetic = False
        elif base_file.lower().endswith(".ipc"):
            self.base_file = base_file
            self.synthetic = True
        else:
            raise ValueError(f"Parameter base_file must be None or have extension .pkl or .ipc")

        self._set_base_dataframes(train_n=train_n, test_n=test_n)
        self._set_data_frames()

    @property
    def base(self) -> DataFrame | None:
        """Complete base dataset in long format.

        Returns:
            DataFrame: Long format with columns: infant, category, risk, feature, value.
                Contains all data (training + testing) before RFE filtering.

        Example:
            >>> bd = BayesianData()
            >>> bd.base.shape
            (70000, 5)  # 1250 samples × 56 features
            >>> bd.base.columns
            ['infant', 'category', 'risk', 'feature', 'value']
        """
        return self._base

    @property
    def base_train(self) -> DataFrame | None:
        """Training subset in long format (category=1).

        Returns:
            DataFrame: Training data including normative (risk=0) and
                sampled at-risk cases (risk=1, category=1).

        Example:
            >>> bd.base_train.filter(pl.col('risk') == 0).shape
            (56000, 5)  # Normative training data
        """
        return self._base_train

    @property
    def base_test(self) -> DataFrame | None:
        """Test subset in long format (category=2).

        Returns:
            DataFrame: Held-out test data, primarily at-risk cases (risk=1).

        Example:
            >>> bd.base_test.filter(pl.col('risk') == 1).shape
            (14000, 5)  # At-risk test cases
        """
        return self._base_test

    @property
    def base_wide(self) -> DataFrame:
        """Complete base dataset in wide format.

        Returns:
            DataFrame: Wide format with infant, category, risk + feature columns.
                Values rounded to 4 decimal places.

        Example:
            >>> bd.base_wide.shape
            (1250, 59)  # 1250 infants × (3 meta + 56 features)
            >>> bd.base_wide.columns[:5]
            ['infant', 'category', 'risk', 'Ankle_L_position_entropy', ...]
        """
        return self._base.pivot(
            index=['infant', 'category', 'risk'],  on=['feature'], values='value'
        ).with_columns(
            pl.col(pl.Float64).round(4)
        )

    @property
    def base_train_wide(self) -> DataFrame:
        """Training subset in wide format.

        Returns:
            DataFrame: Training data (category=1) with features as columns.

        Example:
            >>> bd.base_train_wide.shape
            (1000, 59)  # 1000 training infants
        """
        return self._base_train.pivot(
            index=['infant', 'category', 'risk'],  on=['feature'], values='value'
        ).with_columns(
            pl.col(pl.Float64).round(4)
        )

    @property
    def base_test_wide(self) -> DataFrame:
        """Test subset in wide format.

        Returns:
            DataFrame: Test data (category=2) with features as columns.

        Example:
            >>> bd.base_test_wide.shape
            (250, 59)  # 250 test infants
        """
        return self._base_test.pivot(
            index=['infant', 'category', 'risk'],  on=['feature'], values='value'
        ).with_columns(
            pl.col(pl.Float64).round(4)
        )

    @property
    def model_names(self):
        """List of available model names.

        Returns:
            list[str] | None: Model names in format "{prefix}_k_{num_features}".

        Example:
            >>> bd.model_names
            ['model_trial_k_45', 'model_trial_k_50']
        """
        return list(self._frames.keys()) if self._frames is not None else None

    @property
    def metrics_names(self):
        """List of available metrics result names.

        Returns:
            list[str] | None: Metrics names as provided to run_metrics_from_surprise().

        Example:
            >>> bd.metrics_names
            ['metrics_trial_1', 'metrics_trial_2']
        """
        return list(self._metrics.keys()) if self._frames is not None else None

    # def reference(self, model_name: str) -> DataFrame | None:
    #     frames = self._frames.get(model_name) if self._frames is not None else None
    #     return frames.get("reference", None) if frames is not None else None

    def statistics(self, model_name: str) -> DataFrame | None:
        """Get reference statistics for a model.

        Args:
            model_name (str): Model identifier (e.g., 'model_trial_k_45').

        Returns:
            DataFrame | None: Statistics (mean, std) per feature from normative
                training data (risk=0). Columns: feature, mean, std.

        Example:
            >>> stats = bd.statistics('model_trial_k_45')
            >>> stats.shape
            (45, 3)  # 45 features
            >>> stats.head()
            shape: (5, 3)
            ┌─────────────────┬─────────┬────────┐
            │ feature         │ mean    │ std    │
            │ str             │ f64     │ f64    │
            ╞═════════════════╪═════════╪════════╡
            │ Ankle_L_pos...  │ 0.1234  │ 0.0567 │
            └─────────────────┴─────────┴────────┘
        """
        frames = self._frames.get(model_name) if self._frames is not None else None
        return frames.stats if frames is not None else None
    #
    # def data(self, model_name: str) -> DataFrame | None:
    #     frames = self._frames.get(model_name) if self._frames is not None else None
    #     return frames.data if frames is not None else None

    def train_surprise(self, model_name: str) -> DataFrame | None:
        """Get surprise scores for training data.

        Args:
            model_name (str): Model identifier.

        Returns:
            DataFrame | None: Training surprise scores with columns:
                infant, minus_log_p, z, p_value, risk.

        Example:
            >>> train_surp = bd.train_surprise('model_trial_k_45')
            >>> train_surp.select(['infant', 'z', 'risk']).head(3)
            ┌────────┬────────┬──────┐
            │ infant │ z      │ risk │
            ├────────┼────────┼──────┤
            │ inf_01 │ -0.23  │ 0    │
            │ inf_02 │  1.85  │ 1    │
            │ inf_03 │  0.12  │ 0    │
            └────────┴────────┴──────┘
        """
        frames = self._frames.get(model_name) if self._frames is not None else None
        return frames.train_surprise if frames is not None else None

    def test_surprise(self, model_name: str) -> DataFrame | None:
        """Get surprise scores for test data.

        Args:
            model_name (str): Model identifier.

        Returns:
            DataFrame | None: Test surprise scores with columns:
                infant, minus_log_p, z, p_value, risk.

        Example:
            >>> test_surp = bd.test_surprise('model_trial_k_45')
            >>> test_surp.filter(pl.col('z') > 2).shape[0]
            45  # 45 infants with z > 2 (highly surprising)
        """
        frames = self._frames.get(model_name) if self._frames is not None else None
        return frames.test_surprise if frames is not None else None

    def features(self, model_name: str) -> list[str] | None:
        """Get feature list for a model.

        Args:
            model_name (str): Model identifier.

        Returns:
            list[str] | None: Feature names used in this model.

        Example:
            >>> features = bd.features('model_trial_k_45')
            >>> len(features)
            45
            >>> features[:3]
            ['Ankle_L_position_entropy', 'Wrist_R_velocity_mean', ...]
        """
        frames = self._frames.get(model_name) if self._frames is not None else None
        frame = frames.stats if frames is not None else None
        return frame.select("feature").unique().to_series().to_list() if frame is not None else None

    def rfe(self, rfe_name: str) -> BayesianRfeResult | None:
        """Get RFE result by name.

        Args:
            rfe_name (str): RFE identifier (e.g., 'rfe_90pct').

        Returns:
            BayesianRfeResult | None: RFE result with name, k, and features.

        Example:
            >>> rfe = bd.rfe('rfe_90pct')
            >>> rfe.k
            50
            >>> len(rfe.features)
            50
        """
        return self._rfes.get(rfe_name) if self._rfes is not None else None

    def rfe_k(self, rfe_name):
        """Get number of features selected by RFE.

        Args:
            rfe_name (str): RFE identifier.

        Returns:
            int | None: Number of features selected.

        Example:
            >>> bd.rfe_k('rfe_90pct')
            50
        """
        rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
        return rfe.k if rfe is not None else None

    def rfe_df(self, rfe_name):
        rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
        return rfe.df if rfe is not None else None

    def rfe_x(self, rfe_name):
        rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
        return rfe.x if rfe is not None else None

    def rfe_y(self, rfe_name):
        rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
        return rfe.y if rfe is not None else None

    def rfe_x_sel(self, rfe_name):
        rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
        return rfe.x_sel if rfe is not None else None
    #
    # def rfe_selector(self, rfe_name):
    #     rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
    #     return rfe.selector if rfe is not None else None

    def rfe_features(self, rfe_name):
        """Get features selected by RFE.

        Args:
            rfe_name (str): RFE identifier.

        Returns:
            list[str] | None: Selected feature names.

        Example:
            >>> features = bd.rfe_features('rfe_90pct')
            >>> len(features)
            50
            >>> 'Ankle_L_position_entropy' in features
            True
        """
        rfe = self._rfes.get(rfe_name) if self._rfes is not None else None
        return rfe.features if rfe is not None else None

    def metrics(self, metrics_name: str) -> BayesianRocResult | None:
        """Get complete metrics result by name.

        Args:
            metrics_name (str): Metrics identifier from run_metrics_from_surprise().

        Returns:
            BayesianRocResult | None: Complete ROC analysis results including
                AUC, Youden's J, thresholds, confusion matrix metrics, and plot.

        Example:
            >>> metrics = bd.metrics('metrics_trial')
            >>> metrics.auc
            0.847
            >>> metrics.youdens_j
            0.674
            >>> metrics.threshold_j
            1.85
        """
        metrics = self._metrics.get(metrics_name) if self._metrics is not None else None
        return metrics

    def metrics_rfe_name(self, metrics_name: str):
        metrics = self._metrics.get(metrics_name) if self._metrics is not None else None
        return metrics.rfe_name if metrics is not None else None

    def metrics_features(self, metrics_name: str) -> list[str] | None:
        """Get features used in metrics calculation.

        Args:
            metrics_name (str): Metrics identifier.

        Returns:
            list[str] | None: Feature names used for this model.

        Example:
            >>> features = bd.metrics_features('metrics_trial')
            >>> len(features)
            45
        """
        metrics = self._metrics.get(metrics_name) if self._metrics is not None else None
        return metrics.features if metrics is not None else None

    def metrics_plot_file(self, metrics_name: str) -> str | None:
        """Get path to ROC curve plot.

        Args:
            metrics_name (str): Metrics identifier.

        Returns:
            str | None: Absolute path to saved ROC curve PNG file.

        Example:
            >>> plot_path = bd.metrics_plot_file('metrics_trial')
            >>> plot_path
            '/Volumes/secure/data/early_markers/cribsy/png/roc_model_trial_k_45.png'
        """
        metrics = self._metrics.get(metrics_name) if self._metrics is not None else None
        return metrics.plot_file if metrics is not None else None

    def metrics_df(self, metrics_name: str) -> DataFrame | None:
        """Get metrics DataFrame with confidence intervals.

        Args:
            metrics_name (str): Metrics identifier.

        Returns:
            DataFrame | None: Single-row DataFrame with columns:
                - sensitivity, specificity, ppv, npv, accuracy, f1, auc
                - Each metric has 95% Wilson score confidence interval as tuple

        Example:
            >>> df = bd.metrics_df('metrics_trial')
            >>> df['sensitivity'][0]
            0.824
            >>> df['sensitivity_ci'][0]  # (lower, upper)
            (0.78, 0.87)
            >>> df.columns
            ['sensitivity', 'sensitivity_ci', 'specificity', 'specificity_ci', ...]
        """
        metrics = self._metrics.get(metrics_name) if self._metrics is not None else None
        return metrics.metrics if metrics is not None else None

    def _set_base_dataframes(self, train_n: int | None = None, test_n: int | None = None, reload: bool = False):
        """Load and preprocess base dataset from file.

        Internal method called during __init__() to load data and create
        training/test splits. Handles both real (.pkl) and synthetic (.ipc) data.

        Args:
            train_n (int, optional): Limit training samples to N (random sample).
            test_n (int, optional): Limit test samples to N (random sample).
            reload (bool, optional): Force reload even if already loaded.

        Data Processing (Real Data):
            1. Load raw Pandas pickle from RAW_DATA path
            2. Convert to Polars, filter invalid cases
            3. Transform risk labels: risk_raw ≤ 1 → 0, risk_raw > 1 → 1
            4. Assign categories:
               - category=1: Training (normative + sampled at-risk)
               - category=2: Testing (held-out at-risk)
            5. Create feature names: "{part}_{feature_name}"
            6. Add age_in_weeks as additional feature
            7. Sample N_NORM_TO_TEST normative cases for test set
            8. Optionally stack with synthetic augmentation data

        Data Processing (Synthetic Data):
            1. Load Polars IPC file directly from IPC_DIR
            2. Assume pre-processed format (infant, category, risk, feature, value)

        Sets Attributes:
            self._base (DataFrame): Complete dataset in long format
            self._base_train (DataFrame): Training subset (category=1)
            self._base_test (DataFrame): Test subset (category=2)
            self.base_features (list[str]): List of all feature names

        Example:
            >>> bd = BayesianData()  # Calls _set_base_dataframes() internally
            >>> bd._base.shape
            (70000, 5)
            >>> bd._base_train.select('infant').unique().shape[0]
            1000  # 1000 training infants
            >>> bd._base_test.select('infant').unique().shape[0]
            250   # 250 test infants

        Note:
            - Uses RAND_STATE for reproducible sampling
            - Filters out part='umber' and infant='clin_100_6' (known bad data)
            - category transformation ensures normative data (risk=0) goes to training
            - N_NORM_TO_TEST (e.g., 20) normative cases moved to test for validation

        Raises:
            FileNotFoundError: If base_file doesn't exist at expected path.
        """
        if self.synthetic:
            self._base = pl.read_ipc(IPC_DIR / self.base_file)
        else:
            pd_raw = pd.read_pickle(RAW_DATA)
            df = (
                pl.from_pandas(pd_raw)
                    .filter(pl.col("part") != "umber", pl.col("infant") != "clin_100_6")
                    .rename({"risk": "risk_raw", "Value": "value"})
                    .with_columns(
                        risk=pl.when(pl.col("risk_raw") <= 1).then(0)
                        .otherwise(1),
                    ).with_columns(
                        category=pl.when((pl.col("category") == 0) | (pl.col("risk") == 0)).then(1)
                        .otherwise(2),
                        feature=pl.concat_str(pl.col("part"), pl.col("feature_name"), separator="_")
                    ).drop(["part", "feature_name"])
                )

            df2 = (
                df.select(["infant", "category", "risk", "age_in_weeks"])
                .unique()
                .with_columns(
                    feature=pl.lit("age_in_weeks"),
                    value=pl.col("age_in_weeks")
                )
            ).drop("age_in_weeks")

            df3 = df.select(["infant", "category", "risk", "feature", "value"]).vstack(df2).sort(["infant", "feature"])

            randos = df3.select("infant", "risk", "category").unique().filter(pl.col("category") == 1).sample(n=N_NORM_TO_TEST, seed=RAND_STATE)
            df4 = df3.with_columns(
                category = pl.when((pl.col("category") == 2) | (pl.col("infant").is_in(randos.select("infant")))).then(2).otherwise(1)
            )
            if self.augmented:
                df_real = pl.read_ipc(IPC_DIR / self.base_file)
                self._base = df_real.vstack(df4)
            else:
                self._base = df4

        df_wide = self._base.pivot(
            on=["feature"],
            index=["infant", "category", "risk"],
            values="value"
        )

        self.base_features = [
            col for col in df_wide.columns
            if not col in ["infant", "category", "risk"]
        ]

        self._base_train = self._base.filter(pl.col("category") == 1)
        if train_n is not None:
            train_wide = self._base_train.pivot(
                on=["feature"],
                index=["infant", "category", "risk"],
                values="value"
            ).sample(n=train_n, seed=RAND_STATE)
            self._base_train = train_wide.unpivot(index=["infant", "category", "risk"], variable_name="feature", value_name="value")

        self._base_test = self._base.filter(pl.col("category") == 2)
        if test_n is not None:
            test_wide = self._base_test.pivot(
                on=["feature"],
                index=["infant", "category", "risk"],
                values="value"
            ).sample(n=test_n, seed=RAND_STATE)
            self._base_test = test_wide.unpivot(index=["infant", "category", "risk"], variable_name="feature", value_name="value")

    def _set_data_frames(self, model_prefix: str | None = None, features: list[str] | None = None, overwrite: bool = False):
        """Create model-specific DataFrames with reference statistics.

        Internal method to prepare train/test data for a specific feature subset,
        compute normative reference statistics, and calculate per-feature surprise.

        Args:
            model_prefix (str, optional): Prefix for model name. If None, uses 'base'.
            features (list[str], optional): Feature subset. If None, uses all base_features.
            overwrite (bool, optional): Allow overwriting existing model. Default False.

        Process:
            1. Determine model_name: "{prefix}_k_{num_features}"
            2. Filter training/test data to selected features
            3. Compute reference statistics (mean, std, var) from training data
            4. Join statistics with train/test data
            5. Calculate per-feature surprise: -log P(x | N(μ, σ²))
            6. Store in BayesianFrames structure
            7. Call _set_surprise_data() to aggregate and standardize

        Sets Attributes:
            self._frames[model_name] (BayesianFrames): Contains:
                - stats: Reference statistics per feature
                - train: Training data with minus_log_pfeature column
                - test: Test data with minus_log_pfeature column

        Formula (per feature):
            -log P(x | N(μ, σ²)) = 0.5 × log(2πσ²) + (x - μ)² / (2σ²)

        Example:
            >>> # Internal call during run_surprise_with_features
            >>> bd._set_data_frames('model_trial', features=['Ankle_L_position_entropy', ...])
            >>> frames = bd._frames['model_trial_k_45']
            >>> frames.stats.shape
            (45, 4)  # 45 features × (feature, mean_ref, sd_ref, var_ref)
            >>> frames.train.columns
            ['infant', 'category', 'risk', 'feature', 'value', 'mean_ref', 'sd_ref', 'var_ref', 'minus_log_pfeature']

        Note:
            - Reference statistics computed from training data only (category=1)
            - Uses population std (ddof=0) for consistency
            - Automatically calls _set_surprise_data() to aggregate

        Raises:
            AttributeError: If self._base is None (no base data loaded).
            ValueError: If model_name exists and overwrite=False.
        """
        if self._base is None:
            raise AttributeError("A base DataFrame is not defined (BayesianData.base).")
        if self._frames is None:
            self._frames = {}

        k = (
            len(features) if features is not None
            else len(self.base_features) if self.base_features is not None
            else self._base.group_by("feature").first().height
        )
        model_name = f"{model_prefix}_k_{k}" if model_prefix is not None else f"base_k_{k}"
        if model_name in self._frames and not overwrite:
            raise ValueError(f"DataFrames for model: '{model_name}' already exist.")

        if features is None:
            df_train = self._base_train
            df_test = self._base_test
        else:
            df_train = self._base_train.filter(pl.col("feature").is_in(features))
            df_test = self._base_test.filter(pl.col("feature").is_in(features))

        frames = BayesianFrames()

        df_train = (
            df_train.filter(pl.col("category") == 1).sort("feature")
        )
        df_test = (
            df_test.filter(pl.col("category") == 2).sort("feature")
        )
        frames.stats = (
            df_train
            .group_by("feature").agg(
                pl.col("value").mean().alias("mean_ref"),
                pl.col("value").std(ddof=0).alias("sd_ref"),
                pl.col("value").var(ddof=0).alias("var_ref"),
            )
        )
        frames.train = (
            df_train.join(frames.stats, on="feature", how="inner")
            .with_columns(
                minus_log_pfeature=(
                    -1 * (0.5 * np.log(2 * np.pi * pl.col("var_ref"))
                        + ((pl.col("value") - pl.col("mean_ref"))**2)
                        /(2 * pl.col("var_ref")))
                )
            )
        )
        frames.test = (
            df_test.join(frames.stats, on="feature", how="inner")
            .with_columns(
                minus_log_pfeature=(
                    -1 * (0.5 * np.log(2 * np.pi * pl.col("var_ref"))
                        + ((pl.col("value") - pl.col("mean_ref"))**2)
                        /(2 * pl.col("var_ref")))
                )
            )
        )
        self._frames[model_name] = frames
        if self.base_model_name is None and model_name.startswith("base_"):
            self.base_model_name = model_name
        self._set_surprise_data(model_prefix, k)

    def _set_surprise_data(self, model_prefix: str | None = None, k: int | None = None):
        """Aggregate per-feature surprise and compute z-scores.

        Internal method called by _set_data_frames() to:
        1. Sum -log P across features for each infant
        2. Compute normalization parameters from training data
        3. Standardize to z-scores
        4. Convert to p-values

        Args:
            model_prefix (str, optional): Model prefix for naming.
            k (int, optional): Number of features. If None, uses len(base_features).

        Process:
            Training Data:
            1. Group by infant, sum minus_log_pfeature across features
            2. Compute mean and std of these sums: μ_train, σ_train
            3. Standardize: z = (S - μ_train) / σ_train
            4. P-value: p = 2 × SF(|z|) where SF is survival function

            Test Data:
            1. Group by infant, sum minus_log_pfeature
            2. Standardize using μ_train, σ_train from training data
            3. Compute p-values identically

        Sets Attributes:
            frames.train_surprise (DataFrame): Columns: infant, risk, minus_log_pfeature, z, p
            frames.test_surprise (DataFrame): Columns: infant, risk, minus_log_pfeature, z, p
            self._surprise[model_name] (BayesianSurprise): Normalization parameters

        Mathematical Details:
            For each infant i:
                S_i = Σ_j [-log P(x_ij | N(μ_j, σ_j²))]  # Sum across j features
                z_i = (S_i - mean(S_train)) / std(S_train)
                p_i = 2 × P(Z > |z_i|)  # Two-tailed p-value

        Example:
            >>> # Internal call after _set_data_frames
            >>> frames = bd._frames['model_trial_k_45']
            >>> frames.train_surprise.head(3)
            ┌────────┬──────┬──────────────────┬────────┬───────┐
            │ infant │ risk │ minus_log_pfeature │ z      │ p     │
            ├────────┼──────┼──────────────────┼────────┼───────┤
            │ inf_01 │ 0    │ 52.3              │ -0.23  │ 0.818 │
            │ inf_02 │ 1    │ 68.5              │  1.85  │ 0.064 │
            │ inf_03 │ 0    │ 53.1              │  0.09  │ 0.928 │
            └────────┴──────┴──────────────────┴────────┴───────┘
            >>>
            >>> bd._surprise['model_trial_k_45']
            BayesianSurprise(model_name='model_trial_k_45', k=45, mean_neg_log_p=52.3, sd_neg_log_p=8.7)

        Note:
            - Uses training data statistics to standardize both train and test
            - P-values rounded to 3 decimal places
            - Higher z-scores indicate more "surprising" (deviant) patterns
            - Normative infants (risk=0) should have z ≈ 0
        """
        k_val = k if k is not None else len(self.base_features)
        model_name = f"{model_prefix}_k_{k}" if model_prefix is not None else f"base_k_{k}"

        frames = self._frames[model_name]
        groups = ['infant','risk']
        sorts = ["infant"]

        df_train = frames.train.group_by(groups).agg(pl.col('minus_log_pfeature').sum())
        mean_neg_log_p = df_train.select("minus_log_pfeature").mean().item()
        sd_neg_log_p = df_train.select("minus_log_pfeature").std().item()
        df_surprise = (
            df_train.with_columns(
                z=(pl.col('minus_log_pfeature') - mean_neg_log_p) / sd_neg_log_p
            ).with_columns(
                p=(pl.col('z').abs().map_elements(lambda x: norm.sf(x), return_dtype=pl.Float64) * 2).round(3)
            )
        ).sort(sorts)
        frames.train_surprise = df_surprise

        df_test = frames.test.group_by(groups).agg(pl.col('minus_log_pfeature').sum())
        df_surprise = (
            df_test.with_columns(
                z=(pl.col('minus_log_pfeature') - mean_neg_log_p) / sd_neg_log_p
            ).with_columns(
                p=(pl.col('z').abs().map_elements(lambda x: norm.sf(x), return_dtype=pl.Float64) * 2).round(3)
            )
        ).sort(sorts)
        frames.test_surprise = df_surprise

        if self._surprise is None:
            self._surprise = {}

        self._surprise[model_name] = BayesianSurprise(
            model_name,
            k,
            mean_neg_log_p,
            sd_neg_log_p,
        )

    def run_surprise_with_features(self, model_prefix: str, features: list[str], overwrite: bool = False):
        """Compute Bayesian surprise for specified feature subset.

        Wrapper method that calls _set_data_frames() to create model-specific
        data structures and compute surprise scores for the given features.

        Args:
            model_prefix (str): Prefix for model naming (e.g., 'model_trial').
            features (list[str]): Feature names to include in model.
            overwrite (bool, optional): Allow overwriting existing model. Default False.

        Process:
            1. Validate that base DataFrames exist
            2. Create model_name: "{prefix}_k_{len(features)}"
            3. Call _set_data_frames() which:
               - Filters data to selected features
               - Computes reference statistics
               - Calculates per-feature surprise
               - Aggregates and standardizes to z-scores

        Sets Attributes:
            self._frames[model_name] (BayesianFrames): Model data and statistics
            self._surprise[model_name] (BayesianSurprise): Normalization parameters

        Example:
            >>> # After RFE selects features
            >>> features = bd.rfe_features('rfe_90pct')
            >>> bd.run_surprise_with_features('model_trial', features)
            >>>
            >>> # Access results
            >>> model_name = f'model_trial_k_{len(features)}'
            >>> surprise_df = bd.test_surprise(model_name)
            >>> surprise_df.select(['infant', 'z', 'risk']).head()

        Raises:
            AttributeError: If base DataFrames not initialized.
            ValueError: If model exists and overwrite=False.

        See Also:
            _set_data_frames(): Internal method performing the work
            run_adaptive_rfe(): Combines RFE + surprise computation
        """
        if self._frames is None:
            raise AttributeError(f"Base DataFrames are not set.")

        model_name = f"{model_prefix}_k_{len(features)}" if model_prefix is not None else f"base_k_{len(features)}"
        self._set_data_frames(model_prefix, features, overwrite)

    def run_multi_trial_rfe(self, model_prefix: str | None = None, feature_size: int | None = None):
        """Run simple RFE to select features (legacy method).

        Performs basic Recursive Feature Elimination using Random Forest on
        training data. This is a simpler alternative to run_adaptive_rfe().

        Args:
            model_prefix (str, optional): Prefix for naming results.
            feature_size (int, optional): Target number of features. If None,
                uses len(base_features).

        Returns:
            list[str]: Selected feature names.

        Process:
            1. Extract training data and surprise z-scores
            2. Create feature matrix (infants × features)
            3. Use z-scores as target variable
            4. Fit Random Forest with RFE:
               - n_estimators=N_ESTIMATORS (200)
               - step=1 (remove 1 feature at a time)
               - n_features_to_select=feature_size
            5. Store results in BayesianRfeResult
            6. Call run_surprise_with_features() with selected features

        Sets Attributes:
            self._rfes[rfe_name] (BayesianRfeResult): RFE results
            self._frames[model_name] (BayesianFrames): Model with selected features

        Example:
            >>> # Select 45 best features
            >>> selected = bd.run_multi_trial_rfe('trial_1', feature_size=45)
            >>> len(selected)
            45
            >>>
            >>> # Access RFE results
            >>> rfe = bd.rfe('trial_1_k_45')
            >>> rfe.features[:5]
            ['Ankle_L_position_entropy', ...]

        Note:
            - This is a legacy method, prefer run_adaptive_rfe() for better results
            - Uses single RF fit (no cross-validation or statistical testing)
            - Target (y) is surprise z-scores from base model
            - RepeatedKFold objects created but not used (legacy code)

        Raises:
            AttributeError: If DataFrames not initialized.

        See Also:
            run_adaptive_rfe(): Enhanced version with statistical testing
            run_rfe(): Similar simple RFE method
        """
        outer_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RAND_STATE)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=RAND_STATE)

        k = feature_size if feature_size is not None else len(self.base_features)
        # model_name = f"{model_prefix}_k_{k}" if model_prefix is not None else f"k_{k}"
        rfe_name = f"{model_prefix}_k_{k}" if model_prefix is not None else f"k_{k}"
        if self._frames is None:
            raise AttributeError("DataFrames are not set.")

        frames = self._frames.get(self.base_model_name)
        if frames is None:
            raise AttributeError(f"Base DataFrames are not set.")

        df_raw = frames.train
        if df_raw is None:
            raise AttributeError("Base DataFrame 'train' is not set.")

        df_surprise = frames.train_surprise
        if df_surprise is None:
            raise AttributeError("Base DataFrame 'surprise' is not set.")

        df_rfe = df_raw.join(df_surprise, on="infant", how="inner").sort(["infant", "feature"])

        # generate training samples for RFE feature elimination
        df_x = df_rfe.pivot(index='infant', on='feature', values='value').drop("infant")
        y = df_rfe.group_by('infant', maintain_order=True).agg(pl.col('z').first()).select("z").to_numpy()[:,0]

        model = RandomForestRegressor(random_state=RAND_STATE, n_jobs=N_JOBS, n_estimators=N_ESTIMATORS, max_features=feature_size)
        selector = RFE(model, n_features_to_select=feature_size, step=1, importance_getter='feature_importances_')
        x_selected = selector.fit_transform(df_x.to_numpy(), y)

        if self._rfes is None:
            self._rfes = {}
        # if key not in self._rfes:
        #     self._rfes = {}
        self._rfes[rfe_name] = result = BayesianRfeResult(
            name=rfe_name,
            k=feature_size,
            features=[x for i, x in enumerate(df_x.columns) if selector.support_[i]],
        )
        self.run_surprise_with_features(model_prefix, result.features)
        return result.features

    def run_rfe(self, model_prefix: str | None = None, feature_size: int | None = None):
        """Run basic RFE to select features (legacy method, duplicate of run_multi_trial_rfe).

        Performs simple Recursive Feature Elimination. Functionally identical to
        run_multi_trial_rfe() - kept for backwards compatibility.

        Args:
            model_prefix (str, optional): Prefix for naming results.
            feature_size (int, optional): Target number of features.

        Returns:
            list[str]: Selected feature names.

        Note:
            This is identical to run_multi_trial_rfe(). For new code, prefer
            run_adaptive_rfe() which uses statistical testing and cross-validation.

        See Also:
            run_multi_trial_rfe(): Identical implementation
            run_adaptive_rfe(): Recommended enhanced version
        """
        k = feature_size if feature_size is not None else len(self.base_features)
        rfe_name = f"{model_prefix}_k_{k}" if model_prefix is not None else f"k_{k}"
        if self._frames is None:
            raise AttributeError("DataFrames are not set.")

        frames = self._frames.get(self.base_model_name)
        if frames is None:
            raise AttributeError(f"Base DataFrames are not set.")

        df_raw = frames.train
        if df_raw is None:
            raise AttributeError("Base DataFrame 'train' is not set.")

        df_surprise = frames.train_surprise
        if df_surprise is None:
            raise AttributeError("Base DataFrame 'surprise' is not set.")

        df_rfe = df_raw.join(df_surprise, on="infant", how="inner").sort(["infant", "feature"])

        # generate training samples for RFE feature elimination
        df_x = df_rfe.pivot(index='infant', on='feature', values='value').drop("infant")
        y = df_rfe.group_by('infant', maintain_order=True).agg(pl.col('z').first()).select("z").to_numpy()[:,0]

        model = RandomForestRegressor(random_state=RAND_STATE, n_jobs=N_JOBS, n_estimators=N_ESTIMATORS, max_features=feature_size)
        selector = RFE(model, n_features_to_select=feature_size, step=1, importance_getter='feature_importances_')
        x_selected = selector.fit_transform(df_x.to_numpy(), y)

        if self._rfes is None:
            self._rfes = {}
        self._rfes[rfe_name] = result = BayesianRfeResult(
            name=rfe_name,
            k=feature_size,
            features=[x for i, x in enumerate(df_x.columns) if selector.support_[i]],
        )
        self.run_surprise_with_features(model_prefix, result.features)
        return result.features

    def run_adaptive_rfe(self, model_prefix: str | None = None, features: list[str] | None = None, tot_k: int | None = None):
        """Run Enhanced Adaptive RFE with statistical testing (recommended method).

        Performs advanced feature selection using EnhancedAdaptiveRFE with:
        - Multiple parallel trials (RFE_N_TRIALS, default 50)
        - Repeated cross-validation (15 folds)
        - Statistical significance testing (binomial test)
        - Noise injection for robustness
        - Adaptive percentage adjustment

        Args:
            model_prefix (str, optional): Prefix for naming results.
            features (list[str], optional): Feature pool to select from. If None,
                uses all base_features.
            tot_k (int, optional): Total original feature count for percentage
                calculation. If None, uses len(base_features).

        Returns:
            list[str]: Statistically significant features selected.

        Process:
            1. Join training data with surprise z-scores
            2. Convert to Pandas (required by EnhancedAdaptiveRFE)
            3. Calculate initial target percentage:
               pct = RFE_KEEP_PCT - (1 - len(features)/tot_k) / 2
            4. Run EnhancedAdaptiveRFE:
               - 50 trials with 15 CV folds each = 750 RFE runs
               - Binomial test for feature significance
            5. If no reduction, decrease pct by 0.1 and retry
            6. Store results and compute surprise with selected features

        Sets Attributes:
            self._rfes[rfe_name] (BayesianRfeResult): Significant features
            self._frames[model_name] (BayesianFrames): Model with selected features

        Example:
            >>> # Run enhanced RFE (30-60 minutes)
            >>> features = bd.run_adaptive_rfe('model_best')
            >>> len(features)
            45  # Fewer features than input, statistically validated
            >>>
            >>> # Check which features were selected
            >>> rfe = bd.rfe('model_best_k_45')
            >>> rfe.features[:5]
            ['Ankle_L_position_entropy', 'Wrist_R_velocity_mean', ...]
            >>>
            >>> # Access surprise scores
            >>> surprise = bd.test_surprise('model_best_k_45')
            >>> surprise.filter(pl.col('z') > 2).shape[0]
            38  # 38 infants with highly surprising patterns

        Computational Cost:
            - Trials: RFE_N_TRIALS (50) × 15 folds = 750 RFE fits
            - Each RFE fit: ~200 Random Forest models
            - Total: ~150,000 model fits
            - Wall time: 30-60 minutes on 12 cores

        Note:
            - This is the RECOMMENDED method for production use
            - Features must pass statistical significance (p < 0.05)
            - Adaptive percentage prevents getting stuck
            - Logs progress with timing information

        Raises:
            AttributeError: If DataFrames not initialized.

        See Also:
            EnhancedAdaptiveRFE: Feature selection class used internally
            run_multi_trial_rfe(): Simpler alternative (legacy)
        """
        if self._frames is None:
            raise AttributeError("DataFrames are not set.")

        frames = self._frames.get(self.base_model_name)
        if frames is None:
            raise AttributeError(f"Base DataFrames are not set.")

        df_raw = frames.train
        if df_raw is None:
            raise AttributeError("Base DataFrame 'train' is not set.")

        df_surprise = frames.train_surprise
        if df_surprise is None:
            raise AttributeError("Base DataFrame 'surprise' is not set.")

        features = features if features is not None else self.base_features
        df_rfe = df_raw.join(df_surprise, on="infant", how="inner").sort(["infant", "feature"]).filter(pl.col("feature").is_in(features))

        # generate training samples for RFE feature elimination
        df_x = df_rfe.pivot(index='infant', on='feature', values='value').drop("infant").to_pandas()
        y = df_rfe.group_by('infant', maintain_order=True).agg(pl.col('z').first()).select("z").to_pandas()["z"]

        in_time = datetime.now()
        # Fit on training data
        tick = 0
        pct = RFE_KEEP_PCT - (1 - len(features)/tot_k) / 2

        while True:
            selector = EnhancedAdaptiveRFE(n_trials=RFE_N_TRIALS, alpha=RFE_ALPHA)
            selector.fit(df_x, y, pct)
            # Get statistically significant features
            new_features = selector.get_significant_features()
            if len(new_features) == len(features):
                pct -= 0.1
                if pct < 1.0:
                    break
                #pct = RFE_KEEP_PCT - (1 - len(features)/tot)/2
                logger.debug(f"Smaller feature set not found. Setting features to keep percent to {pct:0.2f}.")
            else:
                break

        out_time = datetime.now()
        logger.debug(f"Selected {len(new_features)} Features in {(out_time - in_time).seconds / 60: 0.2f} Minutes")
        rfe_name = f"{model_prefix}_k_{len(new_features)}" if model_prefix is not None else f"k_{len(new_features)}"

        if self._rfes is None:
            self._rfes = {}
        self._rfes[rfe_name] = result = BayesianRfeResult(
            name=rfe_name,
            k=len(new_features),
            features=new_features,
        )
        self.run_surprise_with_features(model_prefix, result.features)
        return result.features


    def compute_roc_metrics(self, model_prefix: str, feature_size: int | None = None, test_file: str | None = None, overwrite: bool = False):
        """Compute comprehensive ROC analysis and diagnostic metrics.

        Calculates ROC curve, finds optimal threshold via Youden's J, computes
        confusion matrix metrics with Wilson score confidence intervals, generates
        ROC curve plot, and estimates minimum sample sizes.

        Args:
            model_prefix (str): Model prefix for identification.
            feature_size (int, optional): Number of features in model. If None,
                uses len(base_features).
            test_file (str, optional): External test file path. NOT IMPLEMENTED.
            overwrite (bool, optional): Allow overwriting existing results.

        Returns:
            None. Sets self._metrics[model_name] with BayesianRocResult.

        Process:
            1. Extract test surprise z-scores and true risk labels
            2. Compute ROC curve: roc_curve(y_true, -z)  # Note: negative z
            3. Calculate AUC and maximum Youden's J
            4. For each unique z-score threshold:
               a. Apply threshold: y_pred = (z < threshold)
               b. Compute confusion matrix (TP, TN, FP, FN)
               c. Calculate metrics: sens, spec, PPV, NPV, accuracy, F1, Youden's J
               d. Compute 95% Wilson score CIs for all metrics
            5. Find optimal threshold (max Youden's J)
            6. Generate and save ROC curve plot
            7. Estimate rough sample size requirements

        Sets Attributes:
            self._metrics[model_name] (BayesianRocResult): Complete ROC analysis

        Metrics Computed:
            - Sensitivity (TPR, Recall): TP / (TP + FN)
            - Specificity (TNR): TN / (TN + FP)
            - PPV (Precision): TP / (TP + FP)
            - NPV: TN / (TN + FN)
            - Accuracy: (TP + TN) / N
            - F1 Score: 2×TP / (2×TP + FP + FN)
            - Youden's J: Sensitivity + Specificity - 1

        Confidence Intervals:
            - Method: Wilson score (better than Wald for proportions)
            - Level: 95% (alpha=0.05)
            - Applied to all metrics

        Sample Size Estimation:
            - Rough minimum: 2 × k × 10 total (training + testing)
            - Rough maximum: 2 × k × 30 total
            - Uses TEST_PCT to split train/test

        Example:
            >>> # After creating model with surprise scores
            >>> bd.compute_roc_metrics('model_trial', feature_size=45)
            >>>
            >>> # Access results
            >>> result = bd.metrics('model_trial_k_45')
            >>> print(f"AUC: {result.auc:.3f}")
            AUC: 0.847
            >>> print(f"Optimal threshold: {result.threshold_j:.2f}")
            Optimal threshold: 1.85
            >>> print(f"Youden's J: {result.youdens_j:.3f}")
            Youden's J: 0.674
            >>>
            >>> # Access detailed metrics DataFrame
            >>> metrics_df = result.metrics
            >>> optimal_row = metrics_df.filter(pl.col('j') == result.youdens_j)
            >>> print(f"Sensitivity: {optimal_row['sens'][0]:.3f}")
            Sensitivity: 0.824
            >>> print(f"Specificity: {optimal_row['spec'][0]:.3f}")
            Specificity: 0.850
            >>>
            >>> # View ROC plot
            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.image import imread
            >>> img = imread(result.plot_file)
            >>> plt.imshow(img)
            >>> plt.show()

        Outputs:
            - ROC curve plot: {PLOT_DIR}/auc_{model_name}.png
            - Contains:
              * ROC curve with AUC
              * No-skill diagonal reference
              * Model name and feature count

        Note:
            - Uses NEGATIVE z-scores for ROC: higher z = more at-risk
            - Threshold applied as: y_pred = (z < threshold)
            - Optimal threshold maximizes Youden's J
            - Multiple thresholds with same J: chooses highest threshold
            - Sample size estimates are ROUGH guidelines only

        Raises:
            NotImplementedError: If test_file parameter is provided.

        See Also:
            BayesianRocResult: Complete result structure
            BayesianCI: Extract CI information from metrics
            bam.py: Bayesian sample size planning
        """
        k = feature_size if feature_size is not None else len(self.base_features)

        model_name = f"{model_prefix}_k_{k}" if model_prefix is not None else f"k_{k}"

        if test_file is None:
            surp: DataFrame = self.test_surprise(model_name).with_columns(
                y_true = pl.col("risk"),
                neg_z = pl.col("z").neg()
            )
            test_n = surp.height
            train_n = self.train_surprise(model_name).height
            roc_fpr, roc_tpr, roc_thresh = roc_curve(surp["y_true"], surp["neg_z"], drop_intermediate=False)
        else:
            raise NotImplementedError("Ooops!")

        # IMPLEMENT!!!

        roc_auc_val = auc(roc_fpr, roc_tpr)
        roc_youdens_j = (roc_tpr - roc_fpr).max()

        # df_roc = pl.DataFrame(
        #     {
        #         "threshold": roc_thresh,
        #         "j": roc_tpr - roc_fpr,
        #         "roc_fpr": roc_fpr,
        #         "roc_tpr": roc_tpr,
        #     }
        # )
        alpha = 0.05

        tp = []
        fp = []
        tn = []
        fn = []
        n = []

        sens = []  # sens, tpr, recall
        spec = []  # tnr
        tpr = []  # sens, tpr, recall
        fpr = []
        ppv = []  # precision
        npv = []
        acc = []
        f1 = []

        sens_ci = []
        spec_ci = []
        tpr_ci = []
        fpr_ci = []
        ppv_ci = []
        npv_ci = []
        acc_ci = []
        f1_ci = []

        sens_w_ci = []
        spec_w_ci = []
        tpr_w_ci = []
        fpr_w_ci = []
        ppv_w_ci = []
        npv_w_ci = []
        acc_w_ci = []
        f1_w_ci = []

        j = []

        thresh = surp.select("z").unique().sort("z", descending=True).to_series().to_list()
        for thr in thresh:
            df = surp.with_columns(y_pred=pl.when(pl.col("z") < thr).then(1).otherwise(0))
            _tn, _fp, _fn, _tp = confusion_matrix(df["y_true"], df["y_pred"]).ravel()

            tp.append(_tp)
            tn.append(_tn)
            fp.append(_fp)
            fn.append(_fn)
            n.append(_tn +_fp +_fn +_tp)

            # print(f"\nthresh: {thr:.5f}")
            # print(f"thresh: {thr:.5f} | tn: {_tn} | _fp: {_fp} | _fn: {_fn} | _tp: {_tp}")
            _p = _tp + _fn
            _n = _tn + _fp
            _tot = _p + _n
            _p_pred = _tp + _fp
            _n_pred = _tn + _fn
            _t_pred = _tp + _tn
            _2tp = 2 * _tp
            _f1_den = _2tp + _fp + _fn
            _tpr = _tp / _p if _p > 0 else 0
            _fpr = _fp / _n if _n > 0 else 0
            _spec = _tn / _n if _n > 0 else 0
            _sens = _tpr
            _ppv = _tp / _p_pred if _p_pred > 0 else 0
            _npv = _tn / _n_pred if _n_pred > 0 else 0
            _f1 = _2tp / _f1_den if _f1_den > 0 else 0
            _acc = _t_pred / _tot
            _j = _sens + _spec - 1
            # print(f"thresh: {thr:.5f} | _p: {_p} | _n {_n} | _p_pred: {_p_pred} | _n_pred: {_n_pred} | _f1_den: {_f1_den}")
            _sens_lb, _sens_ub = proportion_confint(count=_tp, nobs=_p, alpha=alpha, method='wilson')
            _spec_lb, _spec_ub = proportion_confint(count=_tn, nobs=_n, alpha=alpha, method='wilson')
            _fpr_lb, _fpr_ub = proportion_confint(count=_fp, nobs=_n, alpha=alpha, method='wilson')
            _ppv_lb, _ppv_ub = proportion_confint(count=_tp, nobs=_p_pred, alpha=alpha, method='wilson') if _p_pred > 0 else (0.0, 0.0)
            _npv_lb, _npv_ub = proportion_confint(count=_tn, nobs=_n_pred, alpha=alpha, method='wilson') if _n_pred > 0 else (0.0, 0.0)
            _acc_lb, _acc_ub = proportion_confint(count=_t_pred, nobs=_tot, alpha=alpha, method='wilson') if _tot > 0 else (0.0, 0.0)
            _f1_lb, _f1_ub = proportion_confint(count=_2tp, nobs=_f1_den, alpha=alpha, method='wilson') if _f1_den > 0 else (0.0, 0.0)
            sens.append(_sens)
            spec.append(_spec)
            tpr.append(_tpr)
            fpr.append(_fpr)
            ppv.append(_ppv)
            npv.append(_npv)
            acc.append(_acc)
            f1.append(_f1)
            j.append(_j)
            sens_ci.append((_sens_lb, _sens_ub))
            spec_ci.append((_spec_lb, _spec_ub))
            tpr_ci.append((_sens_lb, _sens_ub))
            fpr_ci.append((_fpr_lb, _fpr_ub))
            ppv_ci.append((_ppv_lb, _ppv_ub))
            npv_ci.append((_npv_lb, _npv_ub))
            acc_ci.append((_acc_lb, _acc_ub))
            f1_ci.append((_f1_lb, _f1_ub))
            sens_w_ci.append(f"{_sens:.3f} [{_sens_lb:.3f}, {_sens_ub:.3f}]")
            spec_w_ci.append(f"{_spec:.3f} [{_spec_lb:.3f}, {_spec_ub:.3f}]")
            tpr_w_ci.append(f"{_tpr:.3f} [{_sens_lb:.3f}, {_sens_ub:.3f}]")
            fpr_w_ci.append(f"{_fpr:.3f} [{_fpr_lb:.3f}, {_fpr_ub:.3f}]")
            ppv_w_ci.append(f"{_ppv:.3f} [{_ppv_lb:.3f}, {_ppv_ub:.3f}]")
            npv_w_ci.append(f"{_npv:.3f} [{_npv_lb:.3f}, {_npv_ub:.3f}]")
            acc_w_ci.append(f"{_acc:.3f} [{_acc_lb:.3f}, {_acc_ub:.3f}]")
            f1_w_ci.append(f"{_f1:.3f} [{_f1_lb:.3f}, {_f1_ub:.3f}]")

        max_j = max(j)
        rough_est_n_train_min = 2 * k * 10
        rough_est_n_test_min = 2 * k * 10 * TEST_PCT
        rough_est_n_total_min = rough_est_n_train_min + rough_est_n_test_min

        rough_est_n_train_max = 2 * k * 30
        rough_est_n_test_max = 2 * k * 30 * TEST_PCT
        rough_est_n_total_max = rough_est_n_train_max + rough_est_n_test_max

        primitives = pl.DataFrame(
            {
                "thresh": thresh,
                "roc_thresh": roc_thresh[1:],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "n": n,
            }
        )
        metrics = pl.DataFrame(
            {
                "threshold": thresh,
                "roc_thresh": roc_thresh[1:],
                "tpr": tpr,
                "fpr": fpr,
                "sens": sens,
                "spec": spec,
                "ppv": ppv,
                "npv": npv,
                "acc": acc,
                "f1": f1,
                "sens_ci": sens_ci,
                "spec_ci": spec_ci,
                "tpr_ci": tpr_ci,
                "fpr_ci": fpr_ci,
                "ppv_ci": ppv_ci,
                "npv_ci": npv_ci,
                "acc_ci": acc_ci,
                "f1_ci": f1_ci,
                "sens_w_ci": sens_w_ci,
                "spec_w_ci": spec_w_ci,
                "tpr_w_ci": tpr_w_ci,
                "fpr_w_ci": fpr_w_ci,
                "ppv_w_ci": ppv_w_ci,
                "npv_w_ci": npv_w_ci,
                "acc_w_ci": acc_w_ci,
                "f1_w_ci": f1_w_ci,
                "j": j,
                "rough_n_min": round(rough_est_n_total_min),
                "rough_n_max": round(rough_est_n_total_max),
            }
        )

        optimals = metrics.filter(pl.col("j").round(5) == round(max_j, 5)).select(["j", "threshold", "sens"])
        if optimals.height != 1:
            if optimals.height > 1:
                optimals = optimals.filter(pl.col("threshold") == pl.col("threshold").max())
                if optimals.height > 1:
                    optimals = optimals.head(1)
            elif optimals.height == 0:
                print(f"No results with J ({max_j} in RFE: {model_name}:\n{optimals.to_dicts()}")
                # raise LookupError(f"Expected exactly one record. Got {optimals.height}. [RFE: {rfe_name} | Youden's J: {youdens_j}]")
        optimal = optimals.select("threshold").item()

        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([0.005, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.suptitle('ROC Curve for Infant Risk Classification')
        plt.title(f'{len(self.features(model_name))} Feature Surprise Model ({model_name})')
        plt.legend(loc="lower right")
        plot_file = f"auc_{model_name}.png"
        fig.savefig(PLOT_DIR / plot_file)
        plt.close()

        result = BayesianRocResult(
            model_name=model_name,
            train_n=train_n,
            test_n=test_n,
            # "method": "validation" if df_test is not None else "resubstitution",
            features=self.features(model_name),
            plot_file=plot_file,
            auc=roc_auc_val,
            youdens_j=max_j,
            roc_youdens_j=roc_youdens_j,
            threshold_j=optimal,
            metrics=metrics,
            primitives=primitives,
            rough_n_min=rough_est_n_total_min,
            rough_n_max=rough_est_n_total_max,
        )
        if self._metrics is None:
            self._metrics = {}
        self._metrics[model_name] = result

    def get_top_n_rfes(self, n: int = 10, rfe_type: RfeType = RfeType.SUMMARY):
        results = []
        for metrics_name in self.metrics_names:
            metrics = self.metrics(metrics_name)
            df_metrics = metrics.metrics
            for row in df_metrics.filter(
                pl.col("j") == pl.col("j").max()
            ).to_dicts():
                row["metrics_name"] = metrics_name
                # row["rfe_name"] = metrics.rfe_name
                row["k"] = len(metrics.features)
                row["auc"] = metrics.auc
                row["rough_n_min"] = int(metrics.rough_n_min)
                row["rough_n_max"] = int(metrics.rough_n_max)
                results.append(row)

        if rfe_type == RfeType.SUMMARY:
            cols = SUMMARY_COLS
            map = SUMMARY_MAP
        else:
            cols = DETAIL_COLS
            map = DETAIL_MAP

        df_summary = DataFrame(results).select(cols)  # .rename(SUMMARY_MAP)

        top_ns = {
            f"auc_{n}": {
                "metric": "AUC",
                "df": df_summary.sort("auc", descending=True).head(n).rename(map, strict=False),
            },
            f"j_{n}": {
                "metric": "Youden's J",
                "df": df_summary.sort("j", descending=True).head(n).rename(map, strict=False),
            },
        }
        return top_ns

    def write_excel_report(self, tag: str | None = None):
        top_tens = self.get_top_n_rfes(25)
        stub = f"{tag}_" if tag is not None else ""
        file_name = f"cribsy_model_{stub}sample_size.xlsx"
        wb = Workbook(XLSX_DIR / file_name)
        formats = set_workbook_formats(wb)
        wb.add_worksheet("Summary")
        row_2 = 1
        row_4 = 3
        row_5 = 4
        row_6 = 5
        row_7 = 6
        row_8 = 7
        row_28 = 27
        row_29 = 28
        row_30 = 29
        # max_df = 0
        # max_fx = 0
        for metrics_name in self.metrics_names:
            # if model_name.startswith("base_"):
            #     continue
            metrics = self.metrics(metrics_name)
            model_name = metrics.model_name
            # is_test = "test" in metrics_name

            ws = wb.add_worksheet(metrics_name)
            col_b = 1
            col_c = 2
            col_d = 3
            col_e = 4
            col_f = 5
            col_g = 6
            col_h = 7
            col_i = 8
            col_j = 9
            col_k = 10
            col_l = 11
            col_m = 12
            col_n = 13
            col_o = 14
            col_p = 15
            col_q = 16
            col_r = 17
            col_s = 18

            ws.merge_range(row_2, col_b, row_2, col_l,
               f"Trial: {metrics_name} | Model: {model_name} | Features: {self.statistics(model_name).height} | "
               f"Train N: {metrics.train_n} | Test N: {metrics.test_n}", formats["heading1"]
           )

            ws.merge_range(row_4, col_b, row_4, col_d, "Selected Features", formats["long_list_heading"])
            row_last = row_5 + self.statistics(model_name).height

            features = self.statistics(model_name).select("feature").to_series().to_list()
            ws.merge_range(row_5, col_b, row_last, col_d, "\n".join(features), formats["long_list"])

            rfe_name = f"{model_name}"
            # metrics = self.metrics(rfe_name)
            ws.merge_range(row_4, col_f, row_4, col_q, f"RFE: {rfe_name}", formats["heading2"])

            ws.insert_image(row_6, col_s, PLOT_DIR / metrics.plot_file)  # , {"x_scale": 1.2, "y_scale": 1.2})

            ws.merge_range(row_6, col_f, row_6, col_q, f"AUC: {metrics.auc:.2f} | Youden's J: {metrics.youdens_j:.3f} | Best Threshold: {metrics.threshold_j:.3f}", formats["heading3"])

            df = metrics.metrics.select("threshold", "tpr_w_ci", "fpr_w_ci", "sens_w_ci", "spec_w_ci", "ppv_w_ci", "npv_w_ci", "f1_w_ci", "acc_w_ci", "j", "rough_n_min", "rough_n_max").rename(METRIC_MAP, strict=False).slice(1)

            df.write_excel(wb, ws, position=(row_7, col_f), table_style="Table Style Medium 1")
            char_f = xl_col_to_name(col_f)
            char_q = xl_col_to_name(col_q)
            rng = f"{char_f}{row_8}:{char_q}{row_8 + df.height}"
            ws.conditional_format(
                rng,
                {
                    "type": "formula",
                    "criteria": f"=ROUND(${char_f}{row_8}, 5)={round(metrics.threshold_j, 5)}",
                    "format": formats["green_bg"]
                }
            )
            ws.set_column(col_e, col_e, 4)
            ws.set_column(col_f, col_f, 12)
            ws.set_column(col_g, col_n, 18)
            ws.set_column(col_o, col_q, 10)
            ws.set_column(col_r, col_r, 4)

            offset = len(df.columns) + 2
            col_b += offset
            col_c += offset
            col_d += offset
            col_e += offset
            col_f += offset
            col_g += offset
            col_h += offset
            col_i += offset
            col_k += offset
            col_m += offset
            col_n += offset
            col_o += offset
            col_p += offset
            col_q += offset
            col_r += offset
            col_s += offset

        # for k, v in AGE_LABELS.items():
        ws = wb.get_worksheet_by_name(f"Summary")
        row = 1
        col = 1

        col_b = 1
        col_c = 2
        col_d = 3
        col_f = 5
        col_g = 6
        col_j = 9
        col_k = 10
        col_l = 11
        col_m = 12
        col_n = 13
        col_0 = 14
        col_p = 15
        col_q = 16
        col_r = 17

        ws.merge_range(row, col_b, row, col_r, f"Top Ten RFE Runs by AUC and Youden's J", formats["heading1"])
        row = 3
        for key, rfe in top_tens.items():
            df = rfe["df"]
            if df.height == 0:
                continue

            summ_auc = key.startswith("auc")

            ws.merge_range(row, col_b, row, col_r, f"Top RFE Runs by {"AUC" if key.startswith("auc") else "Youden's J" }", formats["heading3"])
            row += 1

            row_start = row + 2
            df.write_excel(wb, ws, position=(row, col_b), table_style="Table Style Medium 1")
            row += 1
            models = df.select("Trial").to_series().to_list()
            for model in models:
                ws.write_url(row, col_b, f"internal:{model}!B2", string=model)
                row += 1

            char_b = xl_col_to_name(col_b)
            char_f = xl_col_to_name(col_f)
            char_p = xl_col_to_name(col_p)
            char_r= xl_col_to_name(col_r)
            rng = f"{char_b}{row_start}:{char_r}{row_start + df.height - 1}"
            if summ_auc:
                ws.conditional_format(
                    rng,
                    {
                        "type": "formula",
                        "criteria": f"=ROUND(${char_f}{row_start}, 5)={round(df.select(["AUC"]).max().item(), 5)}",
                        "format": formats["green_bg"]
                    }
                )
            else:
                ws.conditional_format(
                    rng,
                    {
                        "type": "formula",
                        "criteria": f"=ROUND(${char_p}{row_start}, 5)={round(df.select(["Youden's J"]).max().item(), 5)}",
                        "format": formats["green_bg"]
                    }
                )
            ws.set_column(col_b, col_b, 15)
            ws.set_column(col_c, col_c, 15)
            ws.set_column(col_d, col_f, 12)
            ws.set_column(col_g, col_n, 18)
            ws.set_column(col_0, col_p, 12)
            ws.set_column(col_q, col_r, 10)
            row = df.height + 7
            # col_b += offset
            # col_c += offset
            # col_d += offset
            # col_f += offset
            # col_g += offset
            # col_j += offset
            # col_k += offset
            # col_l += offset
            # col_m += offset
            # col_n += offset
            # # col_o += offset
            # col_p += offset
            # col_q += offset
            # col_r += offset
            # col_s += offset
            # row += 1

        wb.close()
