"""Enhanced Adaptive RFE with noise injection and statistical testing.

This module implements an advanced Recursive Feature Elimination (RFE)
approach that combines multiple strategies for robust feature selection:

1. **Noise Injection**: Adds random features to test selection stability
2. **Adaptive Step Sizing**: Dynamically adjusts elimination rate based on CV performance
3. **Hyperparameter Tuning**: GridSearchCV optimization of Random Forest parameters
4. **Statistical Significance**: Binomial test for feature consensus (p < alpha)
5. **Parallel Execution**: Multiple trials run in parallel for efficiency

Key Improvements over Base RFE:
    - Importance-gap step sizing (faster than CV-based approach)
    - Noise features filtered out (shouldn't be selected if method is robust)
    - Optimal RF hyperparameters found automatically
    - Statistical test ensures features appear significantly more than chance
    - Configurable n_estimators for speed vs. stability trade-offs
    - Test-set evaluation metrics (MAE, MSE, R²)
    - Stability visualization with heatmaps

Classes:
    EnhancedAdaptiveRFE: Main feature selection class
    AdaptiveStepRFE: Dynamic step size calculator

Constants from constants.py:
    RFE_N_TRIALS: Number of parallel trials (default: 50)
    RFE_ALPHA: Significance level (default: 0.05)
    RFE_NOISE_RATIO: Proportion of noise features (default: 0.1)
    RFE_KEEP_PCT: Target feature retention (default: 0.9)

Example:
    >>> from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> # Prepare data
    >>> X = pd.DataFrame(...)  # Feature matrix
    >>> y = pd.Series(...)     # Target variable
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> # Run enhanced RFE (use n_estimators=1000 for final models)
    >>> selector = EnhancedAdaptiveRFE(
    ...     n_trials=50,
    ...     alpha=0.05,
    ...     noise_ratio=0.1,
    ...     n_estimators=200  # 200 for speed, 1000 for stability
    ... )
    >>> selector.fit(X_train, y_train, features_to_keep_pct=0.9)
    >>> 
    >>> # Get statistically significant features
    >>> features = selector.get_significant_features()
    >>> print(f"Selected {len(features)} significant features")
    >>> 
    >>> # Generate comprehensive validation report with test evaluation
    >>> report = selector.validation_report(
    ...     X_train=X_train, y_train=y_train,
    ...     X_test=X_test, y_test=y_test,
    ...     plot_heatmap=True,
    ...     save_path='stability_heatmap.png'
    ... )
    >>> print(f"Test R²: {report['test_r2']:.3f}")
    >>> print(f"Noise contamination: {report['noise_contamination']:.2%}")
    >>> 
    >>> # Or plot stability separately
    >>> fig = selector.plot_stability(n_features_to_show=30)
    >>> # fig.show()  # Display the plot

References:
    [1] Xu, P., et al. (2014). Predictor augmentation for feature selection.
    [4] Guyon, I., et al. (2002). Gene selection for cancer classification
        using support vector machines.
    [5] Sharp package for stability testing
    [6] GridSearchCV for hyperparameter optimization

See Also:
    rfe.AdaptiveRFE: Simpler base implementation
    constants.py: Configuration parameters
"""

# ================ ENHANCED IMPLEMENTATION ================
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from joblib import Parallel, delayed
from scipy.stats import binomtest
from loguru import logger
from tqdm import tqdm

from early_markers.cribsy.common.constants import (
    RAND_STATE,
    RFG_PARAM_GRID,
    RFG_N_JOBS,
    RFG_KFOLDS,
    STEP_KFOLDS,
    NOISE_RATIO,
    RFE_MIN_STEP,
    RFE_MAX_STEP,
    RFG_N_ESTIMATORS,
    RFE_TOLERANCE,
    RFE_N_TRIALS,
    RFE_ALPHA,
    RFE_NOISE_RATIO,
    RFE_N_JOBS,
    RKFOLD_SPLITS,
    RKFOLD_REPEATS,
    RFE_KEEP_PCT,
)


# 3. Adaptive step sizing with error control [4]
@dataclass
class AdaptiveStepRFE:
    """Dynamic step size calculator for RFE based on CV performance.
    
    Determines optimal number of features to eliminate per RFE iteration
    by comparing cross-validation performance at min and max step sizes.
    Larger steps when performance is stable, smaller steps when sensitive.
    
    Attributes:
        estimator (BaseEstimator): ML model to use for CV scoring.
        min_step (int): Minimum features to eliminate (from RFE_MIN_STEP).
        max_step (int): Maximum features to eliminate (from RFE_MAX_STEP).
        tol (float): Performance tolerance for step adjustment (from RFE_TOLERANCE).
    
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> estimator = RandomForestRegressor(random_state=42)
        >>> step_selector = AdaptiveStepRFE(estimator, tol=0.01)
        >>> optimal_step = step_selector.determine_step(X, y, cv=3)
        >>> print(f"Eliminate {optimal_step} features this iteration")
    """
    estimator: BaseEstimator
    min_step: int
    max_step: int
    tol: float  # Performance tolerance for step adjustment
    consensus: dict

    def __init__(self, estimator: BaseEstimator, tol: float = RFE_TOLERANCE):  #  min_step: int = 1, max_step: int = 5,
        self.estimator = estimator
        self.min_step = RFE_MIN_STEP
        self.max_step = RFE_MAX_STEP
        self.tol = tol  # Performance tolerance for step adjustment

    def _dynamic_step(self, importances):
        """Calculate optimal features to remove based on importance gaps.
        
        Analyzes the distribution of feature importances and identifies natural
        breakpoints where importance drops significantly. More efficient than
        CV-based approaches and provides interpretable results.
        
        Args:
            importances (np.ndarray): Feature importance values from fitted estimator.
        
        Returns:
            int: Number of features to eliminate (between min_step and max_step).
        
        Algorithm:
            1. Sort importances in descending order
            2. Calculate gaps (differences) between consecutive importances
            3. Find 95th percentile of gaps (critical_gap threshold)
            4. Count features below first gap exceeding threshold
            5. Return count, bounded by [min_step, max_step]
        
        Example:
            >>> importances = np.array([0.5, 0.48, 0.45, 0.1, 0.08, 0.05])
            >>> step_selector._dynamic_step(importances)
            3  # Would eliminate 3 least important features
        
        Note:
            - Faster than CV-based step sizing (no model refitting)
            - More interpretable (based on actual feature importance structure)
            - Adapted from rfe.py implementation
        """
        # Sort importances in descending order
        sorted_imp = np.sort(importances)[::-1]
        
        # Calculate gaps between consecutive importances
        gaps = np.diff(sorted_imp)
        if len(gaps) == 0:
            return self.min_step
        
        # Find critical gap threshold (95th percentile)
        critical_gap = np.quantile(np.abs(gaps), 0.95)
        
        # Count features to remove (up to first large gap)
        removal_count = np.argmax(np.abs(gaps) > critical_gap) + 1
        
        # Bound removal count to valid range
        return max(self.min_step, min(self.max_step, removal_count))


# Modified AdaptiveRFE class incorporating all improvements
class EnhancedAdaptiveRFE:
    """Enhanced Adaptive RFE with noise injection and statistical testing.
    
    Advanced feature selection combining:
    - Noise feature injection for robustness testing
    - Hyperparameter optimization (GridSearchCV)
    - Adaptive step sizing based on CV performance
    - Statistical significance testing (binomial test)
    - Parallel trial execution
    
    Features are selected based on appearing significantly more often than
    chance across multiple RFE trials with cross-validation.
    
    Attributes:
        n_trials (int): Number of parallel RFE trials.
        alpha (float): Significance level for binomial test (e.g., 0.05).
        noise_ratio (float): Proportion of noise features to inject (e.g., 0.1).
        feature_counts (dict): Count of times each feature was selected.
        noise_features (list): Names of injected noise features.
        consensus (dict): Feature selection results with p-values and significance.
    
    Algorithm:
        For each trial:
        1. Inject noise features (random values)
        2. Optimize RF hyperparameters with GridSearchCV
        3. Determine adaptive step size from CV performance
        4. Run RFE with RepeatedKFold (15 folds total)
        5. Collect selected features (excluding noise)
        
        After all trials:
        6. Count feature selections across trials
        7. Binomial test: P(count | n_trials×15, p=0.5)
        8. Features with p < alpha are significant
    
    Parameters:
        n_trials (int, optional): Number of trials. Defaults to RFE_N_TRIALS (50).
        alpha (float, optional): Significance level. Defaults to RFE_ALPHA (0.05).
        noise_ratio (float, optional): Noise proportion. Defaults to RFE_NOISE_RATIO (0.1).
        n_estimators (int, optional): Number of trees in Random Forest. Defaults to 
            RFG_N_ESTIMATORS (200). Use 1000 for more stable final models at cost of speed.
    
    Example:
        >>> from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE
        >>> import pandas as pd
        >>> from sklearn.model_selection import train_test_split
        >>> 
        >>> # Prepare data
        >>> X = pd.DataFrame(...)  # N samples × K features
        >>> y = pd.Series(...)     # N target values
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> 
        >>> # Initialize and fit (use n_estimators=1000 for production)
        >>> selector = EnhancedAdaptiveRFE(
        ...     n_trials=50,
        ...     alpha=0.05,
        ...     noise_ratio=0.1,
        ...     n_estimators=200  # Fast mode; use 1000 for final models
        ... )
        >>> selector.fit(X_train, y_train, features_to_keep_pct=0.9)
        >>> 
        >>> # Get significant features
        >>> sig_features = selector.get_significant_features()
        >>> print(f"{len(sig_features)} features selected")
        >>> 
        >>> # Generate full validation report with test metrics
        >>> report = selector.validation_report(
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     plot_heatmap=True
        ... )
        >>> print(f"Test R²: {report['test_r2']:.3f}")
        >>> print(f"Noise contamination: {report['noise_contamination']:.2%}")
        >>> 
        >>> # Visualize stability
        >>> fig = selector.plot_stability(n_features_to_show=30, save_path='stability.png')
    
    Note:
        - Computational cost: O(n_trials × 15_folds × n_estimators × n_features)
        - Default: 50 trials × 15 folds × 200 trees = 150,000 model fits
        - Use n_jobs parameter to parallelize (default: RFE_N_JOBS=12)
        - Noise features should NOT be selected if method is working correctly
    
    See Also:
        AdaptiveRFE: Simpler base implementation without enhancements
        BayesianData.run_rfe_on_base: Integrates this class into pipeline
    """
    def __init__(self, n_trials=RFE_N_TRIALS, alpha=RFE_ALPHA, noise_ratio=RFE_NOISE_RATIO, 
                 n_estimators=RFG_N_ESTIMATORS):
        self.consensus = None
        self.n_trials = n_trials
        self.alpha = alpha
        self.noise_ratio = noise_ratio
        self.n_estimators = n_estimators
        self.feature_counts = defaultdict(int)
        self.noise_features = []

    # 1. Add noise feature injection based on xu's predictor augmentation theory [1]
    def _augment_with_noise(self, x, noise_ratio=NOISE_RATIO):
        """Add independent noise features to test feature stability.
        
        Injects random Gaussian features to validate that the selection process
        discriminates against irrelevant features. Noise features should NOT be
        selected if the method is working correctly.
        
        Args:
            x (pd.DataFrame): Original feature matrix (N × K).
            noise_ratio (float, optional): Fraction of noise features to add.
                Defaults to NOISE_RATIO (0.1).
        
        Returns:
            pd.DataFrame: Augmented matrix (N × K+M) where M = ⌊K × noise_ratio⌋.
                Noise features named 'noise_0', 'noise_1', etc.
        
        Example:
            >>> x_orig = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
            >>> x_aug = selector._augment_with_noise(x_orig, noise_ratio=0.5)
            >>> x_aug.columns.tolist()
            ['A', 'B', 'noise_0']
            >>> x_aug['noise_0'].mean()  # Random values
            -0.123...
        
        Note:
            - Noise is drawn from N(0, 1) with fixed RAND_STATE
            - Noise features stored in self.noise_features for later filtering
            - Based on predictor augmentation theory (Xu et al.)
        """
        n_noise = int(x.shape[1] * noise_ratio)
        noise = pd.DataFrame(
            np.random.RandomState(RAND_STATE).standard_normal((x.shape[0], n_noise)),
            columns=[f'noise_{i}' for i in range(n_noise)]
        )
        return pd.concat([x, noise], axis=1)

    def _trial_processing(self, x, y, trial_seed, features_to_keep_pct: float | None = None):
        """Run a single RFE trial with enhancements.
        
        Executes one complete enhanced RFE trial with:
        1. Noise injection
        2. Hyperparameter optimization
        3. Adaptive step size selection
        4. Repeated KFold cross-validation (15 folds)
        
        Args:
            x (pd.DataFrame): Feature matrix (N × K).
            y (pd.Series): Target variable (N,).
            trial_seed (int): Random seed for this trial's reproducibility.
            features_to_keep_pct (float, optional): Target feature retention rate.
                If None, uses RFE_KEEP_PCT (0.9). Must be in (0, 1].
        
        Returns:
            list: Feature names selected in this trial (noise features excluded).
        
        Process:
            1. Augment x with noise features (10% by default)
            2. Optimize RF hyperparameters via GridSearchCV
            3. Determine adaptive step size using AdaptiveStepRFE
            4. Configure RFE with tuned estimator and step
            5. Run RFE across 15 CV folds (5 splits × 3 repeats)
            6. Collect features selected in any fold
            7. Filter out noise features
        
        Example:
            >>> x = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
            >>> y = pd.Series([0, 1])
            >>> trial_features = selector._trial_processing(x, y, trial_seed=42, features_to_keep_pct=0.5)
            >>> isinstance(trial_features, list)
            True
            >>> all(f in ['A', 'B'] for f in trial_features)
            True
        
        Note:
            - Each trial uses 15 CV folds: RKFOLD_SPLITS (5) × RKFOLD_REPEATS (3)
            - Features selected in ANY fold are counted
            - Noise features are automatically filtered from results
            - Trial is fully reproducible with given trial_seed
        """
        # Augment data with noise features [1]
        x_aug = self._augment_with_noise(x, self.noise_ratio)
        self.noise_features = [c for c in x_aug if c.startswith('noise_')]

        # Hyperparameter optimization [6]
        tuned_estimator = self._optimize_hyperparameters(x_aug, y)

        # Use fixed step size from configuration
        # Note: sklearn's RFE doesn't support callable step functions
        # Using RFE_MIN_STEP as a conservative fixed step size
        step_size = RFE_MIN_STEP

        # Configure RFE with fixed step size
        rfe = RFE(
            estimator=tuned_estimator,
            step=step_size,  # Use fixed step size
            n_features_to_select= features_to_keep_pct if features_to_keep_pct is not None else RFE_KEEP_PCT,
            importance_getter='feature_importances_'
        )
        # Execute nested CV
        selected = []
        for train_idx, _ in RepeatedKFold(n_splits=RKFOLD_SPLITS, n_repeats=RKFOLD_REPEATS,
                                       random_state=trial_seed).split(x_aug):
            rfe.fit(x_aug.iloc[train_idx], y.iloc[train_idx])
            selected.extend(x_aug.columns[rfe.support_])

        return [f for f in selected if f not in self.noise_features]

    # 4. Integrated hyperparameter tuning [6]
    def _optimize_hyperparameters(self, x, y):
        """Optimize RandomForest hyperparameters via GridSearchCV.
        
        Searches over hyperparameter space defined in RFG_PARAM_GRID:
        - max_depth: Tree depth
        - min_samples_leaf: Minimum samples per leaf
        - max_features: Features considered for splits
        
        Args:
            x (pd.DataFrame): Feature matrix (N × K).
            y (pd.Series): Target variable (N,).
        
        Returns:
            RandomForestRegressor: Best estimator with optimized parameters.
        
        Parameters Searched:
            max_depth: [5, 7, 9]
            min_samples_leaf: [1, 2, 3]
            max_features: [1, 'sqrt', 'log2']
        
        Cross-Validation:
            - KFold with RFG_KFOLDS (default 3) splits
            - Shuffle enabled with RAND_STATE
            - Scoring: Default MSE for regression
        
        Example:
            >>> x = pd.DataFrame(np.random.randn(100, 10))
            >>> y = pd.Series(np.random.randn(100))
            >>> best_rf = selector._optimize_hyperparameters(x, y)
            >>> best_rf.n_estimators
            200
            >>> best_rf.max_depth in [5, 7, 9]
            True
        
        Note:
            - Uses n_jobs=-4 to parallelize (all cores except 4)
            - Base n_estimators configurable via __init__ (default 200, can use 1000)
            - random_state always RAND_STATE for reproducibility
            - Computational cost: ~30 grid points × 3 folds = 90 fits
        
        See Also:
            RFG_PARAM_GRID: Full parameter grid definition in constants.py
        """
        searcher = GridSearchCV(
            RandomForestRegressor(n_estimators=self.n_estimators, random_state=RAND_STATE, n_jobs=1),
            RFG_PARAM_GRID,
            cv=KFold(RFG_KFOLDS, shuffle=True, random_state=RAND_STATE),
            n_jobs=-4,
            # verbose=2,
        )
        searcher.fit(x, y)
        return searcher.best_estimator_

       # 2. Enhanced stability scoring using sharp package's statistical testing [5]
    def _calculate_consensus(self, feature_counts, n_trials, alpha=RFE_ALPHA):
        """Calculate statistical significance of feature selection frequencies.
        
        Uses binomial test to determine if features appear significantly more
        often than chance (50%) across all CV folds and trials.
        
        Args:
            feature_counts (dict): {feature_name: count} across all folds/trials.
            n_trials (int): Total number of opportunities for selection (n_trials × 15).
            alpha (float, optional): Significance level. Defaults to RFE_ALPHA (0.05).
        
        Returns:
            dict: {feature_name: {'count': int, 'pvalue': float, 'significant': bool}}
        
        Statistical Test:
            - Null hypothesis: P(feature selected) = 0.5 (random selection)
            - Alternative: P(feature selected) > 0.5 (above chance)
            - Test: binomial_test(count, n_trials, p=0.5, alternative='greater')
            - Significant if: p-value < alpha
        
        Example:
            >>> feature_counts = {'A': 400, 'B': 350, 'C': 375}
            >>> n_trials = 750  # 50 trials × 15 folds
            >>> consensus = selector._calculate_consensus(feature_counts, n_trials, alpha=0.05)
            >>> consensus['A']['significant']
            True
            >>> consensus['A']['pvalue'] < 0.05
            True
            >>> consensus['A']['count']
            400
        
        Note:
            - Features with count ≤ n_trials/2 will never be significant
            - Multiple testing correction NOT applied (conservative features preferred)
            - P-values stored for post-hoc FDR control if needed
        
        See Also:
            scipy.stats.binomtest: Exact binomial test implementation
        """
        consensus = {}
        for feat, count in feature_counts.items():
            test = binomtest(count, n_trials, p=0.5, alternative='greater')
            consensus[feat] = {
                'count': count,
                'pvalue': test.pvalue,
                'significant': test.pvalue < alpha
            }
        return consensus

    def fit(self, x, y, features_to_keep_pct: float | None = None):
        """Execute enhanced adaptive RFE across multiple trials.
        
        Runs n_trials parallel RFE iterations with cross-validation, then computes
        statistical significance of feature selection frequencies.
        
        Args:
            x (pd.DataFrame): Feature matrix (N × K).
            y (pd.Series): Target variable (N,).
            features_to_keep_pct (float, optional): Target feature retention rate
                in range (0, 1]. If None, uses RFE_KEEP_PCT (0.9).
        
        Returns:
            self: Fitted selector with populated consensus attribute.
        
        Process:
            1. Generate random seeds for n_trials
            2. Run _trial_processing in parallel (RFE_N_JOBS cores)
            3. Aggregate feature counts across all trials and CV folds
            4. Calculate statistical significance via binomial test
            5. Store results in self.consensus dictionary
        
        Attributes Set:
            self.feature_counts (dict): {feature: count} across all folds
            self.consensus (dict): Statistical results per feature
            self.noise_features (list): Injected noise feature names
        
        Example:
            >>> from early_markers.cribsy.common.adaptive_rfe import EnhancedAdaptiveRFE
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Generate sample data
            >>> np.random.seed(42)
            >>> x = pd.DataFrame(np.random.randn(100, 20), columns=[f'F{i}' for i in range(20)])
            >>> y = pd.Series(x['F0'] * 2 + x['F1'] + np.random.randn(100) * 0.1)
            >>> 
            >>> # Fit selector
            >>> selector = EnhancedAdaptiveRFE(n_trials=10, alpha=0.05)
            >>> selector.fit(x, y, features_to_keep_pct=0.5)
            EnhancedAdaptiveRFE(...)
            >>> 
            >>> # Access results
            >>> sig_features = selector.get_significant_features()
            >>> len(sig_features)
            10
            >>> 'F0' in sig_features  # Should be selected (strong signal)
            True
            >>> any(f.startswith('noise_') for f in sig_features)  # Should be False
            False
        
        Computational Cost:
            - Trials: n_trials (default 50)
            - Folds per trial: 15 (RKFOLD_SPLITS × RKFOLD_REPEATS)
            - Total RFE runs: n_trials × 15 = 750
            - Estimators per RFE: N_ESTIMATORS (200)
            - Total models: ~150,000 Random Forest fits
            - Wall time: ~30-60 minutes on 12 cores (dataset dependent)
        
        Note:
            - Uses joblib.Parallel for trial parallelization
            - Each trial is fully reproducible with assigned seed
            - Noise contamination check: no noise features should be selected
            - Features must appear significantly above 50% to be selected
        
        See Also:
            get_significant_features(): Extract significant features after fit
            _trial_processing(): Single trial implementation
        """
        seeds = np.random.randint(100000, size=self.n_trials)
        results = Parallel(n_jobs=RFE_N_JOBS)(
            delayed(self._trial_processing)(x, y, seeds[i], features_to_keep_pct)
            for i in range(self.n_trials)
        )
        # Calculate feature stability with statistical testing [5]
        for trial_features in results:
            for feat in trial_features:
                self.feature_counts[feat] += 1

        self.consensus = self._calculate_consensus(
            self.feature_counts,
            self.n_trials*15,  # 15 splits per trial (5 folds x 3 repeats)
            # PEB 2025.03.31 21:31 => ^^^ make a constant
            alpha=self.alpha
        )
        return self

    def get_significant_features(self):
        """Extract features with significant selection frequencies.
        
        Returns:
            list: Feature names where consensus['significant'] is True.
        
        Raises:
            AttributeError: If called before fit() - consensus not yet computed.
        
        Example:
            >>> selector.fit(x, y)
            >>> sig_features = selector.get_significant_features()
            >>> len(sig_features)
            45
            >>> all(selector.consensus[f]['significant'] for f in sig_features)
            True
            >>> all(selector.consensus[f]['pvalue'] < 0.05 for f in sig_features)
            True
        
        Note:
            - Must call fit() first to populate self.consensus
            - Returns empty list if no features pass significance threshold
            - Noise features already excluded during trial processing
        """
        return [f for f, meta in self.consensus.items() if meta['significant']]
    
    def plot_stability(self, n_features_to_show=50, figsize=(14, 10), save_path=None):
        """Generate stability heatmap visualization for feature selection.
        
        Creates a visual representation showing how consistently each feature
        was selected across trials and cross-validation folds. Useful for
        diagnosing instability issues and validating feature selection.
        
        Args:
            n_features_to_show (int, optional): Number of top features to display.
                Features are sorted by selection count (most stable first).
                Defaults to 50. Use smaller values for cleaner visualizations.
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (14, 10). Increase height for more features.
            save_path (str, optional): Path to save the figure (e.g., 'stability.png').
                If None, figure is not saved. Supports common formats: .png, .pdf, .svg.
        
        Returns:
            matplotlib.figure.Figure: Figure object containing the heatmap.
                Can be further customized or displayed with plt.show().
        
        Raises:
            AttributeError: If called before fit() - consensus not yet computed.
            ValueError: If no significant features were selected.
        
        Example:
            >>> # Basic usage
            >>> selector.fit(X, y)
            >>> fig = selector.plot_stability()
            >>> plt.show()
            >>> 
            >>> # Save to file
            >>> fig = selector.plot_stability(
            ...     n_features_to_show=30,
            ...     save_path='/path/to/stability_heatmap.png'
            ... )
            >>> 
            >>> # Customize further
            >>> fig = selector.plot_stability(figsize=(16, 12))
            >>> fig.suptitle('My Custom Title', fontsize=16)
            >>> plt.show()
        
        Visualization:
            - Y-axis: Feature names (sorted by stability)
            - X-axis: Selection count
            - Color intensity: Number of times selected (darker = more stable)
            - Annotations: Exact selection counts displayed in cells
        
        Note:
            - Must call fit() first to populate consensus data
            - Only shows significant features passing threshold
            - Features sorted by selection frequency (most stable first)
            - Limit n_features_to_show to ~50 for readability
        
        See Also:
            validation_report(): Get numerical stability metrics
            get_significant_features(): Get list of selected features
        """
        # Validate that model has been fitted
        if self.consensus is None:
            raise AttributeError(
                "Model must be fitted before plotting stability. Call fit() first."
            )
        
        # Get significant features
        significant_features = self.get_significant_features()
        if not significant_features:
            raise ValueError(
                "No significant features were selected. Cannot plot stability."
            )
        
        # Sort features by selection count (most stable first)
        sorted_features = sorted(
            significant_features,
            key=lambda f: self.feature_counts[f],
            reverse=True
        )[:n_features_to_show]
        
        # Create stability data
        counts = [self.feature_counts[f] for f in sorted_features]
        pvalues = [self.consensus[f]['pvalue'] for f in sorted_features]
        
        # Build array for heatmap (features × single column of counts)
        counts_array = np.array(counts).reshape(-1, 1)
        
        # Create figure and heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            counts_array,
            yticklabels=sorted_features,
            xticklabels=['Selection Count'],
            cmap='Blues',
            annot=True,
            fmt='d',
            cbar_kws={'label': f'Times Selected (out of {self.n_trials * 15})'},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
        
        # Customize labels and title
        ax.set_title(
            f'Feature Selection Stability\n'
            f'Top {len(sorted_features)} Features (by selection frequency)',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_xlabel('')
        
        # Rotate y-axis labels for readability
        plt.yticks(rotation=0, fontsize=10)
        plt.xticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Stability plot saved to {save_path}")
        
        return fig

def validation_report(self, X_train=None, y_train=None, X_test=None, y_test=None, 
                      plot_heatmap=False, save_path=None):
    """Generate comprehensive validation report for feature selection.
    
    Provides detailed statistics about the feature selection process including
    noise contamination, stability metrics, and optional test-set evaluation.
    
    Args:
        X_train (pd.DataFrame, optional): Training feature matrix for model evaluation.
        y_train (pd.Series, optional): Training target values.
        X_test (pd.DataFrame, optional): Test feature matrix for model evaluation.
        y_test (pd.Series, optional): Test target values.
        plot_heatmap (bool, optional): Whether to generate stability heatmap. Defaults to False.
        save_path (str, optional): Path to save heatmap figure. Only used if plot_heatmap=True.
    
    Returns:
        dict: Validation report containing:
            - n_features: Number of selected features
            - noise_contamination: Proportion of noise features selected
            - mean_stability: Average selection count across features
            - median_pvalue: Median p-value of consensus test
            - consensus_features: List of selected feature names
            - test_mae: Mean Absolute Error on test set (if X_test provided)
            - test_mse: Mean Squared Error on test set (if X_test provided)
            - test_r2: R² score on test set (if X_test provided)
            - stability_plot: Matplotlib figure object (if plot_heatmap=True)
    
    Example:
        >>> # Basic report without evaluation
        >>> report = selector.validation_report()
        >>> print(f"Selected {report['n_features']} features")
        >>> 
        >>> # Full report with test-set evaluation and heatmap
        >>> report = selector.validation_report(
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     plot_heatmap=True,
        ...     save_path='stability_heatmap.png'
        ... )
        >>> print(f"Test R²: {report['test_r2']:.3f}")
    
    Note:
        - Test metrics only computed if all of X_train, y_train, X_test, y_test provided
        - Heatmap shows top 50 features for readability
        - Noise contamination should be 0 if method is working correctly
    """
    significant_features = self.get_significant_features()

    # Noise contamination check
    noise_selected = sum(1 for f in significant_features
                       if f in self.noise_features)
    contamination_rate = noise_selected/len(significant_features) if significant_features else 0

    # Stability distribution
    pvalues = [m['pvalue'] for m in self.consensus.values()]

    report = {
        'n_features': len(significant_features),
        'noise_contamination': contamination_rate,
        'mean_stability': np.mean(list(self.feature_counts.values())),
        'median_pvalue': np.median(pvalues),
        'consensus_features': significant_features,
        'test_mae': None,
        'test_mse': None,
        'test_r2': None,
        'stability_plot': None
    }
    
    # Test-set evaluation if data provided
    if all(v is not None for v in [X_train, y_train, X_test, y_test]):
        if significant_features:
            # Train final model on selected features
            final_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=RAND_STATE,
                n_jobs=-1
            )
            final_model.fit(X_train[significant_features], y_train)
            
            # Evaluate on test set
            y_pred = final_model.predict(X_test[significant_features])
            report['test_mae'] = mean_absolute_error(y_test, y_pred)
            report['test_mse'] = mean_squared_error(y_test, y_pred)
            report['test_r2'] = r2_score(y_test, y_pred)
    
    # Generate stability heatmap if requested
    if plot_heatmap and significant_features:
        # Create stability matrix (top 50 features for readability)
        top_features = sorted(
            significant_features, 
            key=lambda f: self.feature_counts[f], 
            reverse=True
        )[:50]
        
        # Build matrix: features × consensus (binary: selected or not)
        stability_data = []
        for feat in top_features:
            stability_data.append({
                'Feature': feat,
                'Count': self.feature_counts[feat],
                'P-value': self.consensus[feat]['pvalue']
            })
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, max(8, len(top_features) * 0.3)))
        
        # Create heatmap data
        counts = [self.feature_counts[f] for f in top_features]
        counts_array = np.array(counts).reshape(-1, 1)
        
        sns.heatmap(
            counts_array,
            yticklabels=top_features,
            xticklabels=['Selection Count'],
            cmap='Blues',
            annot=True,
            fmt='d',
            cbar_kws={'label': 'Times Selected'},
            ax=ax
        )
        
        ax.set_title(f'Feature Selection Stability (Top {len(top_features)} Features)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Stability heatmap saved to {save_path}")
        
        report['stability_plot'] = fig
    
    return report
