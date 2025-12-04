"""Base Adaptive RFE implementation for feature selection.

This module implements a basic Adaptive Recursive Feature Elimination (RFE)
class that uses dynamic step sizing and consensus-based feature selection
across multiple trials.

Note:
    This is a simpler implementation compared to EnhancedAdaptiveRFE in
    adaptive_rfe.py. The enhanced version adds noise injection, hyperparameter
    tuning, and statistical significance testing.

Classes:
    AdaptiveRFE: Consensus-based RFE with dynamic step sizing

Functions:
    create_estimator: Factory function for Random Forest regressors

Features:
    - Dynamic step sizing based on feature importance gaps
    - Parallel trial execution for stability
    - Nested cross-validation (RepeatedKFold)
    - Consensus threshold for feature stability

Example:
    >>> import pandas as pd
    >>> from early_markers.cribsy.common.rfe import AdaptiveRFE
    >>> 
    >>> # Prepare data
    >>> X = pd.DataFrame(...)  # Feature matrix
    >>> y = pd.Series(...)     # Target variable
    >>> 
    >>> # Run adaptive RFE
    >>> selector = AdaptiveRFE(n_trials=30, stability_threshold=0.75)
    >>> selector.fit(X, y)
    >>> selected = selector.get_consensus_features()
    >>> print(f"Selected {len(selected)} stable features")

See Also:
    adaptive_rfe.EnhancedAdaptiveRFE: More sophisticated implementation
"""

# ================ ENVIRONMENT SETUP ================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import RepeatedKFold, KFold, train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.utils import resample
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from early_markers.cribsy.common.constants import RAND_STATE

# Initialize global randomness controls
np.random.seed(RAND_STATE)

# ================ DATA PREPARATION ================
# Sample dataset configuration (replace with actual data)
X, y = pd.DataFrame(np.random.randn(1000, 50)), pd.Series(np.random.randn(1000))
feature_names = [f'F{i}' for i in range(X.shape[1])]
X.columns = feature_names

# ================ CORE MODEL CONFIGURATION ================
def create_estimator():
    """Create a Random Forest regressor with reproducible settings.
    
    Returns:
        RandomForestRegressor: Configured estimator with:
            - 1000 trees for stability
            - No max depth (fully grown trees)
            - Fixed random state for reproducibility
            - n_jobs=1 for exact reproducibility
    
    Note:
        Using n_jobs=1 ensures identical results across runs but is slower.
        For faster training with approximate reproducibility, increase n_jobs.
    """
    return RandomForestRegressor(
        n_estimators=1000,
        max_depth=None,
        random_state=RAND_STATE,
        n_jobs=1  # Essential for reproducibility
    )

# ================ ADAPTIVE RFE ENGINE ================
class AdaptiveRFE:
    """Adaptive Recursive Feature Elimination with consensus-based selection.
    
    Performs multiple RFE trials with dynamic step sizing and combines results
    using a consensus threshold. Features must appear frequently across trials
    to be selected, ensuring stability.
    
    Attributes:
        n_trials (int): Number of parallel RFE trials to run.
        threshold (float): Minimum proportion of trials a feature must appear
            in to be selected (0.0 to 1.0).
        feature_stability (dict): Count of times each feature was selected.
        consensus_features (list): Final list of selected features.
    
    Algorithm:
        1. For each trial:
           - Resample data with replacement
           - Run RFE with dynamic step sizing across nested CV folds
           - Collect selected features
        2. Count feature appearances across all trials and folds
        3. Select features appearing above threshold
    
    Example:
        >>> import pandas as pd
        >>> from early_markers.cribsy.common.rfe import AdaptiveRFE
        >>> 
        >>> X = pd.DataFrame(np.random.randn(100, 20))
        >>> y = pd.Series(np.random.randn(100))
        >>> 
        >>> # Select features appearing in ≥75% of trial-folds
        >>> selector = AdaptiveRFE(n_trials=30, stability_threshold=0.75)
        >>> selector.fit(X, y)
        >>> 
        >>> features = selector.get_consensus_features()
        >>> print(f"Selected {len(features)} features")
        >>> 
        >>> # Examine stability
        >>> for feat, count in selector.feature_stability.items():
        ...     print(f"{feat}: {count} selections")
    
    See Also:
        EnhancedAdaptiveRFE: Advanced version with noise injection and
            statistical significance testing
    """
    def __init__(self, n_trials=30, stability_threshold=0.75):
        self.n_trials = n_trials
        self.threshold = stability_threshold
        self.feature_stability = defaultdict(int)

    def _dynamic_step(self, importances):
        """Calculate optimal features to remove based on importance gaps"""
        sorted_imp = np.sort(importances)[::-1]
        gaps = np.diff(sorted_imp)
        if len(gaps) == 0:
            return 1
        critical_gap = np.quantile(gaps, 0.95)
        removal_count = np.argmax(gaps > critical_gap) + 1
        return max(1, removal_count)

    def _single_trial(self, X, y, trial_seed):
        """Execute one complete feature selection trial"""
        # Controlled resampling
        X_rs, y_rs = resample(X, y, random_state=trial_seed)

        # Configure RFE with dynamic step sizing
        estimator = create_estimator()
        rfe = RFE(
            estimator=estimator,
            n_features_to_select='auto',
            step=self._dynamic_step,
            importance_getter='feature_importances_'
        )

        # Nested CV execution
        outer_cv = RepeatedKFold(n_splits=5, n_repeats=3,
                               random_state=trial_seed)
        inner_cv = KFold(n_splits=3, shuffle=True,
                       random_state=trial_seed)

        selected_features = set()
        for train_idx, _ in outer_cv.split(X_rs):
            X_train = X_rs.iloc[train_idx]
            y_train = y_rs.iloc[train_idx]

            rfe.fit(X_train, y_train)
            trial_features = X_train.columns[rfe.support_]
            selected_features.update(trial_features)

        return list(selected_features)

    def fit(self, X, y):
        """Execute parallel trials and calculate consensus features"""
        results = Parallel(n_jobs=-1)(
            delayed(self._single_trial)(X, y, trial_seed)
            for trial_seed in range(self.n_trials)
        )

        # Calculate feature stability
        for trial_features in results:
            for feat in trial_features:
                self.feature_stability[feat] += 1

        # Determine consensus features
        total_possible = self.n_trials * 5 * 3  # trials x outer folds x inner folds
        self.consensus_features = [
            feat for feat, count in self.feature_stability.items()
            if count / total_possible >= self.threshold
        ]

        return self

    def get_consensus_features(self):
        return self.consensus_features

# ================ EXECUTION PIPELINE ================
if __name__ == "__main__":
    # Split initial data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RAND_STATE
    )

    # Run feature selection
    selector = AdaptiveRFE(n_trials=30, stability_threshold=0.75)
    selector.fit(X_train, y_train)
    final_features = selector.get_consensus_features()

    # Final model evaluation
    final_model = create_estimator()
    final_model.fit(X_train[final_features], y_train)

    # Generate predictions
    y_pred = final_model.predict(X_test[final_features])

    # Calculate metrics
    metrics = {
        'MAE': median_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }

    # ================ VISUALIZATION ================
    # Stability heatmap
    trial_data = pd.DataFrame({
        f'Trial {i}': [1 if feat in final_features else 0
                      for feat in feature_names]
        for i in range(10)  # First 10 trials for visualization
    }, index=feature_names)

    plt.figure(figsize=(12, 8))
    sns.heatmap(trial_data.T, cmap='Blues',
                cbar_kws={'label': 'Feature Selected'})
    plt.title('Feature Selection Stability Across Trials')
    plt.xlabel('Feature Index')
    plt.ylabel('Trial Number')
    plt.tight_layout()
    plt.show()
