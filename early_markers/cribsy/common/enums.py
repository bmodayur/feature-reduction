"""Enumeration types for the early-markers analysis pipeline.

This module defines enumeration classes used throughout the codebase to ensure
type safety and clarity when specifying analysis options.

Enumerations:
    - RfeType: Output format for RFE (feature selection) results
    - MetricType: Performance metrics for sample size calculations
    - ErrorType: Error estimation methods

Example:
    >>> from early_markers.cribsy.common.enums import MetricType, RfeType
    >>> metric = MetricType.SENSITIVITY
    >>> rfe_format = RfeType.SUMMARY
"""
from enum import Enum


class RfeType(Enum):
    """Output format specification for RFE (Recursive Feature Elimination) results.

    Used by BayesianData.get_top_n_rfes() to control the level of detail in
    output tables and Excel reports.

    Attributes:
        SUMMARY: High-level summary with key metrics only (default)
        DETAIL: Detailed output with all confusion matrix elements and CIs

    Example:
        >>> from early_markers.cribsy.common.enums import RfeType
        >>> bd.get_top_n_rfes(n=10, rfe_type=RfeType.SUMMARY)
    """
    SUMMARY = 1
    DETAIL = 2


class MetricType(Enum):
    """Performance metric types for sample size estimation.

    Used by RocMetricSampleSize class to specify which diagnostic performance
    metric to use for sample size calculations.

    Attributes:
        SENSITIVITY: True positive rate (TPR), recall
        SPECIFICITY: True negative rate (TNR)
        PPV: Positive predictive value, precision
        NPV: Negative predictive value
        F1: F1 score (harmonic mean of precision and recall)
        ACCURACY: Overall classification accuracy

    Example:
        >>> from early_markers.cribsy.common.enums import MetricType
        >>> mss = RocMetricSampleSize()
        >>> n = mss.estimate_n(MetricType.SENSITIVITY, sens=0.85, ...)
    """
    SENSITIVITY = 1
    SPECIFICITY = 2
    PPV = 3
    NPV = 4
    F1 = 5
    ACCURACY = 6


class Metric(Enum):
    SENS = "Sensitivity"
    SPEC = "Specificity"
    PPV = "PPV"
    NPV = "NPV"
    ACC = "Accuracy"


class ErrorType(Enum):
    """Error estimation method specification.

    Used to indicate whether to use standard errors or confidence intervals
    in statistical calculations.

    Attributes:
        SE: Standard error
        CI: Confidence interval

    Note:
        Currently defined but not extensively used in the codebase.
        Kept for future extensibility.
    """
    SE = 1
    CI = 2


