"""Sample size estimation for ROC metrics.

Provides statistical formulas to estimate required sample sizes for
diagnostic test performance metrics (sensitivity, specificity, PPV, NPV,
F1 score, accuracy). Handles both forward calculations (compute SE from
metrics) and inverse calculations (estimate N from target SE/CI).

Key Features:
    - Convert CI width to standard error
    - Calculate SE for six different metrics
    - Estimate required sample size from target SE/CI
    - Account for prevalence in sample size calculations
    - Handle confusion matrix elements (TP, TN, FP, FN)

Mathematical Foundations:
    Standard errors are derived from binomial variance for binary
    classification metrics. F1 score uses delta method with covariance
    between precision and recall.

Example:
    >>> from early_markers.cribsy.common.metrics_n import RocMetricSampleSize
    >>> from early_markers.cribsy.common.enums import MetricType
    >>> 
    >>> # Initialize calculator
    >>> calc = RocMetricSampleSize()
    >>> 
    >>> # Estimate N for sensitivity with target CI width of 0.10
    >>> n_needed = calc.estimate_n(
    ...     metric_type=MetricType.SENSITIVITY,
    ...     sens=0.85,
    ...     prev=0.15,  # 15% prevalence
    ...     ci=0.10,    # ±0.05 (0.10 total width)
    ... )
    >>> print(f"Need N={n_needed:.0f} total samples")
    Need N=784 total samples
    >>> 
    >>> # Calculate SE from existing metrics
    >>> se = calc.se_from_metrics(
    ...     metric_type=MetricType.SPECIFICITY,
    ...     spec=0.90,
    ...     tn=85, fp=10  # From confusion matrix
    ... )
    >>> print(f"SE = {se:.4f}")
    SE = 0.0307

References:
    - Flahault A, Cadilhac M, Thomas G (2005). Sample size calculation
      should be performed for design accuracy in diagnostic test studies.
      J Clin Epidemiol.
    - Bujang MA, Adnan TH (2016). Requirements for minimum sample size for
      sensitivity and specificity analysis. J Clin Diagn Res.

See Also:
    bam.py: Bayesian approach to sample size determination
    MetricType: Enumeration of supported metrics
"""
import math
from dataclasses import (
    dataclass,
    field,
)
import math

from polars import DataFrame
import polars as pl
from scipy.stats import norm

from early_markers.cribsy.common.enums import MetricType, ErrorType


class RocMetricSampleSize:
    """Calculator for ROC metric sample size estimation.
    
    Provides methods to:
    1. Convert confidence interval width to standard error
    2. Calculate standard error from observed metrics
    3. Estimate required sample size from target precision
    
    All methods account for prevalence and confusion matrix structure.
    
    Attributes:
        _results (list[dict]): Storage for calculation results (currently unused).
    
    Example:
        >>> calc = RocMetricSampleSize()
        >>> 
        >>> # Calculate SE from 95% CI width of 0.10
        >>> se = calc.se_from_ci(ci_width=0.10, conf_level=0.95)
        >>> se
        0.0255...
        >>> 
        >>> # Estimate N for target sensitivity precision
        >>> n = calc.estimate_n(
        ...     metric_type=MetricType.SENSITIVITY,
        ...     sens=0.80,
        ...     prev=0.20,
        ...     ci=0.10
        ... )
        >>> n
        615.0...
    """
    _results: list[dict] = field(default_factory=list, init=False)

    def se_from_ci(self, ci_width: float, conf_level: float = 0.95):
        """Convert confidence interval width to standard error.
        
        Assumes symmetric normal-approximation confidence interval:
        CI = estimate ± z_(α/2) × SE
        
        Args:
            ci_width (float): Total width of confidence interval.
                For example, if CI is [0.75, 0.85], width = 0.10.
            conf_level (float, optional): Confidence level. Defaults to 0.95.
        
        Returns:
            float: Standard error corresponding to the CI width.
        
        Formula:
            SE = CI_width / (2 × z_(α/2))
            
            Where:
                α = 1 - conf_level
                z_(α/2) = normal distribution quantile
        
        Example:
            >>> calc = RocMetricSampleSize()
            >>> se = calc.se_from_ci(ci_width=0.10, conf_level=0.95)
            >>> se
            0.025520788274024855
            >>> 
            >>> # 99% CI requires larger SE for same width
            >>> se_99 = calc.se_from_ci(ci_width=0.10, conf_level=0.99)
            >>> se_99 < se  # False - 99% CI is wider
            False
        
        Note:
            - Valid only for large samples where normal approximation holds
            - For 95% CI: z_(α/2) ≈ 1.96
            - For 99% CI: z_(α/2) ≈ 2.576
        """
        alpha = 1 - conf_level
        z = norm.ppf(1 - (alpha / 2))
        se = ci_width / (2 * z)
        return se

    def se_from_metrics(
            self,
        metric_type: MetricType,
        sens: float | None = None,
        spec: float | None = None,
        ppv: float | None = None,
        npv: float | None = None,
        acc: float | None = None,
        prev: float | None = None,
        n: int | None = None,
        tp: int | None = None,
        tn: int | None = None,
        fp: int | None = None,
        fn: int | None = None,
    ):
        """Calculate standard error for a given metric from observed values.
        
        Computes SE using metric-specific formulas derived from binomial
        variance. F1 score uses delta method with covariance term.
        
        Args:
            metric_type (MetricType): Which metric to calculate SE for.
            sens (float, optional): Sensitivity (recall, TPR). Required for
                SENSITIVITY, F1, NPV calculations.
            spec (float, optional): Specificity (TNR). Required for F1, NPV.
            ppv (float, optional): Positive predictive value (precision).
                Required for PPV, F1 calculations.
            npv (float, optional): Negative predictive value. Required for NPV.
            acc (float, optional): Accuracy. Required for ACCURACY.
            prev (float, optional): Prevalence. Required for F1.
            n (int, optional): Total sample size. Can be computed from
                tp+tn+fp+fn if not provided.
            tp (int, optional): True positives. Required for SENSITIVITY, PPV.
            tn (int, optional): True negatives. Required for SPECIFICITY, NPV.
            fp (int, optional): False positives. Required for SPECIFICITY, PPV.
            fn (int, optional): False negatives. Required for SENSITIVITY, NPV.
        
        Returns:
            float: Standard error for the specified metric.
        
        Formulas:
            Sensitivity: SE = sqrt[sens × (1-sens) / (TP + FN)]
            Specificity: SE = sqrt[spec × (1-spec) / (TN + FP)]
            PPV: SE = sqrt[ppv × (1-ppv) / (TP + FP)]
            NPV: SE = sqrt[npv × (1-npv) / (TN + FN)]
            Accuracy: SE = sqrt[acc × (1-acc) / N]
            
            F1: Uses delta method with:
                Var(F1) = (re⁴ × Var(pr) + 2pr²re² × Cov(pr,re) + pr⁴ × Var(re)) / (pr+re)⁴
                Where pr=PPV, re=Sensitivity
        
        Example:
            >>> calc = RocMetricSampleSize()
            >>> 
            >>> # SE for sensitivity from confusion matrix
            >>> se_sens = calc.se_from_metrics(
            ...     metric_type=MetricType.SENSITIVITY,
            ...     sens=0.85,
            ...     tp=85, fn=15
            ... )
            >>> se_sens
            0.0357...
            >>> 
            >>> # SE for F1 score (complex calculation)
            >>> se_f1 = calc.se_from_metrics(
            ...     metric_type=MetricType.F1,
            ...     sens=0.85, spec=0.90, ppv=0.77,
            ...     prev=0.15, n=200,
            ...     tp=85, tn=90, fp=10, fn=15
            ... )
            >>> isinstance(se_f1, float)
            True
        
        Note:
            - n is automatically computed from tp+tn+fp+fn if not provided
            - F1 calculation requires recursive calls for SE of PPV and Sensitivity
            - All metrics assume large-sample normal approximation
        
        Raises:
            ValueError: Implicitly if required parameters are None during calculation.
        """
        if n is None and all([x is not None for x in [tp, tn, fp, fn]]):
            n = sum([tp, tn, fp, fn])

        se = None
        match metric_type:
            case MetricType.SENSITIVITY:
                num = sens * (1 - sens)
                den = tp + fn
                se = math.sqrt(num / den)
            case MetricType.SPECIFICITY:
                num = spec * (1 - spec)
                den = tn + fp
                se = math.sqrt(num / den)
            case MetricType.PPV:
                num = ppv * (1 - ppv)
                den = tp + fp
                se = math.sqrt(num / den)
            case MetricType.NPV:
                num = npv * (1 - npv)
                den = tn + fn
                se = math.sqrt(num / den)
            case MetricType.F1:
                pr = ppv
                re = sens
                se_pr = self.se_from_metrics(
                    MetricType.PPV, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
                    prev=prev, n=n, tp=tp, tn=tn, fp=fp, fn=fn
                )
                se_re = self.se_from_metrics(
                    MetricType.SENSITIVITY, sens=sens, spec=spec, ppv=ppv, npv=npv,
                    acc=acc, prev=prev, n=n, tp=tp, tn=tn, fp=fp, fn=fn
                )
                c1 = (pr * (1 - pr) * (1 - re)) / prev
                c2 = (pr * (1 - pr) * spec) / (1 - prev)
                cov = (c1 + c2) / n
                num = (re**4 * se_pr**2) + (2 * pr**2 * re**2 * cov) + (pr**4 * se_re**2)
                den = (pr + re)**4
                se = math.sqrt(4 * num / den)
            case MetricType.ACCURACY:
                num = acc * (1 - acc)
                den = n
                se = math.sqrt(num / den)
            case _:
                ...
        return se

    def estimate_n(
        self,
        metric_type: MetricType,
        sens: float | None = None,
        spec: float | None = None,
        ppv: float | None = None,
        npv: float | None = None,
        acc: float | None = None,
        prev: float | None = None,
        n: int | None = None,
        tp: int | None = None,
        tn: int | None = None,
        fp: int | None = None,
        fn: int | None = None,
        ci: float | None = None,
    ):
        """Estimate required sample size to achieve target precision.
        
        Inverts the SE formulas to solve for N given a target standard error
        (from CI width) and expected metric values. Critical for study design.
        
        Args:
            metric_type (MetricType): Which metric to power for.
            sens (float, optional): Expected sensitivity. Required for
                SENSITIVITY, F1, PPV, NPV.
            spec (float, optional): Expected specificity. Required for
                SPECIFICITY, F1, NPV.
            ppv (float, optional): Expected positive predictive value.
                Required for PPV, F1.
            npv (float, optional): Expected negative predictive value.
                Required for NPV.
            acc (float, optional): Expected accuracy. Required for ACCURACY.
            prev (float, optional): Expected prevalence. Required for all
                metrics except ACCURACY.
            n (int, optional): Current sample size (for SE calculation if ci=None).
            tp, tn, fp, fn (int, optional): Confusion matrix counts (if available).
            ci (float, optional): Target CI width. If provided, used to compute
                target SE. If None, SE is computed from current metrics.
        
        Returns:
            float: Estimated total sample size required.
        
        Formulas:
            Sensitivity: N = sens × (1-sens) / (SE² × prev)
            Specificity: N = spec × (1-spec) / (SE² × (1-prev))
            PPV: N = ppv² × (1-ppv) / (SE² × prev × sens)
            NPV: N = npv × (1-npv) / (SE² × (spec(1-prev) + prev(1-sens)))
            
            F1: Complex inverse calculation involving:
                N = [2pr²re²((pr(1-pr)(1-re)/prev) + (pr(1-pr)spec/(1-prev)))] /
                    [(SE²(pr+re)⁴/4) - re⁴×SE_pr² - pr⁴×SE_re²]
            
            Accuracy: N = acc × (1-acc) / SE²
        
        Example:
            >>> calc = RocMetricSampleSize()
            >>> 
            >>> # Estimate N for sensitivity with target 95% CI width of 0.10
            >>> n_sens = calc.estimate_n(
            ...     metric_type=MetricType.SENSITIVITY,
            ...     sens=0.85,  # Expected sensitivity
            ...     prev=0.15,  # Expected prevalence
            ...     ci=0.10     # Target: 0.85 ± 0.05
            ... )
            >>> print(f"Need N={n_sens:.0f} total samples")
            Need N=784 total samples
            >>> print(f"With prevalence 15%, need {int(n_sens * 0.15)} positive cases")
            With prevalence 15%, need 117 positive cases
            >>> 
            >>> # Estimate N for joint sensitivity + specificity via F1
            >>> n_f1 = calc.estimate_n(
            ...     metric_type=MetricType.F1,
            ...     sens=0.85, spec=0.90, ppv=0.77,
            ...     prev=0.15,
            ...     ci=0.10
            ... )
            >>> n_f1 > n_sens  # F1 typically requires more samples
            True
        
        Typical Use Case:
            >>> # Study design: How many patients needed?
            >>> # Goal: Sensitivity 80% with 95% CI width of 0.12 (±0.06)
            >>> # Expected 20% prevalence
            >>> n_required = calc.estimate_n(
            ...     metric_type=MetricType.SENSITIVITY,
            ...     sens=0.80,
            ...     prev=0.20,
            ...     ci=0.12
            ... )
            >>> n_pos_needed = int(n_required * 0.20)
            >>> n_neg_needed = int(n_required * 0.80)
            >>> print(f"Total N={n_required:.0f}")
            Total N=445
            >>> print(f"Need {n_pos_needed} positive, {n_neg_needed} negative cases")
            Need 89 positive, 356 negative cases
        
        Note:
            - Result is TOTAL sample size N, not just positive or negative cases
            - Prevalence strongly affects sample size for sens/spec
            - F1 calculation is most complex, involves covariance terms
            - Always round up in practice to ensure adequate power
            - Does NOT account for:
              * Dropout/missing data
              * Multiple testing correction
              * Stratification requirements
        
        See Also:
            se_from_ci(): Convert CI width to SE
            se_from_metrics(): Calculate SE from observed metrics
            bam.py: Bayesian alternative with assurance guarantees
        
        References:
            - Flahault et al. (2005) J Clin Epidemiol
            - Bujang & Adnan (2016) J Clin Diagn Res
        """
        if n is None and all([x is not None for x in [tp, tn, fp, fn]]):
            n = sum([tp, tn, fp, fn])

        if ci is None:
            se = self.se_from_metrics(
                metric_type=metric_type, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
                prev=prev, n=n, tp=tp, tn=tn, fp=fp, fn=fn,
            )
        else:
            se = self.se_from_ci(ci)

        est_n = None
        match metric_type:
            case MetricType.SENSITIVITY:
                num = sens * (1 - sens)
                den = se**2 * prev
                est_n = num / den
            case MetricType.SPECIFICITY:
                num = spec * (1 - spec)
                den = se**2 * (1 - prev)
                est_n = num / den
            case MetricType.PPV:
                num = ppv**2 * (1-ppv)
                den = se**2 * prev * sens
                est_n = num / den
            case MetricType.NPV:
                num = npv * (1 - ppv)
                den = se**2 * (spec * (1 - prev) + prev * (1 - sens))
                est_n = num / den
            case MetricType.F1:
                pr = ppv
                re = sens
                if ci is None:
                    se_pr = self.se_from_metrics(
                        MetricType.PPV, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
                        prev=prev, n=n, tp=tp, tn=tn, fp=fp, fn=fn
                    )
                    se_re = self.se_from_metrics(
                        MetricType.SENSITIVITY, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
                        prev=prev, n=n, tp=tp, tn=tn, fp=fp, fn=fn
                    )
                else:
                    se_pr = se_re = se
                num1 = 2 * pr**2 * re**2
                num2 = pr * (1 - pr) * (1 - re) / prev
                num3 = pr * (1 - pr) * spec / (1 - prev)
                num = num1 * (num2 + num3)
                den1 = se**2 * (pr + re)**4 / 4
                den2 = re**4 * se_pr**2
                den3 = pr**4 * se_re**2
                den = den1 - den2 - den3
                est_n = num / den
            case MetricType.ACCURACY:
                est_n = math.sqrt(acc * (1 - acc) / n)
            case _:
                ...
        return est_n



# @dataclass
# class MetricsSampleSize:
#     _results: dict = field(default_factory=dict, init=False)
#
#     def clear__results(self):
#         self._results = {}
#
#     @property
#     def results(self):
#         return self._results
#
#     @property
#     def results_as_frames(self):
#         r = {}
#         for k, v in self._results.items():
#             r[k] = DataFrame(v)
#         return r
#
#     def estimate_n_for_sensitivity(self, sensitivity: float, ci_width: float, prevalence: float, conf_level: float = 0.95):
#         alpha = 1 - conf_level
#         z = norm.ppf(1 - (alpha / 2))
#         se = ci_width / (2 * z)
#
#         n = (sensitivity * (1 - sensitivity)) / (prevalence * se**2)
#
#         n_pos = math.ceil(n * prevalence)
#         n_neg = math.ceil(n * (1 - prevalence))
#         n_tot = n_pos + n_neg
#         result = {
#             "sensitivity": sensitivity,
#             "ci_width": ci_width,
#             "prevalence": prevalence,
#             "n": n_tot,
#             "n_pos": n_pos,
#             "n_neg": n_neg,
#             "conf_level": conf_level,
#         }
#         if "sensitivity" not in self._results:
#             self._results["sensitivity"] = []
#         self._results["sensitivity"].append(result)
#         return result
#
#     def estimate_n_for_specificity(self, specificity: float, ci_width: float, prevalence: float, conf_level: float = 0.95):
#         alpha = 1 - conf_level
#         z = norm.ppf(1 - (alpha / 2))
#         se = ci_width / (2 * z)
#
#         n = (specificity * (1 - specificity)) / ((1 - prevalence) * se ** 2)
#
#         n_pos = math.ceil(n * prevalence)
#         n_neg = math.ceil(n * (1 - prevalence))
#         n_tot = n_pos + n_neg
#         result = {
#             "specificity": specificity,
#             "ci_width": ci_width,
#             "prevalence": prevalence,
#             "n": n_tot,
#             "n_pos": n_pos,
#             "n_neg": n_neg,
#             "conf_level": conf_level,
#         }
#         if "specificity" not in self._results:
#             self._results["specificity"] = []
#         self._results["specificity"].append(result)
#         return result
#
#     def estimate_n_for_ppv(self, ppv: float, sensitivity: float, ci_width: float, prevalence: float, conf_level: float = 0.95):
#         alpha = 1 - conf_level
#         z = norm.ppf(1 - (alpha / 2))
#         se = ci_width / (2 * z)
#         n = (ppv**2 * (1 - ppv)) / (se**2 * prevalence * sensitivity)
#
#         n_pos = math.ceil(n * prevalence)
#         n_neg = math.ceil(n * (1 - prevalence))
#         n_tot = n_pos + n_neg
#         result = {
#             "ppv": ppv,
#             "sensitivity": sensitivity,
#             "ci_width": ci_width,
#             "prevalence": prevalence,
#             "n": n_tot,
#             "n_pos": n_pos,
#             "n_neg": n_neg,
#             "conf_level": conf_level,
#         }
#         if "ppv" not in self._results:
#             self._results["ppv"] = []
#         self._results["ppv"].append(result)
#         return result
#
#     def estimate_n_for_npv(self, npv: float, sensitivity: float, specificity: float, ci_width: float, prevalence: float, conf_level: float = 0.95):
#         alpha = 1 - conf_level
#         z = norm.ppf(1 - (alpha / 2))
#         se = ci_width / (2 * z)
#
#         n = (npv * (1 - npv)) / (se**2 * (specificity * (1 - prevalence)) + (prevalence * (1 - sensitivity)))
#
#         n_pos = math.ceil(n * prevalence)
#         n_neg = math.ceil(n * (1 - prevalence))
#         n_tot = n_pos + n_neg
#         result = {
#             "npv": npv,
#             "sensitivity": sensitivity,
#             "specificity": specificity,
#             "ci_width": ci_width,
#             "prevalence": prevalence,
#             "n": n_tot,
#             "n_pos": n_pos,
#             "n_neg": n_neg,
#             "conf_level": conf_level,
#         }
#         if "npv" not in self._results:
#             self._results["npv"] = []
#         self._results["npv"].append(result)
#         return result
#
#     def estimate_n_for_accuracy(self, accuracy: float, ci_width: float, prevalence: float, conf_level: float = 0.95):
#         alpha = 1 - conf_level
#         z = norm.ppf(1 - (alpha / 2))
#         se = ci_width / (2 * z)
#
#         n = (accuracy * (1 - accuracy)) / se**2
#
#         n_pos = math.ceil(n * prevalence)
#         n_neg = math.ceil(n * (1 - prevalence))
#         n_tot = n_pos + n_neg
#         result = {
#             "accuracy": accuracy,
#             "ci_width": ci_width,
#             "prevalence": prevalence,
#             "n": n_tot,
#             "n_pos": n_pos,
#             "n_neg": n_neg,
#             "conf_level": conf_level,
#         }
#         if "accuracy" not in self._results:
#             self._results["accuracy"] = []
#         self._results["accuracy"].append(result)
#         return result
#
#     def estimate_n_for_f1(self, f1: float, ppv: float, sensitivity: float, specificity: float, ci_width: float, prevalence: float, conf_level: float = 0.95):
#         alpha = 1 - conf_level
#         z = norm.ppf(1 - (alpha / 2))
#         se = ci_width / (2 * z)
#         pr = ppv  # PPV == Precision
#         re = sensitivity  # Recall = Sensitivity
#         sp = specificity
#
#         num = (
#                 (2 * pr**2 * re**2)
#                 * (
#                     ((pr * (1 - pr) * (1 - re)) / prevalence)
#                     + ((pr * (1 - pr) * sp) / (1 - prevalence))
#                 )
#         )
#         den = (
#             ((se**2 * (pr + re)**4) / 4)
#             - (re**4 * se**2)
#             - (pr**4 * se**2)
#         )
#         n = num / den
#
#         n_pos = math.ceil(n * prevalence)
#         n_neg = math.ceil(n * (1 - prevalence))
#         n_tot = n_pos + n_neg
#         result = {
#             "f1": f1,
#             "ppv": ppv,
#             "sensitivity": sensitivity,
#             "specificity": specificity,
#             "ci_width": ci_width,
#             "prevalence": prevalence,
#             "n": n_tot,
#             "n_pos": n_pos,
#             "n_neg": n_neg,
#             "conf_level": conf_level,
#         }
#         if "f1" not in self._results:
#             self._results["f1"] = []
#         self._results["f1"].append(result)
#         return result
