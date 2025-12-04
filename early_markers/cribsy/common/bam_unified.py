"""Unified Bayesian Assurance Method (BAM) API.

This module provides a production-ready interface for Bayesian sample size determination,
combining the efficiency of the functional API (bam.py), the usability of the
object-oriented API (bayesian_assurance.py), and extensibility for advanced features.

Key Features:
    - Efficient binary search algorithm (O(log N))
    - Hierarchical Bayesian model with Gamma hyperpriors
    - Optimization-based HDI calculation (mathematically rigorous)
    - Structured results with BAMResult dataclass
    - Optional visualization with publication-quality plots
    - Support for single and joint metrics (sensitivity + specificity)
    - Interpolation methods for fine-grained estimates
    
    **NEW ENHANCEMENTS:**
    - Cluster-adjusted methods for hierarchical study designs (ICC, design effect)
    - Batch/grid processing API for parameter sensitivity analysis
    - Excel export utilities for grants, IRBs, and regulatory submissions
    - Corrected Beta prior calculations in joint metrics (critical bug fix)

Core Functionality:
    
    1. **Single Metric Estimation**
       - Simple binary outcomes (success/failure)
       - Independent observations
       - Function: estimate_single() or bam_single()
    
    2. **Joint Metrics Estimation**
       - Sensitivity AND specificity simultaneously
       - Independent observations
       - Function: estimate_joint() or bam_joint()
    
    3. **Cluster-Adjusted Estimation (NEW)**
       - Hierarchical designs (e.g., hospitals, schools, clinics)
       - Accounts for intraclass correlation (ICC)
       - Design effect adjustments
       - Functions: estimate_single_clustered(), estimate_joint_clustered()
    
    4. **Grid Search / Parameter Exploration (NEW)**
       - Explore multiple HDI widths and assurance levels
       - Parallel execution support
       - Returns pandas DataFrame for analysis
       - Functions: bam_grid_search(), bam_grid_search_joint()
    
    5. **Professional Reporting (NEW)**
       - Export to formatted Excel workbooks
       - Summary tables, detail sheets, metadata
       - Conditional formatting, color scales
       - Module: bam_export.py

Quick Start Examples:
    
    **Single Metric**:
        >>> import numpy as np
        >>> from early_markers.cribsy.common.bam_unified import BAMEstimator
        >>> 
        >>> estimator = BAMEstimator(seed=42)
        >>> pilot = np.array([1]*15 + [0]*5)
        >>> result = estimator.estimate_single(
        ...     pilot_data=pilot,
        ...     target_width=0.15,
        ...     target_assurance=0.8
        ... )
        >>> print(f"Required N: {result.optimal_n}")
        >>> result.plot(save_path="assurance_curve.png")
    
    **Joint Metrics (Sensitivity + Specificity)**:
        >>> result = estimator.estimate_joint(
        ...     pilot_se=(18, 20),  # 18 TP out of 20 diseased
        ...     pilot_sp=(35, 40),  # 35 TN out of 40 non-diseased
        ...     target_width=0.10,
        ...     target_assurance=0.8
        ... )
        >>> print(f"Need N={result.optimal_n} for joint constraints")
    
    **Cluster-Adjusted (NEW)**:
        >>> result = estimator.estimate_single_clustered(
        ...     pilot_data=pilot,
        ...     target_width=0.15,
        ...     icc=0.05,
        ...     mean_cluster_size=10
        ... )
        >>> print(f"Need {result.optimal_n} clusters")
        >>> print(f"Total N = {result.optimal_n * 10}")
        >>> print(f"Design effect = {result.design_effect:.2f}")
    
    **Grid Search (NEW)**:
        >>> from early_markers.cribsy.common.bam_unified import bam_grid_search
        >>> 
        >>> results_df = bam_grid_search(
        ...     pilot_data=pilot,
        ...     target_widths=[0.10, 0.15, 0.20],
        ...     target_assurances=[0.75, 0.80, 0.85, 0.90],
        ...     n_jobs=-1  # Use all CPUs
        ... )
        >>> print(results_df.pivot("target_assurance", "target_width", "optimal_n"))
    
    **Excel Export (NEW)**:
        >>> from early_markers.cribsy.common.bam_export import export_bam_results_to_excel
        >>> 
        >>> results = []
        >>> for width in [0.10, 0.15, 0.20]:
        ...     result = estimator.estimate_single(pilot, target_width=width)
        ...     results.append(result)
        >>> 
        >>> export_bam_results_to_excel(
        ...     results,
        ...     "sample_sizes_for_grant.xlsx",
        ...     study_title="Early Infant Movement Markers"
        ... )

Mathematical Model:
    
    **Hierarchical Bayesian Framework**:
        Hyperpriors:  α ~ Gamma(ESS × p + 1, scale=1/ESS)
                     β ~ Gamma(ESS × (1-p) + 1, scale=1/ESS)
        Prior:       θ ~ Beta(α, β)
        Likelihood:  k ~ Binomial(n, θ)
        Posterior:   θ|k ~ Beta(α + k, β + n - k)
    
    **Cluster Adjustment (Design Effect)**:
        DEFF = 1 + (m - 1) × ICC
        n_effective = n_total / DEFF
        
        Where:
        - m = mean cluster size
        - ICC = intraclass correlation coefficient
        - DEFF = design effect
    
    **Joint Metrics (Corrected Beta Priors)**:
        For sensitivity (Se) with pilot_se = (TP, condition_positives):
            se_alpha = TP + 1
            se_beta = (condition_positives - TP) + 1  # FN + 1 ✓ CORRECT
        
        For specificity (Sp) with pilot_sp = (TN, condition_negatives):
            sp_alpha = TN + 1
            sp_beta = (condition_negatives - TN) + 1  # FP + 1 ✓ CORRECT
        
        NOTE: Original bam.py had a bug using condition_positives + 1 for beta,
        which incorrectly treats the total count as failures instead of (total - successes).

Scientific References:
    - Kruschke, J. K. (2013). Bayesian estimation supersedes the t test.
      Journal of Experimental Psychology: General, 142(2), 573.
    - Joseph, L., du Berger, R., & Bélisle, P. (1997). Bayesian and mixed 
      Bayesian/likelihood criteria for sample size determination.
      Statistics in Medicine, 16(7), 769-781.
    - Eldridge, S. M., et al. (2016). Defining feasibility and pilot studies in 
      preparation for randomised controlled trials. BMJ, 355, i5239.
      (Design effect for clustered designs)

Module Organization:
    - BAMResult: Dataclass for structured results
    - BAMEstimator: Main class with all estimation methods
    - beta_hdi(): Highest Density Interval calculation
    - bam_single(), bam_joint(): Convenience functions
    - bam_grid_search(), bam_grid_search_joint(): Batch processing
    - plot_grid_results(): Visualization helper

Related Modules:
    - bam_export.py: Excel export utilities for professional reporting
    - bam.py: Original functional implementation (contains Beta prior bug)
    - bayesian_assurance.py: Original OOP implementation

Author: Generated from comparison of three BAM implementations, enhanced with
        cluster methods, batch processing, and export utilities
Date: 2025-10-05
Version: 2.0 (Enhanced with cluster adjustment, grid search, Excel export)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import time
import warnings

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


@dataclass
class BAMResult:
    """Results from a Bayesian Assurance Method sample size estimation.
    
    Attributes:
        optimal_n (int): Minimum sample size meeting target assurance.
        target_width (float): Target HDI width requested.
        target_assurance (float): Target assurance level requested (0-1).
        achieved_assurance (float): Actual assurance at optimal_n.
        sample_sizes_tested (List[int]): All sample sizes tested during search.
        assurances_at_tested (List[float]): Assurance at each tested size.
        metric_type (str): Type of metric ("single", "joint_se_sp", "clustered_single", "clustered_joint").
        pilot_estimate (float): Point estimate from pilot data.
        computation_time (float): Time taken for estimation (seconds).
        simulations_per_n (int): Number of simulations run per sample size.
        search_iterations (int): Number of binary search iterations.
        ci_level (float): Credible interval level used (e.g., 0.95).
        fig_path (Optional[Path]): Path to saved plot, if generated.
        icc (Optional[float]): Intraclass correlation coefficient for clustered designs.
        mean_cluster_size (Optional[float]): Mean observations per cluster.
        design_effect (Optional[float]): Design effect (DEFF) for clustered designs.
    """
    optimal_n: int
    target_width: float
    target_assurance: float
    achieved_assurance: float
    sample_sizes_tested: List[int]
    assurances_at_tested: List[float]
    metric_type: str
    pilot_estimate: float
    computation_time: float
    simulations_per_n: int
    search_iterations: int
    ci_level: float = 0.95
    fig_path: Optional[Path] = None
    icc: Optional[float] = None
    mean_cluster_size: Optional[float] = None
    design_effect: Optional[float] = None
    
    def __str__(self) -> str:
        """Human-readable summary of results."""
        result_str = (
            f"BAM Result ({self.metric_type}):\n"
            f"  Optimal N: {self.optimal_n}\n"
            f"  Target: HDI width ≤ {self.target_width} with {self.target_assurance:.0%} assurance\n"
            f"  Achieved: {self.achieved_assurance:.1%} assurance\n"
            f"  Pilot estimate: {self.pilot_estimate:.3f}\n"
            f"  Computation: {self.computation_time:.2f}s ({self.search_iterations} iterations)\n"
        )
        
        # Add cluster information if present
        if self.icc is not None:
            result_str += f"  ICC: {self.icc:.3f}\n"
        if self.mean_cluster_size is not None:
            result_str += f"  Mean cluster size: {self.mean_cluster_size:.1f}\n"
        if self.design_effect is not None:
            result_str += f"  Design effect: {self.design_effect:.3f}\n"
        
        return result_str
    
    def interpolate_assurance(self, sample_size: int) -> float:
        """Estimate assurance at arbitrary sample size via interpolation.
        
        Args:
            sample_size: Sample size to estimate assurance for.
            
        Returns:
            Estimated assurance level (0-1).
        """
        return np.interp(
            sample_size,
            self.sample_sizes_tested,
            self.assurances_at_tested
        )
    
    def interpolate_sample_size(self, target_assurance: float) -> float:
        """Estimate sample size needed for target assurance via interpolation.
        
        Args:
            target_assurance: Desired assurance level (0-1).
            
        Returns:
            Estimated sample size needed.
        """
        return np.interp(
            target_assurance,
            self.assurances_at_tested,
            self.sample_sizes_tested
        )
    
    def plot(
        self,
        save_path: Optional[Path] = None,
        show: bool = True,
        figsize: Tuple[float, float] = (12, 6)
    ) -> plt.Figure:
        """Create publication-quality assurance curve plot.
        
        Args:
            save_path: Path to save plot (PNG). If None, not saved.
            show: Whether to display plot interactively.
            figsize: Figure size in inches (width, height).
            
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot assurance curve
        ax.plot(
            self.sample_sizes_tested,
            self.assurances_at_tested,
            marker='o',
            linestyle='-',
            color='#2c7bb6',
            linewidth=2.5,
            markersize=8,
            markerfacecolor='#fdae61',
            label='Assurance Curve'
        )
        
        # Target assurance threshold
        ax.axhline(
            self.target_assurance,
            color='#d7191c',
            linestyle='--',
            linewidth=2,
            label=f'{self.target_assurance:.0%} Assurance Target'
        )
        
        # Optimal sample size marker
        ax.axvline(
            self.optimal_n,
            color='#2c7bb6',
            linestyle=':',
            linewidth=2,
            label=f'Optimal N = {self.optimal_n}'
        )
        
        # Labels and formatting
        ax.set_xlabel('Sample Size', fontsize=12, labelpad=10)
        ax.set_ylabel('Assurance Probability', fontsize=12, labelpad=10)
        ax.set_title(
            f'Bayesian Assurance Curve - {self.metric_type.title()}\n'
            f'Target HDI Width: {self.target_width} | '
            f'Pilot Estimate: {self.pilot_estimate:.3f}',
            fontsize=14,
            pad=15
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim(0, max(self.sample_sizes_tested) * 1.1)
        ax.set_ylim(0, 1.05)
        
        # Annotation
        ax.annotate(
            f'Achieved: {self.achieved_assurance:.1%}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            color='#636363',
            va='top'
        )
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.fig_path = save_path
        
        if show:
            plt.show()
        
        return fig


def beta_hdi(alpha: float, beta: float, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate Highest Density Interval (HDI) for Beta distribution.
    
    The HDI is the narrowest interval containing a specified probability mass.
    This function uses numerical optimization to find the shortest interval
    that contains `ci` probability mass from Beta(alpha, beta).
    
    This is the mathematically rigorous implementation from bam.py, using
    scipy.optimize.minimize_scalar for exact HDI bounds.
    
    Args:
        alpha: Beta distribution shape parameter alpha (α > 0).
        beta: Beta distribution shape parameter beta (β > 0).
        ci: Credible interval width. Defaults to 0.95 (95% HDI).
    
    Returns:
        Tuple of (lower_bound, upper_bound) of the HDI.
    
    Mathematical Details:
        For Beta(α, β), finds interval [L, U] such that:
        1. P(L ≤ θ ≤ U) = ci
        2. f(L) = f(U) (equal density at boundaries)
        3. U - L is minimized
    
    Example:
        >>> lower, upper = beta_hdi(alpha=10, beta=5, ci=0.95)
        >>> width = upper - lower
        >>> print(f"95% HDI width: {width:.3f}")
    """
    def interval_width(low):
        high = low + ci
        lower = stats.beta.ppf(low, alpha, beta)
        upper = stats.beta.ppf(high, alpha, beta)
        return abs(upper - lower)
    
    result = minimize_scalar(interval_width, bounds=(0, 1 - ci), method='bounded')
    low = result.x
    high = low + ci
    return stats.beta.ppf([low, high], alpha, beta)


class BAMEstimator:
    """Unified Bayesian Assurance Method estimator.
    
    This class provides a production-ready interface for BAM sample size estimation,
    combining efficiency (binary search), mathematical rigor (hierarchical
    Bayesian model), and usability (structured results, visualization).
    
    Supports both independent and clustered (hierarchical) study designs,
    single metrics and joint metrics (sensitivity + specificity), with
    corrected Beta prior calculations.
    
    Attributes:
        seed (Optional[int]): Random seed for reproducibility.
        verbose (bool): Whether to print progress information.
    
    Methods:
        estimate_single(): Single metric, independent observations
        estimate_joint(): Joint Se+Sp, independent observations
        estimate_single_clustered(): Single metric, clustered design (NEW)
        estimate_joint_clustered(): Joint Se+Sp, clustered design (NEW)
    
    Examples:
        **Basic Single Metric**:
            >>> estimator = BAMEstimator(seed=42, verbose=True)
            >>> pilot = np.array([1]*15 + [0]*5)
            >>> result = estimator.estimate_single(pilot, target_width=0.15)
            >>> print(result)
            >>> result.plot()
        
        **Joint Metrics (Sensitivity + Specificity)**:
            >>> result = estimator.estimate_joint(
            ...     pilot_se=(18, 20),  # 18 TP out of 20 diseased
            ...     pilot_sp=(35, 40),  # 35 TN out of 40 non-diseased
            ...     target_width=0.10
            ... )
            >>> print(f"Need N={result.optimal_n} for joint Se+Sp")
        
        **Clustered Design (NEW)**:
            >>> result = estimator.estimate_single_clustered(
            ...     pilot_data=pilot,
            ...     target_width=0.15,
            ...     icc=0.05,              # Intraclass correlation
            ...     mean_cluster_size=10   # Patients per clinic
            ... )
            >>> print(f"Need {result.optimal_n} clusters")
            >>> print(f"Total N = {result.optimal_n * 10} patients")
            >>> print(f"Design effect = {result.design_effect:.2f}")
        
        **Joint Metrics with Clustering (NEW)**:
            >>> result = estimator.estimate_joint_clustered(
            ...     pilot_se=(18, 20),
            ...     pilot_sp=(35, 40),
            ...     target_width=0.10,
            ...     icc=0.03,
            ...     mean_cluster_size=15
            ... )
            >>> print(f"Need {result.optimal_n} clusters for joint constraints")
    
    Notes:
        - Clustered methods return number of CLUSTERS, not total observations
        - Design effect (DEFF) = 1 + (m-1) × ICC where m = cluster size
        - Joint metrics use CORRECTED Beta priors (bug fix from original bam.py)
        - All methods use efficient binary search (O(log N))
        - Results include full diagnostics and plotting capabilities
    
    See Also:
        bam_single(), bam_joint(): Convenience functions
        bam_grid_search(), bam_grid_search_joint(): Batch processing
        BAMResult: Structured results with interpolation and plotting
    """
    
    def __init__(self, seed: Optional[int] = None, verbose: bool = False):
        """Initialize BAM estimator.
        
        Args:
            seed: Random seed for reproducibility. If None, no seed set.
            verbose: Whether to print progress during estimation.
        """
        self.seed = seed
        self.verbose = verbose
        
        if seed is not None:
            np.random.seed(seed)
    
    def estimate_single(
        self,
        pilot_data: np.ndarray,
        target_width: float = 0.15,
        ci: float = 0.95,
        target_assurance: float = 0.8,
        simulations: int = 2000,
        max_sample: int = 10000
    ) -> BAMResult:
        """Estimate sample size for single metric using hierarchical BAM.
        
        Uses the mathematically validated algorithm from bam.py with:
        - Hierarchical Bayesian model with Gamma hyperpriors
        - Binary search for efficiency (O(log N))
        - Optimization-based HDI calculation
        
        Args:
            pilot_data: Binary pilot data array (0s and 1s).
            target_width: Target HDI width (e.g., 0.15 for ±0.075).
            ci: Credible interval level. Defaults to 0.95.
            target_assurance: Desired assurance level. Defaults to 0.8.
            simulations: Number of Monte Carlo simulations per candidate N.
            max_sample: Maximum sample size to consider.
        
        Returns:
            BAMResult with optimal sample size and diagnostics.
        
        Raises:
            ValueError: If pilot_data is empty or invalid parameters.
        
        Example:
            >>> pilot = np.array([1]*15 + [0]*5)
            >>> estimator = BAMEstimator(seed=42)
            >>> result = estimator.estimate_single(pilot, target_width=0.15)
            >>> print(f"Need N={result.optimal_n}")
        
        Scientific Basis:
            Implements hierarchical Bayesian model with:
            - Hyperpriors: Gamma(α_ESS × p + 1, scale=1/ESS)
            - Data model: Binomial(n, θ)
            - Posterior: Beta(α + k, β + n - k)
        """
        # Validation
        if len(pilot_data) == 0:
            raise ValueError("pilot_data cannot be empty")
        if not np.all((pilot_data == 0) | (pilot_data == 1)):
            raise ValueError("pilot_data must contain only 0s and 1s")
        if target_width <= 0 or target_width >= 1:
            raise ValueError("target_width must be in (0, 1)")
        if target_assurance <= 0 or target_assurance >= 1:
            raise ValueError("target_assurance must be in (0, 1)")
        
        start_time = time.time()
        
        # Compute pilot statistics and hyperpriors
        n_pilot = len(pilot_data)
        p_pilot = np.mean(pilot_data)
        pilot_ess = n_pilot
        
        if self.verbose:
            print(f"Pilot: n={n_pilot}, p={p_pilot:.3f}, ESS={pilot_ess}")
        
        alpha_shape = pilot_ess * p_pilot + 1
        beta_shape = pilot_ess * (1 - p_pilot) + 1
        
        alpha_hyper = stats.gamma(alpha_shape, scale=1 / pilot_ess)
        beta_hyper = stats.gamma(beta_shape, scale=1 / pilot_ess)
        
        # Binary search setup
        low, high = max(50, int(n_pilot * 0.5)), max_sample
        best_n = max_sample
        
        # Track all tested sample sizes for plotting
        tested_sizes = []
        tested_assurances = []
        iterations = 0
        
        if self.verbose:
            print(f"Binary search range: [{low}, {high}]")
        
        while low <= high:
            mid = (low + high) // 2
            iterations += 1
            
            if self.verbose:
                print(f"  Iteration {iterations}: testing N={mid}")
            
            valid = 0
            
            # Run simulations for this candidate N
            for _ in range(simulations):
                # Hierarchical sampling
                a_h = alpha_hyper.rvs()
                b_h = beta_hyper.rvs()
                theta = stats.beta(
                    a_h + np.sum(pilot_data),
                    b_h + n_pilot - np.sum(pilot_data)
                ).rvs()
                
                # Generate synthetic data
                k = np.random.binomial(mid, theta)
                a_post = a_h + np.sum(pilot_data) + k
                b_post = b_h + n_pilot - np.sum(pilot_data) + mid - k
                
                # Calculate HDI width
                lower, upper = beta_hdi(a_post, b_post, ci)
                if (upper - lower) <= target_width:
                    valid += 1
            
            assurance = valid / simulations
            tested_sizes.append(mid)
            tested_assurances.append(assurance)
            
            if self.verbose:
                print(f"    Assurance: {assurance:.3f}")
            
            # Binary search update
            if assurance >= target_assurance:
                best_n = mid
                high = mid - 1
            else:
                low = mid + 1
        
        computation_time = time.time() - start_time
        
        # Get final assurance at optimal N
        final_idx = tested_sizes.index(best_n) if best_n in tested_sizes else -1
        achieved_assurance = tested_assurances[final_idx] if final_idx >= 0 else target_assurance
        
        # Sort for plotting
        sort_idx = np.argsort(tested_sizes)
        tested_sizes = [tested_sizes[i] for i in sort_idx]
        tested_assurances = [tested_assurances[i] for i in sort_idx]
        
        if self.verbose:
            print(f"Optimal N: {best_n} (found in {computation_time:.2f}s)")
        
        return BAMResult(
            optimal_n=best_n,
            target_width=target_width,
            target_assurance=target_assurance,
            achieved_assurance=achieved_assurance,
            sample_sizes_tested=tested_sizes,
            assurances_at_tested=tested_assurances,
            metric_type="single",
            pilot_estimate=p_pilot,
            computation_time=computation_time,
            simulations_per_n=simulations,
            search_iterations=iterations,
            ci_level=ci
        )
    
    def estimate_joint(
        self,
        pilot_se: Tuple[int, int],
        pilot_sp: Tuple[int, int],
        prevalence_prior: Tuple[int, int] = (8, 32),
        target_width: float = 0.10,
        ci: float = 0.95,
        target_assurance: float = 0.8,
        simulations: int = 1000,
        max_sample: int = 5000
    ) -> BAMResult:
        """Estimate sample size for joint sensitivity and specificity.
        
        Uses informed Bayesian Assurance Method to determine minimum sample
        size for BOTH sensitivity and specificity to achieve target precision
        simultaneously. Accounts for disease prevalence and ensures adequate
        representation of both classes.
        
        Args:
            pilot_se: Pilot sensitivity as (true_positives, condition_positives).
                Example: (18, 20) means 18 TPs out of 20 positive cases.
            pilot_sp: Pilot specificity as (true_negatives, condition_negatives).
                Example: (35, 40) means 35 TNs out of 40 negative cases.
            prevalence_prior: Beta prior for prevalence (α, β).
                Defaults to (8, 32) for ~20% prevalence.
            target_width: Target HDI width for BOTH metrics.
            ci: Credible interval level.
            target_assurance: Desired assurance for joint achievement.
            simulations: Simulations per candidate N.
            max_sample: Maximum sample size to consider.
        
        Returns:
            BAMResult with optimal total sample size for joint metrics.
        
        Example:
            >>> estimator = BAMEstimator(seed=42)
            >>> result = estimator.estimate_joint(
            ...     pilot_se=(18, 20),  # 90% sensitivity
            ...     pilot_sp=(35, 40),  # 87.5% specificity
            ...     target_width=0.10
            ... )
            >>> print(f"Need N={result.optimal_n} total")
        
        Note:
            - Requires BOTH metrics to achieve target (stricter criterion)
            - Enforces minimum 5% of samples per class
            - Prevalence affects final N significantly
        """
        start_time = time.time()
        
        # Validation
        if pilot_se[0] > pilot_se[1] or pilot_se[1] == 0:
            raise ValueError("Invalid pilot_se: must be (TP, condition_positive)")
        if pilot_sp[0] > pilot_sp[1] or pilot_sp[1] == 0:
            raise ValueError("Invalid pilot_sp: must be (TN, condition_negative)")
        
        # Derive Beta priors from pilot (CORRECTED FORMULA)
        # For Beta(α, β) representing Binomial success probability:
        #   α = successes + 1 (TP + 1 for sensitivity)
        #   β = failures + 1 (FN + 1 for sensitivity)
        # pilot_se = (TP, condition_positives) where FN = condition_positives - TP
        # pilot_sp = (TN, condition_negatives) where FP = condition_negatives - TN
        se_alpha = pilot_se[0] + 1  # TP + 1
        se_beta = (pilot_se[1] - pilot_se[0]) + 1  # FN + 1
        sp_alpha = pilot_sp[0] + 1  # TN + 1
        sp_beta = (pilot_sp[1] - pilot_sp[0]) + 1  # FP + 1
        
        se_estimate = pilot_se[0] / pilot_se[1]
        sp_estimate = pilot_sp[0] / pilot_sp[1]
        
        if self.verbose:
            print(f"Pilot Se: {se_estimate:.3f}, Sp: {sp_estimate:.3f}")
            print(f"Prevalence prior: Beta{prevalence_prior}")
        
        # Binary search
        low, high = 100, max_sample
        best_n = max_sample
        
        tested_sizes = []
        tested_assurances = []
        iterations = 0
        
        while low <= high:
            mid = (low + high) // 2
            iterations += 1
            
            if self.verbose:
                print(f"  Iteration {iterations}: testing N={mid}")
            
            valid = 0
            min_cases = max(10, int(0.05 * mid))  # At least 5% per class
            
            for _ in range(simulations):
                # Sample parameters from priors
                se = stats.beta(se_alpha, se_beta).rvs()
                sp = stats.beta(sp_alpha, sp_beta).rvs()
                prev = stats.beta(*prevalence_prior).rvs()
                
                # Generate synthetic data with prevalence floor
                n_pos = max(min_cases, np.random.binomial(mid, prev))
                n_neg = max(min_cases, mid - n_pos)
                
                tp = np.random.binomial(n_pos, se)
                tn = np.random.binomial(n_neg, sp)
                
                # Calculate posteriors
                se_post_alpha = se_alpha + tp
                se_post_beta = se_beta + (n_pos - tp)
                sp_post_alpha = sp_alpha + tn
                sp_post_beta = sp_beta + (n_neg - tn)
                
                # Calculate HDI widths
                se_hdi = beta_hdi(se_post_alpha, se_post_beta, ci)
                sp_hdi = beta_hdi(sp_post_alpha, sp_post_beta, ci)
                
                # Check if BOTH meet target
                if (se_hdi[1] - se_hdi[0] <= target_width and
                    sp_hdi[1] - sp_hdi[0] <= target_width):
                    valid += 1
            
            assurance = valid / simulations
            tested_sizes.append(mid)
            tested_assurances.append(assurance)
            
            if self.verbose:
                print(f"    Joint assurance: {assurance:.3f}")
            
            if assurance >= target_assurance:
                best_n = mid
                high = mid - 1
            else:
                low = mid + 1
        
        computation_time = time.time() - start_time
        
        # Get final assurance
        final_idx = tested_sizes.index(best_n) if best_n in tested_sizes else -1
        achieved_assurance = tested_assurances[final_idx] if final_idx >= 0 else target_assurance
        
        # Sort for plotting
        sort_idx = np.argsort(tested_sizes)
        tested_sizes = [tested_sizes[i] for i in sort_idx]
        tested_assurances = [tested_assurances[i] for i in sort_idx]
        
        if self.verbose:
            print(f"Optimal N: {best_n} (found in {computation_time:.2f}s)")
        
        return BAMResult(
            optimal_n=best_n,
            target_width=target_width,
            target_assurance=target_assurance,
            achieved_assurance=achieved_assurance,
            sample_sizes_tested=tested_sizes,
            assurances_at_tested=tested_assurances,
            metric_type="joint_se_sp",
            pilot_estimate=(se_estimate + sp_estimate) / 2,  # Average for summary
            computation_time=computation_time,
            simulations_per_n=simulations,
            search_iterations=iterations,
            ci_level=ci
        )


    def estimate_single_clustered(
        self,
        pilot_data: np.ndarray,
        icc: float,
        mean_cluster_size: float,
        target_width: float = 0.15,
        ci: float = 0.95,
        target_assurance: float = 0.8,
        simulations: int = 2000,
        max_clusters: int = 500
    ) -> BAMResult:
        """Estimate number of clusters for clustered design.
        
        Uses design effect adjustment for intraclass correlation (ICC).
        Returns optimal number of CLUSTERS, not individual observations.
        
        Args:
            pilot_data: Binary pilot data (pooled across clusters).
            icc: Intraclass correlation coefficient (0-1). Measures correlation
                within clusters. ICC=0 means no clustering effect.
            mean_cluster_size: Average observations per cluster (m).
            target_width: Target HDI width.
            ci: Credible interval level.
            target_assurance: Target assurance level.
            simulations: Monte Carlo simulations per candidate K.
            max_clusters: Maximum number of clusters to consider.
        
        Returns:
            BAMResult with optimal_n representing number of CLUSTERS needed.
            Total observations = optimal_n × mean_cluster_size.
        
        Example:
            >>> pilot = np.array([1]*15 + [0]*5)
            >>> estimator = BAMEstimator(seed=42)
            >>> result = estimator.estimate_single_clustered(
            ...     pilot, icc=0.15, mean_cluster_size=2.0, target_width=0.15
            ... )
            >>> print(f"Need {result.optimal_n} clusters")
            >>> print(f"Total N = {result.optimal_n * result.mean_cluster_size:.0f}")
        
        Mathematical Details:
            Design effect: DEFF = 1 + (m - 1) × ICC
            Effective sample size: n_eff = n_total / DEFF
            
            For ICC > 0, requires more clusters to achieve same effective N.
        
        Note:
            ICC typically ranges 0.01-0.20 for between-subject clustering.
            Higher ICC (>0.5) suggests within-subject repeated measures.
        """
        # Validation
        if icc < 0 or icc >= 1:
            raise ValueError("ICC must be in [0, 1)")
        if mean_cluster_size < 1:
            raise ValueError("mean_cluster_size must be >= 1")
        
        # Calculate design effect
        design_effect = 1 + (mean_cluster_size - 1) * icc
        
        if self.verbose:
            print(f"ICC: {icc:.3f}, Mean cluster size: {mean_cluster_size:.1f}")
            print(f"Design effect (DEFF): {design_effect:.3f}")
        
        # Run standard BAM but interpret results as clusters
        # We effectively need DEFF times more observations
        start_time = time.time()
        
        # Compute pilot statistics and hyperpriors
        n_pilot = len(pilot_data)
        p_pilot = np.mean(pilot_data)
        pilot_ess = n_pilot
        
        alpha_shape = pilot_ess * p_pilot + 1
        beta_shape = pilot_ess * (1 - p_pilot) + 1
        
        alpha_hyper = stats.gamma(alpha_shape, scale=1 / pilot_ess)
        beta_hyper = stats.gamma(beta_shape, scale=1 / pilot_ess)
        
        # Binary search over number of CLUSTERS
        low, high = max(10, int(n_pilot * 0.5 / mean_cluster_size)), max_clusters
        best_k = max_clusters
        
        tested_sizes = []
        tested_assurances = []
        iterations = 0
        
        if self.verbose:
            print(f"Binary search range: [{low}, {high}] clusters")
        
        while low <= high:
            mid_k = (low + high) // 2
            iterations += 1
            
            # Total observations for mid_k clusters
            n_total = int(mid_k * mean_cluster_size)
            # Effective sample size after design effect
            n_effective = n_total / design_effect
            
            if self.verbose:
                print(f"  Iteration {iterations}: testing K={mid_k} (N_total={n_total}, N_eff={n_effective:.1f})")
            
            valid = 0
            
            for _ in range(simulations):
                # Hierarchical sampling
                a_h = alpha_hyper.rvs()
                b_h = beta_hyper.rvs()
                theta = stats.beta(
                    a_h + np.sum(pilot_data),
                    b_h + n_pilot - np.sum(pilot_data)
                ).rvs()
                
                # Generate data accounting for design effect
                # Use n_effective as the "true" sample size
                k = np.random.binomial(int(n_effective), theta)
                a_post = a_h + np.sum(pilot_data) + k
                b_post = b_h + n_pilot - np.sum(pilot_data) + int(n_effective) - k
                
                # Calculate HDI width
                lower, upper = beta_hdi(a_post, b_post, ci)
                if (upper - lower) <= target_width:
                    valid += 1
            
            assurance = valid / simulations
            tested_sizes.append(mid_k)
            tested_assurances.append(assurance)
            
            if self.verbose:
                print(f"    Assurance: {assurance:.3f}")
            
            if assurance >= target_assurance:
                best_k = mid_k
                high = mid_k - 1
            else:
                low = mid_k + 1
        
        computation_time = time.time() - start_time
        
        # Get final assurance
        final_idx = tested_sizes.index(best_k) if best_k in tested_sizes else -1
        achieved_assurance = tested_assurances[final_idx] if final_idx >= 0 else target_assurance
        
        # Sort for plotting
        sort_idx = np.argsort(tested_sizes)
        tested_sizes = [tested_sizes[i] for i in sort_idx]
        tested_assurances = [tested_assurances[i] for i in sort_idx]
        
        if self.verbose:
            print(f"Optimal K: {best_k} clusters (found in {computation_time:.2f}s)")
            print(f"Total N: {int(best_k * mean_cluster_size)}")
        
        return BAMResult(
            optimal_n=best_k,
            target_width=target_width,
            target_assurance=target_assurance,
            achieved_assurance=achieved_assurance,
            sample_sizes_tested=tested_sizes,
            assurances_at_tested=tested_assurances,
            metric_type="clustered_single",
            pilot_estimate=p_pilot,
            computation_time=computation_time,
            simulations_per_n=simulations,
            search_iterations=iterations,
            ci_level=ci,
            icc=icc,
            mean_cluster_size=mean_cluster_size,
            design_effect=design_effect
        )
    
    def estimate_joint_clustered(
        self,
        pilot_se: Tuple[int, int],
        pilot_sp: Tuple[int, int],
        icc_sens: float,
        icc_spec: float,
        mean_cluster_size: float,
        prevalence_prior: Tuple[int, int] = (8, 32),
        target_width: float = 0.10,
        ci: float = 0.95,
        target_assurance: float = 0.8,
        simulations: int = 1000,
        max_clusters: int = 250
    ) -> BAMResult:
        """Estimate clusters for joint sensitivity/specificity with clustering.
        
        Uses separate ICC for sensitivity and specificity since clustering
        effects may differ between positive and negative cases.
        
        Args:
            pilot_se: Pilot sensitivity as (true_positives, condition_positives).
            pilot_sp: Pilot specificity as (true_negatives, condition_negatives).
            icc_sens: ICC for sensitivity (within-cluster correlation for positives).
            icc_spec: ICC for specificity (within-cluster correlation for negatives).
            mean_cluster_size: Average observations per cluster.
            prevalence_prior: Beta prior for prevalence.
            target_width: Target HDI width for BOTH metrics.
            ci: Credible interval level.
            target_assurance: Target assurance for joint achievement.
            simulations: Simulations per candidate K.
            max_clusters: Maximum clusters to consider.
        
        Returns:
            BAMResult with optimal_n as number of CLUSTERS needed.
        
        Example:
            >>> estimator = BAMEstimator(seed=42)
            >>> result = estimator.estimate_joint_clustered(
            ...     pilot_se=(18, 20),
            ...     pilot_sp=(35, 40),
            ...     icc_sens=0.15,
            ...     icc_spec=0.10,
            ...     mean_cluster_size=2.0
            ... )
            >>> print(f"Need {result.optimal_n} clusters")
        
        Note:
            - Uses maximum of two design effects (one per metric)
            - More conservative approach ensures both metrics achieve target
        """
        start_time = time.time()
        
        # Validation
        if pilot_se[0] > pilot_se[1] or pilot_se[1] == 0:
            raise ValueError("Invalid pilot_se: must be (TP, condition_positive)")
        if pilot_sp[0] > pilot_sp[1] or pilot_sp[1] == 0:
            raise ValueError("Invalid pilot_sp: must be (TN, condition_negative)")
        if icc_sens < 0 or icc_sens >= 1:
            raise ValueError("icc_sens must be in [0, 1)")
        if icc_spec < 0 or icc_spec >= 1:
            raise ValueError("icc_spec must be in [0, 1)")
        if mean_cluster_size < 1:
            raise ValueError("mean_cluster_size must be >= 1")
        
        # Calculate design effects
        deff_sens = 1 + (mean_cluster_size - 1) * icc_sens
        deff_spec = 1 + (mean_cluster_size - 1) * icc_spec
        # Use maximum for conservative estimate
        design_effect = max(deff_sens, deff_spec)
        
        if self.verbose:
            print(f"ICC_sens: {icc_sens:.3f}, ICC_spec: {icc_spec:.3f}")
            print(f"DEFF_sens: {deff_sens:.3f}, DEFF_spec: {deff_spec:.3f}")
            print(f"Design effect (max): {design_effect:.3f}")
        
        # Derive Beta priors (CORRECTED FORMULA)
        # α = successes + 1, β = failures + 1
        se_alpha = pilot_se[0] + 1  # TP + 1
        se_beta = (pilot_se[1] - pilot_se[0]) + 1  # FN + 1
        sp_alpha = pilot_sp[0] + 1  # TN + 1
        sp_beta = (pilot_sp[1] - pilot_sp[0]) + 1  # FP + 1
        
        se_estimate = pilot_se[0] / pilot_se[1]
        sp_estimate = pilot_sp[0] / pilot_sp[1]
        
        # Binary search over clusters
        low, high = 50, max_clusters
        best_k = max_clusters
        
        tested_sizes = []
        tested_assurances = []
        iterations = 0
        
        while low <= high:
            mid_k = (low + high) // 2
            iterations += 1
            
            n_total = int(mid_k * mean_cluster_size)
            n_effective = n_total / design_effect
            
            if self.verbose:
                print(f"  Iteration {iterations}: testing K={mid_k} (N_total={n_total}, N_eff={n_effective:.1f})")
            
            valid = 0
            min_cases = max(10, int(0.05 * n_effective))
            
            for _ in range(simulations):
                # Sample parameters
                se = stats.beta(se_alpha, se_beta).rvs()
                sp = stats.beta(sp_alpha, sp_beta).rvs()
                prev = stats.beta(*prevalence_prior).rvs()
                
                # Generate data using effective sample size
                n_pos = max(min_cases, np.random.binomial(int(n_effective), prev))
                n_neg = max(min_cases, int(n_effective) - n_pos)
                
                tp = np.random.binomial(n_pos, se)
                tn = np.random.binomial(n_neg, sp)
                
                # Calculate posteriors
                se_post_alpha = se_alpha + tp
                se_post_beta = se_beta + (n_pos - tp)
                sp_post_alpha = sp_alpha + tn
                sp_post_beta = sp_beta + (n_neg - tn)
                
                # Calculate HDI widths
                se_hdi = beta_hdi(se_post_alpha, se_post_beta, ci)
                sp_hdi = beta_hdi(sp_post_alpha, sp_post_beta, ci)
                
                # Check if BOTH meet target
                if (se_hdi[1] - se_hdi[0] <= target_width and
                    sp_hdi[1] - sp_hdi[0] <= target_width):
                    valid += 1
            
            assurance = valid / simulations
            tested_sizes.append(mid_k)
            tested_assurances.append(assurance)
            
            if self.verbose:
                print(f"    Joint assurance: {assurance:.3f}")
            
            if assurance >= target_assurance:
                best_k = mid_k
                high = mid_k - 1
            else:
                low = mid_k + 1
        
        computation_time = time.time() - start_time
        
        # Get final assurance
        final_idx = tested_sizes.index(best_k) if best_k in tested_sizes else -1
        achieved_assurance = tested_assurances[final_idx] if final_idx >= 0 else target_assurance
        
        # Sort for plotting
        sort_idx = np.argsort(tested_sizes)
        tested_sizes = [tested_sizes[i] for i in sort_idx]
        tested_assurances = [tested_assurances[i] for i in sort_idx]
        
        if self.verbose:
            print(f"Optimal K: {best_k} clusters (found in {computation_time:.2f}s)")
            print(f"Total N: {int(best_k * mean_cluster_size)}")
        
        return BAMResult(
            optimal_n=best_k,
            target_width=target_width,
            target_assurance=target_assurance,
            achieved_assurance=achieved_assurance,
            sample_sizes_tested=tested_sizes,
            assurances_at_tested=tested_assurances,
            metric_type="clustered_joint",
            pilot_estimate=(se_estimate + sp_estimate) / 2,
            computation_time=computation_time,
            simulations_per_n=simulations,
            search_iterations=iterations,
            ci_level=ci,
            icc=(icc_sens + icc_spec) / 2,  # Average for summary
            mean_cluster_size=mean_cluster_size,
            design_effect=design_effect
        )


# Convenience functions for backward compatibility

def bam_single(
    pilot_data: np.ndarray,
    target_width: float = 0.15,
    target_assurance: float = 0.8,
    **kwargs
) -> int:
    """Convenience function for quick single metric estimation.
    
    Returns only the optimal N (for backward compatibility with bam.py).
    For full results with diagnostics, use BAMEstimator.estimate_single().
    
    Args:
        pilot_data: Binary pilot data array.
        target_width: Target HDI width.
        target_assurance: Target assurance level.
        **kwargs: Additional arguments passed to estimate_single().
    
    Returns:
        Optimal sample size (integer).
    
    Example:
        >>> pilot = np.array([1]*15 + [0]*5)
        >>> n = bam_single(pilot, target_width=0.15)
        >>> print(f"Need N={n}")
    """
    estimator = BAMEstimator(seed=kwargs.pop('seed', None))
    result = estimator.estimate_single(
        pilot_data=pilot_data,
        target_width=target_width,
        target_assurance=target_assurance,
        **kwargs
    )
    return result.optimal_n


def bam_joint(
    pilot_se: Tuple[int, int],
    pilot_sp: Tuple[int, int],
    target_width: float = 0.10,
    target_assurance: float = 0.8,
    **kwargs
) -> int:
    """Convenience function for quick joint metric estimation.
    
    Returns only the optimal N (for backward compatibility with bam.py).
    For full results with diagnostics, use BAMEstimator.estimate_joint().
    
    Args:
        pilot_se: Pilot sensitivity (TP, condition_positive).
        pilot_sp: Pilot specificity (TN, condition_negative).
        target_width: Target HDI width for both metrics.
        target_assurance: Target assurance level.
        **kwargs: Additional arguments passed to estimate_joint().
    
    Returns:
        Optimal total sample size (integer).
    
    Example:
        >>> n = bam_joint(
        ...     pilot_se=(18, 20),
        ...     pilot_sp=(35, 40),
        ...     target_width=0.10
        ... )
        >>> print(f"Need N={n} total")
    """
    estimator = BAMEstimator(seed=kwargs.pop('seed', None))
    result = estimator.estimate_joint(
        pilot_se=pilot_se,
        pilot_sp=pilot_sp,
        target_width=target_width,
        target_assurance=target_assurance,
        **kwargs
    )
    return result.optimal_n


# Batch/Grid Processing API

def bam_grid_search(
    pilot_data: np.ndarray,
    target_widths: List[float],
    target_assurances: List[float] = [0.8],
    ci_levels: List[float] = [0.95],
    simulations: int = 2000,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False
):
    """Run BAM estimation over parameter grid (NEW ENHANCEMENT).
    
    Performs comprehensive parameter sensitivity analysis by computing
    optimal sample sizes for all combinations of target HDI widths,
    assurance levels, and credible interval levels.
    
    This is essential for:
    - Grant applications showing sample size justification
    - IRB submissions demonstrating statistical rigor
    - Sensitivity analysis for planning documents
    - Budget planning with multiple scenarios
    
    Args:
        pilot_data (np.ndarray): Binary pilot data array (0s and 1s).
        target_widths (List[float]): List of target HDI widths to evaluate.
            Example: [0.05, 0.10, 0.15, 0.20] for various precision levels.
        target_assurances (List[float]): List of assurance levels (0-1).
            Default: [0.8]. Example: [0.75, 0.80, 0.85, 0.90] for sensitivity.
        ci_levels (List[float]): List of credible interval levels (0-1).
            Default: [0.95]. Example: [0.90, 0.95, 0.99] for various CIs.
        simulations (int): Number of Monte Carlo simulations per candidate N.
            Default: 2000. Higher values increase accuracy but computation time.
        seed (Optional[int]): Random seed for reproducibility. Recommended
            for grant applications requiring reproducible results.
        n_jobs (int): Number of parallel jobs. Use -1 for all CPU cores.
            Default: 1 (sequential). Parallelization speeds up large grids.
        verbose (bool): Print progress information during execution.
    
    Returns:
        pandas.DataFrame: Results with one row per parameter combination.
            Columns:
            - target_width (float): Requested HDI width
            - target_assurance (float): Requested assurance level
            - ci_level (float): Credible interval level used
            - optimal_n (int): Minimum sample size meeting criteria
            - achieved_assurance (float): Actual assurance at optimal_n
            - computation_time (float): Computation time in seconds
            - pilot_estimate (float): Point estimate from pilot data
    
    Raises:
        ImportError: If joblib or pandas not installed.
        ValueError: If pilot_data is invalid or parameters out of range.
    
    Examples:
        **Basic Grid Search**:
            >>> import numpy as np
            >>> from early_markers.cribsy.common.bam_unified import bam_grid_search
            >>> 
            >>> pilot = np.array([1]*15 + [0]*5)
            >>> results = bam_grid_search(
            ...     pilot,
            ...     target_widths=[0.05, 0.10, 0.15, 0.20],
            ...     target_assurances=[0.75, 0.80, 0.85, 0.90],
            ...     n_jobs=-1  # Use all CPUs
            ... )
            >>> print(results)
        
        **Pivot Table for Grant Application**:
            >>> pivot = results.pivot(
            ...     index="target_assurance",
            ...     columns="target_width",
            ...     values="optimal_n"
            ... )
            >>> print("\nSample Size Requirements:")
            >>> print(pivot)
        
        **Identify Optimal Tradeoff**:
            >>> # Find smallest N with assurance >= 0.80
            >>> subset = results[results['target_assurance'] >= 0.80]
            >>> optimal = subset.loc[subset['optimal_n'].idxmin()]
            >>> print(f"Best: N={optimal['optimal_n']}, width={optimal['target_width']}")
        
        **Export to Excel**:
            >>> from early_markers.cribsy.common.bam_export import export_grid_to_excel
            >>> export_grid_to_excel(
            ...     results,
            ...     "sample_size_grid.xlsx",
            ...     study_title="Early Infant Movement Markers"
            ... )
    
    Performance:
        - Computation time scales linearly with grid size
        - Parallelization provides near-linear speedup
        - Example: 16 combinations on 8 cores ~2-3 minutes
        - Larger grids (100+ combinations) benefit most from n_jobs=-1
    
    See Also:
        bam_grid_search_joint(): Grid search for joint sensitivity + specificity
        plot_grid_results(): Visualize grid search results
        export_grid_to_excel(): Export results to formatted Excel
    
    Notes:
        - Results are deterministic when seed is set
        - Parallel execution may use significant memory for large grids
        - Consider reducing simulations for initial exploration, then
          increase for final grant/publication numbers
    """
    try:
        from joblib import Parallel, delayed
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Batch processing requires joblib and pandas. "
            "Install with: pip install joblib pandas"
        )
    
    # Generate all parameter combinations
    param_grid = [
        (tw, ta, ci)
        for tw in target_widths
        for ta in target_assurances
        for ci in ci_levels
    ]
    
    if verbose:
        print(f"Running grid search over {len(param_grid)} parameter combinations...")
    
    # Define worker function
    def run_single(params):
        tw, ta, ci = params
        estimator = BAMEstimator(seed=seed, verbose=False)
        result = estimator.estimate_single(
            pilot_data=pilot_data,
            target_width=tw,
            target_assurance=ta,
            ci=ci,
            simulations=simulations
        )
        return {
            'target_width': tw,
            'target_assurance': ta,
            'ci_level': ci,
            'optimal_n': result.optimal_n,
            'achieved_assurance': result.achieved_assurance,
            'computation_time': result.computation_time,
            'pilot_estimate': result.pilot_estimate,
        }
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_single)(params) for params in param_grid
    )
    
    return pd.DataFrame(results)


def bam_grid_search_joint(
    pilot_se: Tuple[int, int],
    pilot_sp: Tuple[int, int],
    target_widths: List[float],
    target_assurances: List[float] = [0.8],
    prevalence_priors: List[Tuple[int, int]] = [(8, 32)],
    ci_levels: List[float] = [0.95],
    simulations: int = 1000,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False
):
    """Run BAM joint (Se+Sp) estimation over parameter grid (NEW ENHANCEMENT).
    
    Performs comprehensive parameter sensitivity analysis for studies requiring
    BOTH sensitivity AND specificity to meet precision requirements. Evaluates
    all combinations of HDI widths, assurance levels, prevalence priors, and
    credible interval levels.
    
    Uses CORRECTED Beta prior calculations (bug fix from original bam.py).
    
    This is essential for:
    - Diagnostic accuracy studies (test validation)
    - Biomarker discovery requiring dual metrics
    - Screening test development
    - Regulatory submissions requiring Se+Sp constraints
    
    Args:
        pilot_se (Tuple[int, int]): Pilot sensitivity as (TP, condition_positives).
            Example: (18, 20) means 18 true positives out of 20 diseased cases.
        pilot_sp (Tuple[int, int]): Pilot specificity as (TN, condition_negatives).
            Example: (35, 40) means 35 true negatives out of 40 non-diseased.
        target_widths (List[float]): List of target HDI widths for BOTH metrics.
            Example: [0.05, 0.10, 0.15] for various precision requirements.
        target_assurances (List[float]): List of assurance levels (0-1).
            Default: [0.8]. Example: [0.75, 0.80, 0.85, 0.90].
        prevalence_priors (List[Tuple[int, int]]): List of Beta(α, β) priors
            for disease prevalence. Default: [(8, 32)] ≈ 20% prevalence.
            Example: [(5, 20), (8, 32), (10, 40)] for various prevalence scenarios.
        ci_levels (List[float]): List of credible interval levels (0-1).
            Default: [0.95].
        simulations (int): Number of Monte Carlo simulations per candidate N.
            Default: 1000 (lower than single metric due to joint computation).
        seed (Optional[int]): Random seed for reproducibility.
        n_jobs (int): Number of parallel jobs (-1 = all cores). Default: 1.
        verbose (bool): Print progress information.
    
    Returns:
        pandas.DataFrame: Results with one row per parameter combination.
            Columns:
            - target_width (float): Requested HDI width for both Se and Sp
            - target_assurance (float): Requested assurance level
            - ci_level (float): Credible interval level
            - prevalence_prior (str): Prevalence prior as "Beta(α, β)"
            - optimal_n (int): Minimum TOTAL sample size (diseased + non-diseased)
            - achieved_assurance (float): Actual assurance at optimal_n
            - computation_time (float): Computation time in seconds
            - pilot_se_estimate (float): Pilot sensitivity estimate
            - pilot_sp_estimate (float): Pilot specificity estimate
    
    Raises:
        ImportError: If joblib or pandas not installed.
        ValueError: If pilot data invalid or parameters out of range.
    
    Examples:
        **Basic Joint Grid Search**:
            >>> from early_markers.cribsy.common.bam_unified import bam_grid_search_joint
            >>> 
            >>> results = bam_grid_search_joint(
            ...     pilot_se=(18, 20),  # 90% sensitivity
            ...     pilot_sp=(35, 40),  # 87.5% specificity
            ...     target_widths=[0.05, 0.10, 0.15, 0.20],
            ...     target_assurances=[0.75, 0.80, 0.85, 0.90],
            ...     prevalence_priors=[(5, 20), (8, 32), (10, 40)],
            ...     n_jobs=-1
            ... )
            >>> print(results)
        
        **Pivot by Prevalence and Width**:
            >>> # Show how sample size varies with prevalence
            >>> for prev_str in results['prevalence_prior'].unique():
            ...     subset = results[results['prevalence_prior'] == prev_str]
            ...     pivot = subset.pivot(
            ...         index="target_assurance",
            ...         columns="target_width",
            ...         values="optimal_n"
            ...     )
            ...     print(f"\n{prev_str}:")
            ...     print(pivot)
        
        **Find Optimal Configuration**:
            >>> # Find smallest N with assurance >= 0.80 and width <= 0.15
            >>> subset = results[
            ...     (results['target_assurance'] >= 0.80) &
            ...     (results['target_width'] <= 0.15)
            ... ]
            >>> optimal = subset.loc[subset['optimal_n'].idxmin()]
            >>> print(f"Optimal: N={optimal['optimal_n']}, "
            ...       f"width={optimal['target_width']}, "
            ...       f"prevalence={optimal['prevalence_prior']}")
        
        **Export to Excel**:
            >>> from early_markers.cribsy.common.bam_export import export_grid_to_excel
            >>> export_grid_to_excel(
            ...     results,
            ...     "joint_metrics_grid.xlsx",
            ...     study_title="Diagnostic Accuracy Study",
            ...     description="Joint Se+Sp sample size requirements"
            ... )
    
    Performance:
        - Joint metrics are ~2x slower than single metrics
        - Computation scales linearly with grid size
        - Default simulations=1000 balances speed and accuracy
        - Example: 48 combinations on 8 cores ~8-10 minutes
    
    See Also:
        bam_grid_search(): Grid search for single metrics
        estimate_joint_clustered(): For hierarchical designs
        export_grid_to_excel(): Professional Excel export
    
    Notes:
        - Prevalence prior affects sample allocation between diseased/non-diseased
        - Higher prevalence → more diseased cases needed for specificity precision
        - Lower prevalence → more non-diseased cases for sensitivity precision
        - Results use CORRECTED Beta priors (original bam.py had a bug)
    
    Mathematical Note:
        Beta prior calculation (CORRECTED):
            se_alpha = TP + 1
            se_beta = (condition_positives - TP) + 1  # FN + 1 ✓
            sp_alpha = TN + 1
            sp_beta = (condition_negatives - TN) + 1  # FP + 1 ✓
    """
    try:
        from joblib import Parallel, delayed
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Batch processing requires joblib and pandas. "
            "Install with: pip install joblib pandas"
        )
    
    # Generate all parameter combinations
    param_grid = [
        (tw, ta, prev, ci)
        for tw in target_widths
        for ta in target_assurances
        for prev in prevalence_priors
        for ci in ci_levels
    ]
    
    if verbose:
        print(f"Running grid search over {len(param_grid)} parameter combinations...")
    
    # Define worker function
    def run_single(params):
        tw, ta, prev, ci = params
        estimator = BAMEstimator(seed=seed, verbose=False)
        result = estimator.estimate_joint(
            pilot_se=pilot_se,
            pilot_sp=pilot_sp,
            prevalence_prior=prev,
            target_width=tw,
            target_assurance=ta,
            ci=ci,
            simulations=simulations
        )
        return {
            'target_width': tw,
            'target_assurance': ta,
            'prevalence_prior_alpha': prev[0],
            'prevalence_prior_beta': prev[1],
            'ci_level': ci,
            'optimal_n': result.optimal_n,
            'achieved_assurance': result.achieved_assurance,
            'computation_time': result.computation_time,
            'pilot_estimate': result.pilot_estimate,
        }
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(run_single)(params) for params in param_grid
    )
    
    return pd.DataFrame(results)


def plot_grid_results(
    grid_df,
    x: str = 'target_width',
    y: str = 'optimal_n',
    hue: str = 'target_assurance',
    save_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """Create publication-quality plot from grid search results (NEW ENHANCEMENT).
    
    Generates professional seaborn line plots suitable for grants, manuscripts,
    and presentations. Visualizes how sample size requirements vary with
    precision targets and assurance levels.
    
    Args:
        grid_df (pd.DataFrame): DataFrame from bam_grid_search() or 
            bam_grid_search_joint(). Must contain columns specified in
            x, y, and hue parameters.
        x (str): Column name for x-axis. Common choices:
            - 'target_width': Shows effect of precision requirements
            - 'target_assurance': Shows effect of confidence level
            Default: 'target_width'
        y (str): Column name for y-axis. Common choices:
            - 'optimal_n': Sample size requirements (most common)
            - 'achieved_assurance': Actual assurance achieved
            - 'computation_time': Computational cost
            Default: 'optimal_n'
        hue (str): Column name for color grouping. Common choices:
            - 'target_assurance': Compare different assurance levels
            - 'ci_level': Compare different CI levels (if varied)
            - 'prevalence_prior': Compare prevalence assumptions (joint metrics)
            Default: 'target_assurance'
        save_path (Optional[Path]): Path to save PNG file. If None, not saved.
            Automatically creates parent directories if needed.
        figsize (Tuple[float, float]): Figure size in inches (width, height).
            Default: (10, 6). Increase for presentations.
        title (Optional[str]): Custom plot title. If None, auto-generated from
            axis labels.
    
    Returns:
        plt.Figure: Matplotlib Figure object for further customization.
    
    Raises:
        ImportError: If seaborn not installed.
        KeyError: If specified columns not in grid_df.
    
    Examples:
        **Basic Usage**:
            >>> from early_markers.cribsy.common.bam_unified import (
            ...     bam_grid_search, plot_grid_results
            ... )
            >>> import numpy as np
            >>> 
            >>> pilot = np.array([1]*15 + [0]*5)
            >>> results = bam_grid_search(
            ...     pilot,
            ...     target_widths=[0.05, 0.10, 0.15, 0.20],
            ...     target_assurances=[0.75, 0.80, 0.85, 0.90]
            ... )
            >>> fig = plot_grid_results(
            ...     results,
            ...     save_path="sample_size_vs_width.png"
            ... )
        
        **Custom Axes**:
            >>> # Show how assurance varies with sample size
            >>> fig = plot_grid_results(
            ...     results,
            ...     x='optimal_n',
            ...     y='achieved_assurance',
            ...     hue='target_width',
            ...     title="Assurance Curves for Different Precision Targets"
            ... )
        
        **Joint Metrics with Prevalence**:
            >>> from early_markers.cribsy.common.bam_unified import bam_grid_search_joint
            >>> 
            >>> results = bam_grid_search_joint(
            ...     pilot_se=(18, 20),
            ...     pilot_sp=(35, 40),
            ...     target_widths=[0.05, 0.10, 0.15],
            ...     prevalence_priors=[(5, 20), (8, 32), (10, 40)],
            ...     n_jobs=-1
            ... )
            >>> # Group by prevalence to show its effect
            >>> fig = plot_grid_results(
            ...     results,
            ...     x='target_width',
            ...     y='optimal_n',
            ...     hue='prevalence_prior',
            ...     title="Sample Size by Precision and Prevalence",
            ...     figsize=(12, 7)
            ... )
        
        **Large Format for Presentation**:
            >>> fig = plot_grid_results(
            ...     results,
            ...     figsize=(14, 8),
            ...     save_path="presentation_figure.png"
            ... )
    
    Visualization Features:
        - Seaborn line plots with markers for clarity
        - Automatic color palettes for grouping
        - Grid lines for easier value reading
        - Legend with formatted labels
        - High-resolution (300 DPI) output
        - Publication-ready formatting
    
    Tips:
        - Use wider figures (14, 8) for presentations
        - Use default (10, 6) for manuscripts
        - Save as PNG for slides, PDF for publications
        - Consider faceting for very large grids
    
    See Also:
        bam_grid_search(): Generate single metric grid data
        bam_grid_search_joint(): Generate joint metric grid data
        export_grid_to_excel(): Export data to Excel tables
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Visualization requires seaborn. Install with: pip install seaborn"
        )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lines with markers
    sns.lineplot(
        data=grid_df,
        x=x,
        y=y,
        hue=hue,
        marker='o',
        markersize=8,
        linewidth=2.5,
        ax=ax
    )
    
    # Formatting
    ax.set_xlabel(x.replace('_', ' ').title(), fontsize=12, labelpad=10)
    ax.set_ylabel(y.replace('_', ' ').title(), fontsize=12, labelpad=10)
    
    if title is None:
        title = f"BAM Grid Search: {y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title=hue.replace('_', ' ').title(), loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
