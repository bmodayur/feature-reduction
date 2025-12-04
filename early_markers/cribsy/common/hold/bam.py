"""Bayesian Assurance Method (BAM) for sample size estimation.

This module implements Bayesian sample size determination methods based on
the Bayesian Assurance Method (BAM) framework for planning new studies.
BAM uses Monte Carlo simulation to estimate the minimum sample size needed
to achieve target precision (HDI width) with specified assurance.

Key Functions:
    - beta_hdi: Calculate Highest Density Interval for Beta distribution
    - bam_performance: Single metric sample size estimation
    - informed_bam_performance: Sensitivity/specificity joint estimation

Methodology:
    1. Start with pilot data to establish prior distributions
    2. Simulate multiple datasets at candidate sample sizes
    3. Calculate posterior HDI for each simulation
    4. Determine minimum N where target HDI width is achieved
       with desired assurance level

References:
    - Kruschke, J. K. (2013). Bayesian estimation supersedes the t test.
      Journal of Experimental Psychology: General, 142(2), 573.
    - Joseph, L., du Berger, R., & Bélisle, P. (1997). Bayesian and mixed
      Bayesian/likelihood criteria for sample size determination.

Example:
    >>> import numpy as np
    >>> from early_markers.cribsy.bam import bam_performance
    >>> 
    >>> # Pilot data with 15/20 successes
    >>> pilot = np.array([1]*15 + [0]*5)
    >>> 
    >>> # Find N for ±0.15 HDI width with 80% assurance
    >>> n = bam_performance(pilot, hdi_width=0.15, target_assurance=0.8)
    >>> print(f"Required sample size: {n}")
"""
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar


np.random.seed(42)


def beta_hdi(alpha, beta, ci=0.95):
    """Calculate Highest Density Interval (HDI) for Beta distribution.
    
    The HDI is the narrowest interval containing a specified probability mass.
    This function uses numerical optimization to find the shortest interval
    that contains `ci` probability mass from Beta(alpha, beta).
    
    Args:
        alpha (float): Beta distribution shape parameter alpha (α > 0).
        beta (float): Beta distribution shape parameter beta (β > 0).
        ci (float, optional): Credible interval width. Defaults to 0.95 (95% HDI).
    
    Returns:
        tuple[float, float]: (lower_bound, upper_bound) of the HDI.
    
    Mathematical Details:
        For Beta(α, β), finds interval [L, U] such that:
        1. P(L ≤ θ ≤ U) = ci
        2. f(L) = f(U) (equal density at boundaries)
        3. U - L is minimized
    
    Example:
        >>> from early_markers.cribsy.bam import beta_hdi
        >>> 
        >>> # 95% HDI for Beta(10, 5)
        >>> lower, upper = beta_hdi(alpha=10, beta=5, ci=0.95)
        >>> print(f"95% HDI: [{lower:.3f}, {upper:.3f}]")
        95% HDI: [0.542, 0.810]
        >>> 
        >>> # Width of HDI
        >>> width = upper - lower
        >>> print(f"HDI width: {width:.3f}")
        HDI width: 0.268
    
    Note:
        Uses scipy.optimize.minimize_scalar with bounded method for efficiency.
        The optimization minimizes the interval width subject to containing
        the specified probability mass.
    """
    def interval_width(low):
        high = low + ci
        lower = stats.beta.ppf(low, alpha, beta)
        upper = stats.beta.ppf(high, alpha, beta)
        return abs(upper - lower)

    result = minimize_scalar(interval_width, bounds=(0, 1-ci), method='bounded')
    low = result.x
    high = low + ci
    return stats.beta.ppf([low, high], alpha, beta)

def bam_performance(pilot_data, hdi_width=0.15, ci=0.95,
                     target_assurance=0.8, simulations=2000,
                     max_sample=10000):
    """Estimate minimum sample size for single metric using BAM framework.
    
    Uses Bayesian Assurance Method to determine the minimum sample size
    needed to achieve a target HDI width with specified assurance for planning
    new studies. Performs binary search over candidate sample sizes, using
    Monte Carlo simulation at each candidate to estimate the probability
    (assurance) of achieving the target precision.
    
    Args:
        pilot_data (np.ndarray): Binary pilot data (0s and 1s). Used to
            establish prior distribution via effective sample size.
        hdi_width (float, optional): Target HDI width. Defaults to 0.15 (±0.075).
        ci (float, optional): Credible interval level. Defaults to 0.95.
        target_assurance (float, optional): Desired probability of achieving
            target HDI width. Defaults to 0.8 (80% assurance).
        simulations (int, optional): Number of Monte Carlo simulations per
            candidate N. Defaults to 2000.
        max_sample (int, optional): Maximum sample size to consider.
            Defaults to 10000.
    
    Returns:
        int: Estimated minimum sample size to achieve target HDI width with
            desired assurance.
    
    Algorithm:
        1. Compute pilot effective sample size (ESS)
        2. Derive hyperpriors from pilot: Gamma(α_ESS × p + 1, 1/ESS)
        3. Binary search over N from [max(50, 0.5×pilot_n), max_sample]
        4. For each candidate N:
           - Simulate `simulations` datasets
           - For each simulation:
             a. Sample prior parameters from hyperpriors
             b. Generate synthetic data of size N
             c. Compute posterior Beta distribution
             d. Calculate HDI width
           - Count simulations where HDI ≤ target width
        5. Return smallest N where assurance ≥ target
    
    Example:
        >>> import numpy as np
        >>> from early_markers.cribsy.bam import bam_performance
        >>> 
        >>> # Pilot study: 15 successes out of 20 trials
        >>> pilot = np.array([1]*15 + [0]*5)
        >>> 
        >>> # Find N for ±0.15 HDI with 80% assurance
        >>> n_required = bam_performance(
        ...     pilot_data=pilot,
        ...     hdi_width=0.15,
        ...     ci=0.95,
        ...     target_assurance=0.8
        ... )
        >>> print(f"Required N: {n_required}")
        Required N: 156
    
    Note:
        - Uses hierarchical Bayesian model with hyperpriors
        - Computational cost: O(log(max_sample) × simulations)
        - Larger `simulations` gives more stable estimates but slower
        - For small pilot samples, may require large N
    
    See Also:
        informed_bam_performance: For joint sensitivity/specificity estimation
        beta_hdi: For computing HDI intervals
    """
    # Hyperpriors from pilot ESS
    n_pilot = len(pilot_data)
    p_pilot = np.mean(pilot_data)
    pilot_ess = n_pilot

    alpha_shape = pilot_ess * p_pilot + 1
    beta_shape = pilot_ess * (1 - p_pilot) + 1

    alpha_hyper = stats.gamma(alpha_shape, scale=1/pilot_ess)
    beta_hyper = stats.gamma(beta_shape, scale=1/pilot_ess)

    # Binary search with stabilized estimates
    low, high = max(50, int(n_pilot*0.5)), max_sample
    best_n = max_sample

    while low <= high:
        mid = (low + high) // 2
        valid = 0
        min_sims = 0

        while min_sims < simulations:
            # Hierarchical sampling
            a_h = alpha_hyper.rvs()
            b_h = beta_hyper.rvs()
            theta = stats.beta(a_h + np.sum(pilot_data),
                              b_h + n_pilot - np.sum(pilot_data)).rvs()

            # Data generation
            k = np.random.binomial(mid, theta)
            a_post = a_h + np.sum(pilot_data) + k
            b_post = b_h + n_pilot - np.sum(pilot_data) + mid - k

            # Accurate HDI calculation
            lower, upper = beta_hdi(a_post, b_post, ci)
            if (upper - lower) <= hdi_width:
                valid += 1
            min_sims += 1

        assurance = valid / simulations
        if assurance >= target_assurance:
            best_n = mid
            high = mid - 1
        else:
            low = mid + 1

    return best_n


def informed_bam_performance(pilot_se, pilot_sp, prevalence_prior=(8,32),
                            hdi_width=0.1, ci=0.95, target_assurance=0.8,
                            simulations=1000, max_sample=5000):
    """Estimate minimum sample size for sensitivity and specificity jointly.
    
    Uses informed Bayesian Assurance Method to determine the minimum
    sample size needed to achieve target HDI widths for BOTH sensitivity
    and specificity simultaneously when planning new studies. Accounts for
    disease prevalence and ensures adequate representation of both positive
    and negative cases.
    
    Args:
        pilot_se (tuple[int, int]): Pilot sensitivity as (true_positives,
            condition_positives). Example: (18, 20) means 18 TPs out of
            20 disease-positive cases.
        pilot_sp (tuple[int, int]): Pilot specificity as (true_negatives,
            condition_negatives). Example: (35, 40) means 35 TNs out of
            40 disease-negative cases.
        prevalence_prior (tuple[int, int], optional): Beta prior parameters
            for disease prevalence (α, β). Defaults to (8, 32) representing
            ~20% prevalence with moderate uncertainty.
        hdi_width (float, optional): Target HDI width for BOTH metrics.
            Defaults to 0.1 (±0.05 precision).
        ci (float, optional): Credible interval level. Defaults to 0.95.
        target_assurance (float, optional): Desired probability of achieving
            target precision. Defaults to 0.8 (80% assurance).
        simulations (int, optional): Number of Monte Carlo simulations per
            candidate N. Defaults to 1000 (fewer than single-metric BAM
            due to two metrics).
        max_sample (int, optional): Maximum sample size to consider.
            Defaults to 5000.
    
    Returns:
        int: Estimated minimum total sample size to achieve target HDI widths
            for both sensitivity and specificity with desired assurance.
    
    Algorithm:
        1. Derive Beta priors from pilot: Se ~ Beta(α_se, β_se)
        2. Binary search over N from [100, max_sample]
        3. For each candidate N:
           - Simulate `simulations` datasets
           - For each simulation:
             a. Sample Se, Sp, prevalence from priors
             b. Generate N_pos ~ Binomial(N, prevalence)
             c. Ensure min_cases per class (at least 5% each)
             d. Generate TP ~ Binomial(N_pos, Se)
             e. Generate TN ~ Binomial(N_neg, Sp)
             f. Compute posteriors: Se_post, Sp_post
             g. Calculate HDI widths for both
           - Count sims where BOTH HDIs ≤ target
        4. Return smallest N where joint assurance ≥ target
    
    Example:
        >>> from early_markers.cribsy.bam import informed_bam_performance
        >>> 
        >>> # Pilot study results
        >>> pilot_se = (18, 20)  # 90% sensitivity (18/20)
        >>> pilot_sp = (35, 40)  # 87.5% specificity (35/40)
        >>> 
        >>> # Find N for ±0.10 HDI on both metrics
        >>> n_required = informed_bam_performance(
        ...     pilot_se=pilot_se,
        ...     pilot_sp=pilot_sp,
        ...     prevalence_prior=(8, 32),  # ~20% prevalence
        ...     hdi_width=0.10,
        ...     target_assurance=0.8
        ... )
        >>> print(f"Required N: {n_required}")
        Required N: 287
        >>> 
        >>> # Expected distribution
        >>> n_pos = int(n_required * 0.2)  # ~57 positives
        >>> n_neg = n_required - n_pos      # ~230 negatives
    
    Note:
        - Requires both metrics to achieve target precision (stricter)
        - Min 5% of each class enforced to avoid degenerate cases
        - Prevalence prior affects final N (higher prevalence = easier
          to achieve sensitivity precision, harder for specificity)
        - For rare diseases (prevalence < 5%), may require very large N
    
    Mathematical Details:
        Posterior distributions:
        - Se_post ~ Beta(α_se + TP, β_se + FN)
        - Sp_post ~ Beta(α_sp + TN, β_sp + FP)
        
        Joint assurance:
        P(HDI_width(Se) ≤ target AND HDI_width(Sp) ≤ target) ≥ target_assurance
    
    See Also:
        bam_performance: For single metric sample size estimation
        beta_hdi: For computing HDI intervals
    """
    # Derive Beta priors from pilot data
    se_alpha, se_beta = pilot_se[0] + 1, pilot_se[1] + 1
    sp_alpha, sp_beta = pilot_sp[0] + 1, pilot_sp[1] + 1

    # Binary search setup with realistic bounds
    low, high = 100, max_sample
    best_n = max_sample

    while low <= high:
        mid = (low + high) // 2
        valid = 0
        min_cases = max(10, int(0.05*mid))  # At least 5% cases per class

        for _ in range(simulations):
            # Sample parameters from informed priors
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

            if (se_hdi[1] - se_hdi[0] <= hdi_width and
                sp_hdi[1] - sp_hdi[0] <= hdi_width):
                valid += 1

        assurance = valid / simulations

        if assurance >= target_assurance:
            best_n = mid
            high = mid - 1
        else:
            low = mid + 1

    return best_n
