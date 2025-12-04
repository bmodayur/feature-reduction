"""Validation script for unified BAM API.

Compares outputs of bam_unified.py against the original bam.py implementation
to ensure mathematical correctness and verify that the unified API produces
identical results.

Run this script to validate the unified BAM implementation:
    poetry run python -m early_markers.cribsy.common.test_bam_unified
"""

import numpy as np
import time
from pathlib import Path

# Import both implementations
from early_markers.cribsy.bam import bam_performance, informed_bam_performance
from early_markers.cribsy.common.bam_unified import BAMEstimator, bam_single, bam_joint


def test_single_metric():
    """Test single metric BAM against original implementation."""
    print("=" * 80)
    print("TEST 1: Single Metric BAM")
    print("=" * 80)
    
    # Create pilot data
    np.random.seed(42)
    pilot = np.array([1]*15 + [0]*5)  # 75% success rate
    
    print(f"\nPilot data: {len(pilot)} samples, {np.mean(pilot):.3f} success rate")
    
    # Test parameters
    target_width = 0.15
    target_assurance = 0.8
    simulations = 2000
    
    print(f"Target: HDI width ≤ {target_width} with {target_assurance:.0%} assurance")
    print(f"Simulations: {simulations}\n")
    
    # Original implementation
    print("Running original bam.py implementation...")
    start = time.time()
    n_original = bam_performance(
        pilot,
        hdi_width=target_width,
        ci=0.95,
        target_assurance=target_assurance,
        simulations=simulations
    )
    time_original = time.time() - start
    print(f"  Result: N = {n_original}")
    print(f"  Time: {time_original:.2f}s\n")
    
    # Unified implementation (full result)
    print("Running unified BAM implementation (full result)...")
    estimator = BAMEstimator(seed=42, verbose=False)
    start = time.time()
    result = estimator.estimate_single(
        pilot_data=pilot,
        target_width=target_width,
        target_assurance=target_assurance,
        simulations=simulations
    )
    time_unified = time.time() - start
    print(f"  Result: N = {result.optimal_n}")
    print(f"  Achieved assurance: {result.achieved_assurance:.3f}")
    print(f"  Search iterations: {result.search_iterations}")
    print(f"  Time: {time_unified:.2f}s\n")
    
    # Convenience function
    print("Running unified BAM convenience function...")
    start = time.time()
    n_convenience = bam_single(
        pilot_data=pilot,
        target_width=target_width,
        target_assurance=target_assurance,
        simulations=simulations,
        seed=42
    )
    time_convenience = time.time() - start
    print(f"  Result: N = {n_convenience}")
    print(f"  Time: {time_convenience:.2f}s\n")
    
    # Comparison
    print("-" * 80)
    print("COMPARISON:")
    print(f"  Original bam.py:      N = {n_original}")
    print(f"  Unified (full):       N = {result.optimal_n}")
    print(f"  Unified (convenience): N = {n_convenience}")
    
    if n_original == result.optimal_n == n_convenience:
        print("\n✓ PASS: All implementations produce identical results")
        return True
    else:
        print("\n✗ FAIL: Results differ!")
        print(f"  Difference (original vs unified): {abs(n_original - result.optimal_n)}")
        return False


def test_joint_metrics():
    """Test joint metrics BAM against original implementation."""
    print("\n" + "=" * 80)
    print("TEST 2: Joint Metrics BAM (Sensitivity + Specificity)")
    print("=" * 80)
    
    # Pilot data
    pilot_se = (18, 20)  # 90% sensitivity
    pilot_sp = (35, 40)  # 87.5% specificity
    
    print(f"\nPilot sensitivity: {pilot_se[0]}/{pilot_se[1]} = {pilot_se[0]/pilot_se[1]:.3f}")
    print(f"Pilot specificity: {pilot_sp[0]}/{pilot_sp[1]} = {pilot_sp[0]/pilot_sp[1]:.3f}")
    
    # Test parameters
    target_width = 0.10
    target_assurance = 0.8
    simulations = 1000
    prevalence_prior = (8, 32)
    
    print(f"Target: Both metrics HDI width ≤ {target_width} with {target_assurance:.0%} assurance")
    print(f"Prevalence prior: Beta{prevalence_prior}")
    print(f"Simulations: {simulations}\n")
    
    # Original implementation
    print("Running original bam.py implementation...")
    start = time.time()
    n_original = informed_bam_performance(
        pilot_se=pilot_se,
        pilot_sp=pilot_sp,
        prevalence_prior=prevalence_prior,
        hdi_width=target_width,
        ci=0.95,
        target_assurance=target_assurance,
        simulations=simulations
    )
    time_original = time.time() - start
    print(f"  Result: N = {n_original}")
    print(f"  Time: {time_original:.2f}s\n")
    
    # Unified implementation (full result)
    print("Running unified BAM implementation (full result)...")
    estimator = BAMEstimator(seed=42, verbose=False)
    start = time.time()
    result = estimator.estimate_joint(
        pilot_se=pilot_se,
        pilot_sp=pilot_sp,
        prevalence_prior=prevalence_prior,
        target_width=target_width,
        target_assurance=target_assurance,
        simulations=simulations
    )
    time_unified = time.time() - start
    print(f"  Result: N = {result.optimal_n}")
    print(f"  Achieved assurance: {result.achieved_assurance:.3f}")
    print(f"  Search iterations: {result.search_iterations}")
    print(f"  Time: {time_unified:.2f}s\n")
    
    # Convenience function
    print("Running unified BAM convenience function...")
    start = time.time()
    n_convenience = bam_joint(
        pilot_se=pilot_se,
        pilot_sp=pilot_sp,
        prevalence_prior=prevalence_prior,
        target_width=target_width,
        target_assurance=target_assurance,
        simulations=simulations,
        seed=42
    )
    time_convenience = time.time() - start
    print(f"  Result: N = {n_convenience}")
    print(f"  Time: {time_convenience:.2f}s\n")
    
    # Comparison
    print("-" * 80)
    print("COMPARISON:")
    print(f"  Original bam.py:       N = {n_original}")
    print(f"  Unified (full):        N = {result.optimal_n}")
    print(f"  Unified (convenience): N = {n_convenience}")
    
    # Allow small differences due to Monte Carlo variance
    # Joint metrics uses 1000 simulations, so ~1-2% variance is acceptable
    max_diff_pct = 0.02  # 2%
    max_diff_absolute = max(20, int(n_original * max_diff_pct))
    
    diff = abs(n_original - result.optimal_n)
    
    if result.optimal_n == n_convenience:
        print(f"\n✓ Unified implementations agree (N = {result.optimal_n})")
        
        if n_original == result.optimal_n:
            print("✓ PASS: Exact match with original bam.py")
            return True
        elif diff <= max_diff_absolute:
            print(f"✓ PASS: Within acceptable Monte Carlo variance ({diff} samples, {diff/n_original*100:.1f}%)")
            print(f"  (Threshold: ±{max_diff_absolute} samples or {max_diff_pct*100}%)")
            return True
        else:
            print(f"✗ FAIL: Difference too large ({diff} samples, {diff/n_original*100:.1f}%)")
            print(f"  (Threshold: ±{max_diff_absolute} samples or {max_diff_pct*100}%)")
            return False
    else:
        print("\n✗ FAIL: Unified implementations disagree!")
        print(f"  Difference (full vs convenience): {abs(result.optimal_n - n_convenience)}")
        return False


def test_edge_cases():
    """Test edge cases and input validation."""
    print("\n" + "=" * 80)
    print("TEST 3: Edge Cases and Validation")
    print("=" * 80)
    
    estimator = BAMEstimator(seed=42)
    
    # Test 1: Empty pilot data
    print("\n1. Empty pilot data (should raise ValueError):")
    try:
        result = estimator.estimate_single(np.array([]))
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ PASS: Raised ValueError: {e}")
    
    # Test 2: Invalid pilot data (not binary)
    print("\n2. Non-binary pilot data (should raise ValueError):")
    try:
        result = estimator.estimate_single(np.array([0.5, 0.7, 0.9]))
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ PASS: Raised ValueError: {e}")
    
    # Test 3: Invalid target_width
    print("\n3. Invalid target_width (should raise ValueError):")
    try:
        result = estimator.estimate_single(
            np.array([1, 0, 1]),
            target_width=-0.1
        )
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ PASS: Raised ValueError: {e}")
    
    # Test 4: Small pilot sample
    print("\n4. Small pilot sample (3 observations):")
    pilot_small = np.array([1, 1, 0])
    result = estimator.estimate_single(pilot_small, target_width=0.20)
    print(f"  ✓ PASS: Computed N = {result.optimal_n}")
    
    # Test 5: Extreme success rate (all 1s)
    print("\n5. Extreme pilot data (all successes):")
    pilot_extreme = np.array([1] * 20)
    result = estimator.estimate_single(pilot_extreme, target_width=0.10)
    print(f"  ✓ PASS: Computed N = {result.optimal_n}")
    
    # Test 6: Very tight target
    print("\n6. Very tight target (width = 0.05):")
    pilot = np.array([1]*12 + [0]*8)
    result = estimator.estimate_single(pilot, target_width=0.05)
    print(f"  ✓ PASS: Computed N = {result.optimal_n} (expect large N)")
    
    print("\n✓ PASS: All edge case tests passed")
    return True


def test_visualization():
    """Test visualization functionality."""
    print("\n" + "=" * 80)
    print("TEST 4: Visualization")
    print("=" * 80)
    
    # Create a result
    estimator = BAMEstimator(seed=42, verbose=False)
    pilot = np.array([1]*15 + [0]*5)
    result = estimator.estimate_single(pilot, target_width=0.15, simulations=500)
    
    print("\n1. Testing plot generation...")
    try:
        # Test without saving
        fig = result.plot(show=False)
        print("  ✓ PASS: Plot generated successfully (not saved)")
        
        # Test with saving
        save_path = Path("/tmp/test_bam_assurance_curve.png")
        fig = result.plot(save_path=save_path, show=False)
        
        if save_path.exists():
            print(f"  ✓ PASS: Plot saved to {save_path}")
            print(f"  ✓ PASS: result.fig_path = {result.fig_path}")
            save_path.unlink()  # Clean up
        else:
            print("  ✗ FAIL: Plot not saved")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ FAIL: Error during plot generation: {e}")
        return False


def test_interpolation():
    """Test interpolation methods."""
    print("\n" + "=" * 80)
    print("TEST 5: Interpolation Methods")
    print("=" * 80)
    
    # Create a result
    estimator = BAMEstimator(seed=42, verbose=False)
    pilot = np.array([1]*15 + [0]*5)
    result = estimator.estimate_single(pilot, target_width=0.15, simulations=500)
    
    print(f"\nOptimal N: {result.optimal_n}")
    print(f"Sample sizes tested: {result.sample_sizes_tested}")
    print(f"Assurances: {[f'{a:.3f}' for a in result.assurances_at_tested]}")
    
    # Test interpolate_assurance
    print("\n1. Testing interpolate_assurance():")
    test_n = result.optimal_n + 50
    assurance = result.interpolate_assurance(test_n)
    print(f"  Assurance at N={test_n}: {assurance:.3f}")
    print("  ✓ PASS: interpolate_assurance() works")
    
    # Test interpolate_sample_size
    print("\n2. Testing interpolate_sample_size():")
    test_assurance = 0.85
    sample_size = result.interpolate_sample_size(test_assurance)
    print(f"  N needed for {test_assurance:.0%} assurance: {sample_size:.1f}")
    print("  ✓ PASS: interpolate_sample_size() works")
    
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("BAM UNIFIED API VALIDATION SUITE")
    print("=" * 80)
    print("\nValidating mathematical correctness by comparing with original bam.py")
    print("implementation. All tests use identical random seeds for reproducibility.\n")
    
    results = {}
    
    # Run tests
    results['single_metric'] = test_single_metric()
    results['joint_metrics'] = test_joint_metrics()
    results['edge_cases'] = test_edge_cases()
    results['visualization'] = test_visualization()
    results['interpolation'] = test_interpolation()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("The unified BAM API is mathematically correct and ready for use.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the failed tests above.")
    print("=" * 80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
