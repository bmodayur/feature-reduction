#!/usr/bin/env python3
"""
Run BAM sample size calculation for 20-Feature Nov 21 RFE model.

Uses the metrics from compute_roc_after_rfe.py with 10-fold CV and transformation:
- AUC: 0.9018
- Sensitivity: 92.1% (129 TP / 140 total at-risk across 10 folds)
- Specificity: 77.5% (155 TN / 200 total normal across 10 folds)

Generated: 2025-12-03
"""

import json
from datetime import datetime
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from early_markers.cribsy.common.bam_unified import BAMEstimator

# Metrics from compute_roc_after_rfe.py (10-fold CV with transformation)
# From feature_set_comparison_metrics.csv:
# 20-Feature: TP=129, TN=155, FP=45, FN=11
PILOT_TP = 129
PILOT_FN = 11
PILOT_TN = 155
PILOT_FP = 45

PILOT_DISEASED = PILOT_TP + PILOT_FN  # 140
PILOT_NON_DISEASED = PILOT_TN + PILOT_FP  # 200

SENSITIVITY = PILOT_TP / PILOT_DISEASED  # 0.9214
SPECIFICITY = PILOT_TN / PILOT_NON_DISEASED  # 0.775

print("=" * 80)
print("BAM SAMPLE SIZE ESTIMATION: 20-Feature Nov 21 RFE Model")
print("=" * 80)

print(f"\nPilot Data (from 10-fold CV):")
print(f"  Sensitivity: {SENSITIVITY:.4f} ({PILOT_TP}/{PILOT_DISEASED})")
print(f"  Specificity: {SPECIFICITY:.4f} ({PILOT_TN}/{PILOT_NON_DISEASED})")
print(f"  AUC: 0.9018")

print(f"\nBAM Parameters:")
print(f"  Target HDI width: 0.15 (±7.5% precision)")
print(f"  Target assurance: 0.80 (80% confidence)")
print(f"  CI level: 0.95")

# Initialize estimator
estimator = BAMEstimator(seed=20250313)

# Run joint estimation (sensitivity + specificity)
print(f"\nRunning BAM estimation (this may take 1-2 minutes)...")

result = estimator.estimate_joint(
    pilot_se=(PILOT_TP, PILOT_DISEASED),  # (successes, trials) for sensitivity
    pilot_sp=(PILOT_TN, PILOT_NON_DISEASED),  # (successes, trials) for specificity
    target_width=0.15,
    target_assurance=0.80,
    ci=0.95,
    simulations=2000,
    max_sample=1500
)

print(f"\n{'=' * 80}")
print("RESULTS")
print("=" * 80)

print(f"\nOptimal Sample Size: N = {result.optimal_n}")
print(f"Achieved Assurance: {result.achieved_assurance:.3f}")

# Estimate breakdown by prevalence (~20%)
prevalence = 0.20
n_at_risk = int(result.optimal_n * prevalence)
n_normal = result.optimal_n - n_at_risk

print(f"\nEstimated Breakdown (assuming 20% prevalence):")
print(f"  At-risk infants: {n_at_risk}")
print(f"  Normal infants: {n_normal}")

# Sensitivity analysis
print(f"\n{'=' * 80}")
print("SENSITIVITY ANALYSIS")
print("=" * 80)

sensitivity_results = []
for width, assurance, desc in [
    (0.15, 0.80, "Standard (target)"),
    (0.15, 0.90, "Higher assurance"),
    (0.10, 0.80, "Higher precision"),
    (0.20, 0.80, "Lower precision"),
]:
    r = estimator.estimate_joint(
        pilot_se=(PILOT_TP, PILOT_DISEASED),
        pilot_sp=(PILOT_TN, PILOT_NON_DISEASED),
        target_width=width,
        target_assurance=assurance,
        ci=0.95,
        simulations=2000,
        max_sample=2000
    )
    sensitivity_results.append({
        "width": width,
        "assurance": assurance,
        "description": desc,
        "optimal_n": r.optimal_n,
        "achieved_assurance": r.achieved_assurance
    })
    print(f"  Width ±{width:.2f}, Assurance {assurance:.0%}: N = {r.optimal_n} (achieved: {r.achieved_assurance:.3f})")

# Save results
output = {
    "analysis_date": datetime.now().isoformat(),
    "model": "20-Feature Nov 21 RFE (with 1/(1+|x|) transformation)",
    "pilot_data": {
        "sensitivity": SENSITIVITY,
        "specificity": SPECIFICITY,
        "auc": 0.9018,
        "tp": PILOT_TP,
        "fn": PILOT_FN,
        "tn": PILOT_TN,
        "fp": PILOT_FP,
        "n_diseased": PILOT_DISEASED,
        "n_non_diseased": PILOT_NON_DISEASED,
        "evaluation_method": "10-fold CV"
    },
    "recommended_sample_size": {
        "target_width": 0.15,
        "target_assurance": 0.80,
        "ci_level": 0.95,
        "optimal_n_total": result.optimal_n,
        "achieved_assurance": result.achieved_assurance,
        "estimated_at_risk": n_at_risk,
        "estimated_normal": n_normal
    },
    "sensitivity_analysis": sensitivity_results,
    "interpretation": {
        "precision": "±7.5% for both sensitivity and specificity",
        "confidence": "80% assurance (comparable to 80% power)",
        "constraints": "Joint estimation - BOTH metrics must achieve target"
    }
}

output_file = Path("data/json/bam_sample_size_20feature_nov21.json")
with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print("=" * 80)
