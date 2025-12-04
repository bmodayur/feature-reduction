#!/usr/bin/env python3
"""
At-Risk Group Bimodality Analysis

This script analyzes the apparent bimodal distribution observed in the at-risk
group's z-score (Bayesian surprise) distribution. It aims to:

1. Confirm bimodality using statistical tests
2. Identify which infants fall into each mode
3. Examine clinical/demographic characteristics of each subgroup
4. Assess whether the modes correspond to severity differentiation

Usage:
    python scripts/atrisk_group_analysis.py
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# Paths - same as compute_roc_after_rfe.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "pkl"  # Features are in pkl directory
FEATURES_FILE = 'features_merged_20251121_091511.pkl'

# Parameters
SAMPLES = 30
AGE_THRESHOLD = 24
K_FOLDS = 10
RANDOM_STATE = 20250313


def create_train_test(f, samples=30, age_threshold=24):
    """Create train/test split (same as compute_roc_after_rfe.py)."""
    features = f.copy()
    features['train'] = 0

    # train set includes SAMPLES typical infants YOUNGER than age_threshold weeks, sampled
    train_infants = features.loc[
        (features.risk <= 1) & (features.age_in_weeks < age_threshold), 'infant'
    ].drop_duplicates().sample(n=samples, random_state=RANDOM_STATE)

    features.loc[features['infant'].isin(train_infants), 'train'] = 1

    return features


def compute_surprise(train_data, test_data, features_list):
    """
    Compute Bayesian surprise z-scores using log-probability approach.

    This matches compute_roc_after_rfe.py:
    - minus_log_pfeature = log probability under normative Gaussian
    - Higher = more typical, Lower = more atypical
    - More negative z-scores indicate more atypical (at-risk) patterns
    """
    results = []

    for feature in features_list:
        train_vals = train_data[train_data['feature'] == feature]['Value']
        test_subset = test_data[test_data['feature'] == feature].copy()

        if len(train_vals) == 0 or len(test_subset) == 0:
            continue

        # Reference statistics from training (normative) data
        mean_ref = train_vals.mean()
        var_ref = train_vals.var()

        if var_ref > 0:
            # Log probability under Gaussian (same formula as compute_roc_after_rfe.py)
            # minus_log_pfeature = -1 * (0.5*log(2*pi*var) + (x-mu)^2/(2*var))
            # This is actually the log probability (higher = more typical)
            test_subset['minus_log_pfeature'] = -1 * (
                0.5 * np.log(2 * np.pi * var_ref) +
                ((test_subset['Value'] - mean_ref) ** 2) / (2 * var_ref)
            )
        else:
            test_subset['minus_log_pfeature'] = 0

        results.append(test_subset)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def run_kfold_cv(features, feature_list, k_folds=10):
    """
    Run k-fold cross-validation to get z-scores.

    Matches compute_roc_after_rfe.py approach:
    1. Compute minus_log_pfeature for each feature
    2. Sum across features per infant
    3. Normalize to z-scores using training (normative) set statistics
    """
    from sklearn.model_selection import StratifiedKFold

    # Get unique infants in test set (not in train)
    test_infants = features[features['train'] == 0]['infant'].unique()
    test_data_full = features[features['infant'].isin(test_infants)]

    # Get risk labels for stratification
    infant_risk = test_data_full.drop_duplicates('infant')[['infant', 'risk']].set_index('infant')['risk']

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)

    all_test_data = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(test_infants, infant_risk[test_infants])):
        fold_train_infants = test_infants[train_idx]
        fold_test_infants = test_infants[test_idx]

        # Combine original train with fold train (all become normative reference)
        train_data = features[
            (features['train'] == 1) | (features['infant'].isin(fold_train_infants))
        ]
        train_data = train_data[train_data['feature'].isin(feature_list)]

        # All data for this fold (to compute surprise for everyone)
        all_fold_data = features[
            (features['train'] == 1) |
            (features['infant'].isin(fold_train_infants)) |
            (features['infant'].isin(fold_test_infants))
        ]
        all_fold_data = all_fold_data[all_fold_data['feature'].isin(feature_list)]

        # Compute surprise (minus_log_pfeature) for all subjects
        surprise_results = compute_surprise(train_data, all_fold_data, feature_list)

        if len(surprise_results) > 0:
            # Sum minus_log_pfeature across features per infant
            fold_surprise = surprise_results.groupby(['infant', 'age_in_weeks', 'risk']).agg({
                'minus_log_pfeature': 'sum'
            }).reset_index()

            # Mark train vs test
            fold_surprise['is_train'] = fold_surprise['infant'].isin(
                list(features[features['train'] == 1]['infant'].unique()) + list(fold_train_infants)
            )

            # Normalize z-scores using training set statistics
            train_mean = fold_surprise.loc[fold_surprise['is_train'], 'minus_log_pfeature'].mean()
            train_std = fold_surprise.loc[fold_surprise['is_train'], 'minus_log_pfeature'].std()

            if train_std > 0:
                fold_surprise['z'] = (fold_surprise['minus_log_pfeature'] - train_mean) / train_std
            else:
                fold_surprise['z'] = 0

            # Keep only test infants for this fold
            test_surprise = fold_surprise[fold_surprise['infant'].isin(fold_test_infants)].copy()
            test_surprise['fold'] = fold

            all_test_data.append(test_surprise)

    if all_test_data:
        combined = pd.concat(all_test_data, ignore_index=True)
        return combined
    return pd.DataFrame()


def hartigans_dip_test(data, n_boot=1000):
    """
    Perform Hartigan's dip test for unimodality.
    Returns dip statistic and approximate p-value via bootstrap.
    """
    data = np.sort(data)
    n = len(data)

    if n < 4:
        return 0.0, 1.0

    # Compute empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Compute dip statistic (simplified)
    uniform_cdf = (data - data[0]) / (data[-1] - data[0] + 1e-10)
    dip = np.max(np.abs(ecdf - uniform_cdf)) / 2

    # Bootstrap p-value
    boot_dips = []
    for _ in range(n_boot):
        boot_sample = np.sort(np.random.uniform(data.min(), data.max(), n))
        boot_ecdf = np.arange(1, n + 1) / n
        boot_uniform = (boot_sample - boot_sample[0]) / (boot_sample[-1] - boot_sample[0] + 1e-10)
        boot_dip = np.max(np.abs(boot_ecdf - boot_uniform)) / 2
        boot_dips.append(boot_dip)

    p_value = np.mean(np.array(boot_dips) >= dip)
    return dip, p_value


def identify_modes_gmm(data, n_components=2):
    """
    Use Gaussian Mixture Model to identify modes.

    Cluster labeling convention (consistent with z-score interpretation):
    - More negative z = more atypical = MORE SEVERE (cluster 1)
    - Less negative z (closer to 0) = more typical = LESS SEVERE (cluster 0)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("sklearn not available, using median split")
        return np.where(data < np.median(data), 1, 0), None  # More negative = cluster 1

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    data_reshaped = data.reshape(-1, 1)
    gmm.fit(data_reshaped)
    clusters = gmm.predict(data_reshaped)

    # Reorder so cluster 0 = higher z (less severe), cluster 1 = lower z (more severe)
    means = [data[clusters == i].mean() for i in range(n_components)]
    if means[0] < means[1]:
        # Cluster 0 has lower mean, swap so cluster 0 = higher mean = less severe
        clusters = 1 - clusters

    return clusters, gmm


def analyze_bimodality(z_scores, label=""):
    """Analyze whether z-scores show bimodal distribution."""
    print(f"\n{'='*60}")
    print(f"BIMODALITY ANALYSIS: {label}")
    print(f"{'='*60}")

    z_array = np.array(z_scores)
    n = len(z_array)

    print(f"\nSample size: {n}")
    print(f"Mean z-score: {z_array.mean():.3f}")
    print(f"Std z-score: {z_array.std():.3f}")
    print(f"Range: [{z_array.min():.3f}, {z_array.max():.3f}]")

    # Hartigan's dip test
    dip, p_value = hartigans_dip_test(z_array)
    print(f"\nHartigan's Dip Test:")
    print(f"  Dip statistic: {dip:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  -> Significant evidence of multimodality (p < 0.05)")
    else:
        print(f"  -> No significant evidence of multimodality (p >= 0.05)")

    # Shapiro-Wilk test for normality
    if n >= 3:
        stat, p_norm = stats.shapiro(z_array)
        print(f"\nShapiro-Wilk Normality Test:")
        print(f"  W statistic: {stat:.4f}")
        print(f"  P-value: {p_norm:.4f}")
        if p_norm < 0.05:
            print(f"  -> Distribution is NOT normal (p < 0.05)")
        else:
            print(f"  -> Distribution is consistent with normal (p >= 0.05)")

    # GMM clustering
    clusters, gmm = identify_modes_gmm(z_array)

    print(f"\nGaussian Mixture Model (2 components):")
    for i in range(2):
        cluster_data = z_array[clusters == i]
        n_cluster = len(cluster_data)
        if n_cluster > 0:
            severity = 'Less Severe (closer to typical)' if i == 0 else 'More Severe (more atypical)'
            print(f"\n  Cluster {i} ({severity}):")
            print(f"    N = {n_cluster} ({100*n_cluster/n:.1f}%)")
            print(f"    Mean z-score: {cluster_data.mean():.3f}")
            print(f"    Std z-score: {cluster_data.std():.3f}")
            print(f"    Range: [{cluster_data.min():.3f}, {cluster_data.max():.3f}]")

    if gmm is not None:
        print(f"\n  GMM Parameters:")
        for i in range(2):
            print(f"    Component {i}: mean={gmm.means_[i][0]:.3f}, "
                  f"std={np.sqrt(gmm.covariances_[i][0][0]):.3f}, "
                  f"weight={gmm.weights_[i]:.3f}")

    return clusters, gmm


def plot_analysis(baseline_z, transformed_z, baseline_clusters, transformed_clusters,
                  baseline_gmm, transformed_gmm, infant_info):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ['#3498db', '#e74c3c']  # Blue=less severe, Red=more severe
    labels = ['Less Severe', 'More Severe']

    # Row 1: Baseline
    # Histogram with clusters
    ax = axes[0, 0]
    for i in range(2):
        mask = baseline_clusters == i
        ax.hist(baseline_z[mask], bins=15, alpha=0.6, color=colors[i],
                label=labels[i], edgecolor='white')
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Typical mean')
    ax.set_xlabel('Z-Score (Surprise)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('BASELINE: Distribution by Cluster', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Strip plot
    ax = axes[0, 1]
    for i in range(2):
        mask = baseline_clusters == i
        y = baseline_z[mask]
        x = np.random.normal(i, 0.08, len(y))
        ax.scatter(x, y, c=colors[i], alpha=0.7, s=60, edgecolor='white', linewidth=0.5)
        if len(y) > 0:
            ax.errorbar(i, y.mean(), yerr=y.std(), fmt='D', markersize=12,
                       markerfacecolor='white', markeredgecolor=colors[i],
                       markeredgewidth=2, ecolor=colors[i], elinewidth=2,
                       capsize=6, capthick=2, zorder=10)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Z-Score', fontsize=11)
    ax.set_title('BASELINE: Cluster Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # GMM fit
    ax = axes[0, 2]
    x_range = np.linspace(baseline_z.min() - 1, baseline_z.max() + 1, 200)
    ax.hist(baseline_z, bins=20, density=True, alpha=0.5, color='gray', edgecolor='white')
    if baseline_gmm is not None:
        for i in range(2):
            mean = baseline_gmm.means_[i][0]
            std = np.sqrt(baseline_gmm.covariances_[i][0][0])
            weight = baseline_gmm.weights_[i]
            y = weight * stats.norm.pdf(x_range, mean, std)
            ax.plot(x_range, y, color=colors[i], linewidth=2, label=f'{labels[i]}')
        y_total = sum(baseline_gmm.weights_[i] * stats.norm.pdf(x_range, baseline_gmm.means_[i][0],
                      np.sqrt(baseline_gmm.covariances_[i][0][0])) for i in range(2))
        ax.plot(x_range, y_total, 'k--', linewidth=2, label='Combined')
    ax.set_xlabel('Z-Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('BASELINE: GMM Fit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Transformed
    ax = axes[1, 0]
    for i in range(2):
        mask = transformed_clusters == i
        ax.hist(transformed_z[mask], bins=15, alpha=0.6, color=colors[i],
                label=labels[i], edgecolor='white')
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Typical mean')
    ax.set_xlabel('Z-Score (Surprise)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('TRANSFORMED: Distribution by Cluster', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for i in range(2):
        mask = transformed_clusters == i
        y = transformed_z[mask]
        x = np.random.normal(i, 0.08, len(y))
        ax.scatter(x, y, c=colors[i], alpha=0.7, s=60, edgecolor='white', linewidth=0.5)
        if len(y) > 0:
            ax.errorbar(i, y.mean(), yerr=y.std(), fmt='D', markersize=12,
                       markerfacecolor='white', markeredgecolor=colors[i],
                       markeredgewidth=2, ecolor=colors[i], elinewidth=2,
                       capsize=6, capthick=2, zorder=10)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Z-Score', fontsize=11)
    ax.set_title('TRANSFORMED: Cluster Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 2]
    x_range = np.linspace(transformed_z.min() - 1, transformed_z.max() + 1, 200)
    ax.hist(transformed_z, bins=20, density=True, alpha=0.5, color='gray', edgecolor='white')
    if transformed_gmm is not None:
        for i in range(2):
            mean = transformed_gmm.means_[i][0]
            std = np.sqrt(transformed_gmm.covariances_[i][0][0])
            weight = transformed_gmm.weights_[i]
            y = weight * stats.norm.pdf(x_range, mean, std)
            ax.plot(x_range, y, color=colors[i], linewidth=2, label=f'{labels[i]}')
        y_total = sum(transformed_gmm.weights_[i] * stats.norm.pdf(x_range, transformed_gmm.means_[i][0],
                      np.sqrt(transformed_gmm.covariances_[i][0][0])) for i in range(2))
        ax.plot(x_range, y_total, 'k--', linewidth=2, label='Combined')
    ax.set_xlabel('Z-Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('TRANSFORMED: GMM Fit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('At-Risk Group Bimodality Analysis\n(20-Feature NEW Model)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('atrisk_bimodality_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: atrisk_bimodality_analysis.png")


def analyze_characteristics(infant_info, baseline_clusters, transformed_clusters):
    """Analyze characteristics of each cluster."""
    print(f"\n{'='*60}")
    print("CLUSTER CHARACTERISTICS")
    print(f"{'='*60}")

    # Age analysis
    if 'age_in_weeks' in infant_info.columns:
        print(f"\nAge Distribution by Cluster (Baseline):")
        for i in range(2):
            mask = baseline_clusters == i
            ages = infant_info.loc[mask, 'age_in_weeks']
            severity = 'Less Severe' if i == 0 else 'More Severe'
            if len(ages) > 0:
                print(f"  {severity}: N={len(ages)}, mean={ages.mean():.1f}, "
                      f"std={ages.std():.1f}, range=[{ages.min():.0f}, {ages.max():.0f}] weeks")

    # Risk level analysis
    if 'risk' in infant_info.columns:
        print(f"\nRisk Level Distribution by Cluster (Baseline):")
        for i in range(2):
            mask = baseline_clusters == i
            risks = infant_info.loc[mask, 'risk']
            severity = 'Less Severe' if i == 0 else 'More Severe'
            if len(risks) > 0:
                risk_counts = risks.value_counts().sort_index()
                print(f"  {severity}: {dict(risk_counts)}")

    # Cluster consistency
    print(f"\nCluster Consistency (Baseline vs Transformed):")
    agreement = np.sum(baseline_clusters == transformed_clusters)
    total = len(baseline_clusters)
    print(f"  Same cluster assignment: {agreement}/{total} ({100*agreement/total:.1f}%)")

    print(f"\n  Contingency Table:")
    print(f"                      Transformed")
    print(f"                      Less Severe  More Severe")
    print(f"  Baseline Less Severe    {np.sum((baseline_clusters==0) & (transformed_clusters==0)):3d}          "
          f"{np.sum((baseline_clusters==0) & (transformed_clusters==1)):3d}")
    print(f"  Baseline More Severe    {np.sum((baseline_clusters==1) & (transformed_clusters==0)):3d}          "
          f"{np.sum((baseline_clusters==1) & (transformed_clusters==1)):3d}")


def list_infants(infant_info, baseline_z, transformed_z, baseline_clusters, transformed_clusters):
    """List individual infants with their assignments."""
    print(f"\n{'='*60}")
    print("INDIVIDUAL INFANT ASSIGNMENTS")
    print(f"{'='*60}")

    print(f"\n{'Infant':<12} {'Age':>6} {'Risk':>5} {'Base Z':>10} {'Trans Z':>10} "
          f"{'Base Clust':>12} {'Trans Clust':>12}")
    print("-" * 80)

    for idx in range(len(infant_info)):
        infant = infant_info.iloc[idx]['infant']
        age = infant_info.iloc[idx].get('age_in_weeks', 'N/A')
        risk = infant_info.iloc[idx].get('risk', 'N/A')
        b_z = baseline_z[idx]
        t_z = transformed_z[idx]
        b_c = 'Less Severe' if baseline_clusters[idx] == 0 else 'More Severe'
        t_c = 'Less Severe' if transformed_clusters[idx] == 0 else 'More Severe'

        marker = "" if baseline_clusters[idx] == transformed_clusters[idx] else " *"
        age_str = f"{age:.0f}" if isinstance(age, (int, float)) else str(age)

        print(f"{str(infant):<12} {age_str:>6} {risk:>5} {b_z:>10.3f} {t_z:>10.3f} "
              f"{b_c:>12} {t_c:>12}{marker}")

    print("-" * 80)
    print("* = Different cluster between baseline and transformed")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("AT-RISK GROUP BIMODALITY ANALYSIS")
    print("=" * 60)

    # Load features
    features_path = OUTPUT_DIR / FEATURES_FILE
    print(f"\nLoading features from: {features_path}")

    if not features_path.exists():
        print(f"ERROR: File not found: {features_path}")
        return

    f = pd.read_pickle(features_path)
    f['feature'] = f['part'] + '_' + f['feature_name']

    # Add age as feature
    age_rows = f.drop_duplicates(subset='infant').copy()
    age_rows['feature'] = 'age_in_weeks'
    age_rows['Value'] = age_rows['age_in_weeks']
    f = pd.concat([f, age_rows], ignore_index=True)

    # Create train/test split
    features = create_train_test(f, samples=SAMPLES, age_threshold=AGE_THRESHOLD)

    # 20-Feature NEW list
    feature_list = [
        "Ankle_IQRvely", "Ankle_medianvely", "Ankle_meanent", "Ankle_IQRvelx",
        "Ankle_IQRaccy", "Ankle_medianx", "Ankle_mediany", "Ankle_IQRaccx",
        "Ankle_medianvelx", "Knee_IQR_acc_angle", "Knee_lrCorr_angle", "Knee_IQR_vel_angle",
        "Wrist_IQRvelx", "Wrist_medianvely", "Wrist_mediany", "Hip_entropy_angle",
        "Hip_lrCorr_angle", "Elbow_IQR_acc_angle", "Shoulder_stdev_angle", "age_in_weeks"
    ]

    print(f"\nRunning {K_FOLDS}-fold CV for baseline (no transform)...")
    baseline_combined = run_kfold_cv(features, feature_list, K_FOLDS)

    print(f"Running {K_FOLDS}-fold CV for transformed features...")
    # Apply 1/(1+|x|) transform to velocity/acceleration features
    # (same as compute_roc_after_rfe.py)
    TRANSFORM_FEATURES = [
        # Velocity features
        'Ankle_IQRvelx', 'Ankle_medianvelx', 'Ankle_medianvely', 'Ankle_IQRvely',
        'Wrist_medianvely', 'Wrist_IQRvelx',
        'Knee_IQR_vel_angle',
        # Acceleration features
        'Ankle_IQRaccy', 'Ankle_IQRaccx',
        'Elbow_IQR_acc_angle',
        'Knee_IQR_acc_angle'
    ]

    features_transformed = features.copy()
    for feat in TRANSFORM_FEATURES:
        if feat in feature_list:
            mask = features_transformed['feature'] == feat
            if mask.any():
                # Apply 1/(1+|x|) transformation
                features_transformed.loc[mask, 'Value'] = 1 / (1 + np.abs(features_transformed.loc[mask, 'Value']))

    transformed_combined = run_kfold_cv(features_transformed, feature_list, K_FOLDS)

    # Z-scores are already aggregated per infant in run_kfold_cv
    # Just need to handle any duplicate infant entries from different folds
    print("\nAggregating z-scores per infant...")

    baseline_agg = baseline_combined.groupby('infant').agg({
        'z': 'mean',  # Average across folds if infant appears in multiple
        'risk': 'first',
        'age_in_weeks': 'first'
    }).reset_index()

    transformed_agg = transformed_combined.groupby('infant').agg({
        'z': 'mean',
        'risk': 'first',
        'age_in_weeks': 'first'
    }).reset_index()

    # Print overall statistics to verify z-score direction
    print(f"\nZ-score verification (more negative = more atypical):")
    print(f"  Baseline - Typical mean z: {baseline_agg[baseline_agg['risk'] <= 1]['z'].mean():.3f}")
    print(f"  Baseline - At-risk mean z: {baseline_agg[baseline_agg['risk'] > 1]['z'].mean():.3f}")
    print(f"  Transformed - Typical mean z: {transformed_agg[transformed_agg['risk'] <= 1]['z'].mean():.3f}")
    print(f"  Transformed - At-risk mean z: {transformed_agg[transformed_agg['risk'] > 1]['z'].mean():.3f}")

    # Filter to at-risk only
    baseline_atrisk = baseline_agg[baseline_agg['risk'] > 1].copy()
    transformed_atrisk = transformed_agg[transformed_agg['risk'] > 1].copy()

    # Ensure same infants in both
    common_infants = set(baseline_atrisk['infant']) & set(transformed_atrisk['infant'])
    baseline_atrisk = baseline_atrisk[baseline_atrisk['infant'].isin(common_infants)].sort_values('infant')
    transformed_atrisk = transformed_atrisk[transformed_atrisk['infant'].isin(common_infants)].sort_values('infant')

    print(f"\nAt-risk infants for analysis: {len(baseline_atrisk)}")

    baseline_z = baseline_atrisk['z'].values
    transformed_z = transformed_atrisk['z'].values

    # Analyze bimodality
    baseline_clusters, baseline_gmm = analyze_bimodality(baseline_z, "Baseline At-Risk")
    transformed_clusters, transformed_gmm = analyze_bimodality(transformed_z, "Transformed At-Risk")

    # Plot
    try:
        plot_analysis(baseline_z, transformed_z, baseline_clusters, transformed_clusters,
                      baseline_gmm, transformed_gmm, baseline_atrisk)
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {e}")

    # Analyze characteristics
    analyze_characteristics(baseline_atrisk.reset_index(drop=True),
                           baseline_clusters, transformed_clusters)

    # List infants
    list_infants(baseline_atrisk.reset_index(drop=True),
                 baseline_z, transformed_z, baseline_clusters, transformed_clusters)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"""
Sample size: {len(baseline_z)} at-risk infants

Baseline clusters: {np.sum(baseline_clusters==0)} less severe, {np.sum(baseline_clusters==1)} more severe
Transformed clusters: {np.sum(transformed_clusters==0)} less severe, {np.sum(transformed_clusters==1)} more severe
Cluster agreement: {100*np.mean(baseline_clusters==transformed_clusters):.1f}%

Interpretation:
- With N={len(baseline_z)}, {'results should be interpreted cautiously (small sample)' if len(baseline_z) < 30 else 'sample size is adequate'}
- High cluster agreement suggests consistent subgroups
- Clinical validation recommended to confirm severity hypothesis
""")


if __name__ == "__main__":
    main()
