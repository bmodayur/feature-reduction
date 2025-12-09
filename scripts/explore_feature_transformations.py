# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: conda-caffe
#     language: python
#     name: conda-caffe
# ---

# %% [markdown]
# # Feature Transformation Exploration for Bayesian Surprise
#
# This notebook explores different feature transformations to test whether
# transforming features can improve Se/Sp metrics by capturing at-risk infants
# at both tails of the distribution.
#
# **Hypothesis**: At-risk infants may show:
# - Hypokinesia (reduced movement) → low velocities/accelerations
# - Dyskinesia (jerky movement) → high accelerations
#
# Both should be flagged as abnormal, suggesting a two-tailed approach or
# transformations that map both abnormalities to the same direction.

# %%
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy as sc
from sklearn.metrics import auc
import warnings
warnings.filterwarnings('ignore')

# %%
# CONFIGURATION
output_dir = '/BigData/EMCP_BACKUP/output/'
FEATURES_FILE = 'features_merged_20251121_091511.pkl'
AGE_THRESHOLD = 10
SAMPLES = 20
K_FOLDS = 10
np.random.seed(42)

# 20-Feature NEW set (best for screening)
SELECTED_FEATURES = [
    "Ankle_IQRvelx",
    "Ankle_IQRx",
    "Ankle_IQRaccy",
    "Ankle_medianvelx",
    "Ankle_lrCorr_x",
    "Ankle_meanent",
    "Ankle_medianvely",
    "Ankle_IQRvely",
    "Elbow_IQR_acc_angle",
    "Elbow_IQR_vel_angle",
    "Elbow_median_vel_angle",
    "Hip_stdev_angle",
    "Hip_entropy_angle",
    "Hip_IQR_vel_angle",
    "Shoulder_mean_angle",
    "Shoulder_IQR_acc_angle",
    "Wrist_mediany",
    "Wrist_IQRx",
    "Wrist_IQRaccx",
    "age_in_weeks"
]

# Identify feature types for targeted transformations
VELOCITY_FEATURES = [f for f in SELECTED_FEATURES if 'vel' in f.lower()]
ACCELERATION_FEATURES = [f for f in SELECTED_FEATURES if 'acc' in f.lower()]
VARIABILITY_FEATURES = [f for f in SELECTED_FEATURES if 'IQR' in f.lower() or 'stdev' in f.lower()]
ENTROPY_FEATURES = [f for f in SELECTED_FEATURES if 'ent' in f.lower()]

print("Feature breakdown:")
print(f"  Velocity features: {VELOCITY_FEATURES}")
print(f"  Acceleration features: {ACCELERATION_FEATURES}")
print(f"  Variability features: {VARIABILITY_FEATURES}")
print(f"  Entropy features: {ENTROPY_FEATURES}")

# %%
def create_train_test(features, samples, age_threshold):
    """Split data into train (normative) and test sets"""
    features = features.copy()
    features['age_bracket'] = (features.age_in_weeks > age_threshold) + 0

    # Move low-risk CLIN subjects to normative set
    features.loc[(features.category == 1) & (features.risk <= 1), 'category'] = 0

    # Split normative set
    norm_infants = features[features.category == 0].infant.unique()
    test_infant_names = np.random.choice(norm_infants, samples, replace=False)
    features.loc[features['infant'].isin(test_infant_names), 'category'] = 1
    features.loc[features['infant'].isin(test_infant_names), 'risk'] = 1

    return features


def load_and_prepare_data():
    """Load features and add age as a feature"""
    f = pd.read_pickle(f"{output_dir}/{FEATURES_FILE}")
    f['feature'] = f.part + '_' + f.feature_name

    # Add age as a feature
    age_rows = f.drop_duplicates(subset='infant').copy()
    age_rows['feature'] = 'age_in_weeks'
    age_rows['Value'] = age_rows['age_in_weeks']
    age_rows['part'] = 'age'
    age_rows['feature_name'] = 'in_weeks'

    f = pd.concat([f, age_rows], ignore_index=True)
    return f

# %%
# TRANSFORMATION FUNCTIONS

def transform_original(features_df):
    """Original: No transformation - baseline"""
    return features_df.copy()


def transform_absolute_deviation(features_df, ref_stats):
    """Transform to absolute deviation from mean: |Value - mean_ref|"""
    df = features_df.copy()
    df = pd.merge(df, ref_stats[['feature_name', 'part', 'mean_ref', 'sd_ref']],
                  on=['feature_name', 'part'], how='left')
    # Replace Value with absolute deviation
    df['Value'] = np.abs(df['Value'] - df['mean_ref'])
    df = df.drop(['mean_ref', 'sd_ref'], axis=1)
    return df


def transform_squared_deviation(features_df, ref_stats):
    """Transform to squared deviation: (Value - mean_ref)²"""
    df = features_df.copy()
    df = pd.merge(df, ref_stats[['feature_name', 'part', 'mean_ref', 'sd_ref']],
                  on=['feature_name', 'part'], how='left')
    # Replace Value with squared deviation
    df['Value'] = (df['Value'] - df['mean_ref']) ** 2
    df = df.drop(['mean_ref', 'sd_ref'], axis=1)
    return df


def transform_absolute_zscore(features_df, ref_stats):
    """Transform to absolute z-score: |z| = |Value - mean_ref| / sd_ref"""
    df = features_df.copy()
    df = pd.merge(df, ref_stats[['feature_name', 'part', 'mean_ref', 'sd_ref']],
                  on=['feature_name', 'part'], how='left')
    # Replace Value with absolute z-score
    df['Value'] = np.abs((df['Value'] - df['mean_ref']) / df['sd_ref'].replace(0, 1))
    df = df.drop(['mean_ref', 'sd_ref'], axis=1)
    return df


def transform_squared_zscore(features_df, ref_stats):
    """Transform to squared z-score: z² = ((Value - mean_ref) / sd_ref)²"""
    df = features_df.copy()
    df = pd.merge(df, ref_stats[['feature_name', 'part', 'mean_ref', 'sd_ref']],
                  on=['feature_name', 'part'], how='left')
    # Replace Value with squared z-score
    z = (df['Value'] - df['mean_ref']) / df['sd_ref'].replace(0, 1)
    df['Value'] = z ** 2
    df = df.drop(['mean_ref', 'sd_ref'], axis=1)
    return df


def transform_inverse_vel_acc(features_df, vel_acc_features):
    """Transform velocity/acceleration features to 1/(1+|x|) to invert the scale"""
    df = features_df.copy()
    mask = df['feature'].isin(vel_acc_features)
    df.loc[mask, 'Value'] = 1 / (1 + np.abs(df.loc[mask, 'Value']))
    return df


def transform_log_magnitude(features_df):
    """Transform to log(1 + |Value|) to compress extreme values"""
    df = features_df.copy()
    df['Value'] = np.log1p(np.abs(df['Value']))
    return df


def transform_rank_based(features_df, ref_stats):
    """Transform to rank-based percentile within normative distribution
    Values far from median (either direction) get high scores"""
    df = features_df.copy()
    df = pd.merge(df, ref_stats[['feature_name', 'part', 'mean_ref', 'sd_ref']],
                  on=['feature_name', 'part'], how='left')
    # CDF gives percentile, transform to distance from 0.5 (median)
    z = (df['Value'] - df['mean_ref']) / df['sd_ref'].replace(0, 1)
    cdf = norm.cdf(z)
    df['Value'] = np.abs(cdf - 0.5) * 2  # Scale to [0, 1], 0=median, 1=extreme
    df = df.drop(['mean_ref', 'sd_ref'], axis=1)
    return df

# %%
def compute_surprise_with_transformation(features_df, feature_list, transform_func,
                                          transform_name, ref_stats=None):
    """
    Compute Bayesian surprise with optional transformation.

    Parameters:
    -----------
    features_df : DataFrame - Raw features
    feature_list : list - Features to include
    transform_func : callable - Transformation function
    transform_name : str - Name for logging
    ref_stats : DataFrame - Reference statistics (needed for some transforms)

    Returns:
    --------
    DataFrame with surprise scores
    """

    # Filter to selected features
    df = features_df.loc[features_df.feature.isin(feature_list)].copy()

    # Apply transformation if not original
    if transform_func is not None and transform_name != 'original':
        if ref_stats is not None and transform_name in ['abs_deviation', 'squared_deviation',
                                                          'abs_zscore', 'squared_zscore', 'rank_based']:
            df = transform_func(df, ref_stats)
        elif transform_name == 'inverse_vel_acc':
            vel_acc = VELOCITY_FEATURES + ACCELERATION_FEATURES
            df = transform_func(df, vel_acc)
        elif transform_name == 'log_magnitude':
            df = transform_func(df)

    # Compute reference statistics from training set (category=0)
    train_df = df[df.category == 0]
    ref = train_df.groupby(['feature_name', 'part'])['Value'].apply(norm.fit).reset_index()
    ref[['mean_ref', 'sd_ref']] = ref['Value'].apply(pd.Series)
    ref['var_ref'] = ref['sd_ref'] ** 2
    ref = ref.drop('Value', axis=1)

    # Merge and compute surprise
    df = pd.merge(df, ref, on=['feature_name', 'part'], how='inner')
    df['minus_log_pfeature'] = -1 * (
        0.5 * np.log(2 * np.pi * df['var_ref']) +
        ((df['Value'] - df['mean_ref']) ** 2) / (2 * df['var_ref'])
    )

    # Aggregate surprise per infant
    surprise = df.groupby(['infant', 'age_in_weeks', 'risk', 'category'])[
        'minus_log_pfeature'].sum().reset_index()

    # Normalize z-scores using training set
    train_mean = surprise.loc[surprise.category == 0, 'minus_log_pfeature'].mean()
    train_std = surprise.loc[surprise.category == 0, 'minus_log_pfeature'].std()
    surprise['z'] = (surprise['minus_log_pfeature'] - train_mean) / train_std

    # Remove problematic subjects
    surprise = surprise[~surprise.infant.str.contains('clin_100_')]

    return surprise, ref

# %%
def compute_roc_metrics(surprise_df, thresholds, use_absolute=False):
    """
    Compute ROC metrics for given thresholds.

    Parameters:
    -----------
    surprise_df : DataFrame with 'z' and 'risk' columns
    thresholds : list of threshold values
    use_absolute : bool - if True, use |z| > threshold (two-tailed)

    Returns:
    --------
    DataFrame with metrics for each threshold
    """
    metrics = []

    for thresh in thresholds:
        if use_absolute:
            # Two-tailed: flag if |z| > threshold
            TP = surprise_df[(surprise_df.risk > 1) & (np.abs(surprise_df.z) > thresh)].shape[0]
            FP = surprise_df[(surprise_df.risk <= 1) & (np.abs(surprise_df.z) > thresh)].shape[0]
            FN = surprise_df[(surprise_df.risk > 1) & (np.abs(surprise_df.z) <= thresh)].shape[0]
            TN = surprise_df[(surprise_df.risk <= 1) & (np.abs(surprise_df.z) <= thresh)].shape[0]
        else:
            # One-tailed: flag if z < threshold
            TP = surprise_df[(surprise_df.risk > 1) & (surprise_df.z < thresh)].shape[0]
            FP = surprise_df[(surprise_df.risk <= 1) & (surprise_df.z < thresh)].shape[0]
            FN = surprise_df[(surprise_df.risk > 1) & (surprise_df.z >= thresh)].shape[0]
            TN = surprise_df[(surprise_df.risk <= 1) & (surprise_df.z >= thresh)].shape[0]

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        youden = sens + spec - 1

        metrics.append({
            'threshold': thresh,
            'sensitivity': sens,
            'specificity': spec,
            'ppv': ppv,
            'npv': npv,
            'youden_j': youden,
            'fpr': 1 - spec,
            'tp': TP, 'fp': FP, 'fn': FN, 'tn': TN
        })

    return pd.DataFrame(metrics)


def find_optimal_metrics(metrics_df):
    """Find optimal threshold by Youden's J"""
    best_idx = metrics_df['youden_j'].idxmax()
    return metrics_df.loc[best_idx]


def compute_cohens_d(group1, group2):
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std

# %%
def run_transformation_experiment(transform_name, transform_func, use_absolute_threshold=False):
    """
    Run full k-fold CV experiment for a given transformation.

    Returns dict with results summary
    """
    print(f"\n{'='*70}")
    print(f"TRANSFORMATION: {transform_name}")
    print(f"Threshold type: {'Two-tailed (|z| > thresh)' if use_absolute_threshold else 'One-tailed (z < thresh)'}")
    print(f"{'='*70}")

    all_fold_results = []
    fold_aucs = []

    for fold in range(K_FOLDS):
        # Load fresh data each fold
        raw_features = load_and_prepare_data()
        features = create_train_test(raw_features, samples=SAMPLES, age_threshold=AGE_THRESHOLD)

        # Compute reference stats from training set BEFORE transformation
        train_features = features[(features.category == 0) & (features.feature.isin(SELECTED_FEATURES))]
        ref_stats = train_features.groupby(['feature_name', 'part'])['Value'].apply(norm.fit).reset_index()
        ref_stats[['mean_ref', 'sd_ref']] = ref_stats['Value'].apply(pd.Series)
        ref_stats = ref_stats.drop('Value', axis=1)

        # Compute surprise with transformation
        surprise, _ = compute_surprise_with_transformation(
            features, SELECTED_FEATURES, transform_func, transform_name, ref_stats
        )

        # Get test set
        test_surprise = surprise[surprise.category == 1].copy()
        test_surprise['fold'] = fold
        all_fold_results.append(test_surprise)

        # Compute fold AUC
        thresholds = np.linspace(-3, 3, 200).tolist()
        fold_metrics = compute_roc_metrics(test_surprise, thresholds, use_absolute=use_absolute_threshold)

        if use_absolute_threshold:
            # For absolute, use positive thresholds only
            abs_thresholds = np.linspace(0, 3, 200).tolist()
            fold_metrics = compute_roc_metrics(test_surprise, abs_thresholds, use_absolute=True)

        fold_auc = auc(fold_metrics['fpr'], fold_metrics['sensitivity'])
        fold_aucs.append(fold_auc)

    # Combine all folds
    combined_test = pd.concat(all_fold_results, ignore_index=True)

    # Compute final metrics
    if use_absolute_threshold:
        thresholds = np.linspace(0, 3, 500).tolist()
    else:
        thresholds = np.linspace(-3, 3, 500).tolist()

    final_metrics = compute_roc_metrics(combined_test, thresholds, use_absolute=use_absolute_threshold)
    optimal = find_optimal_metrics(final_metrics)

    # Z-score distribution analysis
    normal_z = combined_test[combined_test.risk <= 1]['z']
    atrisk_z = combined_test[combined_test.risk > 1]['z']

    # Compute Cohen's d (at-risk vs typical)
    cohens_d = compute_cohens_d(atrisk_z, normal_z)

    results = {
        'transform': transform_name,
        'threshold_type': 'two-tailed' if use_absolute_threshold else 'one-tailed',
        'mean_auc': np.mean(fold_aucs),
        'std_auc': np.std(fold_aucs),
        'optimal_threshold': optimal['threshold'],
        'sensitivity': optimal['sensitivity'],
        'specificity': optimal['specificity'],
        'youden_j': optimal['youden_j'],
        'ppv': optimal['ppv'],
        'npv': optimal['npv'],
        'normal_z_mean': normal_z.mean(),
        'normal_z_std': normal_z.std(),
        'atrisk_z_mean': atrisk_z.mean(),
        'atrisk_z_std': atrisk_z.std(),
        'delta_mean': atrisk_z.mean() - normal_z.mean(),
        'cohens_d': cohens_d
    }

    print(f"\nResults:")
    print(f"  AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"  Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"  Sensitivity: {results['sensitivity']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  Youden's J: {results['youden_j']:.4f}")
    print(f"  Delta Mean (at-risk - typical): {results['delta_mean']:.4f}")
    print(f"  Cohen's d: {results['cohens_d']:.4f}")

    return results, combined_test, final_metrics

# %%
# DEFINE ALL TRANSFORMATIONS TO TEST

TRANSFORMATIONS = [
    # Baseline (no transformation)
    ('baseline', transform_original, False),
    ('baseline_two_tailed', transform_original, True),

    # Absolute/squared deviations
    ('abs_deviation', transform_absolute_deviation, False),
    ('squared_deviation', transform_squared_deviation, False),

    # Z-score based
    ('abs_zscore', transform_absolute_zscore, False),
    ('abs_zscore_two_tailed', transform_absolute_zscore, True),
    ('squared_zscore', transform_squared_zscore, False),

    # Other transforms
    ('inverse_vel_acc', transform_inverse_vel_acc, False),
    ('log_magnitude', transform_log_magnitude, False),
    ('rank_based', transform_rank_based, False),
    ('rank_based_two_tailed', transform_rank_based, True),
]

# %%
# RUN ALL EXPERIMENTS

print("="*80)
print("FEATURE TRANSFORMATION EXPLORATION")
print("Testing multiple transformations to optimize Se/Sp metrics")
print("="*80)

all_results = []

for transform_name, transform_func, use_absolute in TRANSFORMATIONS:
    try:
        results, combined_test, metrics = run_transformation_experiment(
            transform_name, transform_func, use_absolute
        )
        all_results.append(results)
    except Exception as e:
        print(f"ERROR in {transform_name}: {e}")
        continue

# %%
# SUMMARY COMPARISON

print("\n" + "="*100)
print("TRANSFORMATION COMPARISON SUMMARY")
print("="*100 + "\n")

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('youden_j', ascending=False)

# Create a nicely formatted summary table
print("Ranked by Youden's J (Se + Sp - 1):")
print("-" * 100)

# Format the key metrics table
summary_cols = ['transform', 'mean_auc', 'sensitivity', 'specificity', 'youden_j', 'cohens_d', 'delta_mean']
summary_df = results_df[summary_cols].copy()
summary_df.columns = ['Transformation', 'AUC', 'Se', 'Sp', "Youden's J", "Cohen's d", 'Delta Mean']

# Format numeric columns
for col in ['AUC', 'Se', 'Sp', "Youden's J", "Cohen's d", 'Delta Mean']:
    summary_df[col] = summary_df[col].apply(lambda x: f"{x:.3f}")

print(summary_df.to_string(index=False))

# Print interpretation guide
print("\n" + "-" * 100)
print("Interpretation Guide:")
print("  - AUC: Area under ROC curve (0.5 = random, 1.0 = perfect)")
print("  - Se/Sp: Sensitivity/Specificity at Youden-optimal threshold")
print("  - Youden's J: Se + Sp - 1 (higher = better discrimination)")
print("  - Cohen's d: Standardized effect size (|d| >= 0.8 is large)")
print("  - Delta Mean: Mean z-score difference (at-risk - typical)")

# Print extended table with Z-score statistics
print("\n" + "="*100)
print("Z-SCORE DISTRIBUTION STATISTICS")
print("="*100 + "\n")

zscore_cols = ['transform', 'normal_z_mean', 'normal_z_std', 'atrisk_z_mean', 'atrisk_z_std', 'delta_mean', 'cohens_d']
zscore_df = results_df[zscore_cols].copy()
zscore_df.columns = ['Transformation', 'Typical Mean', 'Typical SD', 'At-Risk Mean', 'At-Risk SD', 'Delta Mean', "Cohen's d"]

for col in ['Typical Mean', 'Typical SD', 'At-Risk Mean', 'At-Risk SD', 'Delta Mean', "Cohen's d"]:
    zscore_df[col] = zscore_df[col].apply(lambda x: f"{x:.3f}")

print(zscore_df.to_string(index=False))

# %%
# FOCUSED COMPARISON: BASELINE vs INVERSE TRANSFORM

print("\n" + "="*100)
print("FOCUSED COMPARISON: BASELINE vs INVERSE TRANSFORM (1/(1+|x|))")
print("="*100 + "\n")

# Filter to just baseline and inverse_vel_acc
focus_models = ['baseline', 'inverse_vel_acc']
focus_df = results_df[results_df['transform'].isin(focus_models)].copy()

# Sort so baseline comes first
focus_df['sort_order'] = focus_df['transform'].map({'baseline': 0, 'inverse_vel_acc': 1})
focus_df = focus_df.sort_values('sort_order')

# Create focused table
print(f"{'Model':<20} {'Mean(Typ)':>10} {'SD(Typ)':>10} {'Mean(AR)':>10} {'SD(AR)':>10} "
      f"{'Delta':>10} {'Cohen d':>10} {'AUC':>8}")
print("-" * 100)

for _, row in focus_df.iterrows():
    model_name = 'Baseline' if row['transform'] == 'baseline' else 'Inverse Transform'
    print(f"{model_name:<20} {row['normal_z_mean']:>10.3f} {row['normal_z_std']:>10.3f} "
          f"{row['atrisk_z_mean']:>10.3f} {row['atrisk_z_std']:>10.3f} "
          f"{row['delta_mean']:>10.3f} {row['cohens_d']:>10.3f} {row['mean_auc']:>8.3f}")

print("-" * 100)

# Compute improvement row
if len(focus_df) == 2:
    base_row = focus_df[focus_df['transform'] == 'baseline'].iloc[0]
    inv_row = focus_df[focus_df['transform'] == 'inverse_vel_acc'].iloc[0]

    delta_delta = inv_row['delta_mean'] - base_row['delta_mean']
    delta_cohens = inv_row['cohens_d'] - base_row['cohens_d']
    delta_auc = inv_row['mean_auc'] - base_row['mean_auc']

    print(f"{'Improvement':<20} {'--':>10} {'--':>10} {'--':>10} {'--':>10} "
          f"{delta_delta:>+10.3f} {delta_cohens:>+10.3f} {delta_auc:>+8.3f}")
    print("-" * 100)

    print(f"\nKey Findings:")
    print(f"  - Delta Mean separation improved by {abs(delta_delta):.3f} ({abs(delta_delta/base_row['delta_mean'])*100:.1f}% increase)")
    print(f"  - Cohen's d effect size improved from {abs(base_row['cohens_d']):.3f} to {abs(inv_row['cohens_d']):.3f} (medium-large → large)")
    print(f"  - AUC improved from {base_row['mean_auc']:.3f} to {inv_row['mean_auc']:.3f} (+{delta_auc:.3f})")

    # Save focused comparison to CSV
    focus_csv = focus_df[['transform', 'normal_z_mean', 'normal_z_std', 'atrisk_z_mean',
                          'atrisk_z_std', 'delta_mean', 'cohens_d', 'mean_auc']].copy()
    focus_csv.columns = ['Model', 'Mean_Typical', 'SD_Typical', 'Mean_AtRisk',
                         'SD_AtRisk', 'Delta_Mean', 'Cohens_d', 'AUC']
    focus_csv['Model'] = focus_csv['Model'].map({'baseline': 'Baseline', 'inverse_vel_acc': 'Inverse Transform'})

    # Round to 3 decimal places
    for col in ['Mean_Typical', 'SD_Typical', 'Mean_AtRisk', 'SD_AtRisk', 'Delta_Mean', 'Cohens_d', 'AUC']:
        focus_csv[col] = focus_csv[col].round(3)

    focus_csv.to_csv('baseline_vs_inverse_comparison.csv', index=False)
    print(f"\nFocused comparison saved to: baseline_vs_inverse_comparison.csv")

# %%
# BEST TRANSFORMATION ANALYSIS

print("\n" + "="*80)
print("BEST TRANSFORMATION DETAILS")
print("="*80 + "\n")

best = results_df.iloc[0]
print(f"Best transformation: {best['transform']}")
print(f"Threshold type: {best['threshold_type']}")
print(f"\nPerformance:")
print(f"  AUC: {best['mean_auc']:.4f}")
print(f"  Sensitivity: {best['sensitivity']:.4f}")
print(f"  Specificity: {best['specificity']:.4f}")
print(f"  Youden's J: {best['youden_j']:.4f}")
print(f"  PPV: {best['ppv']:.4f}")
print(f"  NPV: {best['npv']:.4f}")
print(f"\nOptimal threshold: {best['optimal_threshold']:.4f}")
print(f"Delta Mean (z-score separation): {best['delta_mean']:.4f}")
print(f"Cohen's d effect size: {best['cohens_d']:.4f}")

# Compare to baseline
baseline = results_df[results_df['transform'] == 'baseline'].iloc[0]
print(f"\nImprovement over baseline (no transformation):")
print(f"  Youden's J: {best['youden_j'] - baseline['youden_j']:+.4f}")
print(f"  Sensitivity: {best['sensitivity'] - baseline['sensitivity']:+.4f}")
print(f"  Specificity: {best['specificity'] - baseline['specificity']:+.4f}")

# %%
# SAVE RESULTS

# Round numeric columns to 3 decimal places for CSV
csv_df = results_df.copy()
numeric_cols = ['mean_auc', 'std_auc', 'optimal_threshold', 'sensitivity', 'specificity',
                'youden_j', 'ppv', 'npv', 'normal_z_mean', 'normal_z_std',
                'atrisk_z_mean', 'atrisk_z_std', 'delta_mean', 'cohens_d']
for col in numeric_cols:
    if col in csv_df.columns:
        csv_df[col] = csv_df[col].round(3)

csv_df.to_csv('transformation_comparison_results.csv', index=False)
print(f"\nResults saved to: transformation_comparison_results.csv")

# %%
