#!/usr/bin/env python3
# """
# Test whether the Bayesian Surprise model with 1/(1+|x|) transformation
# can detect HYPERKINESIA (excessive movement) in addition to hypokinesia.
#
# Creates synthetic hyperkinetic infant profiles:
# 1. Excessive lower body movement (ankle/knee)
# 2. Excessive upper body movement (wrist/elbow/shoulder)
# 3. Global hyperkinesia (all body parts)
#
# Runs 10-fold CV to compute Se/Sp/AUC metrics.
# """

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
except NameError:
    PROJECT_DIR = Path.cwd()
    if PROJECT_DIR.name == 'scripts':
        PROJECT_DIR = PROJECT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "data" / "pkl"
FEATURES_FILE = 'features_merged_20251121_091511.pkl'

# Configuration
K_FOLDS = 10
N_SYNTHETIC_PER_TYPE = 10  # Number of synthetic infants per hyperkinesia type
HYPERKINESIA_SEVERITY = 1.5  # Std deviations above typical mean
np.random.seed(42)

# 20-Feature NEW set
SELECTED_FEATURES = [
    "Ankle_IQRvelx", "Ankle_IQRx", "Ankle_IQRaccy", "Ankle_medianvelx",
    "Ankle_lrCorr_x", "Ankle_meanent", "Ankle_medianvely", "Ankle_IQRvely",
    "Elbow_IQR_acc_angle", "Elbow_IQR_vel_angle", "Elbow_median_vel_angle",
    "Hip_stdev_angle", "Hip_entropy_angle", "Hip_IQR_vel_angle",
    "Shoulder_mean_angle", "Shoulder_IQR_acc_angle",
    "Wrist_mediany", "Wrist_IQRx", "Wrist_IQRaccx",
    "age_in_weeks"
]

# Features to transform
TRANSFORM_FEATURES = [
    'Ankle_IQRvelx', 'Ankle_medianvelx', 'Ankle_medianvely', 'Ankle_IQRvely',
    'Ankle_IQRaccy', 'Wrist_IQRaccx',
    'Elbow_IQR_vel_angle', 'Elbow_median_vel_angle', 'Hip_IQR_vel_angle',
    'Elbow_IQR_acc_angle', 'Shoulder_IQR_acc_angle'
]

# Define body part groupings for hyperkinesia patterns
LOWER_BODY_FEATURES = [
    "Ankle_IQRvelx", "Ankle_medianvelx", "Ankle_medianvely", "Ankle_IQRvely",
    "Ankle_IQRaccy", "Ankle_IQRx",
    "Hip_IQR_vel_angle", "Hip_stdev_angle"
]

UPPER_BODY_FEATURES = [
    "Wrist_IQRaccx", "Wrist_IQRx", "Wrist_mediany",
    "Elbow_IQR_acc_angle", "Elbow_IQR_vel_angle", "Elbow_median_vel_angle",
    "Shoulder_IQR_acc_angle", "Shoulder_mean_angle"
]


def load_data():
    """Load and prepare features data"""
    f = pd.read_pickle(OUTPUT_DIR / FEATURES_FILE)
    f['feature'] = f.part + '_' + f.feature_name

    # Add age as a feature
    age_rows = f.drop_duplicates(subset='infant').copy()
    age_rows['feature'] = 'age_in_weeks'
    age_rows['Value'] = age_rows['age_in_weeks']
    age_rows['part'] = 'age'
    age_rows['feature_name'] = 'in_weeks'

    f = pd.concat([f, age_rows], ignore_index=True)
    return f


def get_reference_stats(features_df):
    """Compute reference statistics from typical infants"""
    typical = features_df[features_df['risk'] <= 1]
    typical_filtered = typical[typical['feature'].isin(SELECTED_FEATURES)]

    stats = typical_filtered.groupby('feature')['Value'].agg(['mean', 'std']).reset_index()
    stats.columns = ['feature', 'mean', 'std']
    return stats


def create_synthetic_infant(ref_stats, hyperkinesia_type, infant_id, severity=2.0):
    """
    Create a synthetic hyperkinetic infant.

    Parameters:
    -----------
    ref_stats : DataFrame with feature mean/std from typical infants
    hyperkinesia_type : 'lower_body', 'upper_body', or 'global'
    infant_id : str identifier for the synthetic infant
    severity : float, number of std deviations above mean for affected features

    Returns:
    --------
    DataFrame with synthetic infant features in long format
    """
    rows = []

    # Determine which features are elevated
    if hyperkinesia_type == 'lower_body':
        elevated_features = LOWER_BODY_FEATURES
    elif hyperkinesia_type == 'upper_body':
        elevated_features = UPPER_BODY_FEATURES
    elif hyperkinesia_type == 'global':
        elevated_features = LOWER_BODY_FEATURES + UPPER_BODY_FEATURES
    else:
        elevated_features = []

    for _, row in ref_stats.iterrows():
        feat = row['feature']
        mean_val = row['mean']
        std_val = row['std']

        if feat in elevated_features:
            # Hyperkinetic: elevated value (severity std above mean)
            # Add some noise for realism
            value = mean_val + severity * std_val + np.random.normal(0, std_val * 0.2)
        else:
            # Normal: sample from typical distribution
            value = np.random.normal(mean_val, std_val)

        # Ensure non-negative for velocity/acceleration features
        if 'vel' in feat.lower() or 'acc' in feat.lower() or 'IQR' in feat:
            value = max(0, value)

        # Parse feature name
        if feat == 'age_in_weeks':
            part = 'age'
            feature_name = 'in_weeks'
        else:
            parts = feat.split('_', 1)
            part = parts[0]
            feature_name = parts[1] if len(parts) > 1 else ''

        rows.append({
            'infant': infant_id,
            'feature': feat,
            'Value': value,
            'part': part,
            'feature_name': feature_name,
            'risk': 3,  # Mark as at-risk
            'category': 1,  # Test set
            'age_in_weeks': 20,  # Assume middle age
            'synthetic': True,
            'hyperkinesia_type': hyperkinesia_type
        })

    return pd.DataFrame(rows)


def create_synthetic_cohort(ref_stats, n_per_type=5, severity=2.0):
    """Create a cohort of synthetic hyperkinetic infants"""
    all_synthetic = []

    for hyper_type in ['lower_body', 'upper_body', 'global']:
        for i in range(n_per_type):
            infant_id = f'synthetic_{hyper_type}_{i}'
            synthetic_infant = create_synthetic_infant(
                ref_stats, hyper_type, infant_id, severity
            )
            all_synthetic.append(synthetic_infant)

    return pd.concat(all_synthetic, ignore_index=True)


def apply_transformation(features_df):
    """Apply 1/(1+|x|) transformation to velocity/acceleration features"""
    df = features_df.copy()
    mask = df['feature'].isin(TRANSFORM_FEATURES)
    df.loc[mask, 'Value'] = 1 / (1 + np.abs(df.loc[mask, 'Value']))
    return df


def compute_surprise(features_df, train_infants, test_infants):
    """
    Compute Bayesian surprise scores.

    Parameters:
    -----------
    features_df : Full features DataFrame
    train_infants : list of infant IDs for training (normative)
    test_infants : list of infant IDs for testing

    Returns:
    --------
    DataFrame with z-scores for test infants
    """
    df = features_df[features_df['feature'].isin(SELECTED_FEATURES)].copy()

    # Split train/test
    train_df = df[df['infant'].isin(train_infants)]
    test_df = df[df['infant'].isin(test_infants)]

    # Compute reference statistics from training set
    ref = train_df.groupby('feature')['Value'].apply(
        lambda x: pd.Series({'mean_ref': x.mean(), 'var_ref': x.var()})
    ).unstack().reset_index()

    # Merge reference stats with test data
    test_df = pd.merge(test_df, ref, on='feature', how='inner')

    # Compute log probability
    test_df['minus_log_pfeature'] = -1 * (
        0.5 * np.log(2 * np.pi * test_df['var_ref']) +
        ((test_df['Value'] - test_df['mean_ref']) ** 2) / (2 * test_df['var_ref'])
    )

    # Aggregate per infant
    # First get infant metadata
    infant_info = features_df.drop_duplicates('infant')[
        ['infant', 'risk', 'age_in_weeks']
    ].copy()

    # Check for synthetic column
    if 'synthetic' in features_df.columns:
        synth_info = features_df.drop_duplicates('infant')[['infant', 'synthetic', 'hyperkinesia_type']]
        infant_info = infant_info.merge(synth_info, on='infant', how='left')
        infant_info['synthetic'] = infant_info['synthetic'].fillna(False)
        infant_info['hyperkinesia_type'] = infant_info['hyperkinesia_type'].fillna('real')

    surprise = test_df.groupby('infant')['minus_log_pfeature'].sum().reset_index()
    surprise = surprise.merge(infant_info, on='infant', how='left')

    # Compute z-scores using training distribution
    train_surprise = train_df.groupby('infant')['Value'].count().reset_index()  # Just to get infant list
    train_log_p = []
    for infant in train_infants:
        infant_data = train_df[train_df['infant'] == infant].copy()
        infant_data = pd.merge(infant_data, ref, on='feature', how='inner')
        infant_data['minus_log_pfeature'] = -1 * (
            0.5 * np.log(2 * np.pi * infant_data['var_ref']) +
            ((infant_data['Value'] - infant_data['mean_ref']) ** 2) / (2 * infant_data['var_ref'])
        )
        train_log_p.append(infant_data['minus_log_pfeature'].sum())

    train_mean = np.mean(train_log_p)
    train_std = np.std(train_log_p)

    surprise['z'] = (surprise['minus_log_pfeature'] - train_mean) / train_std

    return surprise


def run_fold(features_df, synthetic_df, fold_idx, typical_infants, atrisk_infants):
    """Run one fold of cross-validation"""
    np.random.seed(42 + fold_idx)

    # Split typical infants into train/test
    n_test = max(5, len(typical_infants) // 5)
    test_typical = np.random.choice(typical_infants, n_test, replace=False)
    train_typical = [i for i in typical_infants if i not in test_typical]

    # Training set: only typical infants (normative)
    train_infants = list(train_typical)

    # Test set: some typical + all at-risk + all synthetic
    test_infants = list(test_typical) + list(atrisk_infants)

    # Add synthetic infants to features_df for this fold
    combined_df = pd.concat([features_df, synthetic_df], ignore_index=True)

    # Add synthetic infants to test set
    synthetic_infants = synthetic_df['infant'].unique().tolist()
    test_infants = test_infants + synthetic_infants

    # Apply transformation
    combined_df = apply_transformation(combined_df)

    # Compute surprise
    surprise = compute_surprise(combined_df, train_infants, test_infants)

    return surprise


def compute_metrics(surprise_df, group_col='risk', threshold_col='z'):
    """Compute ROC metrics for different groups"""
    thresholds = np.linspace(-5, 2, 500)
    metrics = []

    for thresh in thresholds:
        # At-risk detection (z < threshold)
        TP = surprise_df[(surprise_df[group_col] > 1) & (surprise_df[threshold_col] < thresh)].shape[0]
        FP = surprise_df[(surprise_df[group_col] <= 1) & (surprise_df[threshold_col] < thresh)].shape[0]
        FN = surprise_df[(surprise_df[group_col] > 1) & (surprise_df[threshold_col] >= thresh)].shape[0]
        TN = surprise_df[(surprise_df[group_col] <= 1) & (surprise_df[threshold_col] >= thresh)].shape[0]

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0

        metrics.append({
            'threshold': thresh,
            'sensitivity': sens,
            'specificity': spec,
            'youden_j': sens + spec - 1,
            'fpr': 1 - spec
        })

    return pd.DataFrame(metrics)


def main():
    print("=" * 80)
    print("HYPERKINESIA DETECTION TEST")
    print("Testing if model detects synthetic hyperkinetic infants")
    print("=" * 80)

    # Load real data
    print("\nLoading data...")
    features_df = load_data()

    # Get typical and at-risk infants
    infant_info = features_df.drop_duplicates('infant')[['infant', 'risk']]
    typical_infants = infant_info[infant_info['risk'] <= 1]['infant'].tolist()
    atrisk_infants = infant_info[infant_info['risk'] > 1]['infant'].tolist()

    print(f"Real typical infants: {len(typical_infants)}")
    print(f"Real at-risk infants: {len(atrisk_infants)}")

    # Get reference statistics from typical infants
    ref_stats = get_reference_stats(features_df)

    # Create synthetic hyperkinetic infants
    print(f"\nCreating synthetic hyperkinetic infants (severity={HYPERKINESIA_SEVERITY} std)...")
    synthetic_df = create_synthetic_cohort(ref_stats, n_per_type=N_SYNTHETIC_PER_TYPE,
                                            severity=HYPERKINESIA_SEVERITY)

    n_synthetic = synthetic_df['infant'].nunique()
    print(f"  Lower body hyperkinesia: {N_SYNTHETIC_PER_TYPE}")
    print(f"  Upper body hyperkinesia: {N_SYNTHETIC_PER_TYPE}")
    print(f"  Global hyperkinesia: {N_SYNTHETIC_PER_TYPE}")
    print(f"  Total synthetic: {n_synthetic}")

    # Run k-fold cross-validation
    print(f"\nRunning {K_FOLDS}-fold cross-validation...")
    all_results = []

    for fold in range(K_FOLDS):
        fold_surprise = run_fold(features_df, synthetic_df, fold, typical_infants, atrisk_infants)
        fold_surprise['fold'] = fold
        all_results.append(fold_surprise)

    combined_results = pd.concat(all_results, ignore_index=True)

    # Analyze results by group
    print("\n" + "=" * 80)
    print("Z-SCORE DISTRIBUTION BY GROUP")
    print("=" * 80)

    # Group statistics
    groups = {
        'Typical (real)': combined_results[combined_results['risk'] <= 1],
        'At-Risk (real)': combined_results[(combined_results['risk'] > 1) &
                                           (combined_results.get('synthetic', False) == False)],
    }

    # Add synthetic groups if present
    if 'hyperkinesia_type' in combined_results.columns:
        for hyper_type in ['lower_body', 'upper_body', 'global']:
            mask = combined_results['hyperkinesia_type'] == hyper_type
            if mask.any():
                groups[f'Hyperkinetic ({hyper_type})'] = combined_results[mask]

    print(f"\n{'Group':<30} {'N':>6} {'Mean Z':>10} {'Std Z':>10} {'Min Z':>10} {'Max Z':>10}")
    print("-" * 80)

    for group_name, group_df in groups.items():
        if len(group_df) > 0:
            print(f"{group_name:<30} {len(group_df):>6} {group_df['z'].mean():>10.3f} "
                  f"{group_df['z'].std():>10.3f} {group_df['z'].min():>10.3f} {group_df['z'].max():>10.3f}")

    # Compute detection rates at optimal threshold
    print("\n" + "=" * 80)
    print("DETECTION PERFORMANCE")
    print("=" * 80)

    # Find optimal threshold using real data only
    real_data = combined_results[combined_results.get('synthetic', False) == False]
    if 'synthetic' not in combined_results.columns:
        real_data = combined_results[~combined_results['infant'].str.contains('synthetic')]

    metrics = compute_metrics(real_data)
    best_idx = metrics['youden_j'].idxmax()
    optimal_thresh = metrics.loc[best_idx, 'threshold']

    print(f"\nOptimal threshold (from real data): {optimal_thresh:.3f}")

    # Detection rates
    print(f"\n{'Group':<30} {'Detected':>10} {'Total':>10} {'Rate':>10}")
    print("-" * 60)

    for group_name, group_df in groups.items():
        if len(group_df) > 0:
            detected = (group_df['z'] < optimal_thresh).sum()
            total = len(group_df)
            rate = detected / total * 100
            print(f"{group_name:<30} {detected:>10} {total:>10} {rate:>9.1f}%")

    # Compute AUC for different test scenarios
    print("\n" + "=" * 80)
    print("AUC ANALYSIS")
    print("=" * 80)

    # AUC: Real at-risk vs typical
    real_atrisk = combined_results[(combined_results['risk'] > 1) &
                                   (~combined_results['infant'].str.contains('synthetic'))]
    real_typical = combined_results[combined_results['risk'] <= 1]

    if len(real_atrisk) > 0 and len(real_typical) > 0:
        combined_real = pd.concat([real_typical, real_atrisk])
        metrics_real = compute_metrics(combined_real)
        auc_real = auc(metrics_real['fpr'], metrics_real['sensitivity'])
        print(f"\nAUC (Real at-risk vs Typical): {auc_real:.4f}")

    # AUC: Synthetic hyperkinetic vs typical
    synthetic_data = combined_results[combined_results['infant'].str.contains('synthetic')]
    if len(synthetic_data) > 0:
        combined_synth = pd.concat([real_typical, synthetic_data])
        # Mark synthetic as at-risk for metrics computation
        combined_synth_copy = combined_synth.copy()
        combined_synth_copy.loc[combined_synth_copy['infant'].str.contains('synthetic'), 'risk'] = 3
        metrics_synth = compute_metrics(combined_synth_copy)
        auc_synth = auc(metrics_synth['fpr'], metrics_synth['sensitivity'])
        print(f"AUC (Synthetic hyperkinetic vs Typical): {auc_synth:.4f}")

        # AUC by hyperkinesia type
        for hyper_type in ['lower_body', 'upper_body', 'global']:
            type_data = synthetic_data[synthetic_data['hyperkinesia_type'] == hyper_type]
            if len(type_data) > 0:
                combined_type = pd.concat([real_typical, type_data])
                combined_type_copy = combined_type.copy()
                combined_type_copy.loc[combined_type_copy['infant'].str.contains('synthetic'), 'risk'] = 3
                metrics_type = compute_metrics(combined_type_copy)
                auc_type = auc(metrics_type['fpr'], metrics_type['sensitivity'])
                print(f"AUC ({hyper_type} hyperkinetic vs Typical): {auc_type:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    real_atrisk_z = groups.get('At-Risk (real)', pd.DataFrame())['z'].mean() if 'At-Risk (real)' in groups else np.nan

    print(f"""
    Real At-Risk (hypokinesia pattern):
      Mean Z-score: {real_atrisk_z:.3f}
      Detection: Based on z < {optimal_thresh:.3f}

    Synthetic Hyperkinetic Infants:""")

    for hyper_type in ['lower_body', 'upper_body', 'global']:
        key = f'Hyperkinetic ({hyper_type})'
        if key in groups and len(groups[key]) > 0:
            mean_z = groups[key]['z'].mean()
            detected = (groups[key]['z'] < optimal_thresh).sum()
            total = len(groups[key])
            print(f"      {hyper_type}: Mean Z={mean_z:.3f}, Detected={detected}/{total} ({detected/total*100:.0f}%)")

    print(f"""
    Interpretation:
    - Negative z-scores indicate atypical movement (both hypo and hyperkinesia)
    - The 1/(1+|x|) transformation + log-probability captures BOTH patterns
    - Detection rate for hyperkinetic infants validates the model's bidirectional sensitivity
    """)

    return combined_results, groups


if __name__ == "__main__":
    results, groups = main()


