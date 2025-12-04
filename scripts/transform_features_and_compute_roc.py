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
# # Transform Features and Compute ROC
#
# This script:
# 1. Loads the original features file
# 2. Applies inverse transformation to velocity/acceleration features
# 3. Saves the transformed features to a new pickle file
# 4. Computes ROC metrics using the pre-transformed features
#
# The inverse transformation: `x_transformed = 1 / (1 + |x|)`
# Applied to velocity and acceleration features only.

# %%
import pandas as pd
import os
import numpy as np
from scipy.stats import norm
import scipy as sc
import random
from sklearn.metrics import auc
from datetime import datetime

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

output_dir = '/BigData/EMCP_BACKUP/output/'

# Original features file
ORIGINAL_FEATURES_FILE = 'features_merged_20251121_091511.pkl'

# Transformed features file (will be created)
TRANSFORMED_FEATURES_FILE = None  # Will be auto-generated with timestamp

FEATURE_SIZE = 20
AGE_THRESHOLD = 10
SAMPLES = 20
K_FOLDS = 10

# Features to transform with inverse: 1/(1+|x|)
VELOCITY_FEATURES = [
    'Ankle_IQRvelx', 'Ankle_medianvelx', 'Ankle_medianvely', 'Ankle_IQRvely',
    'Elbow_IQR_vel_angle', 'Elbow_median_vel_angle', 'Hip_IQR_vel_angle'
]

ACCELERATION_FEATURES = [
    'Ankle_IQRaccy', 'Elbow_IQR_acc_angle', 'Shoulder_IQR_acc_angle', 'Wrist_IQRaccx'
]

TRANSFORM_FEATURES = VELOCITY_FEATURES + ACCELERATION_FEATURES

print("=" * 80)
print("FEATURE TRANSFORMATION AND ROC COMPUTATION")
print("=" * 80)
print(f"\nOriginal features file: {ORIGINAL_FEATURES_FILE}")
print(f"\nFeatures to transform ({len(TRANSFORM_FEATURES)} total):")
print(f"  Velocity features ({len(VELOCITY_FEATURES)}): {VELOCITY_FEATURES}")
print(f"  Acceleration features ({len(ACCELERATION_FEATURES)}): {ACCELERATION_FEATURES}")

# %%
# =============================================================================
# STEP 1: LOAD AND TRANSFORM FEATURES
# =============================================================================

print(f"\n{'='*80}")
print("STEP 1: LOADING AND TRANSFORMING FEATURES")
print(f"{'='*80}")

# Load original features
print(f"\nLoading: {ORIGINAL_FEATURES_FILE}")
f_original = pd.read_pickle(os.path.join(output_dir, ORIGINAL_FEATURES_FILE))
f_original['feature'] = f_original.part + '_' + f_original.feature_name

print(f"  Original shape: {f_original.shape}")
print(f"  Unique features: {f_original.feature.nunique()}")
print(f"  Unique infants: {f_original.infant.nunique()}")

# Show original statistics for features to be transformed
print(f"\nOriginal feature statistics (features to transform):")
transform_mask = f_original['feature'].isin(TRANSFORM_FEATURES)
original_stats = f_original[transform_mask].groupby('feature')['Value'].agg(['mean', 'std', 'min', 'max'])
print(original_stats.round(4).to_string())

# %%
# Apply inverse transformation: 1/(1+|x|)
print(f"\n{'='*80}")
print("APPLYING INVERSE TRANSFORMATION: 1/(1+|x|)")
print(f"{'='*80}")

f_transformed = f_original.copy()

# Apply transformation
mask = f_transformed['feature'].isin(TRANSFORM_FEATURES)
n_transformed = mask.sum()

print(f"\nTransforming {n_transformed} values across {len(TRANSFORM_FEATURES)} features...")

# Store original values for comparison
original_values = f_transformed.loc[mask, 'Value'].copy()

# Apply transformation
f_transformed.loc[mask, 'Value'] = 1 / (1 + np.abs(f_transformed.loc[mask, 'Value']))

# Show transformed statistics
print(f"\nTransformed feature statistics:")
transformed_stats = f_transformed[mask].groupby('feature')['Value'].agg(['mean', 'std', 'min', 'max'])
print(transformed_stats.round(4).to_string())

# Summary comparison
print(f"\nTransformation summary:")
print(f"  Original mean:     {original_values.mean():.4f}")
print(f"  Transformed mean:  {f_transformed.loc[mask, 'Value'].mean():.4f}")
print(f"  Original range:    [{original_values.min():.4f}, {original_values.max():.4f}]")
print(f"  Transformed range: [{f_transformed.loc[mask, 'Value'].min():.4f}, {f_transformed.loc[mask, 'Value'].max():.4f}]")

# %%
# =============================================================================
# STEP 2: SAVE TRANSFORMED FEATURES
# =============================================================================

print(f"\n{'='*80}")
print("STEP 2: SAVING TRANSFORMED FEATURES")
print(f"{'='*80}")

# Generate filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
TRANSFORMED_FEATURES_FILE = f'features_merged_inverse_vel_acc_{timestamp}.pkl'

# Save to pickle
output_path = os.path.join(output_dir, TRANSFORMED_FEATURES_FILE)
f_transformed.to_pickle(output_path)

print(f"\nSaved transformed features to:")
print(f"  {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# Verify the saved file
f_verify = pd.read_pickle(output_path)
print(f"\nVerification:")
print(f"  Shape matches: {f_verify.shape == f_transformed.shape}")
print(f"  Values match: {(f_verify['Value'] == f_transformed['Value']).all()}")

# %%
# =============================================================================
# STEP 3: DEFINE FEATURE SETS (same as compute_roc_after_rfe.py)
# =============================================================================

# 20-Feature NEW: Best for screening
selected_features_20_new = [
    "Ankle_IQRvelx", "Ankle_IQRx", "Ankle_IQRaccy", "Ankle_medianvelx",
    "Ankle_lrCorr_x", "Ankle_meanent", "Ankle_medianvely", "Ankle_IQRvely",
    "Elbow_IQR_acc_angle", "Elbow_IQR_vel_angle", "Elbow_median_vel_angle",
    "Hip_stdev_angle", "Hip_entropy_angle", "Hip_IQR_vel_angle",
    "Shoulder_mean_angle", "Shoulder_IQR_acc_angle",
    "Wrist_mediany", "Wrist_IQRx", "Wrist_IQRaccx", "age_in_weeks"
]

# 15-Feature NEW
selected_features_15_new = [
    "Ankle_IQRvelx", "Ankle_IQRx", "Ankle_IQRaccy", "Ankle_medianvelx",
    "Ankle_lrCorr_x", "Ankle_meanent", "Ankle_medianvely",
    "Elbow_IQR_acc_angle", "Elbow_IQR_vel_angle",
    "Hip_stdev_angle", "Hip_entropy_angle",
    "Wrist_mediany", "Wrist_IQRx", "Wrist_IQRaccx", "age_in_weeks"
]

# 19-Feature NEW
selected_features_19_new = [
    "Ankle_IQRvelx", "Ankle_IQRx", "Ankle_IQRaccy", "Ankle_medianvelx",
    "Ankle_lrCorr_x", "Ankle_meanent", "Ankle_medianvely", "Ankle_IQRvely",
    "Elbow_IQR_acc_angle", "Elbow_IQR_vel_angle", "Elbow_median_vel_angle",
    "Hip_stdev_angle", "Hip_entropy_angle", "Hip_IQR_vel_angle",
    "Shoulder_mean_angle",
    "Wrist_mediany", "Wrist_IQRx", "Wrist_IQRaccx", "age_in_weeks"
]

feature_sets = {
    '20-Feature (NEW)': selected_features_20_new,
    '15-Feature (NEW)': selected_features_15_new,
    '19-Feature (NEW)': selected_features_19_new,
}

# %%
# =============================================================================
# STEP 4: HELPER FUNCTIONS
# =============================================================================

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


def run_kfold_cv(features_file, feature_list, feature_set_name, k_folds=10):
    """Run k-fold cross-validation using pre-transformed features"""
    np.random.seed(42)
    random.seed(42)

    fold_results = []

    print(f"\n{'='*80}")
    print(f"Processing: {feature_set_name} ({len(feature_list)} features)")
    print(f"Using pre-transformed features (no runtime transformation)")
    print(f"K-FOLD CROSS-VALIDATION (K={k_folds})")
    print(f"{'='*80}")

    for fold_idx in range(k_folds):
        print(f"  [FOLD {fold_idx+1}/{k_folds}]", end=" ")

        # Load pre-transformed features
        f = pd.read_pickle(os.path.join(output_dir, features_file))
        f['feature'] = f.part + '_' + f.feature_name

        # Add age_in_weeks as a feature
        age_rows = f.drop_duplicates(subset='infant').copy()
        age_rows['feature'] = 'age_in_weeks'
        age_rows['Value'] = age_rows['age_in_weeks']
        age_rows['part'] = 'age'
        age_rows['feature_name'] = 'in_weeks'
        f = pd.concat([f, age_rows], ignore_index=True)

        # Create train/test split
        features = create_train_test(f, samples=SAMPLES, age_threshold=AGE_THRESHOLD)
        features['feature'] = features.part + '_' + features.feature_name

        # NO TRANSFORMATION HERE - features are already transformed

        # Compute reference statistics from TRAINING set (category=0)
        ref_stats = features[features.category == 0].groupby(['feature_name', 'part'])['Value']\
            .apply(norm.fit).reset_index()
        ref_stats[['mean_ref', 'sd_ref']] = ref_stats['Value'].apply(pd.Series)
        ref_stats['var_ref'] = ref_stats['sd_ref']**2
        ref_stats = ref_stats.drop('Value', axis=1)

        # Merge and compute surprise
        features = pd.merge(features, ref_stats, on=['feature_name', 'part'], how='inner')
        features['minus_log_pfeature'] = -1 * (
            0.5 * np.log(2 * np.pi * features['var_ref']) +
            ((features['Value'] - features['mean_ref'])**2) / (2 * features['var_ref'])
        )
        features['feature'] = features.part + '_' + features.feature_name

        # Filter to selected features
        features = features[features.feature.isin(feature_list)]

        # Compute surprise
        fold_surprise = features.groupby(['infant', 'age_in_weeks', 'risk', 'category'])[
            'minus_log_pfeature'].sum().reset_index()

        # Normalize z-scores
        train_mean = fold_surprise.loc[fold_surprise.category == 0, 'minus_log_pfeature'].mean()
        train_std = fold_surprise.loc[fold_surprise.category == 0, 'minus_log_pfeature'].std()
        fold_surprise['z'] = (fold_surprise['minus_log_pfeature'] - train_mean) / train_std
        fold_surprise['p'] = (sc.stats.norm.sf(np.abs(fold_surprise['z'])) * 2).round(3)

        # Remove problematic subjects
        fold_surprise = fold_surprise[~fold_surprise.infant.str.contains('clin_100_')]

        fold_results.append(fold_surprise)

        n_train = fold_surprise[fold_surprise.category == 0]['infant'].nunique()
        n_test = fold_surprise[fold_surprise.category == 1]['infant'].nunique()
        print(f"Train: {n_train} | Test: {n_test}")

    # Combine all test sets
    all_test_data = []
    for fold_idx, fold_df in enumerate(fold_results):
        test_data = fold_df[fold_df.category == 1].copy()
        test_data['fold'] = fold_idx
        all_test_data.append(test_data)

    combined_test = pd.concat(all_test_data, ignore_index=True)

    return combined_test, fold_results


def compute_roc_metrics(combined_test, thresholds):
    """Compute ROC metrics for all thresholds"""
    metrics = []

    for thresh in thresholds:
        TP = combined_test[(combined_test.risk > 1) & (combined_test.z < thresh)].shape[0]
        FP = combined_test[(combined_test.risk <= 1) & (combined_test.z < thresh)].shape[0]
        FN = combined_test[(combined_test.risk > 1) & (combined_test.z >= thresh)].shape[0]
        TN = combined_test[(combined_test.risk <= 1) & (combined_test.z >= thresh)].shape[0]

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        youden = sens + spec - 1
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        metrics.append({
            'threshold': thresh,
            'sensitivity': sens,
            'specificity': spec,
            'ppv': ppv,
            'npv': npv,
            'youden_j': youden,
            'accuracy': acc,
            'tp': TP, 'fp': FP, 'fn': FN, 'tn': TN
        })

    return pd.DataFrame(metrics)

# %%
# =============================================================================
# STEP 5: RUN K-FOLD CV WITH PRE-TRANSFORMED FEATURES
# =============================================================================

print(f"\n{'='*80}")
print("STEP 5: RUNNING K-FOLD CV WITH PRE-TRANSFORMED FEATURES")
print(f"{'='*80}")

all_results = {}

for feature_set_name, feature_list in feature_sets.items():
    combined_test, fold_results = run_kfold_cv(
        TRANSFORMED_FEATURES_FILE, feature_list, feature_set_name, K_FOLDS
    )
    all_results[feature_set_name] = {
        'combined_test': combined_test,
        'fold_results': fold_results
    }

# %%
# =============================================================================
# STEP 6: COMPUTE AND DISPLAY METRICS
# =============================================================================

print(f"\n{'='*80}")
print("STEP 6: ROC METRICS WITH PRE-TRANSFORMED FEATURES")
print(f"{'='*80}")

thresholds = np.linspace(-3.0, 3.0, 500).tolist()

final_metrics = {}

for feature_set_name in feature_sets.keys():
    combined_test = all_results[feature_set_name]['combined_test']

    # Compute ROC metrics
    roc_df = compute_roc_metrics(combined_test, thresholds)

    # Find optimal by Youden's J
    best_idx = roc_df['youden_j'].idxmax()
    optimal = roc_df.loc[best_idx]

    # Compute AUC
    fold_aucs = []
    for fold_df in all_results[feature_set_name]['fold_results']:
        fold_test = fold_df[fold_df.category == 1]
        if len(fold_test) > 0:
            fold_roc = compute_roc_metrics(fold_test, thresholds)
            fold_auc = auc(1 - fold_roc['specificity'], fold_roc['sensitivity'])
            fold_aucs.append(fold_auc)
    mean_auc = np.mean(fold_aucs)

    final_metrics[feature_set_name] = {
        'auc': mean_auc,
        'sensitivity': optimal['sensitivity'],
        'specificity': optimal['specificity'],
        'youden_j': optimal['youden_j'],
        'ppv': optimal['ppv'],
        'npv': optimal['npv'],
        'accuracy': optimal['accuracy'],
        'optimal_threshold': optimal['threshold']
    }

    print(f"\n{feature_set_name}")
    print("-" * 60)
    print(f"  AUC:          {mean_auc:.4f}")
    print(f"  Sensitivity:  {optimal['sensitivity']:.4f} ({optimal['sensitivity']*100:.1f}%)")
    print(f"  Specificity:  {optimal['specificity']:.4f} ({optimal['specificity']*100:.1f}%)")
    print(f"  Youden's J:   {optimal['youden_j']:.4f}")
    print(f"  PPV:          {optimal['ppv']:.4f}")
    print(f"  NPV:          {optimal['npv']:.4f}")
    print(f"  Threshold:    {optimal['threshold']:.4f}")

# %%
# =============================================================================
# STEP 7: SUMMARY TABLE
# =============================================================================

print(f"\n{'='*80}")
print("SUMMARY: PRE-TRANSFORMED FEATURES ROC METRICS")
print(f"{'='*80}\n")

summary_df = pd.DataFrame(final_metrics).T
summary_df = summary_df.round(4)

print(summary_df[['auc', 'sensitivity', 'specificity', 'youden_j', 'ppv', 'npv', 'optimal_threshold']].to_string())

# Save to CSV
summary_df.to_csv('pretransformed_features_roc_metrics.csv')
print(f"\nResults saved to: pretransformed_features_roc_metrics.csv")

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")

print(f"\n1. Original features file:")
print(f"   {ORIGINAL_FEATURES_FILE}")

print(f"\n2. Transformed features file (NEW):")
print(f"   {TRANSFORMED_FEATURES_FILE}")

print(f"\n3. Transformation applied:")
print(f"   Formula: x_transformed = 1 / (1 + |x|)")
print(f"   Applied to {len(TRANSFORM_FEATURES)} velocity/acceleration features")

print(f"\n4. Best model: 20-Feature (NEW)")
best = final_metrics['20-Feature (NEW)']
print(f"   AUC:         {best['auc']:.4f}")
print(f"   Sensitivity: {best['sensitivity']*100:.1f}%")
print(f"   Specificity: {best['specificity']*100:.1f}%")
print(f"   Youden's J:  {best['youden_j']:.4f}")

print(f"\n5. To use the pre-transformed features in other scripts:")
print(f"   FEATURES_FILE = '{TRANSFORMED_FEATURES_FILE}'")
print(f"   # No additional transformation needed")

# %%
