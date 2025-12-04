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

# %%
import pandas as pd
import os

from scipy.stats import norm
import numpy as np
import scipy as sc
import random
from sklearn.metrics import auc
def create_train_test(features, samples, age_threshold):
    features['age_bracket'] = (features.age_in_weeks>age_threshold) + 0
    
    # CLIN subjects with risk = 1 (low risk) will have category 1
    # CLIN subjects risk can be 1, 2, or 3
    # MOVE CLIN subjects with risk=1 (low risk) to NORMATIVE SET for now
    features.loc[(features.category==1) & (features.risk <= 1), 'category'] = 0
    # setting category=0 makes it normative


    # add to the TEST set which consists of only CLIN presently
    
    # split NORMATIVE set to training and the rest TEST
    norm_infants = features[features.category==0].infant.unique()
    # pick some samples from this set and make it TEST set
    
    test_infant_names = np.random.choice(norm_infants, samples, replace=False)
    # set category 1 so it can move to TEST set
    features.loc[features['infant'].isin(test_infant_names), 'category'] = 1
    # all category=0 subjects were LOW RISK
    features.loc[features['infant'].isin(test_infant_names), 'risk'] = 1

    return features

# %%

output_dir = '/BigData/emcp_test_30s/output_files_yt_clin_emcp_merged'
output_dir = '/BigData/EMCP_BACKUP/mp4test/output_files'
output_dir = '/BigData/EMCP_BACKUP/output/'
FEATURE_SIZE = 20 # number of features to use through RFE (recursive feature elimination)
AGE_THRESHOLD = 10 #age threshold in weeks
FEATURES_FILE = 'features_merged_20251121_091511.pkl'
# two age groups, below is age_bracket 0, above is age_bracket 1

# if age is a feature, we don't need age brackets to divide the dataset by
SAMPLES = 20 # number of samples from normative set to move to TEST

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
# Random seed for reproducible k-fold cross-validation results
RANDOM_SEED = 20250313  # Matches RAND_STATE in constants.py

# ============================================================================
# INTERACTIVE MODEL SELECTION
# ============================================================================
# Choose which model to use for interactive ROC curve visualization
# Options: '15-Feature (NEW)', '19-Feature (NEW)', '20-Feature (NEW)',
#          '15-Feature (LEGACY)', '18-Feature (LEGACY)', '20-Feature (LEGACY)'
INTERACTIVE_MODEL = '20-Feature'  # Change this to explore different models

# ============================================================================
# FEATURE TRANSFORMATION CONFIGURATION
# ============================================================================
# Apply inverse transformation to velocity/acceleration features
# This improves specificity by ~16.5% while maintaining sensitivity
# See: docs/FEATURE_TRANSFORMATION_METHODOLOGY.md for details

APPLY_INVERSE_TRANSFORM = True  # Set to False to use original (baseline) features

# Run comparison mode: runs BOTH transformed and non-transformed, then compares
RUN_COMPARISON_MODE = True  # Set to True to compare WITH vs WITHOUT transform

# Features to transform with 1/(1+|x|) - velocity and acceleration features
VELOCITY_FEATURES = [
    'Ankle_IQRvelx', 'Ankle_medianvelx', 'Ankle_medianvely', 'Ankle_IQRvely',
    'Elbow_IQR_vel_angle', 'Elbow_median_vel_angle', 'Hip_IQR_vel_angle'
]
ACCELERATION_FEATURES = [
    'Ankle_IQRaccy', 'Elbow_IQR_acc_angle', 'Shoulder_IQR_acc_angle', 'Wrist_IQRaccx'
]
TRANSFORM_FEATURES = VELOCITY_FEATURES + ACCELERATION_FEATURES


def apply_inverse_transform(features_df, features_to_transform):
    """
    Apply inverse transformation: 1/(1+|x|) to specified features.

    This transformation:
    - Compresses high values (jerky movements) toward 0
    - Expands differentiation at lower values (hypokinesia)
    - Creates better separation between at-risk and normal infants

    Parameters:
    -----------
    features_df : DataFrame - Features in long format with 'feature' and 'Value' columns
    features_to_transform : list - Feature names to transform

    Returns:
    --------
    DataFrame with transformed values
    """
    df = features_df.copy()
    mask = df['feature'].isin(features_to_transform)
    df.loc[mask, 'Value'] = 1 / (1 + np.abs(df.loc[mask, 'Value']))
    return df

#this is the starting point for compute_surprise.py
f = pd.read_pickle(os.path.join(output_dir,FEATURES_FILE))
f['feature'] = f.part +'_'+ f.feature_name


# %%
# Get unique subjects and their age_in_weeks (assuming age_in_weeks is the same for all rows per subject)
age_rows = f.drop_duplicates(subset='infant').copy()

age_rows['feature'] = 'age_in_weeks'  # Must match feature list name

age_rows['Value'] = age_rows['age_in_weeks']
age_rows['part'] = 'age'
age_rows['feature_name'] = 'in_weeks'  # part + feature_name = 'age_in_weeks'

# Combine with original data
f = pd.concat([f, age_rows], ignore_index=True)
# f will now have 59 features
#
f

# %%
f.feature.unique()

# %%
f.infant.unique().shape

# %%
# Move SAMPLES from EMMA and YT (normative) dataset with risk=1
# threshold age bracket (at 6 months)
features = create_train_test(f, samples=SAMPLES, age_threshold=AGE_THRESHOLD)

# %%
# %%
# USE PRE-SELECTED FEATURES (from RFE computed elsewhere)
# Do NOT perform RFE in this script
# All 6 feature sets from verified 10-fold cross-validation analysis (Nov 21, 2025)

# ============================================================================
# NEW FEATURE SETS (Recommended for clinical use)
# ============================================================================
# NOTE: NEW feature sets fix critical bugs:
#   - 60 FPS frame rate normalization ✅ FIXED
#   - Coordinate system rotation correction ✅ FIXED

# 15-Feature NEW: Simplified variant, good performance (AUC 0.7379)
# Features: 14 movement + age
selected_features_15_new = [
    "Elbow_IQR_acc_angle",
    "Ankle_IQRaccy",
    "Ankle_medianvelx",
    "Knee_IQR_acc_angle",
    "Hip_lrCorr_angle",
    "Ankle_meanent",
    "Ankle_IQRvelx",
    "Wrist_medianx",
    "Ankle_medianvely",
    "Ankle_IQRaccx",
    "Knee_lrCorr_angle",
    "Wrist_medianvely",
    "Shoulder_stdev_angle",
    "Ankle_mediany",
    "age_in_weeks"
]

# 19-Feature NEW: Poor performance (AUC 0.6330) - NOT RECOMMENDED
# Features: 18 movement + age
selected_features_19_new = [
    "Knee_IQR_vel_angle",
    "Elbow_IQR_vel_angle",
    "Shoulder_entropy_angle",
    "Elbow_IQR_acc_angle",
    "Ankle_IQRvely",
    "Knee_entropy_angle",
    "Ankle_IQRaccy",
    "Knee_lrCorr_x",
    "Elbow_mean_angle",
    "Ankle_medianvely",
    "Ankle_IQRvelx",
    "Wrist_IQRvelx",
    "Wrist_mediany",
    "Hip_lrCorr_angle",
    "Wrist_IQRvely",
    "Ankle_medianvelx",
    "Wrist_IQRy",
    "Ankle_IQRaccx",
    "age_in_weeks"
]

# 20-Feature NEW: BEST FOR SCREENING (AUC 0.9018 with transform, Sensitivity 92.1%) ✅
# Features: 19 movement + age
# ORIGIN: Generated 2025-11-21 10:35 AM via run_rfe_with_age_forced.py
#         Selected on: features_merged.pkl (OLD file)
#         Evaluated on: features_merged_20251121_091511.pkl (NEW file)
#         Evidence: rfe_20_feature_20251121.log
selected_features_20_new = [
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

# ============================================================================
# 20-Feature RFE DEC 2025: VERIFIED PROVENANCE
# ============================================================================
# Generated: 2025-12-03 via scripts/run_rfe_with_age_forced.py
# Data file: features_merged_20251121_091511.pkl (CORRECT data file)
# Method: Enhanced Adaptive RFE (50 trials, 15-fold CV) → top 19 by RF importance + forced age
# Single-split AUC: 0.830, Sensitivity: 85.7%, Specificity: 72.5%
# NOTE: This set has VERIFIED provenance from traceable RFE run
selected_features_20_rfe_dec2025 = [
    "Hip_stdev_angle",       # importance: 0.1943
    "Hip_mean_angle",        # importance: 0.0700
    "Elbow_IQR_acc_angle",   # importance: 0.0610
    "Hip_entropy_angle",     # importance: 0.0559
    "Ankle_IQRaccy",         # importance: 0.0332
    "Ankle_IQRaccx",         # importance: 0.0283
    "Shoulder_IQR_acc_angle",# importance: 0.0282
    "Ankle_medianvelx",      # importance: 0.0266
    "Shoulder_lrCorr_angle", # importance: 0.0251
    "Elbow_IQR_vel_angle",   # importance: 0.0216
    "Ankle_meanent",         # importance: 0.0215
    "Wrist_IQRaccx",         # importance: 0.0191
    "Ankle_mediany",         # importance: 0.0176
    "Ankle_IQRx",            # importance: 0.0173
    "Shoulder_stdev_angle",  # importance: 0.0166
    "Wrist_IQRvelx",         # importance: 0.0161
    "Knee_lrCorr_angle",     # importance: 0.0158
    "Ankle_IQRvely",         # importance: 0.0147
    "Shoulder_mean_angle",   # importance: 0.0146
    "age_in_weeks"           # FORCED inclusion
]

# ============================================================================
# LEGACY FEATURE SETS (Reference only - contains known bugs)
# ============================================================================
# WARNING: LEGACY feature sets contain known bugs (60 FPS error, rotation error)
# Use for reference/comparison only - NOT recommended for clinical validation

# 15-Feature LEGACY: Balanced performance (AUC 0.7614)
# Features: 14 movement + age
selected_features_15_legacy = [
    "Ankle_IQRaccx",
    "Ankle_IQRaccy",
    "Ankle_IQRvelx",
    "Ankle_IQRvely",
    "Ankle_medianvelx",
    "Ankle_medianvely",
    "Elbow_IQR_acc_angle",
    "Elbow_IQR_vel_angle",
    "Elbow_median_vel_angle",
    "Hip_IQR_acc_angle",
    "Hip_IQR_vel_angle",
    "Hip_entropy_angle",
    "Hip_median_vel_angle",
    "Hip_stdev_angle",
    "age_in_weeks"
]

# 18-Feature LEGACY: Similar to 15-feature (AUC 0.7611)
# Features: 18 movement + age
selected_features_18_legacy = [
    "Ankle_IQRaccx",
    "Ankle_IQRaccy",
    "Ankle_IQRvelx",
    "Ankle_IQRvely",
    "Ankle_medianvelx",
    "Ankle_medianvely",
    "Elbow_IQR_acc_angle",
    "Elbow_IQR_vel_angle",
    "Elbow_median_vel_angle",
    "Hip_IQR_acc_angle",
    "Hip_IQR_vel_angle",
    "Hip_entropy_angle",
    "Hip_median_vel_angle",
    "Hip_stdev_angle",
    "Knee_IQR_acc_angle",
    "Knee_IQR_vel_angle",
    "Knee_entropy_angle",
    "Knee_median_vel_angle",
    "age_in_weeks"
]

# 20-Feature LEGACY: BEST OVERALL LEGACY (AUC 0.7884, highest AUC)
# Features: 19 movement + age
selected_features_20_legacy = [
    "Ankle_IQRaccx",
    "Ankle_IQRaccy",
    "Ankle_IQRvelx",
    "Ankle_IQRvely",
    "Ankle_medianvelx",
    "Ankle_medianvely",
    "Ankle_mediany",
    "Elbow_IQR_acc_angle",
    "Elbow_IQR_vel_angle",
    "Elbow_median_vel_angle",
    "Hip_IQR_acc_angle",
    "Hip_IQR_vel_angle",
    "Hip_entropy_angle",
    "Hip_stdev_angle",
    "Knee_IQR_acc_angle",
    "Wrist_IQRaccy",
    "Wrist_IQRvelx",
    "Wrist_IQRvely",
    "Wrist_medianvelx",
    "age_in_weeks"
]

# all 38 features
all_38_features = ['Ankle_medianx','Wrist_medianx','Ankle_mediany','Wrist_mediany',
                'Knee_mean_angle','Elbow_mean_angle',
                'Ankle_IQRx', 'Wrist_IQRx','Ankle_IQRy', 'Wrist_IQRy',
                'Knee_stdev_angle', 'Elbow_stdev_angle',
                'Ankle_medianvelx','Wrist_medianvelx','Ankle_medianvely','Wrist_medianvely',
                'Knee_median_vel_angle','Elbow_median_vel_angle',
                'Ankle_IQRvelx','Wrist_IQRvelx','Ankle_IQRvely','Wrist_IQRvely',
                'Knee_IQR_vel_angle','Elbow_IQR_vel_angle',
                'Ankle_IQRaccx','Wrist_IQRaccx','Ankle_IQRaccy','Wrist_IQRaccy',
                'Knee_IQR_acc_angle','Elbow_IQR_acc_angle',
                'Ankle_meanent', 'Wrist_meanent','Knee_entropy_angle', 'Elbow_entropy_angle',
                'Ankle_lrCorr_x', 'Wrist_lrCorr_x','Knee_lrCorr_angle', 'Elbow_lrCorr_angle', 'age_in_weeks']

all_59_features = ['Ankle_IQRaccx', 'Ankle_IQRaccy', 'Ankle_IQRvelx', 'Ankle_IQRvely',
       'Ankle_IQRx', 'Ankle_IQRy', 'Ankle_lrCorr_x', 'Ankle_meanent',
       'Ankle_medianvelx', 'Ankle_medianvely', 'Ankle_medianx',
       'Ankle_mediany', 'Ear_lrCorr_x', 'Elbow_IQR_acc_angle',
       'Elbow_IQR_vel_angle', 'Elbow_entropy_angle', 'Elbow_lrCorr_angle',
       'Elbow_lrCorr_x', 'Elbow_mean_angle', 'Elbow_median_vel_angle',
       'Elbow_stdev_angle', 'Eye_lrCorr_x', 'Hip_IQR_acc_angle',
       'Hip_IQR_vel_angle', 'Hip_entropy_angle', 'Hip_lrCorr_angle',
       'Hip_lrCorr_x', 'Hip_mean_angle', 'Hip_median_vel_angle',
       'Hip_stdev_angle', 'Knee_IQR_acc_angle', 'Knee_IQR_vel_angle',
       'Knee_entropy_angle', 'Knee_lrCorr_angle', 'Knee_lrCorr_x',
       'Knee_mean_angle', 'Knee_median_vel_angle', 'Knee_stdev_angle',
       'Shoulder_IQR_acc_angle', 'Shoulder_IQR_vel_angle',
       'Shoulder_entropy_angle', 'Shoulder_lrCorr_angle',
       'Shoulder_lrCorr_x', 'Shoulder_mean_angle',
       'Shoulder_median_vel_angle', 'Shoulder_stdev_angle',
       'Wrist_IQRaccx', 'Wrist_IQRaccy', 'Wrist_IQRvelx', 'Wrist_IQRvely',
       'Wrist_IQRx', 'Wrist_IQRy', 'Wrist_lrCorr_x', 'Wrist_meanent',
       'Wrist_medianvelx', 'Wrist_medianvely', 'Wrist_medianx',
       'Wrist_mediany', 'age_in_weeks']


# ============================================================================
# FEATURE SET SELECTION FOR ANALYSIS
# ============================================================================
# Select one of the 6 feature sets below to analyze:
# - selected_features_15_new: 15-Feature NEW (simplified)
# - selected_features_19_new: 19-Feature NEW (NOT RECOMMENDED)
# - selected_features_20_new: 20-Feature NEW (BEST FOR SCREENING) ✅
# - selected_features_15_legacy: 15-Feature LEGACY (reference only)
# - selected_features_18_legacy: 18-Feature LEGACY (reference only)
# - selected_features_20_legacy: 20-Feature LEGACY (reference only)

# Default to 20-Feature NEW (best overall for screening)
selected_features = selected_features_20_new


print(f"Using {len(selected_features)} pre-selected features")

# %%
# FUNCTION DEFINITIONS FOR ROC ANALYSIS

def compute_roc(t, s):

    typical_category = 0
    lowrisk_category = 1
    modrisk_category = 2
    highrisk_category = 3

    tpr = []
    fpr = []
    for z_threshold in t:
        tp = s[(s.risk>lowrisk_category)].shape[0]

        fp = s[(s.risk<=lowrisk_category)&(s.z < z_threshold)].shape[0]

        fn = s[(s.risk>lowrisk_category)&(s.z >= z_threshold)].shape[0]

        tn = s[(s.risk<=lowrisk_category)&(s.z > z_threshold)].shape[0]

        tpr_val = tp / (tp+fn) #sens
        tpr.append(tpr_val)

        spec = tn / (tn + fp)
        fpr_val = 1-spec
        fpr.append(fpr_val)

    return tpr, fpr


def compute_roc_interval(t, s):

    typical_category = 0
    lowrisk_category = 1
    modrisk_category = 2
    highrisk_category = 3

    tpr = []
    fpr = []
    for z_threshold in t:
        z_threshold_abs = abs(z_threshold)
        tp = s[(s.risk>lowrisk_category)].shape[0]

        fp = s[(s.risk<=lowrisk_category)&( not(abs(s.z) <= z_threshold_abs) )].shape[0]

        fn = s[(s.risk>lowrisk_category)&(abs(s.z) < z_threshold_abs )].shape[0]

        tn = s[(s.risk<=lowrisk_category)& (abs(s.z)<=z_threshold_abs)].shape[0]

        tpr_val = tp / (tp+fn) #sens
        tpr.append(tpr_val)

        spec = tn / (tn + fp)
        fpr_val = 1-spec
        fpr.append(fpr_val)

    return tpr, fpr

# %% jupyter={"outputs_hidden": true}
# FUNCTION: Run k-fold CV for a feature set
def run_kfold_cv_for_feature_set(feature_list, feature_set_name, k_folds=10, apply_transform=None):
    """Run k-fold cross-validation for a specific feature set

    Parameters:
    -----------
    feature_list : list - Features to use
    feature_set_name : str - Name for display
    k_folds : int - Number of folds
    apply_transform : bool or None - If None, uses global APPLY_INVERSE_TRANSFORM
    """
    np.random.seed(42)
    random.seed(42)

    # Use parameter if provided, otherwise fall back to global setting
    use_transform = apply_transform if apply_transform is not None else APPLY_INVERSE_TRANSFORM

    fold_results = []

    transform_status = "WITH inverse transform" if use_transform else "WITHOUT transform (baseline)"
    print(f"\n{'='*80}")
    print(f"Processing: {feature_set_name} ({len(feature_list)} features)")
    print(f"Transform: {transform_status}")
    print(f"K-FOLD CROSS-VALIDATION (K={k_folds})")
    print(f"{'='*80}")

    for fold_idx in range(k_folds):
        print(f"  [FOLD {fold_idx+1}/{k_folds}]", end=" ")

        # Load fresh features for each fold
        f = pd.read_pickle(os.path.join(output_dir,FEATURES_FILE))
        f['feature'] = f.part + '_' + f.feature_name

        # Add age_in_weeks as a feature (must match feature list name)
        age_rows = f.drop_duplicates(subset='infant').copy()
        age_rows['feature'] = 'age_in_weeks'
        age_rows['Value'] = age_rows['age_in_weeks']
        age_rows['part'] = 'age'
        age_rows['feature_name'] = 'in_weeks'
        f = pd.concat([f, age_rows], ignore_index=True)

        # Create train/test split
        features = create_train_test(f, samples=SAMPLES, age_threshold=AGE_THRESHOLD)

        # Add feature column for transformation matching (refresh after train/test split)
        features['feature'] = features.part + '_' + features.feature_name

        # Apply inverse transformation to velocity/acceleration features if enabled
        if use_transform:
            features = apply_inverse_transform(features, TRANSFORM_FEATURES)

        # Compute reference statistics from TRAINING set (category=0)
        ref_stats = pd.DataFrame()
        ref_stats = features[features.category == 0].groupby(['feature_name' , 'part'])['Value']\
        .apply(norm.fit).reset_index()
        ref_stats[['mean_ref', 'sd_ref']] = ref_stats['Value'].apply(pd.Series)
        ref_stats['var_ref'] = ref_stats['sd_ref']**2
        ref_stats = ref_stats.reset_index().drop('Value', axis=1)

        # Merge reference statistics with full features
        features = pd.merge(features, ref_stats, on=['feature_name', 'part'], how='inner')
        features['minus_log_pfeature'] = -1*(.5*np.log(2*np.pi*features['var_ref']) + ((features['Value']-features['mean_ref'])**2)/(2*features['var_ref']))
        features['feature'] = features.part +'_'+ features.feature_name

        # Use pre-selected features only
        features = features.loc[np.isin(features.feature, feature_list)]

        # Compute surprise for all subjects
        fold_surprise = features.groupby(['infant', 'age_in_weeks','risk', 'category'])['minus_log_pfeature'].sum().reset_index()

        # Normalize Z-scores using training set statistics (category=0)
        train_mean = fold_surprise.loc[fold_surprise.category==0, 'minus_log_pfeature'].mean()
        train_std = fold_surprise.loc[fold_surprise.category==0, 'minus_log_pfeature'].std()

        fold_surprise['z'] = (fold_surprise['minus_log_pfeature'] - train_mean) / train_std
        fold_surprise['p'] = (sc.stats.norm.sf(np.abs(fold_surprise['z']))*2).round(3)

        # Remove problematic subjects
        index = fold_surprise[fold_surprise.infant.str.contains('clin_100_')].index
        fold_surprise.drop(index, inplace=True)

        # Extract test set (category=1)
        test_surprise = fold_surprise[fold_surprise.category==1].copy()
        test_surprise['fold'] = fold_idx

        fold_results.append(fold_surprise)

        n_train = fold_surprise[fold_surprise.category==0]['infant'].nunique()
        n_test = test_surprise['infant'].nunique()
        print(f"Train: {n_train} | Test: {n_test}")

    # Combine all test sets
    all_test_data = []
    for fold_idx, fold_df in enumerate(fold_results):
        test_data = fold_df[fold_df.category==1].copy()
        test_data['fold'] = fold_idx
        all_test_data.append(test_data)

    combined_test = pd.concat(all_test_data, ignore_index=True)

    return combined_test, fold_results

# K-FOLD CROSS-VALIDATION FOR ALL FEATURE SETS
K_FOLDS = 10  # number of cross-validation folds

# Define feature sets dictionary
feature_sets = {
    '15-Feature': selected_features_15_new,
    '19-Feature': selected_features_19_new,
    '20-Feature': selected_features_20_new,
    '15-Feature (LEGACY)': selected_features_15_legacy,
    '18-Feature (LEGACY)': selected_features_18_legacy,
    '20-Feature (LEGACY)': selected_features_20_legacy,
    'All 38 Features': all_38_features,
    'All 59 Features': all_59_features
}

# ============================================================================
# SET RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
# Set random seed before k-fold CV to ensure reproducible train/test splits
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
print(f"\nRandom seed set to {RANDOM_SEED} for reproducible k-fold CV")

# ============================================================================
# RUN K-FOLD CV (with optional comparison mode)
# ============================================================================

if RUN_COMPARISON_MODE:
    print("\n" + "="*80)
    print("COMPARISON MODE: Running WITH and WITHOUT inverse transformation")
    print("="*80)

    # Run WITHOUT transform (baseline)
    # Reset seed to ensure same train/test splits as Phase 2
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print("\n" + "-"*80)
    print("PHASE 1: BASELINE (No Transformation)")
    print("-"*80)

    baseline_results = {}
    for feature_set_name, feature_list in feature_sets.items():
        combined_test, fold_results = run_kfold_cv_for_feature_set(
            feature_list, feature_set_name, K_FOLDS, apply_transform=False
        )
        baseline_results[feature_set_name] = {'combined_test': combined_test, 'fold_results': fold_results}

    # Run WITH transform
    # Reset seed to ensure same train/test splits as Phase 1
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print("\n" + "-"*80)
    print("PHASE 2: WITH INVERSE TRANSFORMATION")
    print("-"*80)

    transformed_results = {}
    for feature_set_name, feature_list in feature_sets.items():
        combined_test, fold_results = run_kfold_cv_for_feature_set(
            feature_list, feature_set_name, K_FOLDS, apply_transform=True
        )
        transformed_results[feature_set_name] = {'combined_test': combined_test, 'fold_results': fold_results}

    # Use transformed results as the main results
    all_feature_set_results = transformed_results

else:
    # Single mode - use global APPLY_INVERSE_TRANSFORM setting
    all_feature_set_results = {}

    for feature_set_name, feature_list in feature_sets.items():
        combined_test, fold_results = run_kfold_cv_for_feature_set(feature_list, feature_set_name, K_FOLDS)
        all_feature_set_results[feature_set_name] = {'combined_test': combined_test, 'fold_results': fold_results}

        print(f"\n  Combined test set summary:")
        print(f"    Total test subjects: {combined_test['infant'].nunique()}")
        print(f"    Total test data points: {len(combined_test)}")
        print(f"    Risk=2 (moderate): {(combined_test.risk==2).sum()}")
        print(f"    Risk=3 (high): {(combined_test.risk==3).sum()}")
        print(f"    Risk=1 (low/normal): {(combined_test.risk<=1).sum()}")

# Use first feature set for reference (20-Feature NEW)
combined_test = all_feature_set_results['20-Feature']['combined_test']
fold_results = all_feature_set_results['20-Feature']['fold_results']

print(f"\n{'='*80}")
print("K-fold cross-validation complete for all feature sets")
print(f"{'='*80}")


# %%
# AGGREGATE RESULTS FROM ALL FOLDS
# Combine test sets from all folds for overall ROC analysis

# Combine all test sets
all_test_data = []
for fold_idx, fold_df in enumerate(fold_results):
    test_data = fold_df[fold_df.category==1].copy()
    test_data['fold'] = fold_idx
    all_test_data.append(test_data)

combined_test = pd.concat(all_test_data, ignore_index=True)

print(f"\nCombined test set from all {K_FOLDS} folds:")
print(f"  Total test subjects: {combined_test['infant'].nunique()}")
print(f"  Total test data points: {len(combined_test)}")
print(f"  Risk=2 (moderate): {(combined_test.risk==2).sum()}")
print(f"  Risk=3 (high): {(combined_test.risk==3).sum()}")
print(f"  Risk=1 (low/normal): {(combined_test.risk<=1).sum()}")

# %%
# Z-SCORE DISTRIBUTION ANALYSIS BY RISK GROUP
# This helps understand why one-tailed vs two-tailed thresholding performs differently

def compute_zscore_stats(data, label=""):
    """Compute z-score statistics for normal and at-risk groups."""
    normal_z = data[data.risk <= 1]['z']
    atrisk_z = data[data.risk > 1]['z']

    stats = {
        'label': label,
        'normal_mean': normal_z.mean(),
        'normal_median': normal_z.median(),
        'normal_std': normal_z.std(),
        'normal_min': normal_z.min(),
        'normal_max': normal_z.max(),
        'atrisk_mean': atrisk_z.mean(),
        'atrisk_median': atrisk_z.median(),
        'atrisk_std': atrisk_z.std(),
        'atrisk_min': atrisk_z.min(),
        'atrisk_max': atrisk_z.max(),
        'mean_separation': atrisk_z.mean() - normal_z.mean(),
        'median_separation': atrisk_z.median() - normal_z.median(),
    }
    return stats

def print_zscore_stats(stats):
    """Print z-score statistics for a single condition."""
    print(f"\nNormal (risk <= 1):")
    print(f"  Mean Z:   {stats['normal_mean']:.3f}")
    print(f"  Median Z: {stats['normal_median']:.3f}")
    print(f"  Std Z:    {stats['normal_std']:.3f}")
    print(f"  Range:    [{stats['normal_min']:.3f}, {stats['normal_max']:.3f}]")

    print(f"\nAt-Risk (risk > 1):")
    print(f"  Mean Z:   {stats['atrisk_mean']:.3f}")
    print(f"  Median Z: {stats['atrisk_median']:.3f}")
    print(f"  Std Z:    {stats['atrisk_std']:.3f}")
    print(f"  Range:    [{stats['atrisk_min']:.3f}, {stats['atrisk_max']:.3f}]")

    print(f"\nGroup Separation:")
    print(f"  Mean difference (at-risk - normal):   {stats['mean_separation']:.3f}")
    print(f"  Median difference (at-risk - normal): {stats['median_separation']:.3f}")

print(f"\n{'='*80}")
print("Z-SCORE DISTRIBUTION BY RISK GROUP")
print(f"{'='*80}")

# Check if we're in comparison mode and have baseline data
if RUN_COMPARISON_MODE and 'baseline_results' in dir():
    # Get baseline data for 20-Feature (NEW)
    baseline_fold_results = baseline_results['20-Feature']['fold_results']
    baseline_test_data = []
    for fold_idx, fold_df in enumerate(baseline_fold_results):
        test_data = fold_df[fold_df.category==1].copy()
        test_data['fold'] = fold_idx
        baseline_test_data.append(test_data)
    baseline_combined = pd.concat(baseline_test_data, ignore_index=True)

    # Compute stats for both conditions
    baseline_stats = compute_zscore_stats(baseline_combined, "Baseline (No Transform)")
    transformed_stats = compute_zscore_stats(combined_test, "With Inverse Transform")

    # Print comparison
    print(f"\n{'─'*80}")
    print("BASELINE (No Transformation) - 20-Feature Model")
    print(f"{'─'*80}")
    print_zscore_stats(baseline_stats)

    print(f"\n{'─'*80}")
    print("WITH INVERSE TRANSFORMATION - 20-Feature Model")
    print(f"{'─'*80}")
    print_zscore_stats(transformed_stats)

    # Print improvement summary
    print(f"\n{'='*80}")
    print("Z-SCORE SEPARATION IMPROVEMENT (Baseline → Transformed)")
    print(f"{'='*80}")

    mean_sep_improvement = abs(transformed_stats['mean_separation']) - abs(baseline_stats['mean_separation'])
    median_sep_improvement = abs(transformed_stats['median_separation']) - abs(baseline_stats['median_separation'])

    print(f"\n{'Metric':<30} | {'Baseline':>12} | {'Transformed':>12} | {'Change':>12}")
    print(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'Mean Separation (|Δ|)':<30} | {abs(baseline_stats['mean_separation']):>12.3f} | {abs(transformed_stats['mean_separation']):>12.3f} | {mean_sep_improvement:>+12.3f}")
    print(f"{'Median Separation (|Δ|)':<30} | {abs(baseline_stats['median_separation']):>12.3f} | {abs(transformed_stats['median_separation']):>12.3f} | {median_sep_improvement:>+12.3f}")
    print(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'Normal Mean Z':<30} | {baseline_stats['normal_mean']:>12.3f} | {transformed_stats['normal_mean']:>12.3f} | {transformed_stats['normal_mean'] - baseline_stats['normal_mean']:>+12.3f}")
    print(f"{'Normal Std Z':<30} | {baseline_stats['normal_std']:>12.3f} | {transformed_stats['normal_std']:>12.3f} | {transformed_stats['normal_std'] - baseline_stats['normal_std']:>+12.3f}")
    print(f"{'-'*30}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'At-Risk Mean Z':<30} | {baseline_stats['atrisk_mean']:>12.3f} | {transformed_stats['atrisk_mean']:>12.3f} | {transformed_stats['atrisk_mean'] - baseline_stats['atrisk_mean']:>+12.3f}")
    print(f"{'At-Risk Std Z':<30} | {baseline_stats['atrisk_std']:>12.3f} | {transformed_stats['atrisk_std']:>12.3f} | {transformed_stats['atrisk_std'] - baseline_stats['atrisk_std']:>+12.3f}")

    print(f"\nInterpretation:")
    if mean_sep_improvement > 0:
        print(f"  ✓ Transformation INCREASES mean z-score separation by {mean_sep_improvement:.3f}")
        print(f"  ✓ Better discrimination between normal and at-risk groups")
    else:
        print(f"  ✗ Transformation DECREASES mean z-score separation by {abs(mean_sep_improvement):.3f}")

    # =========================================================================
    # Z-SCORE DISTRIBUTION VISUALIZATION: BASELINE vs TRANSFORMED
    # =========================================================================
    print(f"\n{'='*80}")
    print("GENERATING Z-SCORE DISTRIBUTION VISUALIZATION")
    print(f"{'='*80}")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score

    # Compute AUC values for both conditions
    # For Bayesian surprise: lower z-scores indicate higher risk (at-risk infants have lower surprise)
    # So we need to negate the z-scores for AUC computation (higher = more likely at-risk)
    baseline_labels = (baseline_combined['risk'] > 1).astype(int)
    baseline_auc = roc_auc_score(baseline_labels, -baseline_combined['z'])

    transformed_labels = (combined_test['risk'] > 1).astype(int)
    transformed_auc = roc_auc_score(transformed_labels, -combined_test['z'])

    # Prepare data for visualization
    # Add condition label to baseline data
    baseline_viz = baseline_combined[['z', 'risk']].copy()
    baseline_viz['Condition'] = 'Baseline'
    baseline_viz['Risk Group'] = baseline_viz['risk'].apply(lambda x: 'Typical' if x <= 1 else 'At-Risk')

    # Add condition label to transformed data
    transformed_viz = combined_test[['z', 'risk']].copy()
    transformed_viz['Condition'] = 'Transformed'
    transformed_viz['Risk Group'] = transformed_viz['risk'].apply(lambda x: 'Typical' if x <= 1 else 'At-Risk')

    # Combine for plotting
    viz_data = pd.concat([baseline_viz, transformed_viz], ignore_index=True)

    # Create figure with side-by-side panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Z-Score Distribution: Baseline vs Inverse Transformed Features\n(20-Feature Model)',
                 fontsize=14, fontweight='bold', y=1.02)

    # Color palette
    palette = {'Typical': '#2ecc71', 'At-Risk': '#e74c3c'}

    def make_half_violin(ax, side_typical='left', side_atrisk='left', offset=0):
        """
        Modify violin plots to show density on one side only, with independent control per group.

        Parameters:
        - side_typical: 'left' or 'right' - direction for Typical group (at x~0)
        - side_atrisk: 'left' or 'right' - direction for At-Risk group (at x~1)
        - offset: horizontal shift to apply after clipping

        'right': density extends to the right, flat edge on left
        'left': density extends to the left, flat edge on right
        """
        from matplotlib.collections import PolyCollection
        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                for path in collection.get_paths():
                    vertices = path.vertices
                    # Find the center x for this violin
                    center_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2

                    # Determine which group this violin belongs to based on center position
                    # Typical is at x~0, At-Risk is at x~1
                    if center_x < 0.5:
                        side = side_typical
                    else:
                        side = side_atrisk

                    if side == 'right':
                        # Keep only right side: clip left vertices to center
                        vertices[:, 0] = np.clip(vertices[:, 0], center_x, None)
                    else:
                        # Keep only left side: clip right vertices to center
                        vertices[:, 0] = np.clip(vertices[:, 0], None, center_x)
                    # Apply horizontal offset
                    vertices[:, 0] += offset

    # Visualization flags
    SHOW_SWARM = True  # Set to False to hide swarm plots for debugging

    # Half-violin direction control (independent for each group)
    # 'left' = density extends left, flat edge on right (at category position)
    # 'right' = density extends right, flat edge on left (at category position)
    VIOLIN_SIDE_TYPICAL = 'left'   # Direction for Typical (green) half-violin
    VIOLIN_SIDE_ATRISK = 'right'    # Direction for At-Risk (red) half-violin

    # Fixed x-coordinates for each component (explicit positioning)
    # Typical group centered at x=0, At-Risk group centered at x=1
    TYPICAL_X = 0.0
    ATRISK_X = 1.0
    VIOLIN_OFFSET = 0.0     # No offset - violins stay at category position
    SWARM_OFFSET = 0.0      # No offset - swarms stay at category position

    # Helper function to get swarm center x-positions after drawing
    def get_swarm_centers(ax, n_typical, n_atrisk):
        """Calculate actual mean x-position of each swarm group."""
        # Find PathCollection objects (swarm points)
        swarm_collections = [c for c in ax.collections if hasattr(c, 'get_offsets') and len(c.get_offsets()) > 0]
        if not swarm_collections:
            return TYPICAL_X + SWARM_OFFSET, ATRISK_X + SWARM_OFFSET

        # Get all offsets from all collections
        all_offsets = np.vstack([c.get_offsets() for c in swarm_collections])

        # Split by x-position (typical ~0, at-risk ~1)
        typical_mask = all_offsets[:, 0] < 0.5
        atrisk_mask = all_offsets[:, 0] >= 0.5

        typical_center = all_offsets[typical_mask, 0].mean() if typical_mask.any() else TYPICAL_X + SWARM_OFFSET
        atrisk_center = all_offsets[atrisk_mask, 0].mean() if atrisk_mask.any() else ATRISK_X + SWARM_OFFSET

        return typical_center, atrisk_center

    # Left panel: Baseline
    ax_baseline = axes[0]

    # Create violin plot (will be modified to half-violin)
    sns.violinplot(data=baseline_viz, x='Risk Group', y='z', hue='Risk Group',
                   palette=palette, ax=ax_baseline, inner=None,
                   order=['Typical', 'At-Risk'], legend=False, alpha=0.5, width=0.6)
    # Convert to half-violins with independent direction control per group
    make_half_violin(ax_baseline, side_typical=VIOLIN_SIDE_TYPICAL, side_atrisk=VIOLIN_SIDE_ATRISK, offset=VIOLIN_OFFSET)

    # Add swarm plot with circle markers (if enabled)
    if SHOW_SWARM:
        sns.swarmplot(data=baseline_viz, x='Risk Group', y='z', hue='Risk Group',
                      palette=palette, ax=ax_baseline, alpha=0.6, size=3,
                      order=['Typical', 'At-Risk'], legend=False, marker='o',
                      edgecolor='white', linewidth=0.5)
        # Offset swarm plot points to the right
        for collection in ax_baseline.collections:
            if hasattr(collection, 'get_offsets'):
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    offsets[:, 0] += SWARM_OFFSET
                    collection.set_offsets(offsets)

    # Remove any auto-generated legend
    if ax_baseline.get_legend():
        ax_baseline.get_legend().remove()

    # Add mean markers for baseline
    baseline_typical_mean = baseline_stats['normal_mean']
    baseline_atrisk_mean = baseline_stats['atrisk_mean']
    baseline_typical_std = baseline_stats['normal_std']
    baseline_atrisk_std = baseline_stats['atrisk_std']

    # Get actual swarm centers for error bar positioning
    n_typical_baseline = len(baseline_viz[baseline_viz['Risk Group'] == 'Typical'])
    n_atrisk_baseline = len(baseline_viz[baseline_viz['Risk Group'] == 'At-Risk'])
    errorbar_x_typical, errorbar_x_atrisk = get_swarm_centers(ax_baseline, n_typical_baseline, n_atrisk_baseline)

    # Add error bars (mean ± 1 SD) with diamond markers for baseline
    ax_baseline.errorbar(errorbar_x_typical, baseline_typical_mean, yerr=baseline_typical_std,
                         fmt='D', markersize=10, markerfacecolor='white', markeredgecolor='#27ae60',
                         markeredgewidth=2.5, ecolor='#27ae60', elinewidth=2.5, capsize=8, capthick=2.5, zorder=6)
    ax_baseline.errorbar(errorbar_x_atrisk, baseline_atrisk_mean, yerr=baseline_atrisk_std,
                         fmt='D', markersize=10, markerfacecolor='white', markeredgecolor='#c0392b',
                         markeredgewidth=2.5, ecolor='#c0392b', elinewidth=2.5, capsize=8, capthick=2.5, zorder=6)

    # Annotate separation (arrow between the two groups, centered between tick marks)
    sep_baseline = abs(baseline_stats['mean_separation'])
    arrow_x = 0.5  # Centered between Typical (x=0) and At-Risk (x=1) tick marks
    mid_y = (baseline_typical_mean + baseline_atrisk_mean) / 2
    ax_baseline.annotate('', xy=(arrow_x, baseline_atrisk_mean), xytext=(arrow_x, baseline_typical_mean),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax_baseline.text(arrow_x + 0.05, mid_y, f'Δ={sep_baseline:.2f}', fontsize=9, fontweight='bold', va='center')

    ax_baseline.set_title(f'BASELINE (No Transform)\nAUC = {baseline_auc:.3f}',
                          fontsize=12, fontweight='bold')
    ax_baseline.set_xlabel('Risk Group', fontsize=11, fontweight='bold')
    ax_baseline.set_ylabel('Z-Score (Surprise)', fontsize=11, fontweight='bold')
    ax_baseline.grid(True, alpha=0.3, axis='y')

    # Add stats annotation box for baseline
    stats_text_baseline = f'Typical: μ={baseline_typical_mean:.2f}, σ={baseline_stats["normal_std"]:.2f}\nAt-Risk: μ={baseline_atrisk_mean:.2f}, σ={baseline_stats["atrisk_std"]:.2f}'
    ax_baseline.text(0.3, 0.02, stats_text_baseline, transform=ax_baseline.transAxes, fontsize=9, va='top',
                     fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                     family='monospace')

    # Right panel: Transformed
    ax_transformed = axes[1]

    # Create violin plot (will be modified to half-violin)
    sns.violinplot(data=transformed_viz, x='Risk Group', y='z', hue='Risk Group',
                   palette=palette, ax=ax_transformed, inner=None,
                   order=['Typical', 'At-Risk'], legend=False, alpha=0.5, width=0.6)
    # Convert to half-violins with independent direction control per group
    make_half_violin(ax_transformed, side_typical=VIOLIN_SIDE_TYPICAL, side_atrisk=VIOLIN_SIDE_ATRISK, offset=VIOLIN_OFFSET)

    # Add swarm plot with circle markers (if enabled)
    if SHOW_SWARM:
        sns.swarmplot(data=transformed_viz, x='Risk Group', y='z', hue='Risk Group',
                      palette=palette, ax=ax_transformed, alpha=0.6, size=3,
                      order=['Typical', 'At-Risk'], legend=False, marker='o',
                      edgecolor='white', linewidth=0.5)
        # Offset swarm plot points to the right
        for collection in ax_transformed.collections:
            if hasattr(collection, 'get_offsets'):
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    offsets[:, 0] += SWARM_OFFSET
                    collection.set_offsets(offsets)

    # Remove any auto-generated legend
    if ax_transformed.get_legend():
        ax_transformed.get_legend().remove()

    # Add mean markers for transformed
    transformed_typical_mean = transformed_stats['normal_mean']
    transformed_atrisk_mean = transformed_stats['atrisk_mean']
    transformed_typical_std = transformed_stats['normal_std']
    transformed_atrisk_std = transformed_stats['atrisk_std']

    # Get actual swarm centers for error bar positioning
    n_typical_transformed = len(transformed_viz[transformed_viz['Risk Group'] == 'Typical'])
    n_atrisk_transformed = len(transformed_viz[transformed_viz['Risk Group'] == 'At-Risk'])
    errorbar_x_typical_t, errorbar_x_atrisk_t = get_swarm_centers(ax_transformed, n_typical_transformed, n_atrisk_transformed)

    # Add error bars (mean ± 1 SD) with diamond markers for transformed
    ax_transformed.errorbar(errorbar_x_typical_t, transformed_typical_mean, yerr=transformed_typical_std,
                            fmt='D', markersize=10, markerfacecolor='white', markeredgecolor='#27ae60',
                            markeredgewidth=2.5, ecolor='#27ae60', elinewidth=2.5, capsize=8, capthick=2.5, zorder=6)
    ax_transformed.errorbar(errorbar_x_atrisk_t, transformed_atrisk_mean, yerr=transformed_atrisk_std,
                            fmt='D', markersize=10, markerfacecolor='white', markeredgecolor='#c0392b',
                            markeredgewidth=2.5, ecolor='#c0392b', elinewidth=2.5, capsize=8, capthick=2.5, zorder=6)

    # Annotate separation (arrow between the two groups, centered between tick marks)
    sep_transformed = abs(transformed_stats['mean_separation'])
    arrow_x_t = 0.5  # Centered between Typical (x=0) and At-Risk (x=1) tick marks
    mid_y_t = (transformed_typical_mean + transformed_atrisk_mean) / 2
    ax_transformed.annotate('', xy=(arrow_x_t, transformed_atrisk_mean), xytext=(arrow_x_t, transformed_typical_mean),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax_transformed.text(arrow_x_t + 0.05, mid_y_t, f'Δ={sep_transformed:.2f}', fontsize=9, fontweight='bold', va='center')

    ax_transformed.set_title(f'WITH INVERSE TRANSFORM\nAUC = {transformed_auc:.3f}',
                             fontsize=12, fontweight='bold')
    ax_transformed.set_xlabel('Risk Group', fontsize=11, fontweight='bold')
    ax_transformed.set_ylabel('')  # Share y-axis label
    ax_transformed.grid(True, alpha=0.3, axis='y')

    # Add stats annotation box for transformed
    stats_text_transformed = f'Typical: μ={transformed_typical_mean:.2f}, σ={transformed_stats["normal_std"]:.2f}\nAt-Risk: μ={transformed_atrisk_mean:.2f}, σ={transformed_stats["atrisk_std"]:.2f}'
    ax_transformed.text(0.3, 0.02, stats_text_transformed, transform=ax_transformed.transAxes, fontsize=9, va='top',
                        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                        family='monospace')

    # Add custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='gray', markerfacecolor='white', markeredgecolor='black',
               markersize=10, label='Mean (μ)', linestyle='-', linewidth=2.5),
        Line2D([0], [0], marker='_', color='gray', markersize=12, markeredgewidth=2.5,
               linestyle='None', label='±1 SD')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.98, 0.98))

    # Set x-axis limits to reduce white space (constrain to show all elements tightly)
    xlim_left = -0.5   # Room for left-extending violin
    xlim_right = 1.5   # Room for right-extending swarm
    ax_baseline.set_xlim(xlim_left, xlim_right)
    ax_transformed.set_xlim(xlim_left, xlim_right)

    # Keep x-tick labels at category centers (0 and 1)
    ax_baseline.set_xticks([0, 1])
    ax_baseline.set_xticklabels(['Typical', 'At-Risk'])
    ax_transformed.set_xticks([0, 1])
    ax_transformed.set_xticklabels(['Typical', 'At-Risk'])

    # Add improvement annotation at bottom
    improvement_text = f'Mean Separation Improvement: {sep_baseline:.2f} → {sep_transformed:.2f} (+{sep_transformed - sep_baseline:.2f})'
    fig.text(0.5, -0.02, improvement_text, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('zscore_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Z-score distribution visualization saved to: zscore_distribution_comparison.png")

else:
    # Single mode - just show current results
    normal_z = combined_test[combined_test.risk <= 1]['z']
    atrisk_z = combined_test[combined_test.risk > 1]['z']

    print("\nNormal (risk <= 1):")
    print(f"  Mean Z:   {normal_z.mean():.3f}")
    print(f"  Median Z: {normal_z.median():.3f}")
    print(f"  Std Z:    {normal_z.std():.3f}")
    print(f"  Range:    [{normal_z.min():.3f}, {normal_z.max():.3f}]")

    print("\nAt-Risk (risk > 1):")
    print(f"  Mean Z:   {atrisk_z.mean():.3f}")
    print(f"  Median Z: {atrisk_z.median():.3f}")
    print(f"  Std Z:    {atrisk_z.std():.3f}")
    print(f"  Range:    [{atrisk_z.min():.3f}, {atrisk_z.max():.3f}]")

    print(f"\nInterpretation:")
    if atrisk_z.mean() < normal_z.mean():
        print(f"  At-risk infants have LOWER z-scores (mean diff: {atrisk_z.mean() - normal_z.mean():.3f})")
        print(f"  → One-tailed threshold (z < threshold) is appropriate")
    else:
        print(f"  At-risk infants have HIGHER z-scores (mean diff: {atrisk_z.mean() - normal_z.mean():.3f})")
        print(f"  → One-tailed threshold (z > threshold) would be appropriate")

# %%
# combined_test dataframe contains results from the k-fold validation (all k folds)
# obtained after movement metrics for the chosen set (20 features new) have been inverse transformed
#
# baseline_combined contains results with the raw movement features, untransformed
#


# %%
# Generate 500 evenly-spaced z-score thresholds from -3.0 to 3.0
# This provides fine resolution (0.012 step) for smooth ROC curves
thresholds = np.linspace(-3.0, 3.0, 500).tolist()

print(f"Testing {len(thresholds)} thresholds")
print(f"Range: [{thresholds[0]:.2f}, {thresholds[-1]:.2f}]")
print(f"Step size: {(thresholds[1] - thresholds[0]):.4f}")



# %% [markdown]
# # SENS/SPEC with CONFIDENCE INTERVALS (K-FOLD CV)

# %% jupyter={"outputs_hidden": true}
from statsmodels.stats.proportion import proportion_confint

print(f"\n{'='*80}")
print("SENSITIVITY & SPECIFICITY WITH 95% CONFIDENCE INTERVALS")
print(f"(Based on K-Fold Combined Test Set: {K_FOLDS} folds)")
print(f"{'='*80}\n")

typical_category = 0
lowrisk_category = 1
modrisk_category = 2
highrisk_category = 3

tpr = []
fpr = []
for z_threshold in thresholds:
    TP = combined_test[(combined_test.risk>lowrisk_category)].shape[0]

    FP = combined_test[(combined_test.risk<=lowrisk_category)&(combined_test.z < z_threshold)].shape[0]

    FN = combined_test[(combined_test.risk>lowrisk_category)&(combined_test.z > z_threshold)].shape[0]

    TN = combined_test[(combined_test.risk<=lowrisk_category)&(combined_test.z > z_threshold)].shape[0]

    sensitivity = TP / (TP+FN) if (TP+FN) > 0 else 0
    specificity = TN / (TN+FP) if (TN+FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Confidence Intervals (95% CI)
    alpha = 0.05
    sensitivity_ci = proportion_confint(count=TP, nobs=TP + FN, alpha=alpha, method='wilson') if (TP+FN) > 0 else (0, 0)
    specificity_ci = proportion_confint(count=TN, nobs=TN + FP, alpha=alpha, method='wilson') if (TN+FP) > 0 else (0, 0)
    accuracy_ci = proportion_confint(count=TP + TN, nobs=TP + TN + FP + FN, alpha=alpha, method='wilson') if (TP + TN + FP + FN) > 0 else (0, 0)

    print('Threshold: {:5.1f} | TP={:3d} FP={:3d} FN={:3d} TN={:3d}'.format(z_threshold, TP, FP, FN, TN))
    print(f"\tSensitivity: {sensitivity:.3f} (95% CI: {sensitivity_ci[0]:.3f}, {sensitivity_ci[1]:.3f})")
    print(f"\tSpecificity: {specificity:.3f} (95% CI: {specificity_ci[0]:.3f}, {specificity_ci[1]:.3f})")
    print(f"\tAccuracy: {accuracy:.3f} (95% CI: {accuracy_ci[0]:.3f}, {accuracy_ci[1]:.3f})\n")

# %%
# OPTIMAL THRESHOLD DETERMINATION

print(f"\n{'='*80}")
print("OPTIMAL THRESHOLD ANALYSIS")
print(f"{'='*80}\n")

# Function to find optimal thresholds
def find_optimal_thresholds(surprise_data, min_sensitivity=0.85, min_specificity=0.75):
    """
    Find optimal thresholds based on:
    1. Youden's J statistic (max sensitivity + specificity - 1)
    2. Thresholds meeting clinical constraints (Sens >= min_sensitivity, Spec >= min_specificity)

    Args:
        surprise_data: DataFrame with 'z' and 'risk' columns
        min_sensitivity: Minimum required sensitivity
        min_specificity: Minimum required specificity

    Returns:
        dict with optimal threshold info
    """

    # Get unique z-score thresholds sorted descending
    unique_z_scores = np.sort(surprise_data['z'].unique())[::-1]

    threshold_metrics = []

    for z_threshold in unique_z_scores:
        # At-risk: risk > 1 (threshold below z_threshold)
        tp = surprise_data[(surprise_data.risk>1) & (surprise_data.z < z_threshold)].shape[0]
        fn = surprise_data[(surprise_data.risk>1) & (surprise_data.z >= z_threshold)].shape[0]

        # Normal: risk <= 1 (threshold above z_threshold)
        tn = surprise_data[(surprise_data.risk<=1) & (surprise_data.z >= z_threshold)].shape[0]
        fp = surprise_data[(surprise_data.risk<=1) & (surprise_data.z < z_threshold)].shape[0]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # Youden's J statistic
        youden_j = sensitivity + specificity - 1

        # Check if meets clinical constraints
        meets_constraints = (sensitivity >= min_sensitivity) and (specificity >= min_specificity)

        threshold_metrics.append({
            'threshold': z_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'accuracy': accuracy,
            'youden_j': youden_j,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'meets_constraints': meets_constraints
        })

    threshold_df = pd.DataFrame(threshold_metrics)

    # Find optimal threshold by Youden's J
    optimal_youden_idx = threshold_df['youden_j'].idxmax()
    optimal_youden = threshold_df.loc[optimal_youden_idx]

    # Find thresholds meeting constraints
    constraint_thresholds = threshold_df[threshold_df['meets_constraints'] == True]

    # If any meet constraints, find the one closest to balanced (middle ground)
    if len(constraint_thresholds) > 0:
        # Find threshold with best balance between sensitivity and specificity
        constraint_thresholds['balance_score'] = np.abs(
            constraint_thresholds['sensitivity'] - constraint_thresholds['specificity']
        )
        best_balanced_idx = constraint_thresholds['balance_score'].idxmin()
        optimal_constrained = threshold_df.loc[best_balanced_idx]
    else:
        optimal_constrained = None

    return {
        'all_thresholds': threshold_df,
        'optimal_youden': optimal_youden.to_dict(),
        'optimal_constrained': optimal_constrained.to_dict() if optimal_constrained is not None else None,
        'n_meeting_constraints': len(constraint_thresholds),
        'constraint_thresholds': constraint_thresholds
    }

# Analyze optimal thresholds on combined test set
optimal_thresholds_info = find_optimal_thresholds(
    combined_test,
    min_sensitivity=0.85,
    min_specificity=0.75
)

print("OPTIMAL THRESHOLD (Youden's J Statistic):")
print(f"  Threshold: {optimal_thresholds_info['optimal_youden']['threshold']:.4f}")
print(f"  Sensitivity: {optimal_thresholds_info['optimal_youden']['sensitivity']:.3f}")
print(f"  Specificity: {optimal_thresholds_info['optimal_youden']['specificity']:.3f}")
print(f"  Accuracy: {optimal_thresholds_info['optimal_youden']['accuracy']:.3f}")
print(f"  Youden's J: {optimal_thresholds_info['optimal_youden']['youden_j']:.3f}")
print(f"  PPV: {optimal_thresholds_info['optimal_youden']['ppv']:.3f}")
print(f"  NPV: {optimal_thresholds_info['optimal_youden']['npv']:.3f}")

print(f"\nCLINICAL CONSTRAINTS (Sensitivity >= 0.85, Specificity >= 0.75):")
print(f"  Thresholds meeting constraints: {optimal_thresholds_info['n_meeting_constraints']}")

if optimal_thresholds_info['optimal_constrained'] is not None:
    print(f"\n  RECOMMENDED THRESHOLD (Best Balance within Constraints):")
    print(f"    Threshold: {optimal_thresholds_info['optimal_constrained']['threshold']:.4f}")
    print(f"    Sensitivity: {optimal_thresholds_info['optimal_constrained']['sensitivity']:.3f}")
    print(f"    Specificity: {optimal_thresholds_info['optimal_constrained']['specificity']:.3f}")
    print(f"    Accuracy: {optimal_thresholds_info['optimal_constrained']['accuracy']:.3f}")
    print(f"    PPV: {optimal_thresholds_info['optimal_constrained']['ppv']:.3f}")
    print(f"    NPV: {optimal_thresholds_info['optimal_constrained']['npv']:.3f}")

    print(f"\n  All thresholds meeting constraints:")
    constraint_display = optimal_thresholds_info['constraint_thresholds'][
        ['threshold', 'sensitivity', 'specificity', 'accuracy', 'ppv', 'npv']
    ].copy()
    constraint_display = constraint_display.round(4)
    print(constraint_display.to_string(index=False))
else:
    print(f"  ⚠ No thresholds meet both constraints (Sensitivity >= 0.85 AND Specificity >= 0.75)")
    print(f"  Suggested: Use Youden's J optimal threshold or relax constraints")

# %%
# INTERACTIVE ROC CURVE ANALYSIS

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplcursors
import seaborn as sns
# %matplotlib widget

def compute_roc_metrics_for_data(surprise_data, thresholds):
    """Compute ROC metrics for a dataset across all thresholds."""
    roc_metrics = []
    for z_threshold in thresholds:
        TP = surprise_data[(surprise_data.risk > 1) & (surprise_data.z < z_threshold)].shape[0]
        FP = surprise_data[(surprise_data.risk <= 1) & (surprise_data.z < z_threshold)].shape[0]
        FN = surprise_data[(surprise_data.risk > 1) & (surprise_data.z >= z_threshold)].shape[0]
        TN = surprise_data[(surprise_data.risk <= 1) & (surprise_data.z >= z_threshold)].shape[0]

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        fpr = 1 - specificity
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        youden_j = sensitivity + specificity - 1

        roc_metrics.append({
            'threshold': z_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'ppv': ppv,
            'npv': npv,
            'youden_j': youden_j,
            'tp': TP,
            'fp': FP,
            'fn': FN,
            'tn': TN
        })
    return pd.DataFrame(roc_metrics)


def plot_interactive_roc(surprise_data, thresholds, optimal_threshold_youden=None, output_prefix='roc_analysis',
                         baseline_data=None, baseline_optimal_threshold=None):
    """
    Create interactive ROC curve visualization with threshold exploration.

    Parameters:
    -----------
    surprise_data : DataFrame
        Combined test set with 'z' and 'risk' columns (transformed data if in comparison mode)
    thresholds : list
        List of thresholds to evaluate
    optimal_threshold_youden : float
        Optimal threshold from Youden's J (will be highlighted)
    output_prefix : str
        Prefix for output files
    baseline_data : DataFrame, optional
        Baseline (non-transformed) test set for comparison
    baseline_optimal_threshold : float, optional
        Optimal threshold for baseline data
    """

    # Compute ROC metrics for transformed data
    roc_df = compute_roc_metrics_for_data(surprise_data, thresholds)

    # Compute ROC metrics for baseline data if provided
    baseline_roc_df = None
    if baseline_data is not None:
        baseline_roc_df = compute_roc_metrics_for_data(baseline_data, thresholds)

    # Determine figure layout based on whether we have baseline data
    if baseline_data is not None:
        # 3 rows: 2x2 grid on top, comparison panel on bottom
        fig = plt.figure(figsize=(12, 15), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
        ax_roc = fig.add_subplot(gs[0, 0])
        ax_tradeoff = fig.add_subplot(gs[0, 1])
        ax_pv = fig.add_subplot(gs[1, 0])
        ax_cm = fig.add_subplot(gs[1, 1])
        ax_comparison = fig.add_subplot(gs[2, 0])  # Span both columns
    else:
        # Original 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
        ax_roc = axes[0, 0]
        ax_tradeoff = axes[0, 1]
        ax_pv = axes[1, 0]
        ax_cm = axes[1, 1]

    # 1. ROC CURVE (Transformed data)
    line_roc, = ax_roc.plot(roc_df['fpr'], roc_df['sensitivity'], 'b-', linewidth=2, label='ROC Curve')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    # Highlight Youden's optimal point
    if optimal_threshold_youden is not None:
        # Find closest threshold if exact match not found
        closest_idx = (roc_df['threshold'] - optimal_threshold_youden).abs().idxmin()
        youden_row = roc_df.loc[closest_idx]
        ax_roc.plot(youden_row['fpr'], youden_row['sensitivity'], 'r*', markersize=15,
                   label=f"Youden's J Optimal\n(Threshold: {optimal_threshold_youden:.3f})")

    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax_roc.set_title('ROC Curve: Explore Trade-offs', fontsize=12, fontweight='bold')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ax_roc.set_xticks(np.arange(0, 1.1, 0.1))
    ax_roc.set_yticks(np.arange(0, 1.1, 0.1))

    # 2. SENSITIVITY vs SPECIFICITY TRADE-OFF
    line_sens, = ax_tradeoff.plot(roc_df['threshold'], roc_df['sensitivity'], 'g-', linewidth=2, label='Sensitivity')
    line_spec, = ax_tradeoff.plot(roc_df['threshold'], roc_df['specificity'], 'r-', linewidth=2, label='Specificity')
    line_youden, = ax_tradeoff.plot(roc_df['threshold'], roc_df['youden_j'], 'b--', linewidth=2, label="Youden's J")

    if optimal_threshold_youden is not None:
        # Find closest threshold if exact match not found
        closest_idx = (roc_df['threshold'] - optimal_threshold_youden).abs().idxmin()
        youden_row = roc_df.loc[closest_idx]
        ax_tradeoff.axvline(optimal_threshold_youden, color='r', linestyle='--', alpha=0.5,
                           label=f"Youden Optimal: {optimal_threshold_youden:.3f}")
        ax_tradeoff.plot(optimal_threshold_youden, youden_row['sensitivity'], 'g*', markersize=12)
        ax_tradeoff.plot(optimal_threshold_youden, youden_row['specificity'], 'r*', markersize=12)

    ax_tradeoff.set_xlabel('Z-Score Threshold', fontsize=11, fontweight='bold')
    ax_tradeoff.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax_tradeoff.set_title('Se/Sp Trade-off Across Thresholds', fontsize=12, fontweight='bold')
    ax_tradeoff.legend(loc='best', fontsize=9)
    ax_tradeoff.grid(True, alpha=0.3)
    ax_tradeoff.set_xticks(np.arange(-3.0, 3.2, 0.5))
    ax_tradeoff.set_yticks(np.arange(-0.5, 1.1, 0.1))

    # 3. PPV vs NPV TRADE-OFF
    line_ppv, = ax_pv.plot(roc_df['threshold'], roc_df['ppv'], 'purple', linewidth=2, label='PPV (Positive Pred. Value)')
    line_npv, = ax_pv.plot(roc_df['threshold'], roc_df['npv'], 'orange', linewidth=2, label='NPV (Negative Pred. Value)')

    if optimal_threshold_youden is not None:
        # Find closest threshold if exact match not found
        closest_idx = (roc_df['threshold'] - optimal_threshold_youden).abs().idxmin()
        youden_row = roc_df.loc[closest_idx]
        ax_pv.axvline(optimal_threshold_youden, color='r', linestyle='--', alpha=0.5)
        ax_pv.plot(optimal_threshold_youden, youden_row['ppv'], 'o', color='purple', markersize=10)
        ax_pv.plot(optimal_threshold_youden, youden_row['npv'], 'o', color='orange', markersize=10)

    ax_pv.set_xlabel('Z-Score Threshold', fontsize=11, fontweight='bold')
    ax_pv.set_ylabel('Predictive Value', fontsize=11, fontweight='bold')
    ax_pv.set_title('PPV vs NPV: How Trustworthy Are Results?', fontsize=12, fontweight='bold')
    ax_pv.legend(loc='best')
    ax_pv.grid(True, alpha=0.3)
    ax_pv.set_ylim([0, 1])
    ax_pv.set_xticks(np.arange(-3.0, 3.2, 0.5))
    ax_pv.set_yticks(np.arange(0, 1.1, 0.1))

    # 4. CONFUSION MATRIX HEATMAP FOR SELECTED THRESHOLD
    ax_cm.axis('off')

    # Display text information about thresholds
    info_text = "THRESHOLD SELECTION GUIDE:\n\n"
    info_text += f"Total At-Risk Cases: {(surprise_data.risk > 1).sum()}\n"
    info_text += f"Total Normal Cases: {(surprise_data.risk <= 1).sum()}\n\n"

    info_text += "PRESET THRESHOLDS:\n"
    info_text += "─" * 50 + "\n"

    # Show a few key thresholds
    key_thresholds = [
        ('High Sensitivity (92%+)', 0.92),
        ('High Specificity (85%+)', 0.85),
        ("Youden's J Optimal", None)
    ]

    for label, target_sens in key_thresholds:
        if target_sens is None:  # Youden's J
            if optimal_threshold_youden is not None:
                # Find closest threshold if exact match not found
                closest_idx = (roc_df['threshold'] - optimal_threshold_youden).abs().idxmin()
                row = roc_df.loc[closest_idx]
                info_text += f"\n{label}:\n"
                info_text += f"  Threshold: {row['threshold']:7.3f}\n"
                info_text += f"  Se: {row['sensitivity']:.3f} | Sp: {row['specificity']:.3f}\n"
                info_text += f"  PPV: {row['ppv']:.3f} | NPV: {row['npv']:.3f}\n"
        else:
            # Find closest threshold to target
            closest_idx = (roc_df['sensitivity'] - target_sens).abs().idxmin()
            row = roc_df.loc[closest_idx]
            info_text += f"\n{label}:\n"
            info_text += f"  Threshold: {row['threshold']:7.3f}\n"
            info_text += f"  Se: {row['sensitivity']:.3f} | Sp: {row['specificity']:.3f}\n"
            info_text += f"  PPV: {row['ppv']:.3f} | NPV: {row['npv']:.3f}\n"

    ax_cm.text(0.05, 0.95, info_text, transform=ax_cm.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # INTERACTIVE HOVER CURSORS (mplcursors)
    # =========================================================================

    # ROC Curve hover - show all metrics at this point
    cursor_roc = mplcursors.cursor(line_roc, hover=True)
    @cursor_roc.connect("add")
    def on_add_roc(sel):
        idx = int(sel.index)
        row = roc_df.iloc[idx]
        sel.annotation.set_text(
            f"Threshold: {row['threshold']:.3f}\n"
            f"Sensitivity: {row['sensitivity']:.3f}\n"
            f"Specificity: {row['specificity']:.3f}\n"
            f"FPR: {row['fpr']:.3f}\n"
            f"Youden's J: {row['youden_j']:.3f}"
        )
        sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.95)

    # Se/Sp Trade-off hover - comprehensive metrics
    cursor_sens = mplcursors.cursor(line_sens, hover=True)
    @cursor_sens.connect("add")
    def on_add_sens(sel):
        idx = int(sel.index)
        row = roc_df.iloc[idx]
        sel.annotation.set_text(
            f"Z-Threshold: {row['threshold']:.3f}\n"
            f"─────────────────\n"
            f"Sensitivity: {row['sensitivity']:.3f}\n"
            f"Specificity: {row['specificity']:.3f}\n"
            f"Youden's J: {row['youden_j']:.3f}\n"
            f"─────────────────\n"
            f"PPV: {row['ppv']:.3f}\n"
            f"NPV: {row['npv']:.3f}\n"
            f"─────────────────\n"
            f"TP: {row['tp']:.0f} | FP: {row['fp']:.0f}\n"
            f"FN: {row['fn']:.0f} | TN: {row['tn']:.0f}"
        )
        sel.annotation.get_bbox_patch().set(fc="lightgreen", alpha=0.95)

    cursor_spec = mplcursors.cursor(line_spec, hover=True)
    @cursor_spec.connect("add")
    def on_add_spec(sel):
        idx = int(sel.index)
        row = roc_df.iloc[idx]
        sel.annotation.set_text(
            f"Z-Threshold: {row['threshold']:.3f}\n"
            f"─────────────────\n"
            f"Sensitivity: {row['sensitivity']:.3f}\n"
            f"Specificity: {row['specificity']:.3f}\n"
            f"Youden's J: {row['youden_j']:.3f}\n"
            f"─────────────────\n"
            f"PPV: {row['ppv']:.3f}\n"
            f"NPV: {row['npv']:.3f}\n"
            f"─────────────────\n"
            f"TP: {row['tp']:.0f} | FP: {row['fp']:.0f}\n"
            f"FN: {row['fn']:.0f} | TN: {row['tn']:.0f}"
        )
        sel.annotation.get_bbox_patch().set(fc="lightcoral", alpha=0.95)

    cursor_youden = mplcursors.cursor(line_youden, hover=True)
    @cursor_youden.connect("add")
    def on_add_youden(sel):
        idx = int(sel.index)
        row = roc_df.iloc[idx]
        sel.annotation.set_text(
            f"Z-Threshold: {row['threshold']:.3f}\n"
            f"─────────────────\n"
            f"Youden's J: {row['youden_j']:.3f}\n"
            f"Sensitivity: {row['sensitivity']:.3f}\n"
            f"Specificity: {row['specificity']:.3f}"
        )
        sel.annotation.get_bbox_patch().set(fc="lightblue", alpha=0.95)

    # PPV/NPV hover
    cursor_ppv = mplcursors.cursor(line_ppv, hover=True)
    @cursor_ppv.connect("add")
    def on_add_ppv(sel):
        idx = int(sel.index)
        row = roc_df.iloc[idx]
        sel.annotation.set_text(
            f"Z-Threshold: {row['threshold']:.3f}\n"
            f"─────────────────\n"
            f"PPV: {row['ppv']:.3f}\n"
            f"NPV: {row['npv']:.3f}\n"
            f"─────────────────\n"
            f"Se: {row['sensitivity']:.3f} | Sp: {row['specificity']:.3f}"
        )
        sel.annotation.get_bbox_patch().set(fc="plum", alpha=0.95)

    cursor_npv = mplcursors.cursor(line_npv, hover=True)
    @cursor_npv.connect("add")
    def on_add_npv(sel):
        idx = int(sel.index)
        row = roc_df.iloc[idx]
        sel.annotation.set_text(
            f"Z-Threshold: {row['threshold']:.3f}\n"
            f"─────────────────\n"
            f"PPV: {row['ppv']:.3f}\n"
            f"NPV: {row['npv']:.3f}\n"
            f"─────────────────\n"
            f"Se: {row['sensitivity']:.3f} | Sp: {row['specificity']:.3f}"
        )
        sel.annotation.get_bbox_patch().set(fc="moccasin", alpha=0.95)

    print("✓ Interactive hover enabled - move mouse over lines to see metrics")

    # =========================================================================
    # 5. COMPARISON ROC CURVE (Baseline vs Transformed) - Only if baseline provided
    # =========================================================================
    if baseline_data is not None and baseline_roc_df is not None:
        # Compute AUC for both curves
        baseline_auc = auc(baseline_roc_df['fpr'], baseline_roc_df['sensitivity'])
        transformed_auc = auc(roc_df['fpr'], roc_df['sensitivity'])

        # Plot both ROC curves
        line_baseline, = ax_comparison.plot(baseline_roc_df['fpr'], baseline_roc_df['sensitivity'],
                                            'r-', linewidth=2.5, label=f'Baseline (AUC={baseline_auc:.3f})')
        line_transformed, = ax_comparison.plot(roc_df['fpr'], roc_df['sensitivity'],
                                               'b-', linewidth=2.5, label=f'Transformed (AUC={transformed_auc:.3f})')
        ax_comparison.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

        # Highlight optimal points
        if baseline_optimal_threshold is not None:
            closest_idx = (baseline_roc_df['threshold'] - baseline_optimal_threshold).abs().idxmin()
            baseline_opt = baseline_roc_df.loc[closest_idx]
            ax_comparison.plot(baseline_opt['fpr'], baseline_opt['sensitivity'], 'r*', markersize=15,
                              label=f"Baseline Optimal (Se={baseline_opt['sensitivity']:.2f}, Sp={1-baseline_opt['fpr']:.2f})")

        if optimal_threshold_youden is not None:
            closest_idx = (roc_df['threshold'] - optimal_threshold_youden).abs().idxmin()
            transformed_opt = roc_df.loc[closest_idx]
            ax_comparison.plot(transformed_opt['fpr'], transformed_opt['sensitivity'], 'b*', markersize=15,
                              label=f"Transformed Optimal (Se={transformed_opt['sensitivity']:.2f}, Sp={1-transformed_opt['fpr']:.2f})")

        ax_comparison.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
        ax_comparison.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
        ax_comparison.set_title('ROC Comparison: Baseline vs Inverse Transformed Features', fontsize=12, fontweight='bold')
        ax_comparison.legend(loc='lower right', fontsize=9)
        ax_comparison.grid(True, alpha=0.3)
        ax_comparison.set_xlim([0, 1])
        ax_comparison.set_ylim([0, 1])
        ax_comparison.set_xticks(np.arange(0, 1.1, 0.1))
        ax_comparison.set_yticks(np.arange(0, 1.1, 0.1))

        # Add text annotation showing improvement
        auc_improvement = transformed_auc - baseline_auc
        improvement_text = f"AUC Improvement: {auc_improvement:+.3f} ({auc_improvement/baseline_auc*100:+.1f}%)"
        # removing because of clutter
#         ax_comparison.text(0.98, 0.02, improvement_text, transform=ax_comparison.transAxes,
#                           fontsize=10, fontweight='bold', ha='right', va='bottom',
#                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # Add interactive hover for comparison curves
        cursor_baseline = mplcursors.cursor(line_baseline, hover=True)
        @cursor_baseline.connect("add")
        def on_add_baseline(sel):
            idx = int(sel.index)
            row = baseline_roc_df.iloc[idx]
            sel.annotation.set_text(
                f"BASELINE\n"
                f"Threshold: {row['threshold']:.3f}\n"
                f"Sensitivity: {row['sensitivity']:.3f}\n"
                f"Specificity: {1-row['fpr']:.3f}\n"
                f"Youden's J: {row['youden_j']:.3f}"
            )
            sel.annotation.get_bbox_patch().set(fc="lightsalmon", alpha=0.95)

        cursor_transformed = mplcursors.cursor(line_transformed, hover=True)
        @cursor_transformed.connect("add")
        def on_add_transformed(sel):
            idx = int(sel.index)
            row = roc_df.iloc[idx]
            sel.annotation.set_text(
                f"TRANSFORMED\n"
                f"Threshold: {row['threshold']:.3f}\n"
                f"Sensitivity: {row['sensitivity']:.3f}\n"
                f"Specificity: {1-row['fpr']:.3f}\n"
                f"Youden's J: {row['youden_j']:.3f}"
            )
            sel.annotation.get_bbox_patch().set(fc="lightblue", alpha=0.95)

        print("✓ Comparison ROC panel added (baseline vs transformed)")

    # Only use tight_layout when not using GridSpec (baseline_data is None)
    if baseline_data is None:
        plt.tight_layout()
    plt.savefig(f'{output_prefix}_interactive.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {output_prefix}_interactive.png")

    return roc_df, fig


def display_threshold_metrics(surprise_data, threshold, label="Selected Threshold"):
    """
    Display detailed metrics for a specific threshold.

    Parameters:
    -----------
    surprise_data : DataFrame
        Combined test set with 'z' and 'risk' columns
    threshold : float
        Z-score threshold to evaluate
    label : str
        Label for display
    """

    TP = surprise_data[(surprise_data.risk > 1) & (surprise_data.z < threshold)].shape[0]
    FP = surprise_data[(surprise_data.risk <= 1) & (surprise_data.z < threshold)].shape[0]
    FN = surprise_data[(surprise_data.risk > 1) & (surprise_data.z >= threshold)].shape[0]
    TN = surprise_data[(surprise_data.risk <= 1) & (surprise_data.z >= threshold)].shape[0]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    youden_j = sensitivity + specificity - 1

    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"Threshold: {threshold:.4f}\n")
    print("PERFORMANCE METRICS:")
    print(f"  Sensitivity (Se):  {sensitivity:.4f} ({TP}/{TP+FN} caught at-risk)")
    print(f"  Specificity (Sp):  {specificity:.4f} ({TN}/{TN+FP} normal identified)")
    print(f"  PPV:               {ppv:.4f} (when flagged at-risk, {ppv:.1%} actually are)")
    print(f"  NPV:               {npv:.4f} (when flagged normal, {npv:.1%} actually are)")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Youden's J:        {youden_j:.4f}\n")
    print("CONFUSION MATRIX:")
    print(f"  TP={TP:3d}  FN={FN:3d}  |  Sensitivity = {TP}/{TP+FN}")
    print(f"  FP={FP:3d}  TN={TN:3d}  |  Specificity = {TN}/{TN+FP}")


# %%
# K-FOLD SUMMARY STATISTICS

print(f"\n{'='*80}")
print("K-FOLD CROSS-VALIDATION SUMMARY")
print(f"{'='*80}\n")

# Compute per-fold AUC values
fold_aucs = []
print("Per-Fold AUC Scores:")
for fold_idx, fold_df in enumerate(fold_results):
    fold_test = fold_df[fold_df.category==1].copy()
    fold_tpr, fold_fpr = compute_roc(thresholds, fold_test)
    fold_auc = auc(fold_fpr, fold_tpr)
    fold_aucs.append(fold_auc)
    print(f"  Fold {fold_idx+1}/{K_FOLDS}: AUC = {fold_auc:.4f}")

print(f"\nOverall K-Fold Metrics:")
print(f"  Mean AUC: {np.mean(fold_aucs):.4f}")
print(f"  Std AUC:  {np.std(fold_aucs):.4f}")
print(f"  Min AUC:  {np.min(fold_aucs):.4f}")
print(f"  Max AUC:  {np.max(fold_aucs):.4f}")

# Compute aggregated metrics on combined test set
print(f"\nCombined Test Set Metrics (All {K_FOLDS} Folds):")
print(f"  Total test subjects: {combined_test['infant'].nunique()}")
print(f"  Total test data points: {len(combined_test)}")
print(f"  Risk=2 (moderate): {(combined_test.risk==2).sum()}")
print(f"  Risk=3 (high): {(combined_test.risk==3).sum()}")
print(f"  Risk=1 (low/normal): {(combined_test.risk<=1).sum()}")
print(f"  Mean AUC (across all folds): {np.mean(fold_aucs):.4f}")

print(f"\n{'='*80}")
print("K-Fold cross-validation and ROC analysis complete")
print(f"{'='*80}\n")

# %%
# GENERATE INTERACTIVE ROC CURVE VISUALIZATIONS
# %matplotlib widget
INTERACTIVE_MODEL = '20-Feature (NEW)'

print(f"\n{'='*80}")
print("GENERATING INTERACTIVE ROC CURVE VISUALIZATIONS")
print(f"{'='*80}\n")

# Validate selected model
if INTERACTIVE_MODEL not in all_feature_set_results:
    print(f"⚠️  Warning: Model '{INTERACTIVE_MODEL}' not found!")
    print(f"   Available models: {list(all_feature_set_results.keys())}")
    print(f"   Using default: '20-Feature'\n")
    selected_model = '20-Feature'
else:
    selected_model = INTERACTIVE_MODEL

print(f"Selected Model for Interactive Analysis: {selected_model}")
print(f"{'='*80}\n")

# Get the combined test set for the selected model
interactive_combined_test = all_feature_set_results[selected_model]['combined_test']

# Compute optimal thresholds for the selected model
interactive_optimal_thresholds = find_optimal_thresholds(
    interactive_combined_test[['z', 'risk']].copy(),
    min_sensitivity=0.85,
    min_specificity=0.75
)

# Generate ROC curve with all thresholds for the selected model
# Create a clean prefix from model name
model_prefix = selected_model.lower().replace(' ', '_').replace('(', '').replace(')', '')

# Check if we have baseline data for comparison (only in comparison mode)
baseline_test_data = None
baseline_opt_threshold = None
if RUN_COMPARISON_MODE and 'baseline_results' in dir() and selected_model in baseline_results:
    # Get baseline combined test data
    baseline_fold_results = baseline_results[selected_model]['fold_results']
    baseline_test_list = []
    for fold_idx, fold_df in enumerate(baseline_fold_results):
        test_data = fold_df[fold_df.category==1].copy()
        test_data['fold'] = fold_idx
        baseline_test_list.append(test_data)
    baseline_test_data = pd.concat(baseline_test_list, ignore_index=True)

    # Compute optimal threshold for baseline
    baseline_thresholds = find_optimal_thresholds(
        baseline_test_data[['z', 'risk']].copy(),
        min_sensitivity=0.85,
        min_specificity=0.75
    )
    baseline_opt_threshold = baseline_thresholds['optimal_youden']['threshold']
    print(f"Baseline data available for comparison (n={len(baseline_test_data)})")
    print(f"Baseline Youden's J Optimal Threshold: {baseline_opt_threshold:.4f}")

roc_curve_df, roc_fig = plot_interactive_roc(
    interactive_combined_test,
    thresholds,
    optimal_threshold_youden=interactive_optimal_thresholds['optimal_youden']['threshold'],
    output_prefix=f'roc_analysis_{model_prefix}',
    baseline_data=baseline_test_data,
    baseline_optimal_threshold=baseline_opt_threshold
)

print(f"\n✓ ROC visualization generated for: {selected_model}")
print(f"✓ Youden's J Optimal Threshold: {interactive_optimal_thresholds['optimal_youden']['threshold']:.4f}")
print("\nHover over the lines to see metrics at each threshold.")
print("For interactive hover in Jupyter, use: %matplotlib widget")
print("Use the generated plots to explore the Se/Sp trade-off and select your preferred threshold.\n")

# %%
# INTERACTIVE THRESHOLD EXPLORATION
# Uncomment and modify the threshold below to explore metrics for any z-score threshold

custom_threshold=-0.234
print("EXAMPLE: Metrics for different thresholds")
print(f"(Model: {selected_model})")
print("-" * 80)

# Display metrics for a few strategic thresholds
example_thresholds = [
    (interactive_optimal_thresholds['optimal_youden']['threshold'], "Youden's J Optimal"),
    (-0.5, "High Sensitivity Focus (z < -0.5)"),
    (0.0, "Neutral Threshold (z < 0.0)"),
    (custom_threshold, "User-defined threshold"),
]

for threshold, description in example_thresholds:
    display_threshold_metrics(interactive_combined_test, threshold, description)

print("\n" + "="*80)
print("TO EXPLORE DIFFERENT THRESHOLDS:")
print("="*80)
print("Uncomment the line below and change the threshold value to explore any z-score:")
print(">>> display_threshold_metrics(interactive_combined_test, threshold=YOUR_THRESHOLD_HERE, label='Your Label')")
print("\nTO SWITCH TO A DIFFERENT MODEL:")
print(">>> Edit line 68: INTERACTIVE_MODEL = '15-Feature (NEW)'  # or other model")
print(">>> Then re-run the script")
print("="*80 + "\n")

# %%
# COMPREHENSIVE MULTI-MODEL ANALYSIS
# Compute metrics for all feature sets using their own k-fold CV results

print(f"\n{'='*80}")
print("COMPREHENSIVE FEATURE SET COMPARISON")
print(f"{'='*80}\n")

# Store results for all models
all_model_results = {}

for model_name in feature_sets.keys():
    print(f"\n{model_name}")
    print("-" * 80)

    # Get the combined test set for this feature set
    feature_set_combined_test = all_feature_set_results[model_name]['combined_test']

    # Analyze optimal thresholds using this feature set's combined test set
    model_thresholds = find_optimal_thresholds(
        feature_set_combined_test[['z', 'risk']].copy(),
        min_sensitivity=0.85,
        min_specificity=0.75
    )

    optimal = model_thresholds['optimal_youden']

    # Compute AUC for this feature set
    fold_results_this_set = all_feature_set_results[model_name]['fold_results']
    fold_aucs = []
    for fold_df in fold_results_this_set:
        fold_test = fold_df[fold_df.category==1].copy()
        if len(fold_test) > 0:
            fold_tpr, fold_fpr = compute_roc(thresholds, fold_test)
            fold_auc = auc(fold_fpr, fold_tpr)
            fold_aucs.append(fold_auc)

    mean_auc = np.mean(fold_aucs) if fold_aucs else np.nan

    all_model_results[model_name] = {
        'n_features': len(feature_sets[model_name]),
        'auc': mean_auc,
        'sensitivity': optimal['sensitivity'],
        'specificity': optimal['specificity'],
        'accuracy': optimal['accuracy'],
        'ppv': optimal['ppv'],
        'npv': optimal['npv'],
        'youdens_j': optimal['youden_j'],
        'optimal_threshold': optimal['threshold'],
        'tp': optimal['tp'],
        'tn': optimal['tn'],
        'fp': optimal['fp'],
        'fn': optimal['fn']
    }

    print(f"  Threshold: {optimal['threshold']:.4f}")
    print(f"  Sensitivity: {optimal['sensitivity']:.3f}")
    print(f"  Specificity: {optimal['specificity']:.3f}")
    print(f"  Accuracy: {optimal['accuracy']:.3f}")
    print(f"  Youden's J: {optimal['youden_j']:.3f}")
    print(f"  AUC: {mean_auc:.4f}")
    print(f"  PPV: {optimal['ppv']:.3f}")
    print(f"  NPV: {optimal['npv']:.3f}")

# %%
# PRINT SUMMARY TABLE

print(f"\n{'='*80}")
print("FEATURE SET COMPARISON SUMMARY (K-FOLD CV)")
print(f"{'='*80}\n")

summary_df = pd.DataFrame(all_model_results).T
summary_df = summary_df.round(4)
print(summary_df[['n_features', 'auc', 'sensitivity', 'specificity', 'accuracy', 'youdens_j', 'optimal_threshold']].to_string())

# Also save to CSV
summary_df.to_csv('feature_set_comparison_metrics.csv')
print(f"\nResults saved to: feature_set_comparison_metrics.csv")

# %%
# TRANSFORMATION COMPARISON (if comparison mode enabled)

if RUN_COMPARISON_MODE:
    print(f"\n{'='*80}")
    print("TRANSFORMATION COMPARISON: BASELINE vs INVERSE TRANSFORM")
    print(f"{'='*80}\n")

    # Compute metrics for baseline results
    baseline_model_results = {}
    for model_name in feature_sets.keys():
        baseline_combined_test = baseline_results[model_name]['combined_test']
        baseline_thresholds_result = find_optimal_thresholds(
            baseline_combined_test[['z', 'risk']].copy(),
            min_sensitivity=0.85,
            min_specificity=0.75
        )
        optimal = baseline_thresholds_result['optimal_youden']

        # Compute AUC for baseline
        baseline_fold_results = baseline_results[model_name]['fold_results']
        fold_aucs = []
        for fold_df in baseline_fold_results:
            fold_test = fold_df[fold_df.category==1].copy()
            if len(fold_test) > 0:
                fold_tpr, fold_fpr = compute_roc(thresholds, fold_test)
                fold_auc = auc(fold_fpr, fold_tpr)
                fold_aucs.append(fold_auc)
        mean_auc = np.mean(fold_aucs) if fold_aucs else np.nan

        baseline_model_results[model_name] = {
            'auc': mean_auc,
            'sensitivity': optimal['sensitivity'],
            'specificity': optimal['specificity'],
            'youdens_j': optimal['youden_j'],
        }

    # Compute metrics for transformed results (already done in all_model_results)
    transformed_model_results = {}
    for model_name in feature_sets.keys():
        transformed_model_results[model_name] = {
            'auc': all_model_results[model_name]['auc'],
            'sensitivity': all_model_results[model_name]['sensitivity'],
            'specificity': all_model_results[model_name]['specificity'],
            'youdens_j': all_model_results[model_name]['youdens_j'],
        }

    # Create comparison table
    print(f"{'Feature Set':<22} | {'Metric':<12} | {'Baseline':>10} | {'Transformed':>12} | {'Change':>10}")
    print("-" * 80)

    for model_name in ['20-Feature', '15-Feature', '19-Feature',
                       '20-Feature (LEGACY)', '18-Feature (LEGACY)', '15-Feature (LEGACY)', 'All 38 Features', 'All 59 Features']:
        base = baseline_model_results[model_name]
        trans = transformed_model_results[model_name]

        print(f"\n{model_name}")
        print(f"  {'AUC':<20} | {base['auc']:>10.4f} | {trans['auc']:>12.4f} | {trans['auc']-base['auc']:>+10.4f}")
        print(f"  {'Sensitivity':<20} | {base['sensitivity']:>10.4f} | {trans['sensitivity']:>12.4f} | {trans['sensitivity']-base['sensitivity']:>+10.4f}")
        print(f"  {'Specificity':<20} | {base['specificity']:>10.4f} | {trans['specificity']:>12.4f} | {trans['specificity']-base['specificity']:>+10.4f}")
        print(f"  {'Youden J':<20} | {base['youdens_j']:>10.4f} | {trans['youdens_j']:>12.4f} | {trans['youdens_j']-base['youdens_j']:>+10.4f}")

    # Summary comparison for best model (20-Feature
    print(f"\n{'='*80}")
    print("SUMMARY: 20-Feature - Best Model")
    print(f"{'='*80}")
    base = baseline_model_results['20-Feature']
    trans = transformed_model_results['20-Feature']

    print(f"\n{'Metric':<15} | {'Baseline':>12} | {'Transformed':>12} | {'Improvement':>12}")
    print("-" * 60)
    print(f"{'AUC':<15} | {base['auc']:>12.4f} | {trans['auc']:>12.4f} | {trans['auc']-base['auc']:>+12.4f}")
    print(f"{'Sensitivity':<15} | {base['sensitivity']*100:>11.1f}% | {trans['sensitivity']*100:>11.1f}% | {(trans['sensitivity']-base['sensitivity'])*100:>+11.1f}%")
    print(f"{'Specificity':<15} | {base['specificity']*100:>11.1f}% | {trans['specificity']*100:>11.1f}% | {(trans['specificity']-base['specificity'])*100:>+11.1f}%")
    print(f"{'Youden J':<15} | {base['youdens_j']:>12.4f} | {trans['youdens_j']:>12.4f} | {trans['youdens_j']-base['youdens_j']:>+12.4f}")

    # Save comparison to CSV
    comparison_data = []
    for model_name in feature_sets.keys():
        base = baseline_model_results[model_name]
        trans = transformed_model_results[model_name]
        comparison_data.append({
            'model': model_name,
            'baseline_auc': base['auc'],
            'transformed_auc': trans['auc'],
            'auc_change': trans['auc'] - base['auc'],
            'baseline_sensitivity': base['sensitivity'],
            'transformed_sensitivity': trans['sensitivity'],
            'sensitivity_change': trans['sensitivity'] - base['sensitivity'],
            'baseline_specificity': base['specificity'],
            'transformed_specificity': trans['specificity'],
            'specificity_change': trans['specificity'] - base['specificity'],
            'baseline_youdens_j': base['youdens_j'],
            'transformed_youdens_j': trans['youdens_j'],
            'youdens_j_change': trans['youdens_j'] - base['youdens_j'],
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('transformation_comparison_metrics.csv', index=False)
    print(f"\nComparison results saved to: transformation_comparison_metrics.csv")

# %%
feature_sets.keys()

# %%
