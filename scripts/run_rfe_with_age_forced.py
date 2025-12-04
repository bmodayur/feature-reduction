#!/usr/bin/env python3
"""
Run Enhanced Adaptive RFE with forced age_in_weeks inclusion.

This script implements a modified RFE workflow:
1. Run RFE on 58 movement features (excluding age_in_weeks)
2. Select top 19 features from RFE results
3. Add age_in_weeks to get exactly 20 features
4. Compute surprise scores and ROC metrics on the 20-feature model

This ensures age_in_weeks is always included in the final feature set
while allowing RFE to optimize selection of movement features.
"""
from early_markers.cribsy.common.bayes import BayesianData
from early_markers.cribsy.common.constants import PKL_DIR, JSON_DIR, N_ESTIMATORS, RAND_STATE
from sklearn.ensemble import RandomForestRegressor
import polars as pl
import json
from pathlib import Path

print("=" * 80)
print("MODIFIED RFE: FORCED age_in_weeks INCLUSION")
print("=" * 80)

# Step 1: Initialize BayesianData with real data
# IMPORTANT: Use the same data file as compute_roc_after_rfe.py
print("\nStep 1: Initialize BayesianData")
print("-" * 80)
bd = BayesianData(base_file="features_merged_20251121_091511.pkl")
print(f"✓ Loaded: {bd.base_file}")
print(f"  Total infants: {bd.base_wide.shape[0]}")
print(f"  Total features: {len(bd.base_features)}")

# Step 2: Separate age_in_weeks from movement features
print("\nStep 2: Separate age_in_weeks from movement features")
print("-" * 80)
features_no_age = [f for f in bd.base_features if f != 'age_in_weeks']
print(f"  Movement features: {len(features_no_age)}")
print(f"  Age feature: age_in_weeks (will be added later)")

# Step 3: Run Enhanced Adaptive RFE on movement features only
print("\nStep 3: Run Enhanced Adaptive RFE on 58 movement features")
print("-" * 80)
print("  Configuration:")
print("    • Method: Enhanced Adaptive RFE")
print("    • Trials: 50 (parallel)")
print("    • CV Folds: 15")
print("    • Statistical testing: Binomial test (α=0.05)")
print("    • Target: ~30-35 features (will be reduced to 19)")
print()
print("  ⏳ Starting RFE (this will take 30-60 minutes)...")
print("     Press Ctrl+C to cancel if needed")
print()

# Run adaptive RFE on features without age
selected_features_no_age = bd.run_adaptive_rfe(
    model_prefix='rfe_no_age',
    features=features_no_age,
    tot_k=len(bd.base_features)
)

print(f"\n✓ RFE completed!")
print(f"  Selected features: {len(selected_features_no_age)}")

# Step 4: If we have more than 19 features, reduce to top 19 by importance
print("\nStep 4: Select top 19 features by Random Forest importance")
print("-" * 80)

if len(selected_features_no_age) > 19:
    print(f"  RFE returned {len(selected_features_no_age)} features")
    print(f"  Reducing to top 19 by importance...")

    # Get training data with selected features
    frames = bd._frames.get(f'rfe_no_age_k_{len(selected_features_no_age)}')
    if frames is None:
        raise ValueError(f"Could not find frames for rfe_no_age_k_{len(selected_features_no_age)}")

    df_train = frames.train
    df_surprise = frames.train_surprise
    df_rfe = df_train.join(df_surprise, on="infant", how="inner").sort(["infant", "feature"])

    # Pivot to wide format for RF
    df_x = df_rfe.pivot(index='infant', on='feature', values='value').drop("infant")
    y = df_rfe.group_by('infant', maintain_order=True).agg(pl.col('z').first()).select("z").to_numpy()[:,0]

    # Train RF to get feature importances (use Regressor since y is continuous z-scores)
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RAND_STATE, n_jobs=8)
    rf.fit(df_x.to_numpy(), y)

    # Rank features by importance
    feature_importance = list(zip(selected_features_no_age, rf.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Take top 19
    selected_19 = [f for f, imp in feature_importance[:19]]

    print(f"  ✓ Selected top 19 features:")
    for i, (feat, imp) in enumerate(feature_importance[:19], 1):
        print(f"      {i:2d}. {feat:40s} (importance: {imp:.4f})")

    print(f"\n  Excluded {len(selected_features_no_age) - 19} features:")
    for feat, imp in feature_importance[19:]:
        print(f"      - {feat:40s} (importance: {imp:.4f})")

elif len(selected_features_no_age) == 19:
    print(f"  ✓ RFE returned exactly 19 features (perfect!)")
    selected_19 = selected_features_no_age

else:  # Less than 19
    print(f"  ⚠️  RFE returned only {len(selected_features_no_age)} features")
    print(f"  Using all {len(selected_features_no_age)} features")
    selected_19 = selected_features_no_age

# Step 5: Add age_in_weeks to get final 20 features
print("\nStep 5: Add age_in_weeks to get final feature set")
print("-" * 80)
final_features = selected_19 + ['age_in_weeks']
print(f"  Final feature count: {len(final_features)}")
print(f"  ✓ age_in_weeks included")

# Show final features by body part
from collections import defaultdict
body_parts = defaultdict(list)
for feat in final_features:
    if feat == 'age_in_weeks':
        body_parts['Age'].append(feat)
    else:
        part = feat.split('_')[0]
        body_parts[part].append(feat)

print("\n  Final features by body part:")
for part in sorted(body_parts.keys()):
    print(f"    • {part}: {len(body_parts[part])} features")

# Step 6: Run surprise with final features
print("\nStep 6: Compute Bayesian surprise with final 20 features")
print("-" * 80)
bd.run_surprise_with_features('final20_forced_age', final_features)
print(f"  ✓ Surprise scores computed")

# Step 7: Compute ROC metrics
print("\nStep 7: Compute ROC metrics")
print("-" * 80)
bd.compute_roc_metrics('final20_forced_age', len(final_features))
print(f"  ✓ ROC metrics computed")

# Step 8: Display results
print("\nStep 8: Display Results")
print("=" * 80)

# The model name is stored as {prefix}_k_{size}
model_name = f'final20_forced_age_k_{len(final_features)}'
metrics_df = bd.metrics_df(model_name)
optimal_row_df = metrics_df.filter(pl.col('j') == pl.col('j').max())
if optimal_row_df.height > 1:
    optimal_row_df = optimal_row_df.filter(pl.col('threshold') == pl.col('threshold').max())
optimal_row = optimal_row_df.row(0, named=True)

roc_result = bd._metrics[model_name]

print("\nPerformance Metrics:")
print(f"  AUC:                {roc_result.auc:.4f}")
print(f"  Sensitivity:        {optimal_row['sens']:.4f}")
print(f"  Specificity:        {optimal_row['spec']:.4f}")
print(f"  Youden's J:         {optimal_row['j']:.4f}")
print(f"  Optimal Threshold:  {optimal_row['threshold']:.4f}")

# Get confusion matrix from primitives
primitives_df = roc_result.primitives
primitives_row_df = primitives_df.filter(pl.col('thresh') == optimal_row['threshold'])
if primitives_row_df.height > 0:
    primitives_row = primitives_row_df.row(0, named=True)
    tp = int(primitives_row['tp'])
    fn = int(primitives_row['fn'])
    tn = int(primitives_row['tn'])
    fp = int(primitives_row['fp'])

    print(f"\nConfusion Matrix:")
    print(f"  True Positives:     {tp}")
    print(f"  False Negatives:    {fn}")
    print(f"  True Negatives:     {tn}")
    print(f"  False Positives:    {fp}")
    print(f"  Total:              {tp + fn + tn + fp}")

# Step 9: Save results
print("\nStep 9: Save Results")
print("=" * 80)

# Save model
model_file = PKL_DIR / "bd_final20_forced_age.pkl"
import pickle
with open(model_file, 'wb') as f:
    pickle.dump(bd, f)
print(f"  ✓ Model saved: {model_file}")

# Save feature list
feature_file = JSON_DIR / "final_20_features_forced_age.json"
feature_data = {
    "model_name": "final20_forced_age_k_20",
    "features": final_features,
    "n_features": len(final_features),
    "includes_age": True,
    "age_forced": True,
    "metrics": {
        "auc": roc_result.auc,
        "sensitivity": optimal_row['sens'],
        "specificity": optimal_row['spec'],
        "threshold": optimal_row['threshold'],
        "youdens_j": optimal_row['j']
    }
}

with open(feature_file, 'w') as f:
    json.dump(feature_data, f, indent=2)
print(f"  ✓ Features saved: {feature_file}")

# Save detailed feature list
feature_detail_file = JSON_DIR / "final_20_features_forced_age_detail.json"
feature_detail = {
    "model_name": "final20_forced_age_k_20",
    "total_features": len(final_features),
    "features_by_body_part": {part: feats for part, feats in body_parts.items()},
    "feature_list": final_features,
    "methodology": {
        "rfe_method": "Enhanced Adaptive RFE",
        "rfe_features": len(features_no_age),
        "rfe_selected": len(selected_features_no_age),
        "final_movement_features": len(selected_19),
        "age_forced": True
    },
    "performance": {
        "auc": roc_result.auc,
        "sensitivity": optimal_row['sens'],
        "specificity": optimal_row['spec'],
        "threshold": optimal_row['threshold'],
        "youdens_j": optimal_row['j'],
        "confusion_matrix": {
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp
        }
    }
}

with open(feature_detail_file, 'w') as f:
    json.dump(feature_detail, f, indent=2)
print(f"  ✓ Detailed results saved: {feature_detail_file}")

print("\n" + "=" * 80)
print("COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nFinal model with {len(final_features)} features (including age_in_weeks):")
print(f"  • AUC:         {roc_result.auc:.4f}")
print(f"  • Sensitivity: {optimal_row['sens']:.4f}")
print(f"  • Specificity: {optimal_row['spec']:.4f}")
print()
print("Next steps:")
print("  1. Review feature list in: data/json/final_20_features_forced_age.json")
print("  2. Compare with previous 20-feature model")
print("  3. Run BAM sample size estimation if needed")
print()
