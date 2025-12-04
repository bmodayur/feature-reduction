"""Legacy data loading utilities for early-markers project.

This module provides functions for loading and processing merged infant
movement feature data. These functions are legacy utilities that have
largely been superseded by the BayesianData class methods.

Functions:
    get_merged_dataframe: Load and transform merged CSV data
    get_dataframes: Generate training/test splits with statistics

Data Format:
    Input CSV columns: infant, part, feature_name, Value, risk, category, age_bracket
    
    Risk encoding:
        - risk_raw <= 1 -> risk = 0 (normal/typical development)
        - risk_raw > 1 -> risk = 1 (at-risk/atypical development)
    
    Category encoding:
        - category = 0: Training data (normative set)
        - category = 1: Test data (held-out for validation)

Note:
    This module is maintained for backward compatibility but new code
    should use BayesianData class in bayes.py for data management.

Warning:
    References to FEATURES_MERGED_CSV constant which may not be defined
    in current constants.py. Use with caution or update imports.

Example:
    >>> from early_markers.cribsy.common.data import get_merged_dataframe
    >>> df = get_merged_dataframe()
    >>> print(df.columns)
    ['infant', 'part', 'feature_name', 'Value', 'risk_raw', 'risk', ...]
"""
from polars import DataFrame
import polars as pl

from early_markers.cribsy.common.constants import (
    SURPRISE_DIR,
    # FEATURES_MERGED_CSV,  # Note: May not be defined in current constants.py
    # FEATURE_SET_38,
)

# PEB 2025.03.27 21:06 => Remove features_38... nothing special.
#  Do not define max features statically.
#  Create synthetic data on risk in [0,1].
#  Ensure Synthetic training and testing data have 90/10, 85/15 ratios of risk 0/1
#  Setup for cross validation and feature selection to arrive at optimal estimates

# From B:
# - Move the risk=1, category=1 (CLIN data) to normative set (by setting its category field to 0)
# - Randomly sample a subset of this normative set (all risk=1) and flag it as TEST (by setting category=1)
# - These happen in the create_train_test() function

# Using 20 movement features and setting age threshold at 6 mos (age_bracket 0 would mean age under 6 mos and 1 would mean age over 6)
#
# - Produced a TRAIN set size of 94 and TEST set size of 54
# - The TRAIN set has a mix of YT, EMMA, and CLIN (all with risk=1)
# - The TEST set has a mix of CLIN (risk is 2 or 3) and YT (risk =1 ) and EMMA (risk = 1)
# - I used 20 features for RFE (uses your random seed and with 200 estimators)
#
# Using a threshold of -0.6, we get decent sens and spec. AUC is 0.88

def get_merged_dataframe(reload: bool = False) -> DataFrame:
    """Load and transform merged movement feature data from CSV.
    
    Reads the merged CSV file containing infant movement features and
    applies binary risk encoding transformation.
    
    Args:
        reload (bool, optional): Legacy parameter, currently unused.
            Kept for API compatibility. Defaults to False.
    
    Returns:
        DataFrame: Polars DataFrame with columns:
            - infant: Infant identifier
            - part: Body part (Ankle, Wrist, etc.)
            - feature_name: Metric name (medianx, IQRvelx, etc.)
            - Value: Feature value
            - risk_raw: Original risk score
            - risk: Binary risk (0=normal, 1=at-risk)
            - category: Train/test split indicator
            - age_bracket: Age category
    
    Transformations:
        - Renames 'risk' to 'risk_raw'
        - Creates binary 'risk': 0 if risk_raw ≤ 1, else 1
    
    Note:
        Requires FEATURES_MERGED_CSV to be defined in constants.
        This function is deprecated in favor of BayesianData class.
    
    Example:
        >>> df = get_merged_dataframe()
        >>> print(df.filter(pl.col('risk') == 1).height)  # Count at-risk
    """
    return (
        pl.read_csv(SURPRISE_DIR / FEATURES_MERGED_CSV, infer_schema_length=1000)
            .rename({"risk": "risk_raw"})
            .with_columns(
                risk=pl.when(pl.col("risk_raw") <= 1).then(0)
                .otherwise(1)
            )
    )


def get_dataframes(reload: bool = False) -> dict[str, DataFrame]:
    """Generate training/test splits with computed statistics.
    
    Creates a dictionary of DataFrames split by category (train/test) and
    age bracket, along with precomputed statistics (mean, SD) for each
    feature within each split.
    
    Args:
        reload (bool, optional): Legacy parameter, currently unused.
            Defaults to False.
    
    Returns:
        dict[str, DataFrame]: Dictionary with keys:
            - 'train': All training data (category=0)
            - 'train_lt_10': Training data, age < 10 weeks
            - 'train_ge_10': Training data, age >= 10 weeks
            - 'test': All test data (category=1)
            - 'test_lt_10': Test data, age < 10 weeks
            - 'test_ge_10': Test data, age >= 10 weeks
            - 'train_stats_59': Statistics for all 59 features (training)
            - 'test_stats_59': Statistics for all 59 features (test)
            - 'train_stats_38': Statistics for 38-feature subset (training)
            - 'test_stats_38': Statistics for 38-feature subset (test)
    
    Statistics DataFrames:
        Columns: part, feature_name, mean, sd
        One row per feature with computed mean and standard deviation
    
    Note:
        This function is deprecated. Use BayesianData class which provides
        more flexible data management and statistics computation.
        
        The 38-feature subset functionality is currently commented out in
        the implementation.
    
    Example:
        >>> data = get_dataframes()
        >>> train_df = data['train']
        >>> train_stats = data['train_stats_59']
        >>> print(f"Training samples: {train_df.height}")
        >>> print(f"Features with stats: {train_stats.height}")
    
    See Also:
        BayesianData: Modern replacement with better functionality
        get_merged_dataframe: Loads the source data
    """
    df_merged = get_merged_dataframe(reload)
    data = {
        # "features_38": DataFrame(
        #     {
        #         "part": [p for p, _ in FEATURE_SET_38],
        #         "feature_name": [f for _, f in FEATURE_SET_38],
        #     }
        # ),
        "train": df_merged.filter(pl.col("category") == 0),
        "train_lt_10": df_merged.filter(pl.col("category") == 0, pl.col("age_bracket") == 0),
        "train_ge_10": df_merged.filter(pl.col("category") == 0, pl.col("age_bracket") == 1),
        "test": df_merged.filter(pl.col("category") == 1),
        "test_lt_10": df_merged.filter(pl.col("category") == 1, pl.col("age_bracket") == 0),
        "test_ge_10": df_merged.filter(pl.col("category") == 1, pl.col("age_bracket") == 1),
    }

    for cat in ["train", "test"]:
        data[f"{cat}_stats_59"] = data[cat].group_by("part", "feature_name").agg(
            mean=pl.col("Value").mean(),
            sd=pl.col("Value").std(),
        ).sort("part", "feature_name")

        data[f"{cat}_stats_38"] = data[cat].join(data["features_38"], on=(["part", "feature_name"]), how="inner").group_by("part", "feature_name").agg(
            mean=pl.col("Value").mean(),
            sd=pl.col("Value").std(),
        ).sort("part", "feature_name")

    return data
