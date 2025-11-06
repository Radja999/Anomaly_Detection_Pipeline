import numpy as np
import pickle
import pandas as pd

def log_safe(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def apply_log_scaling(df, features):
    df_scaled = df.copy()
    for col in features:
        if col in df_scaled.columns:
            df_scaled[col] = log_safe(df_scaled[col].values)
    return df_scaled

def robust_minmax_transform(df, bounds, exclude_cols=None):
    df_scaled = df.copy()
    for col, (lower, upper) in bounds.items():
        if col not in df_scaled.columns:
            continue
        if exclude_cols and col in exclude_cols:
            df_scaled[col] = 0.0
            continue
        if upper == lower or (upper - lower) < 1e-9:
            df_scaled[col] = 0.0
        else:
            x = np.clip(df_scaled[col].values, lower, upper)
            df_scaled[col] = (x - lower) / (upper - lower)
    return df_scaled.fillna(0.0)


def save_schema_and_bounds(train_scaled, minmax_features, cat_feature, no_scale_features, filename_schema, filename_bounds):
    """Save schema (columns, features) and robust scaling bounds."""
    cont_cols = [c for c in train_scaled.columns if c != cat_feature]
    bounds = {}
    for col in minmax_features:
        if col in train_scaled.columns:
            lower = 0.0
            upper = train_scaled[col].quantile(0.999)
            bounds[col] = (lower, upper)

    with open(filename_schema, "wb") as f:
        pickle.dump({
            "cat_feature": cat_feature,
            "cont_cols": cont_cols,
            "minmax_features": minmax_features,
            "no_scale_features": no_scale_features
        }, f)

    with open(filename_bounds, "wb") as f:
        pickle.dump(bounds, f)

    print(f"Saved schema ({len(cont_cols)} continuous cols) and {len(bounds)} feature bounds.")
