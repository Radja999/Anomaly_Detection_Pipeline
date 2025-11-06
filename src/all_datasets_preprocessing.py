"""
preprocess_all_sets.py
----------------------
Comprehensive preprocessing pipeline for CICIDS2017 and private benign datasets.
Performs data cleaning, consistency checks, and feature alignment
for three output sets:
    - public_benign_set.csv
    - attack_public_set.csv
    - private_benign_set.csv


"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ========================================
# STEP 1 – Load and Merge Public CSV Files
# ========================================


# -------------------------------
# Repository directory structure
# -------------------------------
RAW_PUBLIC_DIR = "Anomaly_Detection_Pipeline/data/raw/CIC-IDS2017_CSVs"
RAW_PRIVATE_DIR = "Anomaly_Detection_Pipeline/data/raw/Lab_Generated_Benign_Dataset"
PROCESSED_DIR = "Anomaly_Detection_Pipeline/data/processed/"

PRIVATE_FILE = os.path.join(RAW_PRIVATE_DIR, "merged_flows.csv")
PUBLIC_BENIGN_FILE = os.path.join(PROCESSED_DIR, "public_benign_set.csv")
ATTACK_FILE = os.path.join(PROCESSED_DIR, "attack_set.csv")
PRIVATE_BENIGN_FILE = os.path.join(PROCESSED_DIR, "private_benign_set.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(RAW_PUBLIC_DIR, "*.csv")))
print(f"Found {len(csv_files)} CSV files.")

# Merge all CSV files
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
print(f"Merged public dataset shape: {df.shape}")


# ========================================
# STEP 2 – Basic Cleaning
# ========================================

df.columns = df.columns.str.strip()

# Check correlation and invalid values between duplicate header columns
if 'Fwd Header Length.1' in df.columns and 'Fwd Header Length' in df.columns:
    print("Correlation between duplicate header columns:",
          df['Fwd Header Length'].corr(df['Fwd Header Length.1']))
    neg_fwd = (df['Fwd Header Length'] < 0).sum()
    neg_fwd_1 = (df['Fwd Header Length.1'] < 0).sum()
    print(f"Negative values in 'Fwd Header Length': {neg_fwd}")
    print(f"Negative values in 'Fwd Header Length.1': {neg_fwd_1}")

# Drop redundant duplicate header column if exists
drop_cols = [c for c in ["Fwd Header Length.1"] if c in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped redundant columns: {drop_cols}")

# Remove rows with invalid header lengths
neg_count = (df['Fwd Header Length'] < 0).sum()
print(f"Rows with negative 'Fwd Header Length': {neg_count}")
df = df[df['Fwd Header Length'] >= 0].copy()
print(f"Shape after removing invalid header rows: {df.shape}")


# ========================================
# STEP 3 – Duplicate and Constant Columns
# ========================================

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates dropped.")

# Identify constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
if constant_cols:
    print(f"Constant columns detected ({len(constant_cols)}): {constant_cols}")
    df.drop(columns=constant_cols, inplace=True)
else:
    print("No constant columns detected.")


# ========================================
# STEP 4 – Mixed Type and Categorical Columns
# ========================================

mixed_type_cols = []
for col in df.columns:
    if df[col].map(type).nunique() > 1:
        mixed_type_cols.append(col)
if mixed_type_cols:
    print(f"Columns with mixed data types: {mixed_type_cols}")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")


# ========================================
# STEP 5 – Missing and Infinite Values
# ========================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

nan_summary = df.isna().sum()
nan_cols = nan_summary[nan_summary > 0]
print(f"Columns with missing values: {list(nan_cols.index)}")

# Drop rows with missing rate features
df = df.dropna(subset=["Flow Bytes/s", "Flow Packets/s"]).reset_index(drop=True)
print(f"Shape after dropping rows with missing rate values: {df.shape}")


# ========================================
# STEP 6 – Negative Value Checks
# ========================================

negative_cols = df.columns[(df < 0).any()].tolist()
print(f"Features with negative values ({len(negative_cols)}): {negative_cols}")

invalid_cols = [
    'Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]

neg_rows = df[df[invalid_cols].lt(0).any(axis=1)]
print(f"Rows containing invalid negatives: {len(neg_rows)}")


# ========================================
# STEP 7 – Recalculation and Correction
# ========================================

# Recalculate flow rate features
df['Flow Packets/s_recalc'] = (df['Total Fwd Packets'] + df['Total Backward Packets']) / (df['Flow Duration'] / 1e6)
df.loc[df['Flow Duration'] <= 0, 'Flow Packets/s_recalc'] = np.nan

df['Flow Bytes/s_recalc'] = (
    (df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets'])
    / (df['Flow Duration'] / 1000)
)

# Correlation checks
corr_packets = df[['Flow Packets/s', 'Flow Packets/s_recalc']].corr().iloc[0, 1]
corr_bytes = df[['Flow Bytes/s', 'Flow Bytes/s_recalc']].corr().iloc[0, 1]
print(f"Correlation (Flow Packets/s vs recalculated): {corr_packets:.4f}")
print(f"Correlation (Flow Bytes/s vs recalculated): {corr_bytes:.4f}")


# ========================================
# STEP 8 – Apply Corrections
# ========================================

time_features = ['Flow Duration', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min']
rate_features = ['Flow Bytes/s', 'Flow Packets/s']
window_features = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'min_seg_size_forward']
corrected_features = time_features + rate_features + window_features

df_before = df[corrected_features].copy()

# Apply absolute value to time and rate features
df[time_features] = df[time_features].abs()
df[rate_features] = df[rate_features].abs()

# Replace negatives in window features with median
for col in window_features:
    median_val = df.loc[df[col] >= 0, col].median()
    df.loc[df[col] < 0, col] = median_val

print("Negative values corrected using absolute or median substitution.")


# ========================================
# STEP 9 – Verify and Drop Temporary Columns
# ========================================

remaining_neg = (df[corrected_features] < 0).sum()
print("Remaining negative values per corrected feature:")
print(remaining_neg[remaining_neg > 0] if remaining_neg.any() else "None remaining.")

recalc_cols = [c for c in df.columns if 'recalc' in c]
if recalc_cols:
    df.drop(columns=recalc_cols, inplace=True)
    print(f"Dropped recalculated columns: {recalc_cols}")


# ========================================
# STEP 10 – Split into Benign and Attack Sets
# ========================================

label_cols = [c for c in df.columns if "label" in c.lower()]
label_col = label_cols[0] if label_cols else None

if label_col:
    print(f"Detected label column: {label_col}")
    public_benign_df = df[df[label_col].str.contains("BENIGN", case=False, na=False)].copy()
    attack_df = df[~df[label_col].str.contains("BENIGN", case=False, na=False)].copy()
else:
    raise ValueError("Label column not found in dataset.")

print(f"Public benign shape: {public_benign_df.shape}")
print(f"Attack traffic shape: {attack_df.shape}")


# ========================================
# STEP 11 – Save Cleaned Outputs
# ========================================

public_benign_df.to_csv(os.path.join(PROCESSED_DIR, "public_benign_set.csv"), index=False)
attack_df.to_csv(os.path.join(PROCESSED_DIR, "attack_public_set.csv"), index=False)
print("Public benign and attack sets saved successfully.")


# ========================================
# STEP 12 – Process Private Benign Dataset
# ========================================

private_df = pd.read_csv(PRIVATE_FILE)
print(f"Private raw dataset shape: {private_df.shape}")

# Rename columns to align with public dataset
rename_mapping = {
    "dst_port": "Destination Port",
    "flow_duration": "Flow Duration",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_seg_size_min": "min_seg_size_forward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "flow_iat_std": "Flow IAT Std",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "cwr_flag_count": "CWE Flag Count",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_byts": "Subflow Bwd Bytes"
}

private_df.rename(columns=rename_mapping, inplace=True)
private_df.columns = private_df.columns.str.strip()

# Drop non-feature columns
private_df.drop(columns=["src_ip", "dst_ip", "protocol", "timestamp", "src_port"], inplace=True, errors="ignore")

# Drop 8 sparse bulk-related columns
cols_to_drop = [
    "Bwd PSH Flags", "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"
]
private_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

print(f"Private benign shape after cleaning: {private_df.shape}")

private_df.to_csv(os.path.join(PROCESSED_DIR, "private_benign_set.csv"), index=False)
print("Private benign set saved successfully.")
