import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_datasets(public_path, attack_path=None, private_path=None):
    """Load preprocessed CSV datasets."""
    df_public = pd.read_csv(public_path)
    df_attack = pd.read_csv(attack_path) if attack_path else None
    df_private = pd.read_csv(private_path) if private_path else None
    return df_public, df_attack, df_private


def split_data(df, test_size=0.2, val_size=0.2, seed=42):
    """Split dataset into train/validation/test sets."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, shuffle=True)
    return train_df, val_df, test_df


def df_to_tensor(df, cat_feature):
    """Convert a DataFrame into categorical and continuous tensors."""
    cat = torch.as_tensor(df[cat_feature].values, dtype=torch.long)
    cont = torch.as_tensor(df.drop(columns=[cat_feature]).values, dtype=torch.float32)
    return cat, cont


def build_dataloaders(train_df, val_df, test_df, cat_feature, device, batch_size_gpu=4096, batch_size_cpu=1024):
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    cat_train, cont_train = df_to_tensor(train_df, cat_feature)
    cat_val, cont_val = df_to_tensor(val_df, cat_feature)
    cat_test, cont_test = df_to_tensor(test_df, cat_feature)

    train_ds = TensorDataset(cat_train, cont_train)
    val_ds = TensorDataset(cat_val, cont_val)
    test_ds = TensorDataset(cat_test, cont_test)

    batch_size = batch_size_gpu if device.type == "cuda" else batch_size_cpu

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=2)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader, test_loader
