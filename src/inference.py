import os
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from utils.scaler import apply_log_scaling, robust_minmax_transform
from utils.visualization import plot_reconstruction_distribution
from utils.dataset_loader import load_datasets
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

# =========================================
# Paths and Setup
# =========================================
BASE_DIR = "Anomaly_Detection_Pipeline"
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data/processed")

PUBLIC_FILE = os.path.join(DATA_DIR, "public_benign_set.csv")
PRIVATE_FILE = os.path.join(DATA_DIR, "private_benign_set.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# Load Model and Schema
# =========================================
with open(os.path.join(MODEL_DIR, "schema.pkl"), "rb") as f:
    schema = pickle.load(f)
with open(os.path.join(MODEL_DIR, "bounds.pkl"), "rb") as f:
    bounds = pickle.load(f)

cat_feature = schema["cat_feature"]
cont_cols = schema["cont_cols"]
minmax_features = schema["minmax_features"]
no_scale_features = schema["no_scale_features"]

model_path = os.path.join(MODEL_DIR, "best_autoencoder.pt")


# =========================================
# Define Model Class (same as training)
# =========================================
class DeepAutoencoder(torch.nn.Module):
    def __init__(self, n_cont, max_ports=65535, emb_dim=128, latent_dim=128):
        super().__init__()
        self.emb = torch.nn.Embedding(max_ports + 1, emb_dim)
        in_dim = n_cont + emb_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 1024), torch.nn.ReLU(),
            torch.nn.Linear(1024, n_cont)
        )

    def forward(self, cat, cont):
        emb = self.emb(cat)
        x = torch.cat([emb, cont], dim=1)
        z = self.encoder(x)
        return self.decoder(z)


# =========================================
# Load model and datasets
# =========================================
model = DeepAutoencoder(len(cont_cols)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Loaded trained model successfully.")

df_public, _, df_private = load_datasets(PUBLIC_FILE, private_path=PRIVATE_FILE)


# =========================================
# Helper function: compute MSE for a dataset
# =========================================
def compute_mse(df):
    df_scaled = apply_log_scaling(df, minmax_features)
    for col in no_scale_features:
        if col in df.columns:
            df_scaled[col] = df[col]
    df_scaled[cat_feature] = df[cat_feature].astype(int)
    df_scaled = robust_minmax_transform(df_scaled, bounds)
    cat_tensor = torch.tensor(df_scaled[cat_feature].values, dtype=torch.long)
    cont_tensor = torch.tensor(df_scaled[cont_cols].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(cat_tensor, cont_tensor), batch_size=1024, shuffle=False)
    mse = []
    with torch.no_grad():
        for cat, cont in loader:
            cat, cont = cat.to(device), cont.to(device)
            out = model(cat, cont)
            batch_mse = torch.mean((out - cont) ** 2, dim=1).cpu().numpy()
            mse.extend(batch_mse)
    return np.array(mse)


# =========================================
# Compute MSEs for both benign sets
# =========================================
mse_public = compute_mse(df_public)
mse_private = compute_mse(df_private)

print(f"Public Benign: mean MSE={np.mean(mse_public):.6f}, std={np.std(mse_public):.6f}")
print(f"Private Benign: mean MSE={np.mean(mse_private):.6f}, std={np.std(mse_private):.6f}")


# =========================================
# Visualize and Compare Distributions
# =========================================
plt.figure(figsize=(10, 5))
sns.kdeplot(mse_public, color='blue', fill=True, label='Public Benign')
sns.kdeplot(mse_private, color='green', fill=True, label='Private Benign')
plt.xscale('log')
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density (log scale)")
plt.title("MSE Distribution Comparison — Public vs Private Benign Sets")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.ecdfplot(mse_public, color='blue', label='Public Benign')
sns.ecdfplot(mse_private, color='green', label='Private Benign')
plt.xscale('log')
plt.xlabel("Reconstruction Error (log scale)")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution of Reconstruction Errors")
plt.legend()
plt.show()


# =========================================
# Divergence Statistics (KS-Test & Overlap)
# =========================================
ks_stat, ks_p = ks_2samp(mse_public, mse_private)
print(f"Kolmogorov–Smirnov Test → KS-stat = {ks_stat:.4f}, p-value = {ks_p:.4e}")

bins = np.logspace(
    np.log10(min(mse_public.min(), mse_private.min()) + 1e-9),
    np.log10(max(mse_public.max(), mse_private.max()) + 1e-9),
    200
)
hist_pub, _ = np.histogram(mse_public, bins=bins, density=True)
hist_priv, _ = np.histogram(mse_private, bins=bins, density=True)

hist_pub /= np.sum(hist_pub)
hist_priv /= np.sum(hist_priv)
overlap_area = np.sum(np.minimum(hist_pub, hist_priv))

print("\n=== Divergence Summary ===")
print(f"KS Statistic  : {ks_stat:.4f}")
print(f"KS p-value    : {ks_p:.4e}")
print(f"Overlap Area  : {overlap_area:.4f}")

if overlap_area < 0.5:
    print("Strong distributional divergence detected.")
elif overlap_area < 0.8:
    print("Moderate divergence detected.")
else:
    print("High similarity detected.")

# =========================================
# Save comparison results
# =========================================
np.save(os.path.join(MODEL_DIR, "mse_public.npy"), mse_public)
np.save(os.path.join(MODEL_DIR, "mse_private.npy"), mse_private)
print("Saved reconstruction error arrays for reuse.")
