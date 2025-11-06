import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.dataset_loader import load_datasets, split_data, build_dataloaders
from utils.scaler import apply_log_scaling, save_schema_and_bounds
from utils.visualization import plot_loss_curves, plot_grad_norms

# === Paths ===
PROCESSED_DIR = "Anomaly_Detection_Pipeline/data/processed"
MODEL_DIR = "Anomaly_Detection_Pipeline/models"

PUBLIC_BENIGN_FILE = os.path.join(PROCESSED_DIR, "public_benign_set.csv")
ATTACK_FILE = os.path.join(PROCESSED_DIR, "attack_set.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# === Device setup ===
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load datasets ===
df, attack_df, _ = load_datasets(PUBLIC_BENIGN_FILE, attack_path=ATTACK_FILE)
train_df, val_df, test_df = split_data(df)

# === Feature grouping ===
cat_feature = "Destination Port"
candidate_no_scale = [
    "Fwd PSH Flags", "URG Flag Count",
    "Fwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bytes/Bulk", "Fwd Avg Bulk Rate",
    "Fwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "CWE Flag Count", "ECE Flag Count", "RST Flag Count",
    "FIN Flag Count", "SYN Flag Count", "Bwd URG Flags"
]
no_scale_features = [f for f in candidate_no_scale if f in df.columns]
minmax_features = [c for c in df.columns if c not in no_scale_features + [cat_feature, "Label"]]

# === Log scaling ===
train_scaled = apply_log_scaling(train_df, minmax_features)
val_scaled = apply_log_scaling(val_df, minmax_features)
test_scaled = apply_log_scaling(test_df, minmax_features)

# === Save schema & bounds ===
save_schema_and_bounds(train_scaled, minmax_features, cat_feature, no_scale_features,
                       os.path.join(MODEL_DIR, "schema.pkl"),
                       os.path.join(MODEL_DIR, "bounds.pkl"))

# === Data loaders ===
train_loader, val_loader, test_loader = build_dataloaders(train_scaled, val_scaled, test_scaled, cat_feature, device)

# === Model definition ===
class DeepAutoencoder(nn.Module):
    def __init__(self, n_cont, max_ports=65535, emb_dim=128, latent_dim=128):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=max_ports + 1, embedding_dim=emb_dim)
        in_dim = n_cont + emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, n_cont)
        )

    def forward(self, cat, cont):
        emb = self.emb(cat)
        x = torch.cat([emb, cont], dim=1)
        z = self.encoder(x)
        return self.decoder(z)

# === Initialize ===
n_cont = len(train_scaled.columns) - 1
model = DeepAutoencoder(n_cont).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = nn.MSELoss()

# === Training loop ===
EPOCHS = 255
PATIENCE = 20
best_val_loss = float("inf")
epochs_no_improve = 0
train_losses, val_losses, grad_norms = [], [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss, total_grad_norm = 0, 0

    for cat, cont in train_loader:
        cat, cont = cat.to(device), cont.to(device)
        optimizer.zero_grad()
        output = model(cat, cont)
        loss = criterion(output, cont)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        total_grad_norm += grad_norm.item()
        optimizer.step()
        total_train_loss += loss.item() * cont.size(0)

    total_train_loss /= len(train_loader.dataset)
    train_losses.append(total_train_loss)
    grad_norms.append(total_grad_norm / len(train_loader))

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for cat, cont in val_loader:
            cat, cont = cat.to(device), cont.to(device)
            output = model(cat, cont)
            total_val_loss += criterion(output, cont).item() * cont.size(0)
    total_val_loss /= len(val_loader.dataset)
    val_losses.append(total_val_loss)

    print(f"Epoch {epoch}/{EPOCHS} | Train MSE: {total_train_loss:.6f} | Val MSE: {total_val_loss:.6f}")

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_autoencoder.pt"))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

plot_loss_curves(train_losses, val_losses)
plot_grad_norms(grad_norms)

