import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_grad_norms(grad_norms):
    plt.figure(figsize=(8,4))
    plt.plot(grad_norms, color='orange', label='Average Gradient Norm')
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reconstruction_distribution(errors_benign, errors_attack):
    plt.figure(figsize=(8,5))
    sns.histplot(errors_benign, bins=100, color='green', alpha=0.6, label='Benign')
    sns.histplot(errors_attack, bins=100, color='red', alpha=0.5, label='Attack')
    plt.yscale('log')
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.show()
