from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_auc_scores(true_labels, recon_errors):
    roc_auc = roc_auc_score(true_labels, recon_errors)
    precision, recall, thresholds = precision_recall_curve(true_labels, recon_errors)
    pr_auc = np.trapz(precision, recall)
    return roc_auc, pr_auc, precision, recall, thresholds

def plot_roc_curve(fpr, tpr, auc_value):
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_pr_curve(precision, recall, auc_value):
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, color='purple', label=f"AUC = {auc_value:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.show()

def plot_confusion_matrix(true_labels, preds, labels=('Benign','Attack')):
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(true_labels, preds, target_names=labels))
