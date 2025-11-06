# AutoEncoder Network for Anomaly Detection Pipeline

## Overview
This project implements an **AutoEncoder-based anomaly detection pipeline** for identifying cyberattacks in network traffic data. It integrates classical preprocessing, feature engineering, scaling, and feature alignment techniques with deep learning for anomaly detection.

The implementation supports two datasets:
1. **Public dataset:** CIC-IDS2017 (benign and attack flows)
2. **Private dataset:** A lab-generated benign dataset, created in a controlled LAN environment.

All stages — from data preprocessing, feature scaling, model training, evaluation, and reconstruction error analysis — are covered in this repository.


---
## Dataset Details

### 1. CIC-IDS2017 Dataset (Public)
Download the 8 CSV files of the **CIC-IDS2017 dataset** from:
 [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

After downloading, place all the CSVs into: **CIC-IDS2017_CSVs** folder.

### 2. Private Lab-Generated Dataset
This dataset was **personally generated** in a simulated network environment using **VMware Workstation Pro**.  
The setup included:
- **Kali Linux VM** as the attacker.
- **Windows 10 VM** and **Ubuntu VM** as benign clients.
- **Offensive Security Firewall VM** acting as the network gateway with bridged (WAN) and host-only (LAN) adapters.
- **Host (Windows 10)** included in the same LAN.
- **Tcpdump** was used to capture all LAN ingoing and outgoing traffic for 5 hours.
- Captured packets were redirected to the Ubuntu VM for preprocessing and flow extraction.

The result is the file:

This dataset serves as a **private benign set** for validating model generalization against unseen internal traffic.

---


## How to Use

### 1. Clone the Repository 
```bash
git clone https://github.com/Radja999/Anomaly_Detection_Pipeline.git
cd AutoEncoder-Network-Anomaly-Detection-Pipeline.
```
## 2. Install Dependencies
You can install all dependencies using:
```bash
pip install -r requirements.txt
```
## 3. Run the Notebooks (Colab or Local Jupyter)

You can run each notebook step by step:

**Preprocessing:** All_Sets_Preprocessing.ipynb

**Training:** model_training.ipynb

**Inference and Evaluation:** inference_model.ipynb

**Analysis:** data_comparison_analysis.ipynb

For Google Colab, clone the repository using the following command: 
```bash
!git clone https://github.com/Radja999/Anomaly_Detection_Pipeline.git
```
**Note:** after running the **All_Sets_Preprcessing** notebook, save the output files and then upload them to **processed** folder to run the next 3 notebooks since all the preprcoessed datasets are required as inputs to the next phases of the pipeline.
## 4. Docker (Optional)

You can containerize the project using the provided **Dockerfile**.
Build the image:
```bash
docker build -t anomaly-detection-pipeline .
```
Run the container:
```bash
docker run -it anomaly-detection-pipeline
```
## 5. Outputs
After running the notebooks or scripts, the following files will be generated:

- **Trained AutoEncoder model** → models/best_autoencoder.pt

- **Feature schema** → models/schema.pkl

- **Scaling bounds** → models/bounds.pkl

- **Evaluation reports and plots** → inside notebook outputs

For the sake of the pipeline completeness and outputs comparison, best_autoencoder, schema and bounds files were included inside the **~/models/reference** folder.




---

## Why the Private Dataset Matters — Evaluating Generalization

One of the most important goals of this work was to evaluate **how well the trained Autoencoder generalizes** to unseen benign network traffic captured in a different environment.

While both datasets — the *public CIC-IDS2017 benign flows* and the *lab-generated private benign flows* — share similar structural properties (number of features, overall flow composition, etc.), they differ in their **data distributions**.

These differences arise naturally from:

* Network topology and routing behaviors
* Background noise and protocol patterns
* Host operating systems and software versions
* Timing and packet scheduling differences

The model’s performance clearly illustrates that **shape similarity between datasets (i.e., same number of features and roughly similar distributions)** does **not** guarantee good generalization.
Autoencoders trained on one environment tend to learn the **joint feature distribution** — i.e., the internal correlations and co-dependencies between features — rather than the independent marginal distributions of each variable.

This means:

> Even if two datasets look similar after scaling and normalization, their feature-to-feature relationships can differ significantly.
> Consequently, the model perceives unseen but benign patterns as anomalies.

Such an evaluation using a **private benign dataset** is therefore essential for assessing whether the learned model captures general network behavior or is merely overfitting to the statistical properties of the training dataset.

---

## Limitations and Future Work — Beyond Spatial Feature Correlation

The current Autoencoder model focuses purely on **spatial correlations** between features — that is, it learns how combinations of flow-level attributes (packet counts, durations, byte rates, flags, etc.) relate within a single record.

However, real network traffic exhibits strong **temporal dependencies**.
Events and flow patterns evolve over time, and many anomalies or attacks can only be distinguished by analyzing these sequential dynamics.

To enhance generalization and robustness, future improvements could include:

1. **Temporal modeling**

   * Incorporating sequence-aware architectures such as LSTM, GRU, or Temporal Convolutional Networks (TCN).
   * Sliding-window representations of flows to model time-series dependencies.

2. **Hybrid reconstruction–prediction models**

   * Jointly training an Autoencoder with a forecasting head to capture both spatial and temporal anomalies.

3. **Self-supervised or contrastive learning approaches**

   * Allowing the model to learn invariant representations of benign behavior across multiple environments.

4. **Domain adaptation or fine-tuning**

   * Using a small amount of data from a new network (like the private set) to recalibrate the latent space without full retraining.































