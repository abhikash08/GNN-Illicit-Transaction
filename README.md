# Illicit Transaction Detection in Bitcoin using Graph Neural Networks

## Project Overview
This project implements an Anti-Money Laundering (AML) detection system for the Bitcoin blockchain. Instead of treating transactions as isolated tabular data, this system models the entire Bitcoin network as a connected Graph (nodes = transactions, edges = flow of Bitcoin). 

Using PyTorch Geometric, the project leverages Graph Neural Networks (GNNs) to analyze transaction neighborhoods and detect illicit activity.

## Core Challenges & Solutions
Real-world AML detection is notoriously difficult. This project was specifically engineered to solve two major data science challenges:

1. **Severe Class Imbalance:** Only ~10% of the labeled dataset represents illicit transactions. Standard models achieve high accuracy simply by predicting "Licit" every time.
   * **Solution:** Implemented **Focal Loss** to mathematically penalize the model heavily for missing the minority "Illicit" class, prioritizing Recall over generic accuracy.
2. **Temporal Concept Drift:** Money launderers constantly change their network structures and tactics over time. A model trained on past data quickly becomes obsolete.
   * **Solution:** Replaced random data splitting with a strict **Time-Based Split** (training on Timesteps 1-34, testing on Timesteps 35-49) to prove the model can catch previously unseen future tactics. We then applied **Threshold Calibration** to optimize the precision-recall trade-off.

## Dataset
This project utilizes the **Elliptic Bitcoin Dataset**.
* **Nodes:** 203,769 transactions (Nodes contain 165 features, including local transaction data and aggregated neighborhood features).
* **Edges:** 234,355 directed edges representing the flow of Bitcoin.
* **Classes:** Licit (0), Illicit (1), and Unknown (-1).

**Note:** Due to GitHub file size limitations, the raw CSV files are not included in this repository. 
You must download the dataset directly from Kaggle: [Elliptic Data Set (Kaggle)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set?resource=download)

Place the extracted `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, and `elliptic_txs_features.csv` files directly into the root folder before running the scripts.

## Project Structure
* `preprocess.py`: Ingests the raw CSVs, maps transaction IDs to sequential integers, creates attention masks for known/unknown labels, and builds the PyTorch Geometric `Data` object.
* `train.py`: Contains the Baseline Graph Convolutional Network (GCN).
* `train_sage.py`: Upgrades the architecture to GraphSAGE and introduces Focal Loss to combat class imbalance.
* `train_advanced.py`: The final production model. Implements Time-Based splitting to simulate real-world conditions and automates threshold calibration.

## Setup & Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install pandas numpy torch torch-geometric scikit-learn
```
## Progress & Results

The project was developed iteratively, with each stage addressing a critical limitation identified in the previous model.

### 1. Baseline Model (GCN)

The initial implementation used a standard **Graph Convolutional Network (GCN)** with a conventional **80/20 random train-test split**.

| Metric | Value |
|--------|-------|
| **Precision** | 0.81 |
| **Recall** | 0.60 |
| **F1-Score** | 0.69 |

**Key Takeaway:**  
While precision was reasonably high, the model detected only **60% of illicit transactions**, missing approximately **40% of money laundering activities**. Such a recall is inadequate for real-world Anti-Money Laundering (AML) systems where missing suspicious transactions carries significant risk.

---

### 2. Architecture Upgrade (GraphSAGE + Focal Loss)

To improve the detection of minority-class transactions, the standard Cross-Entropy loss was replaced with **Focal Loss**, forcing the model to focus more heavily on difficult and underrepresented illicit samples. Simultaneously, the GCN architecture was upgraded to **GraphSAGE**, enabling more effective neighborhood aggregation and improving representation of complex transaction networks.

| Metric | Value |
|--------|-------|
| **Precision** | 0.48 |
| **Recall** | 0.96 |
| **F1-Score** | 0.64 |

**Key Takeaway:**  
Recall increased dramatically to **96%**, meaning the model successfully identified nearly all illicit transactions. However, because the evaluation still relied on a random train-test split, the model unintentionally learned from future transaction patterns, resulting in overly optimistic performance estimates.

---

### 3. Real-World Evaluation (Temporal Split + Threshold Calibration)

To simulate a realistic deployment scenario, the dataset was split chronologically instead of randomly.

- **Training:** Timesteps **1-34**
- **Testing:** Timesteps **35-49**

This prevents information leakage from future transactions and evaluates the model on previously unseen laundering strategies.

Since prediction probabilities shifted on future data, **threshold calibration** was performed to identify the optimal operating point balancing Precision and Recall.

#### Threshold Calibration Results

| Decision Threshold | Precision | Recall | F1-Score |
|-------------------|----------:|--------:|---------:|
| **> 50%** | 0.2274 | 0.7775 | 0.3519 |
| **> 60%** | 0.3654 | 0.7091 | 0.4823 |
| **> 70%** | **0.6889** | **0.6094** | **0.6467** |
| **> 80%** | 0.9486 | 0.2558 | 0.4029 |

### Final Outcome

A **70% decision threshold** provided the best trade-off between precision and recall, achieving the highest F1-Score while maintaining strong detection capability on completely unseen future transactions.

This final pipeline demonstrates a practical AML workflow by combining:

- **GraphSAGE** for improved graph representation learning.
- **Focal Loss** to mitigate severe class imbalance.
- **Temporal train-test splitting** to account for concept drift.
- **Threshold calibration** to optimize deployment performance.