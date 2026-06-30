import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_elliptic_dataset(data_dir):
    print("--- Step 1: Loading Raw CSV Files ---")
    # 1. Read classes, features, and edge list files
    classes_df = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_classes.csv'))
    features_df = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_features.csv'), header=None)
    edges_df = pd.read_csv(os.path.join(data_dir, 'elliptic_txs_edgelist.csv'))
    
    # Give descriptive names to crucial structural columns
    # features_df: column 0 is txId, column 1 is timestep, columns 2 to 166 are local/aggregated features
    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
    
    print(f"Total nodes in features file: {features_df.shape[0]}")
    print(f"Total edges in edgelist file: {edges_df.shape[0]}")

    print("\n--- Step 2: Mapping Labels & Creating Node ID Dictionary ---")
    # Map classes into numeric labels:
    # '1' -> Illicit (Class 1)
    # '2' -> Licit (Class 0)
    # 'unknown' -> Map to -1 so we can easily create training/testing masks later
    classes_df['label'] = classes_df['class'].map({'1': 1, '2': 0, 'unknown': -1})
    
    # Merge features and classes on transaction ID to guarantee perfect alignment
    merged_df = pd.merge(features_df, classes_df, on='txId', how='left')
    
    # Graph neural networks require node IDs to be sequential integers starting from 0 (0, 1, 2, ... N-1).
    # Currently, txId values are large random numbers (e.g., 230425881). We must remap them.
    node_id_map = {tx_id: idx for idx, tx_id in enumerate(merged_df['txId'])}
    
    print("\n--- Step 3: Structuring Graph Attributes (PyTorch Tensors) ---")
    # Extract node features (columns 2 through 166)
    feature_cols = [c for c in merged_df.columns if c not in ['txId', 'timestep', 'class', 'label']]
    x = torch.tensor(merged_df[feature_cols].values, dtype=torch.float)
    
    # Extract labels and timesteps
    y = torch.tensor(merged_df['label'].values, dtype=torch.long)
    timesteps = torch.tensor(merged_df['timestep'].values, dtype=torch.long)
    
    print(f"Node feature tensor shape (x): {x.shape}")
    print(f"Label tensor shape (y): {y.shape}")

    print("\n--- Step 4: Re-mapping and Constructing Edges ---")
    # Filter edges to make sure both source and destination exist in our features mapping dictionary
    edges_df = edges_df[edges_df['txId1'].isin(node_id_map) & edges_df['txId2'].isin(node_id_map)]
    
    # Map old transaction IDs to our new sequential 0-indexed integers
    mapped_src = edges_df['txId1'].map(node_id_map).values
    mapped_dst = edges_df['txId2'].map(node_id_map).values
    
    # In PyTorch Geometric, edge list must have shape [2, num_edges] (source array row, destination array row)
    edge_index = torch.tensor(np.array([mapped_src, mapped_dst]), dtype=torch.long)
    print(f"Edge index tensor shape: {edge_index.shape}")

    print("\n--- Step 5: Handling Unknown Labels (Masking) ---")
    # Create masks to isolate known nodes (licit & illicit) from unknown nodes
    # This prevents the model from attempting to calculate loss or accuracy on unknown transaction labels.
    known_mask = (y != -1)
    
    print(f"Total labeled (Licit/Illicit) nodes: {known_mask.sum().item()}")
    print(f"Total unlabeled (Unknown) nodes: {(~known_mask).sum().item()}")

    print("\n--- Step 6: Creating the PyG Data Object ---")
    # Wrap everything up into a unified PyTorch Geometric Data object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        timestep=timesteps,
        known_mask=known_mask
    )
    
    return graph_data

# --- EXECUTION ---
# Change this directory path to where you extracted your zip archive files
DATA_DIR = "." 

# Run the pipeline
graph = load_elliptic_dataset(DATA_DIR)
print("\n==================================================")
print("SUCCESS: Your PyTorch Geometric Graph Object is Ready!")
print(graph)
print("==================================================")