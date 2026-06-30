import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# This imports the function we wrote in your preprocess.py file!
from preprocess import load_elliptic_dataset 

print("--- Step 1: Loading Processed Graph Data ---")
# The "." tells it to look in the current folder for the CSVs
graph = load_elliptic_dataset(".") 

print("\n--- Step 2: Defining the GCN Architecture ---")
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # First Graph Convolutional Layer (takes all 165 features and condenses them)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Second Layer (takes condensed features and outputs Licit (0) or Illicit (1))
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Pass features and network topology through the first layer
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Activation function (keeps it non-linear)
        # Pass through second layer
        x = self.conv2(x, edge_index)
        return x

# Initialize the model: 165 input features -> 64 hidden neurons -> 2 output classes
model = GCN(in_channels=graph.num_node_features, hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# --- ADVANCED FEATURE: FOCAL LOSS ---
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # We assign a weight of 0.25 to Licit (Class 0) and 0.75 to Illicit (Class 1)
        # This forces the model to care 3x more about catching criminals
        self.weight = torch.tensor([0.25, 0.75]) 

    def forward(self, inputs, targets):
        # Calculate standard cross entropy with our custom class weights
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        # Apply the Focal Loss formula to focus on hard-to-classify nodes
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Replace the old criterion with our new custom Focal Loss
criterion = FocalLoss(gamma=2.0)

print("\n--- Step 3: Creating Train/Test Splits ---")
# We CANNOT train on "Unknown" nodes. We must only split the nodes we actually have labels for.
known_indices = graph.known_mask.nonzero(as_tuple=False).view(-1)
num_known = known_indices.size(0)

# Randomly shuffle the known nodes
indices = known_indices[torch.randperm(num_known)]

# Take 80% for training, 20% for testing
train_size = int(0.8 * num_known)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create boolean masks for PyTorch
train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
test_mask[test_indices] = True

print(f"Training on {train_mask.sum().item()} nodes")
print(f"Testing on {test_mask.sum().item()} nodes")

print("\n--- Step 4: Training Loop ---")
def train():
    model.train()
    optimizer.zero_grad() # Clear old gradients
    
    # Forward pass: Feed the ENTIRE graph through the model
    out = model(graph.x, graph.edge_index)
    
    # Calculate loss ONLY on the training nodes using our mask
    loss = criterion(out[train_mask], graph.y[train_mask])
    
    loss.backward() # Backpropagation
    optimizer.step() # Update weights
    return loss.item()

# Train for 50 epochs (iterations)
for epoch in range(1, 51):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

print("\nTraining Complete! You just built your first GNN.")
from sklearn.metrics import classification_report

print("\n--- Phase 3: Evaluation ---")
# Set the model to evaluation mode (turns off training-specific layers like dropout)
model.eval() 

# Tell PyTorch we don't need to calculate gradients anymore, saving memory
with torch.no_grad(): 
    # Forward pass on the whole graph
    out = model(graph.x, graph.edge_index)
    
    # The model outputs two numbers per node (probability of Licit vs Illicit). 
    # argmax(dim=1) picks the class with the highest probability.
    pred = out.argmax(dim=1)
    
    # Filter the predictions and true labels to ONLY look at our 20% test set
    test_pred = pred[test_mask].cpu().numpy()
    test_true = graph.y[test_mask].cpu().numpy()

# Print the official classification report
print("\nFinal Baseline GCN Results:")
print(classification_report(test_true, test_pred, target_names=['Licit (0)', 'Illicit (1)'], zero_division=0))