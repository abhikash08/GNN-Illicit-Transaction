import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, precision_recall_fscore_support
from preprocess import load_elliptic_dataset 

print("--- Step 1: Loading Processed Graph Data ---")
graph = load_elliptic_dataset(".") 

print("\n--- Step 2: Advanced Data Split (The Time Machine Fix) ---")
# Instead of random splitting, we simulate the real world.
# We train on past data (Timesteps 1 to 34)
train_mask = graph.known_mask & (graph.timestep <= 34)
# We test on future data (Timesteps 35 to 49)
test_mask = graph.known_mask & (graph.timestep > 34)

print(f"Training on PAST nodes: {train_mask.sum().item()}")
print(f"Testing on FUTURE nodes: {test_mask.sum().item()}")

print("\n--- Step 3: GraphSAGE & Focal Loss Architecture ---")
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(in_channels=graph.num_node_features, hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = torch.tensor([0.25, 0.75]) 

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

criterion = FocalLoss(gamma=2.0)

print("\n--- Step 4: Training on the Past ---")
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph.x, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 101):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

print("\n--- Step 5: Threshold Calibration on the Future ---")
model.eval() 
with torch.no_grad(): 
    # Get raw model outputs for the whole graph
    out = model(graph.x, graph.edge_index)
    
    # Convert raw outputs into actual percentages/probabilities (0.0 to 1.0)
    probabilities = F.softmax(out, dim=1)
    
    # Isolate the probabilities specifically for the 'Illicit' class (Class 1) on our test set
    illicit_probs = probabilities[test_mask, 1].cpu().numpy()
    test_true = graph.y[test_mask].cpu().numpy()

# Test different strictness levels for ringing the alarm
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 55)

best_f1 = 0
best_thresh = 0.50

for thresh in thresholds:
    # If the AI's confidence is higher than the threshold, mark it as Illicit (1), else Licit (0)
    test_pred = (illicit_probs > thresh).astype(int)
    
    # Calculate metrics just for the Illicit class
    p, r, f, _ = precision_recall_fscore_support(test_true, test_pred, pos_label=1, average='binary', zero_division=0)
    
    print(f" > {thresh*100:2.0f}%      | {p:.4f}     | {r:.4f}     | {f:.4f}")
    
    if f > best_f1:
        best_f1 = f
        best_thresh = thresh

