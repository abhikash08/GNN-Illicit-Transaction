import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report
from preprocess import load_elliptic_dataset 

print("--- Step 1: Loading Processed Graph Data ---")
graph = load_elliptic_dataset(".") 

print("\n--- Step 2: Defining the GraphSAGE Architecture ---")
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # Using SAGEConv instead of GCNConv
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Adding Dropout: Randomly turns off 50% of neurons during training to prevent memorization
        x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)
        return x

# Initialize GraphSAGE
model = GraphSAGE(in_channels=graph.num_node_features, hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Keeping the Focal Loss that gave you that amazing 0.93 Recall!
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = torch.tensor([0.25, 0.75]) 

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

criterion = FocalLoss(gamma=2.0)

print("\n--- Step 3: Creating Train/Test Splits ---")
known_indices = graph.known_mask.nonzero(as_tuple=False).view(-1)
num_known = known_indices.size(0)

# Randomly shuffle the known nodes (Using a fixed seed here so results are somewhat stable)
torch.manual_seed(42) 
indices = known_indices[torch.randperm(num_known)]

train_size = int(0.8 * num_known)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
test_mask[test_indices] = True

print("\n--- Step 4: Training GraphSAGE ---")
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph.x, graph.edge_index)
    loss = criterion(out[train_mask], graph.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Train for 100 epochs this time, as GraphSAGE sometimes needs a bit longer to settle
for epoch in range(1, 101):
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

print("\n--- Phase 3: Evaluation ---")
model.eval() 
with torch.no_grad(): 
    out = model(graph.x, graph.edge_index)
    pred = out.argmax(dim=1)
    
    test_pred = pred[test_mask].cpu().numpy()
    test_true = graph.y[test_mask].cpu().numpy()

print("\nFinal GraphSAGE Results:")
print(classification_report(test_true, test_pred, target_names=['Licit (0)', 'Illicit (1)'], zero_division=0))