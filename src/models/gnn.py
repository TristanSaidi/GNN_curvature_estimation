import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GCNRegressor(torch.nn.Module):
	def __init__(self, num_node_features, hidden_channels):
		super(GCNRegressor, self).__init__()
		self.conv1 = GCNConv(num_node_features, hidden_channels, dtype=torch.float32)
		self.conv2 = GCNConv(hidden_channels, hidden_channels, dtype=torch.float32)
		self.conv3 = GCNConv(hidden_channels, hidden_channels, dtype=torch.float32)
		self.lin = Linear(hidden_channels, 1) # regress to scalar curvature est for central vertex of graph

	def forward(self, x, edge_index, edge_attrs, batch):
		x = self.conv1(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv3(x, edge_index, edge_attrs)
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		x = F.dropout(x, p=0.1, training=self.training)
		x = self.lin(x)
		return x
	
	def get_num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
	

class GATRegressor(torch.nn.Module):
	def __init__(self, num_node_features, hidden_channels, heads=2):
		super(GATRegressor, self).__init__()
		self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.1)
		self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.1)
		self.conv3 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=0.1)
		self.lin = Linear(hidden_channels*heads, 1) # regress to scalar curvature est for central vertex of graph

	def forward(self, x, edge_index, edge_attrs, batch):
		x = self.conv1(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv3(x, edge_index, edge_attrs)
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		x = self.lin(x)
		return x
	
	def get_num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)