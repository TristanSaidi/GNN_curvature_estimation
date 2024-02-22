import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import degree

class GCNRegressor(torch.nn.Module):
	def __init__(self, num_node_features, hidden_channels, degree_features=False):
		super(GCNRegressor, self).__init__()
		self.num_node_features = num_node_features if not degree_features else 1
		self.degree_features = degree_features
		self.conv1 = GCNConv(self.num_node_features, hidden_channels, dtype=torch.float32)
		self.conv2 = GCNConv(hidden_channels, hidden_channels, dtype=torch.float32)
		self.conv3 = GCNConv(hidden_channels, hidden_channels, dtype=torch.float32)
		self.lin = Linear(hidden_channels, 1) # regress to scalar curvature est for central vertex of graph

	def forward(self, data):
		x, edge_index, edge_attrs, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
		if self.degree_features:
			x = degree(edge_index[0], data.num_nodes).unsqueeze(1).float()
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
	def __init__(self, num_node_features, hidden_channels, heads=2, dropout=0.1, degree_features=False):
		super(GATRegressor, self).__init__()
		self.num_node_features = num_node_features if not degree_features else 1
		self.degree_features = degree_features
		self.conv1 = GATConv(self.num_node_features, hidden_channels, heads=heads, dropout=dropout)
		self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout)
		self.conv3 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout)
		self.conv4 = GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout)
		self.lin1 = Linear(hidden_channels*heads, hidden_channels) # regress to scalar curvature est for central vertex of graph
		self.lin2 = Linear(hidden_channels, 1)
	
	def forward(self, data):
		x, edge_index, edge_attrs, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
		if self.degree_features:
			x = degree(edge_index[0], data.num_nodes).unsqueeze(1).float()
		x = self.conv1(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv3(x, edge_index, edge_attrs)
		x = F.relu(x)
		x = self.conv4(x, edge_index, edge_attrs)
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		x = self.lin1(x)
		x = F.relu(x)
		x = self.lin2(x)
		return x
	
	def get_num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)