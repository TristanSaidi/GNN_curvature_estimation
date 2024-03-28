import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import degree
from torch_geometric.utils import dropout_edge, dropout_node


class GCNRegressor(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, num_layers, dropout):
		super(GCNRegressor, self).__init__()
		self.num_layers = num_layers
		self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
		for i in range(2, num_layers + 1):
			setattr(self, f'conv{i}', GCNConv(hidden_channels, hidden_channels, normalize=False))
		self.lin1 = Linear(hidden_channels, hidden_channels//2) # regress to scalar curvature est for central vertex of graph
		self.lin2 = Linear(hidden_channels//2, hidden_channels//4)
		self.lin3 = Linear(hidden_channels//4, 1)
		self.dropout = dropout
		
	def forward(self, x, edge_index, edge_weight, batch):
		# dropout edges
		# edge_index, edge_mask = dropout_edge(edge_index, p=0.1, training=self.training)
		# edge_weight = edge_weight[edge_mask]

		for i in range(1, self.num_layers + 1):
			x = eval(f'self.conv{i}')(x, edge_index, edge_weight)
			x = F.relu(x)
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.lin1(x)
		x = F.relu(x)
		x = self.lin2(x)
		x = F.relu(x)
		x = self.lin3(x)
		return x
	
	def get_num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
	
class GCNClassifier(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, num_layers, dropout):
		super(GCNClassifier, self).__init__()
		self.num_layers = num_layers
		self.conv1 = GCNConv(in_channels, hidden_channels, normalize=False)
		for i in range(2, num_layers + 1):
			setattr(self, f'conv{i}', GCNConv(hidden_channels, hidden_channels, normalize=False))
		self.lin1 = Linear(hidden_channels, hidden_channels//2) # regress to scalar curvature est for central vertex of graph
		self.lin2 = Linear(hidden_channels//2, hidden_channels//4)
		self.lin3 = Linear(hidden_channels//4, 3) # 3 classes [negative, zero, positive]
		self.dropout = dropout
		
	def forward(self, x, edge_index, edge_weight, batch):
		# dropout edges
		# edge_index, edge_mask = dropout_edge(edge_index, p=0.1, training=self.training)
		# edge_weight = edge_weight[edge_mask]

		for i in range(1, self.num_layers + 1):
			x = eval(f'self.conv{i}')(x, edge_index, edge_weight)
			x = F.relu(x)
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.lin1(x)
		x = F.relu(x)
		x = self.lin2(x)
		x = F.relu(x)
		x = self.lin3(x)
		return x
	
	def get_num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)




class GATRegressor(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, num_layers, dropout, heads=1):
		super(GATRegressor, self).__init__()
		self.num_layers = num_layers
		self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
		for i in range(2, num_layers + 1):
			setattr(self, f'conv{i}', GATConv(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout))
		self.lin = Linear(hidden_channels*heads, 1)
		self.dropout = dropout

	def forward(self, x, edge_index, edge_weight, batch):
		for i in range(1, self.num_layers + 1):
			x = F.relu(eval(f'self.conv{i}')(x, edge_index=edge_index, edge_attr=edge_weight))
		x = global_mean_pool(x, batch)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.lin(x)
		return x