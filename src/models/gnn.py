import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNRegressor(torch.nn.Module):
	def __init__(self, num_node_features, hidden_channels):
		super(GCNRegressor, self).__init__()
		self.conv1 = GCNConv(num_node_features, hidden_channels, dtype=torch.float32)
		self.lin = Linear(hidden_channels, 1) # regress to scalar curvature est for central vertex of graph

	def forward(self, x, edge_index, batch):
		# 1. Obtain node embeddings 
		x = self.conv1(x, edge_index)
		# 2. Readout layer
		x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		# 3. Apply a final linear transf.
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin(x)
		return x