import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import os
from tqdm import tqdm

class ManifoldGraphDataset(Dataset):
    def __init__(self, root, subgraph_k):
        super(ManifoldGraphDataset, self).__init__()
        self.file_list = os.listdir(root)
        self.subgraph_k = subgraph_k
        self.data = self.load(root, self.file_list)

    def load(self, root, file_list):
        data_list = []
        for file in file_list:
            full_graph = torch.load(os.path.join(root, file))
            for i in tqdm(range(full_graph.x.shape[0]), desc=f'Processing {file}'):
                subgraph_nodes, subgraph_edges, inv, edge_mask = k_hop_subgraph(i, self.subgraph_k, full_graph.edge_index.transpose(0,1).long(), relabel_nodes=False, directed=False)
                # print(len(subgraph_nodes), len(subgraph_edges[0]))
                edge_indices = edge_mask.nonzero(as_tuple=False).squeeze()
                subgraph = Data(x=full_graph.x[subgraph_nodes], edge_index=subgraph_edges, edge_attr=full_graph.edge_attr[edge_indices], y=full_graph.y[i])
                data_list.append(subgraph)
        return data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]