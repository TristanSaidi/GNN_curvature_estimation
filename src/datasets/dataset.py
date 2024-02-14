import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import os
from tqdm import tqdm

class ManifoldGraphDataset(Dataset):
    def __init__(self, root):
        super(ManifoldGraphDataset, self).__init__(root)
        self.file_list = os.listdir(root)
        self.data = self.load(root, self.file_list)

    def load(self, root, file_list):
        data_list = []
        for file in file_list:
            full_graph = torch.load(os.path.join(root, file))
            for i in tqdm(range(full_graph.x.shape[0]), desc=f'Processing {file}'):
                subgraph_nodes, subgraph_edges, mapping, _ = k_hop_subgraph(i, 10, full_graph.edge_index.transpose(0,1).long(), relabel_nodes=True, directed=False)
                subgraph = Data(x=full_graph.x[subgraph_nodes], edge_index=subgraph_edges, edge_attr=full_graph.edge_attr[subgraph_edges[0]], y=full_graph.y[i])
                data_list.append(subgraph)
        return data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]