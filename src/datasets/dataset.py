import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import os
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree


class ManifoldGraphDataset(Dataset):
    def __init__(self, full_graphs, subgraph_k, degree_features, subsample_pctg=0.1):
        super(ManifoldGraphDataset, self).__init__()
        self.subgraph_k = subgraph_k
        self.subsample_pctg = subsample_pctg
        self.full_graphs = [full_graphs] if not isinstance(full_graphs, list) else full_graphs
        self.degree_features = degree_features
        if self.degree_features:
            self.transform = OneHotDegree(max_degree=10)
        self.prepare_data()

    def prepare_data(self):
        data_list = []
        for full_graph in self.full_graphs:
            indices = np.random.choice(full_graph.x.shape[0], int(self.subsample_pctg * full_graph.x.shape[0]), replace=False).tolist()
            for idx in tqdm(indices, desc=f'Processing new graph'):
                subgraph_nodes, subgraph_edges, inv, edge_mask = k_hop_subgraph(idx, self.subgraph_k, full_graph.edge_index.transpose(0,1).long(), relabel_nodes=True, directed=False)
                edge_indices = edge_mask.clone().nonzero(as_tuple=False)
                if len(edge_indices.shape) == 2:
                    edge_indices = edge_indices.squeeze(1)
                subgraph = Data(x=full_graph.x[subgraph_nodes], edge_index=subgraph_edges, edge_attr=full_graph.edge_attr[edge_indices].clone(), y=full_graph.y[idx])
                if self.degree_features:
                    subgraph = self.transform(subgraph)
                data_list.append(subgraph)
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]