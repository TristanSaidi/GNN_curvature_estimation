import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import os
from tqdm import tqdm
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree, GDC
from torch_geometric.utils import dropout_node, dropout_edge, to_undirected
from src.utils import StandardScalerTorch
import src.manifold as manifold
from termcolor import cprint

# from torch_geometric.utils import dropout_adj, dropout_node

class ManifoldGraphDataset(Dataset):
    def __init__(self, full_graphs, subgraph_k, degree_features, subsample_pctg=0.05, avoid_boundary=True, scale_features=False, edge_attrs="distance"):
        super(ManifoldGraphDataset, self).__init__()
        self.subgraph_k = subgraph_k
        self.subsample_pctg = subsample_pctg
        self.full_graphs = full_graphs
        self.degree_features = degree_features
        self.avoid_boundary = avoid_boundary
        self.scale_features = scale_features # whether to scale features - only use when features are 'nbr_distances'
        cprint(f'Scaling features', "red", attrs=['bold']) if self.scale_features else cprint('Not scaling features')
        self.edge_attrs = edge_attrs
        assert self.edge_attrs in ["distance", "connectivity", "affinity"]
        if self.degree_features:
            self.one_hot_transform = OneHotDegree(max_degree=10)
        self.prepare_data()

    def prepare_data(self):
        data_list = []
        # iterate through full graphs
        for (name, data) in self.full_graphs.items():
            full_graph = data['graph']
            # if graph is not undirected, make it undirected
            full_graph.edge_index, full_graph.edge_attr = to_undirected(full_graph.edge_index.transpose(0,1), full_graph.edge_attr)
            full_graph.edge_index = full_graph.edge_index.transpose(0,1)
            X = data['coords']
            # choose random subset of indices for constructing subgraphs
            if self.subsample_pctg < 1:
                indices = np.random.choice(full_graph.x.shape[0], int(self.subsample_pctg * full_graph.x.shape[0]), replace=False).tolist()
            else:
                indices = list(range(full_graph.x.shape[0]))

            # if we are dealing with Poincare disks, we want to ignore nodes too close to the boundary
            if 'poincare' in name and self.avoid_boundary:
                # fetch curvature from name
                str_list = name.split('_')
                # get index of 'K'
                idx = str_list.index('K')
                # get curvature value
                K = float(str_list[idx+1])
                # get hyperbolic radius
                idx = str_list.index('Rh') if 'Rh' in str_list else -1
                if idx == -1:
                    Rh = 1
                else:
                    Rh = float(str_list[idx+1])
                print(f'Curvature: {K}, Hyperbolic radius: {Rh}')
                # get distance from boundary
                norms = []
                assert X.shape[0] == full_graph.x.shape[0]
                for row in range(X.shape[0]):
                    norm = manifold.PoincareDisk.norm(X[row], K)
                    norms.append(norm)
                norms = np.stack(norms)
                near_boundary_indices = np.argwhere(norms >= Rh*0.85)[:,0]
                indices = list(set(indices) - set(near_boundary_indices))
                print(f'Number of nodes after removing boundary nodes (Poincare): {len(indices)}')

            if 'euclidean' in name and self.avoid_boundary:
                str_list = name.split('_')
                idx = str_list.index('rad') if 'rad' in str_list else -1
                if idx == -1:
                    rad = 1
                else:
                    rad = float(str_list[idx+1])
                print(f'Euclidean radius: {rad}')
                # get distance from boundary
                norms = []
                assert X.shape[0] == full_graph.x.shape[0]
                for row in range(X.shape[0]):
                    norm = np.linalg.norm(X[row])
                    norms.append(norm)
                norms = np.stack(norms)
                near_boundary_indices = np.argwhere(norms >= rad*0.85)[:,0]
                indices = list(set(indices) - set(near_boundary_indices))
                print(f'Number of nodes after removing boundary nodes (Euclidean): {len(indices)}')

            nodes_in_subgraph = []
            edges_in_subgraph = []
            subgraph_node_indices = [] # store node masks for each subgraph
            for idx in tqdm(indices, desc=f'Processing new graph: {name}'):
                subgraph_nodes, subgraph_edges, inv, edge_mask = k_hop_subgraph(idx, self.subgraph_k, full_graph.edge_index.transpose(0,1).long(), relabel_nodes=True, directed=False)
                edge_indices = edge_mask.clone().nonzero(as_tuple=False)
                if len(edge_indices.shape) == 2:
                    edge_indices = edge_indices.squeeze(1)
                
                # fetch edge attributes
                if self.edge_attrs == "distance":
                    edge_attrs = full_graph.edge_attr[edge_indices].clone()
                elif self.edge_attrs == "connectivity":
                    edge_attrs = torch.ones_like(full_graph.edge_attr[edge_indices]).float()
                elif self.edge_attrs == "affinity":
                    edge_attrs = torch.exp(-full_graph.edge_attr[edge_indices].clone()).float()
                
                x = full_graph.x[subgraph_nodes].clone()
                y = full_graph.y[idx].clone()
                # store node indices
                subgraph_node_indices.append(subgraph_nodes)
                nodes_in_subgraph.append(x.shape[0])
                edges_in_subgraph.append(edge_attrs.shape[0])

                scale = None
                # if we are scaling features
                if self.scale_features:
                    # fetch scale
                    central_node_features = full_graph.x[idx].clone()
                    scale = central_node_features[-1]  
                    # scale features
                    x = x / scale
                    # scale labels
                    y = y / (scale ** 2)

                subgraph = Data(x=x, edge_index=subgraph_edges, edge_attr=edge_attrs, y=y, scale=scale)
                if self.degree_features:
                    subgraph = self.one_hot_transform(subgraph)
                data_list.append(subgraph)
            self.subgraph_node_indices = subgraph_node_indices
            print(f'Average num of nodes in subgraph: {sum(nodes_in_subgraph)/len(nodes_in_subgraph)}')
            print(f'Average num of edges in subgraph: {sum(edges_in_subgraph)/len(edges_in_subgraph)}\n')
        
        self.data = data_list

    def len(self):
        return len(self.data)
    
    def get(self, idx):
        desired_data = self.data[idx].clone()
        return desired_data