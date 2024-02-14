import numpy as np
import manifold
import matplotlib.pyplot as plt
from sklearn import neighbors
import torch
from torch_geometric.data import Data
import os
import argparse

def adjmat_to_edgelist(adjmat):
    rows, cols = adjmat.nonzero()
    edges = np.array([rows, cols]).T
    edge_attrs = np.array(adjmat[rows, cols]).T
    return edges, edge_attrs

def create_sphere_graph(R, N, d=2, k=10, device='cpu', path=None):
    # Sample from manifold
    X = manifold.Sphere.sample(N, d, R)
    # Create nearest neighbors graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features = X
    node_features = torch.tensor(X).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    curvature = 1/(R ** 2)
    node_labels = np.ones((node_features.shape[0], 1)) * curvature
    node_labels = torch.tensor(node_labels).to(device)

    assert node_features.shape == (N, 3)
    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    sphere_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)
    if path is not None:
        torch.save(sphere_data, path)
    return sphere_data

def create_torus_graph(inner_radius, outer_radius, N, k=10, device='cpu', path=None):
    # Sample from manifold
    X, thetas = manifold.Torus.sample(N, inner_radius, outer_radius)
    # Create nearest neighbors graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features = torch.tensor(X).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    curvature = manifold.Torus.exact_curvatures(thetas, inner_radius, outer_radius)
    node_labels = torch.tensor(curvature).unsqueeze(1).to(device)
    torus_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)

    assert node_features.shape == (N, 3)
    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    if path is not None:
        torch.save(torus_data, path)
    return torus_data


def create_euclidean_graph(N, d, rad, k=10, device='cpu', path=None):
    # Sample from manifold
    X = manifold.Euclidean.sample(N, d, rad)
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features = torch.tensor(X).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    curvature = 0
    node_labels = torch.tensor(np.ones((N, 1)) * curvature).to(device)
    euclidean_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)

    assert node_features.shape == (N, d)
    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)
    
    if path is not None:
        torch.save(euclidean_data, path)
    return euclidean_data


def create_poincare_graph(N, K, k, Rh, device='cpu', path=None):
    X = manifold.PoincareDisk.sample(N, K, Rh)
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features = torch.tensor(X).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    curvature = K
    node_labels = torch.tensor(np.ones((N, 1)) * curvature).to(device)
    poincare_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)

    assert node_features.shape == (N, 2)
    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    if path is not None:
        torch.save(poincare_data, path)
    return poincare_data


def main():
    argparser = argparse.ArgumentParser(
        description='Generate data for curvature estimation experiments'
    )
    argparser.add_argument(
        '--output_dir', 
        type=str, 
        default='data', 
        help='Directory to save generated data'
    )
    argparser.add_argument(
        '--N', 
        type=int, 
        default=5000, 
        help='Number of nodes in the graph for each manifold'
    )
    argparser.add_argument(
        '--k', 
        type=int, 
        default=10, 
        help='Number of neighbors for each node in the graph'
    )

    args = argparser.parse_args()
    output_dir = args.output_dir
    
    N = args.N # Number of nodes
    k = args.k # Number of neighbors

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create sphere data
    R = 1
    d = 2
    create_sphere_graph(R, N, d, k, path=os.path.join(output_dir, f'sphere_rad_{R}_nodes_{N}_dim_{d}_k_{k}.pt'))
    # Create torus data
    inner_radius = 1
    outer_radius = 2
    create_torus_graph(inner_radius, outer_radius, N, k, path=os.path.join(output_dir, f'torus_inrad_{inner_radius}_outrad_{outer_radius}_nodes_{N}_k_{k}.pt'))
    # Create euclidean data
    d = 3
    rad = 2
    create_euclidean_graph(N, d, rad, k, path=os.path.join(output_dir, f'euclidean_dim_{d}_rad_{rad}_nodes_{N}_k_{k}.pt'))
    # Create poincare data
    # K = -1
    # Rh = 1
    # create_poincare_graph(N, K, k, Rh, path=os.path.join(output_dir, f'poincare_K_{K}_nodes_{N}_k_{k}.pt'))
    return

if __name__ == '__main__':
    main()