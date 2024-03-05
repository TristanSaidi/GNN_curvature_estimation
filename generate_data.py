import numpy as np
import src.manifold as manifold
import matplotlib.pyplot as plt
from sklearn import neighbors
import torch
from torch_geometric.data import Data
from src.curvature import *
import os
import argparse
from gtda.graphs import KNeighborsGraph
from scipy import sparse

def get_features(d, X, features_max_k, pairwise_dists=None, feature_type='ball_ratios'):
    assert feature_type in ['ball_ratios', 'nbr_distances', 'ball_volumes', 'e_radius']
    sce = scalar_curvature_est(d, X, Rdist=pairwise_dists, verbose=True)
    pairwise_dists = sce.Rdist
    if feature_type == 'e_radius':
        v_i = get_e_radius_features(features_max_k, pairwise_dists)
        return v_i, sce
    v_i = [] # list of features for each vertex
    for i in range(X.shape[0]):
        data = eval(f'sce.{feature_type}')(i, max_k=features_max_k)
        if feature_type == 'ball_ratios' or feature_type == 'ball_volumes':
            _, features = data
        if feature_type == 'nbr_distances':
            features, _ = data
        v_i.append(features)
    v_i = np.array(v_i)
    return v_i, sce


def get_e_radius_features(features_max_k, pairwise_dists, max_epsilon=1):
    # create features_max_k bins from 0 to max_pairwise_dist
    bins = np.linspace(0, max_epsilon, features_max_k+1)
    # for each vertex, compute number of elements in each bin
    v_i = []
    for i in range(pairwise_dists.shape[0]):
        hist, _ = np.histogram(pairwise_dists[i], bins=bins)
        # normalize histogram by number of points in this subpatch
        hist = hist/pairwise_dists.shape[0]
        v_i.append(hist)
    v_i = np.array(v_i)
    return v_i


def adjmat_to_edgelist(adjmat):
    rows, cols = adjmat.nonzero()
    edges = np.array([rows, cols]).T
    edge_attrs = np.array(adjmat[rows, cols]).T
    return edges, edge_attrs

def create_sphere_dataset(R, N, d=2, k=10, features_max_k=2500, device='cpu', path=None, scale_labels=False, features='ball_ratios'):
    print(f'Creating sphere dataset with {N} nodes, dimension {d}, and radius {R}')
    # Sample from manifold
    X = manifold.Sphere.sample(N, d, R=R)
    # Create nearest neighbors graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features, sce = get_features(d, X, features_max_k, feature_type=features)
    node_features = torch.tensor(node_features).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    curvature = (d)*(d-1)/(R ** 2)
    node_labels = np.ones((node_features.shape[0], 1)) * curvature
    if scale_labels:
        scaling_factors = []
        for i in range(X.shape[0]):
            nbrs = sce.nbr_distances(i, max_k=features_max_k)
            scaling_factors.append(nbrs[0][-1]**2)
        scaling_factors = np.expand_dims(np.array(scaling_factors), -1)
        node_labels = node_labels / scaling_factors
    node_labels = torch.tensor(node_labels).to(device)

    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    sphere_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)
    if path is not None:
        data_dict = {
            "coords" : X,
            "graph" : sphere_data
        }
        torch.save(data_dict, path)
    return sphere_data, X

def create_torus_dataset(inner_radius, outer_radius, N, k=10, features_max_k=2500, device='cpu', path=None, scale_labels=False, features='ball_ratios'):
    print(f'Creating torus dataset with {N} nodes, inner radius {inner_radius}, and outer radius {outer_radius}')
    # Sample from manifold
    X, thetas = manifold.Torus.sample(N, inner_radius, outer_radius)
    # Create nearest neighbors graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features, sce = get_features(2, X, features_max_k, feature_type=features)
    node_features = torch.tensor(node_features).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.expand_dims(manifold.Torus.exact_curvatures(thetas, inner_radius, outer_radius), -1)
    if scale_labels:
        scaling_factors = []
        for i in range(X.shape[0]):
            nbrs = sce.nbr_distances(i, max_k=features_max_k)
            scaling_factors.append(nbrs[0][-1]**2)
        scaling_factors = np.expand_dims(np.array(scaling_factors), -1)
        node_labels = node_labels / scaling_factors
    node_labels = torch.tensor(node_labels).to(device)
    torus_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)

    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    if path is not None:
        data_dict = {
            "coords" : X,
            "graph" : torus_data
        }
        torch.save(data_dict, path)
    return torus_data, X


def create_euclidean_dataset(N, d, rad, k=10, features_max_k=2500, device='cpu', path=None, features='ball_ratios'):
    print(f'Creating euclidean dataset with {N} nodes, dimension {d}, and radius {rad}')
    # Sample from manifold
    X = manifold.Euclidean.sample(N, d, rad)
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features, sce = get_features(d, X, features_max_k, feature_type=features)
    node_features = torch.tensor(node_features).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.zeros((N, 1))
    node_labels = torch.tensor(node_labels).to(device)

    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)
    
    euclidean_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)

    if path is not None:
        data_dict = {
            "coords" : X,
            "graph" : euclidean_data
        }
        torch.save(data_dict, path)
    return euclidean_data, X


def create_poincare_dataset(N, K, k, Rh, features_max_k=2500, device='cpu', path=None, scale_labels=False, features='ball_ratios'):
    print(f'Creating poincare dataset with {N} nodes, curvature {K}, and radius {Rh}')
    X = manifold.PoincareDisk.sample(N=N, K=K, Rh=Rh)
    # data isn't embedded in euclidean space --> need to compute hyperbolic distances for kNN
    pairwise_dists = manifold.PoincareDisk.Rdist_array(X, K=K)
    sparse_pairwise_dists = sparse.csr_matrix(pairwise_dists)
    knn_graph = KNeighborsGraph(n_neighbors=k, metric='precomputed')
    adjacency_mat = knn_graph.fit_transform([sparse_pairwise_dists])[0]
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)

    dim = 2
    node_features, sce = get_features(dim, X, features_max_k, pairwise_dists, feature_type=features)
    node_features = torch.tensor(node_features).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.ones((N, 1)) * K
    if scale_labels:
        scaling_factors = []
        for i in range(X.shape[0]):
            nbrs = sce.nbr_distances(i, max_k=features_max_k)
            scaling_factors.append(nbrs[0][-1]**2)
        scaling_factors = np.expand_dims(np.array(scaling_factors), -1)
        node_labels = node_labels / scaling_factors
    node_labels = torch.tensor(node_labels).to(device)

    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    poincare_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)
    if path is not None:
        data_dict = {
            "coords" : X,
            "graph" : poincare_data
        }
        torch.save(data_dict, path)
    return poincare_data, X

def create_hyperbolic_dataset(N, k=10, features_max_k=2500, device='cpu', path=None, scale_labels=False, features='ball_ratios'):
    print(f'Creating hyperbolic dataset with {N} nodes')
    X = manifold.Hyperboloid.sample(N)
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    node_features, sce = get_features(2, X, features_max_k, feature_type=features)
    node_features = torch.tensor(node_features).to(device)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.expand_dims(manifold.Hyperboloid.S(X[:, -1]), -1)
    if scale_labels:
        scaling_factors = []
        for i in range(X.shape[0]):
            nbrs = sce.nbr_distances(i, max_k=features_max_k)
            scaling_factors.append(nbrs[0][-1]**2)
        scaling_factors = np.expand_dims(np.array(scaling_factors), -1)
        node_labels = node_labels / scaling_factors
    node_labels = torch.tensor(node_labels).to(device)

    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    hyperbolic_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)
    if path is not None:
        data_dict = {
            "coords" : X,
            "graph" : hyperbolic_data
        }
        torch.save(data_dict, path)
    return hyperbolic_data, X



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
    argparser.add_argument(
        '--features_max_k',
        type=int,
        default=2500,
        help='Maximum k for computing ball ratios'
    )
    argparser.add_argument(
        '--scale_labels',
        type=bool,
        default=False,
    )
    argparser.add_argument(
        '--features',
        type=str,
        choices=['ball_ratios', 'nbr_distances', 'ball_volumes', 'e_radius'],
        help='node features to create'
    )

    args = argparser.parse_args()
    output_dir = args.output_dir
    scale_labels=args.scale_labels

    N = args.N # Number of nodes
    k = args.k # Number of neighbors
    features = args.features
    features_max_k = args.features_max_k # Maximum k for computing ball ratios

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create 2-sphere data
    d = 2
    rs = [2.82, 2, 1.633, 1.414, 1.265, 1.15, 1.069, 1] # curvatures [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    for R in rs:
        create_sphere_dataset(R, N, d, k, path=os.path.join(output_dir, f'sphere_dim_{d}_rad_{R}_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    rs = [2.31, 1.032]
    for R in rs:
        create_sphere_dataset(R, N, d, k, path=os.path.join(output_dir, f'sphere_dim_{d}_rad_{R}_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    # Create 3-sphere data
    # d = 3
    # rs = [1, 2] # curvatures = (6, 1.5)
    # for R in rs:
    #     create_sphere_dataset(R, N, d, k, path=os.path.join(output_dir, f'sphere_dim_{d}_rad_{R}_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    # # # Create 5-sphere data
    # # d = 5
    # # rs = [3, 4] # curvatures = (20/9, 5/4)
    # # for R in rs:
    # #     create_sphere_dataset(R, N, d, k, path=os.path.join(output_dir, f'sphere_dim_{d}_rad_{R}_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    # # Create torus data
    # rads = [(1, 2)]
    # for inner_radius, outer_radius in rads:
    #     create_torus_dataset(inner_radius, outer_radius, N, k, path=os.path.join(output_dir, f'torus_inrad_{inner_radius}_outrad_{outer_radius}_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    # # Create euclidean data
    # rad = 1
    # # ds = [2, 3, 4]
    # # for d in ds:
    # d = 2
    # create_euclidean_dataset(N, d, rad, k, path=os.path.join(output_dir, f'euclidean_dim_{d}_rad_{rad}_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, features=features)
    # # Create poincare data
    Rh = 2
    Ks = [-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]
    for K in Ks:
        create_poincare_dataset(N, K, k, Rh, path=os.path.join(output_dir, f'poincare_K_{K}_nodes_{N}_Rh_{Rh}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    Ks = [-0.375, -1.875]
    for K in Ks:
        create_poincare_dataset(N, K, k, Rh, path=os.path.join(output_dir, f'poincare_K_{K}_nodes_{N}_Rh_{Rh}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    # # Create hyperbolic data
    # create_hyperbolic_dataset(N, k, path=os.path.join(output_dir, f'hyperbolic_nodes_{N}_k_{k}.pt'), features_max_k=features_max_k, scale_labels=scale_labels, features=features)
    return

if __name__ == '__main__':
    main()