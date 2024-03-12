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

def get_ball_ratio_features_rmax(N, sce, features_max_k, rmax):
    r_values = np.linspace(0, rmax, features_max_k)
    v_i = [] # list of ball ratios for each vertex
    for i in range(N):
        nbr_distances, ball_ratios = sce.ball_ratios(i, rmax=rmax)
        # linearly space r values from 0 to rmax based on features_max_k
        # for each r_value take the ball_volume of closest radius
        ball_ratios = np.array([ball_ratios[np.argmin(np.abs(nbr_distances - r))] for r in r_values])
        v_i.append(ball_ratios)
    v_i = np.array(v_i)
    return v_i

def get_ball_ratio_features(N, sce, features_max_k):
    v_i = [] # list of features for each vertex
    for i in range(N):
        _, ball_ratios = sce.ball_ratios(i=i, max_k=features_max_k)
        v_i.append(ball_ratios)
    v_i = np.array(v_i)
    return v_i

def get_e_radius_features(N, sce, features_max_k, rmax):
    pairwise_dists = sce.Rdist
    # create features_max_k bins from 0 to max_pairwise_dist
    bins = np.linspace(0, rmax, features_max_k+1)
    # for each vertex, compute number of elements in each bin
    v_i = []
    for i in range(N):
        hist, _ = np.histogram(pairwise_dists[i], bins=bins)
        # normalize histogram by number of points in this subpatch
        hist = hist/pairwise_dists.shape[0]
        v_i.append(hist)
    v_i = np.array(v_i)
    return v_i

def get_nbr_distance_features(N, sce, features_max_k, scale=False):
    v_i = [] # list of features for each vertex
    for i in range(N):
        features, _ = sce.nbr_distances(i=i, max_k=features_max_k)
        v_i.append(features)
    v_i = np.array(v_i)
    if scale:
        v_i = v_i / v_i[:, -1][:, None]
    return v_i

def get_features(sce, N, features_max_k, feature_type='ball_ratios', rmax=None):
    assert feature_type in ['ball_ratios', 'nbr_distances', 'e_radius']
    # ball ratio features
    if feature_type == 'ball_ratios':
        if rmax is not None:
            v_i = get_ball_ratio_features_rmax(N=N, sce=sce, features_max_k=features_max_k, rmax=rmax)
            return v_i, sce
        else:
            v_i = get_ball_ratio_features(N=N, sce=sce, features_max_k=features_max_k)
            return v_i, sce
    
    # e-rad features
    if feature_type == 'e_radius':
        assert rmax is not None, 'rmax must be specified for e_radius features'
        v_i = get_e_radius_features(N=N, sce=sce, features_max_k=features_max_k, rmax=rmax)
        return v_i, sce
    
    # nbr distance features
    if feature_type == 'nbr_distances':
        v_i = get_nbr_distance_features(N=N, sce=sce, features_max_k=features_max_k)
        return v_i, sce


def adjmat_to_edgelist(adjmat):
    rows, cols = adjmat.nonzero()
    edges = np.array([rows, cols]).T
    edge_attrs = np.array(adjmat[rows, cols]).T
    return edges, edge_attrs


def create_sphere_dataset(
        R, 
        N, 
        d=2, 
        k=10, 
        features_max_k=2500, 
        device='cpu', 
        path=None, 
        features='ball_ratios', 
        rmax=None
    ):
    print(f'Creating sphere dataset with {N} nodes, dimension {d}, and radius {R}')
    # Sample from manifold
    X = manifold.Sphere.sample(N=N, n=d, R=R)
    sce = scalar_curvature_est(n=d, X=X, n_nbrs=k, Rdist=None, verbose=True)
    # grab adjacency matrix for kNN graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # get node features
    node_features, _ = get_features(sce=sce, N=N, features_max_k=features_max_k, feature_type=features, rmax=rmax)
    node_features = torch.tensor(node_features).to(device)
    # Create node labels (scalar curvature in this case)
    curvature = (d)*(d-1)/(R ** 2)
    node_labels = np.ones((node_features.shape[0], 1)) * curvature
    node_labels = torch.tensor(node_labels).to(device)
    # print(edge_list.shape, edge_attrs.shape, node_features.shape, node_labels.shape)
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


def create_torus_dataset(
        inner_radius, 
        outer_radius, 
        N, 
        k=10, 
        features_max_k=2500, 
        device='cpu', 
        path=None, 
        features='ball_ratios', 
        rmax=None
    ):
    print(f'Creating torus dataset with {N} nodes, inner radius {inner_radius}, and outer radius {outer_radius}')
    # Sample from manifold
    X, thetas = manifold.Torus.sample(N, inner_radius, outer_radius)
    sce = scalar_curvature_est(n=2, X=X, n_nbrs=k, Rdist=None, verbose=True)
    # grab adjacency matrix for kNN graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # get node features
    node_features, _ = get_features(sce=sce, N=N, features_max_k=features_max_k, feature_type=features, rmax=rmax)
    node_features = torch.tensor(node_features).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.expand_dims(manifold.Torus.exact_curvatures(thetas, inner_radius, outer_radius), -1)
    node_labels = torch.tensor(node_labels).to(device)
    # print(edge_list.shape, edge_attrs.shape, node_features.shape, node_labels.shape)
    assert edge_list.shape == (N*k, 2)
    assert edge_attrs.shape == (N*k,1)
    assert node_labels.shape == (N,1)

    torus_data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_attrs, y=node_labels)
    if path is not None:
        data_dict = {
            "coords" : X,
            "graph" : torus_data
        }
        torch.save(data_dict, path)
    return torus_data, X


def create_euclidean_dataset(
        N, 
        d, 
        rad, 
        k=10, 
        features_max_k=2500, 
        device='cpu', 
        path=None, 
        features='ball_ratios', 
        rmax=None
    ):
    print(f'Creating euclidean dataset with {N} nodes, dimension {d}, and radius {rad}')
    # Sample from manifold
    X = manifold.Euclidean.sample(N=N, n=d, R=rad)
    sce = scalar_curvature_est(n=d, X=X, n_nbrs=k, Rdist=None, verbose=True)
    # grab adjacency matrix for kNN graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # get node features
    node_features, _ = get_features(sce=sce, N=N, features_max_k=features_max_k, feature_type=features, rmax=rmax)
    node_features = torch.tensor(node_features).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.zeros((node_features.shape[0], 1))
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


def create_poincare_dataset(
        N, 
        K, 
        k, 
        Rh, 
        features_max_k=2500, 
        device='cpu', 
        path=None, 
        features='ball_ratios', 
        rmax=None
    ):
    print(f'Creating poincare dataset with {N} nodes, curvature {K}, and radius {Rh}')
    X = manifold.PoincareDisk.sample(N=N, K=K, Rh=Rh)
    # data isn't embedded in euclidean space --> need to compute hyperbolic distances for kNN
    pairwise_dists = manifold.PoincareDisk.Rdist_array(X, K=K)
    sparse_pairwise_dists = sparse.csr_matrix(pairwise_dists)
    
    sce = scalar_curvature_est(n=2, X=X, n_nbrs=k, Rdist=pairwise_dists, verbose=True)
    # create kNN graph
    knn_graph = KNeighborsGraph(n_neighbors=k, metric='precomputed', mode='distance')
    adjacency_mat = knn_graph.fit_transform([sparse_pairwise_dists])[0]
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # get node features
    node_features, _ = get_features(sce=sce, N=N, features_max_k=features_max_k, feature_type=features, rmax=rmax)
    node_features = torch.tensor(node_features).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.ones((N, 1)) * K
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



def create_hyperbolic_dataset(
        N, 
        k=10, 
        features_max_k=2500, 
        device='cpu', 
        path=None, 
        features='ball_ratios', 
        rmax=None
    ):
    print(f'Creating hyperbolic dataset with {N} nodes')
    X = manifold.Hyperboloid.sample(N=N)
    sce = scalar_curvature_est(n=2, X=X, n_nbrs=k, Rdist=None, verbose=True)
    # grab adjacency matrix for kNN graph
    adjacency_mat = neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    # Convert adjacency matrix to edge list
    edge_list, edge_attrs = adjmat_to_edgelist(adjacency_mat)
    edge_list = torch.tensor(edge_list).to(device)
    edge_attrs = torch.tensor(edge_attrs).to(device)
    # get node features
    node_features, _ = get_features(sce=sce, N=N, features_max_k=features_max_k, feature_type=features, rmax=rmax)
    node_features = torch.tensor(node_features).to(device)
    # Create node labels (scalar curvature in this case)
    node_labels = np.expand_dims(manifold.Hyperboloid.S(X[:, -1]), -1)
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
        '--features',
        type=str,
        choices=['ball_ratios', 'nbr_distances', 'e_radius'],
        help='node features to create'
    )
    argparser.add_argument(
        '--rmax',
        default=None,
    )

    args = argparser.parse_args()
    output_dir = args.output_dir

    N = args.N # Number of nodes
    k = args.k # Number of neighbors
    features = args.features
    features_max_k = args.features_max_k # Maximum k for computing ball ratios
    rmax = float(args.rmax) if args.rmax is not None else None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
                    # edge_attrs = torch.exp(-1*edge_attrs) # edge attributes are distances, so we want to convert to affinities

    # Create 2-sphere data
    d = 2
    rs = [2.82, 2, 1.633, 1.414, 1.265, 1.15, 1.069, 1] # curvatures [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    for R in rs:
        create_sphere_dataset(
            R=R, 
            N=N, 
            d=d, 
            k=k, 
            path=os.path.join(output_dir, f'sphere_dim_{d}_rad_{R}_nodes_{N}_k_{k}.pt'), 
            features_max_k=features_max_k, 
            features=features, 
            rmax=rmax
        )
    rs = [2.31, 1.032]
    for R in rs:
        create_sphere_dataset(
            R=R, 
            N=N, 
            d=d, 
            k=k, 
            path=os.path.join(output_dir, f'sphere_dim_{d}_rad_{R}_nodes_{N}_k_{k}.pt'), 
            features_max_k=features_max_k, 
            features=features, 
            rmax=rmax
        )
    rads = [(1, 2)]
    for inner_radius, outer_radius in rads:
        create_torus_dataset(
            inner_radius=inner_radius, 
            outer_radius=outer_radius, 
            N=N, 
            k=k, 
            path=os.path.join(output_dir, f'torus_inrad_{inner_radius}_outrad_{outer_radius}_nodes_{N}_k_{k}.pt'), 
            features_max_k=features_max_k, 
            features=features, 
            rmax=rmax
        )
    # # Create euclidean data
    rad = 1
    d = 2
    create_euclidean_dataset(
        N=N, 
        d=d, 
        rad=rad, 
        k=k, 
        path=os.path.join(output_dir, f'euclidean_dim_{d}_rad_{rad}_nodes_{N}_k_{k}.pt'), 
        features_max_k=features_max_k, 
        features=features, 
        rmax=rmax
    )
    # # Create poincare data
    Rh = 2
    Ks = [-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]
    for K in Ks:
        create_poincare_dataset(
            N=N, 
            K=K, 
            k=k, 
            Rh=Rh, 
            path=os.path.join(output_dir, f'poincare_K_{K}_nodes_{N}_Rh_{Rh}_k_{k}.pt'), 
            features_max_k=features_max_k, 
            features=features, 
            rmax=rmax
        )
    Ks = [-0.375, -1.875]
    for K in Ks:
        create_poincare_dataset(
            N=N, 
            K=K, 
            k=k, 
            Rh=Rh, 
            path=os.path.join(output_dir, f'poincare_K_{K}_nodes_{N}_Rh_{Rh}_k_{k}.pt'), 
            features_max_k=features_max_k, 
            features=features, 
            rmax=rmax
        )
    # Create hyperbolic data
    create_hyperbolic_dataset(
        N=N, 
        k=k, 
        path=os.path.join(output_dir, f'hyperbolic_nodes_{N}_k_{k}.pt'), 
        features_max_k=features_max_k, 
        features=features, 
        rmax=rmax
    )
    return

if __name__ == '__main__':
    main()