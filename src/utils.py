import torch
import logging
import community as community_louvain
import networkx as nx
import numpy as np
from functools import partial, partialmethod

logging.TRACE = logging.DEBUG + 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)

logger = logging.getLogger("GraphRicciCurvature")

EPSILON = 1e-5


def set_verbose(verbose="ERROR"):
    """
    Set up the verbose level of the GraphRicciCurvature.
    """
    if verbose == "INFO":
        logger.setLevel(logging.INFO)
    elif verbose == "TRACE":
        logger.setLevel(logging.TRACE)
    elif verbose == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif verbose == "ERROR":
        logger.setLevel(logging.ERROR)
    else:
        print('Incorrect verbose level, option:["INFO","DEBUG","ERROR"], use "ERROR instead."')
        logger.setLevel(logging.ERROR)


def cut_graph_by_cutoff(G_origin, cutoff, weight="weight"):
    """
    Remove graph's edges with "weight" greater than "cutoff".
    """
    assert nx.get_edge_attributes(G_origin, weight), "No edge weight detected, abort."

    G = G_origin.copy()
    edge_trim_list = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cutoff:
            edge_trim_list.append((n1, n2))
    G.remove_edges_from(edge_trim_list)
    return G


def get_rf_metric_cutoff(G_origin, weight="weight", cutoff_step=0.025, drop_threshold=0.01):
    """
    Get good clustering cutoff points for Ricci flow metric by detect the change of modularity while removing edges.
    """

    G = G_origin.copy()
    modularity, ari = [], []
    maxw = max(nx.get_edge_attributes(G, weight).values())
    cutoff_range = np.arange(maxw, 1, -cutoff_step)

    for cutoff in cutoff_range:
        G = cut_graph_by_cutoff(G, cutoff, weight=weight)
        # Get connected component after cut as clustering
        clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
        # Compute modularity
        modularity.append(community_louvain.modularity(clustering, G, weight))

    good_cuts = []
    mod_last = modularity[-1]

    # check drop from 1 -> maxw
    for i in range(len(modularity) - 1, 0, -1):
        mod_now = modularity[i]
        if mod_last > mod_now > 1e-4 and abs(mod_last - mod_now) / mod_last > drop_threshold:
            logger.trace("Cut detected: cut:%f, diff:%f, mod_now:%f, mod_last:%f" % (
                cutoff_range[i+1], mod_last - mod_now, mod_now, mod_last))
            good_cuts.append(cutoff_range[i+1])
        mod_last = mod_now

    return good_cuts

class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )
