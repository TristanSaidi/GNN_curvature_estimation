import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

from src.orc import OllivierRicci

    
@functional_transform('local_curvature_profile')
class LocalCurvatureProfile(BaseTransform):
    """
    This class computes the local curvature profile structural encoding for each node in a graph.
    """
    def __init__(self, attr_name = 'lcp_se'):
        self.attr_name = attr_name
        

    def compute_orc(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_orc = []
        max_orc = []
        mean_orc = []
        std_orc = []
        median_orc = []

        for node in graph.nodes():
            if len(neighbors[node]) > 0:
                min_orc.append(min([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                max_orc.append(max([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                mean_orc.append(np.mean([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                std_orc.append(np.std([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                median_orc.append(np.median([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
            else:
                min_orc.append(0)
                max_orc.append(0)
                mean_orc.append(0)
                std_orc.append(0)
                median_orc.append(0)
                                                                      
        lcp_pe = torch.tensor([min_orc, max_orc, mean_orc, std_orc, median_orc]).T

        # move lcps to the GPU if available
        # if torch.cuda.is_available():
            # lcp_pe = lcp_pe.cuda()
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

    def forward(self, data: Data) -> Data:
        """
        Compute orc approximation. Serves as default, especially for large graphs/ datasets.
        """
        graph = to_networkx(data)
            
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        min_orc = []
        max_orc = []

        def compute_upper_bound(node_1, node_2):
            deg_node_1 = len(neighbors[node_1])
            deg_node_2 = len(neighbors[node_2])
            num_triangles = len([neighbor for neighbor in neighbors[node_1] if neighbor in neighbors[node_2]])
            return num_triangles / np.max([deg_node_1, deg_node_2])

        def compute_lower_bound(node_1, node_2):
            deg_node_1 = len(neighbors[node_1])
            deg_node_2 = len(neighbors[node_2])
            num_triangles = len([neighbor for neighbor in neighbors[node_1] if neighbor in neighbors[node_2]])
            return -np.max([0, 1 - 1/deg_node_1 - 1/deg_node_2 - num_triangles/np.min([deg_node_1, deg_node_2])]) - np.max([0, 1 - 1/deg_node_1 - 1/deg_node_2 - num_triangles/np.max([deg_node_1, deg_node_2])]) + num_triangles/np.max([deg_node_1, deg_node_2])

        for node in graph.nodes():
            if len(neighbors[node]) > 0:
                min_orc.append(min([compute_lower_bound(node, neighbor) for neighbor in neighbors[node]]))
                max_orc.append(max([compute_upper_bound(node, neighbor) for neighbor in neighbors[node]]))
            else:
                min_orc.append(0)
                max_orc.append(0)
                                                                      
        lcp_pe = torch.tensor([min_orc, max_orc]).T
    
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x

            if data.x.device != lcp_pe.device:
                lcp_pe = lcp_pe.to(data.x.device)
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data    
   

@functional_transform('alt_local_curvature_profile')
class AltLocalCurvatureProfile(BaseTransform):
    """
    This class computes the LCP based on the most extreme ORC values for each node in a graph.
    """
    def __init__(self, attr_name = 'alt_lcp_se'):
        self.attr_name = attr_name
        

    def compute_orc(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        ordered_orc = []
        for node in graph.nodes():
            if len(neighbors[node]) > 0:
                ordered_orc.append(sorted([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
            else:
                ordered_orc.append([0])

        cropped_orc = []
        for node in graph.nodes():
            if len(ordered_orc[node]) > 5:
                cropped_orc.append(ordered_orc[node][:3] + ordered_orc[node][-2:])
            elif len(ordered_orc[node]) < 5:
                cropped_orc.append(ordered_orc[node] + [0] * (5 - len(ordered_orc[node])))
            else:
                cropped_orc.append(ordered_orc[node])

        alt_lcp_pe = torch.tensor(cropped_orc)
    
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, alt_lcp_pe), dim=-1)
        else:
            data.x = torch.cat(alt_lcp_pe, dim=-1)

        return data