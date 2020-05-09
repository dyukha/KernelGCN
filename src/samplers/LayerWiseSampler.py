from typing import List

from Graph import Subgraph, Vertex
from networks import LayerWiseSamplingGCN
from samplers.Sampler import Sampler

import random

# TODO. Not ready at all

class LayerWiseSampler(Sampler):
    """
    Collects at most neighbors_count neighbors of each vertex
    """
    def __init__(self, gcn: LayerWiseSamplingGCN, layers_count: int, nodes_per_layer: int):
        super().__init__(layers_count)
        self._gcn: LayerWiseSamplingGCN = gcn
        self.nodes_per_layer = nodes_per_layer

    def sample(self, root_vertices: List[Vertex]) -> List[Subgraph]:
        return self.__collect(root_vertices, self._layers_count - 1)

    def p_cond(self, u: Vertex, v: Vertex):
        g = self._gcn.g.data()


    def __collect(self, vertices: List[Vertex], layer: int) -> List[Subgraph]:
        collected_vertices: List[List[Vertex]] = [vertices]
        for layer in range(0, self._layers_count - 1):
            cur_vertices = collected_vertices[layer]



        for v in vertices:
            if layer == 0:
                neighbors_subgraphs = []
            else:
                neighbors = v.neighbors \
                    if len(v.neighbors) <= self._neighbors_count \
                    else random.sample(v.neighbors, self._neighbors_count)
                neighbors_subgraphs = self.__collect(neighbors, layer - 1)
            res.append(Subgraph(v.id, v.degree, neighbors_subgraphs))
        return res
