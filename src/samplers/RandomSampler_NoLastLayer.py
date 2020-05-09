from typing import List

from Graph import Subgraph, Vertex
from samplers.Sampler import Sampler

import random

class RandomSampler_NoLastLayer(Sampler):
    """
    Collects at most neighbors_count neighbors of each vertex
    """

    def __init__(self, layers_count: int, neighbors_count: int):
        super().__init__(layers_count)
        self._neighbors_count = neighbors_count

    def sample(self, root_vertices: List[Vertex]) -> List[Subgraph]:
        return self.__collect(root_vertices, self._layers_count - 2)  # note -2 instead of -1

    def __collect(self, vertices: List[Vertex], layer: int) -> List[Subgraph]:
        res: List[Subgraph] = []
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
