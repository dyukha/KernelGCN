from typing import List, Dict

from Graph import Subgraph, Vertex
from samplers.Sampler import Sampler

class TrivialSampler(Sampler):
    """
    Collects all neighbors of the vertex
    """

    def __init__(self, layers_count: int):
        super().__init__(layers_count)
        self.cache: List[Dict[Vertex, Subgraph]] = [{} for _ in range(layers_count)]
        """ Cache of subgraphs """

    def sample(self, root_vertices: List[Vertex]) -> List[Subgraph]:
        return self.__collect(root_vertices, self._layers_count - 1)

    def __collect(self, vertices: List[Vertex], layer: int) -> List[Subgraph]:
        res: List[Subgraph] = []
        for v in vertices:
            if v in self.cache[layer]:
                res.append(self.cache[layer][v])
            else:
                n_subgraphs = [] if layer == 0 else self.__collect(v.neighbors, layer - 1)
                subgraph = Subgraph(v.id, v.degree, n_subgraphs)
                self.cache[layer][v] = subgraph
                res.append(subgraph)
        return res