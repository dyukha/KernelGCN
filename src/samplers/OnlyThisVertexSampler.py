from typing import List

from Graph import Subgraph, Vertex
from samplers.Sampler import Sampler

class OnlyThisVertexSampler(Sampler):
    """
    Collects all neighbors of the vertex
    """

    def __init__(self, layers_count: int):
        super().__init__(layers_count)

    def sample(self, root_vertices: List[Vertex]) -> List[Subgraph]:
        return self.__collect(root_vertices, self._layers_count - 1)

    def __collect(self, vertices: List[Vertex], layer: int) -> List[Subgraph]:
        return [Subgraph(v.id, v.degree, [] if layer == 0 else self.__collect([v], layer - 1))
                for v in vertices]
