from typing import List

from Graph import Subgraph, Vertex
from samplers.Sampler import Sampler

class EmptySampler(Sampler):
    """
    Collects 0 neighbors of the vertex
    """

    def __init__(self, layers_count: int):
        super().__init__(layers_count)

    def sample(self, root_vertices: List[Vertex]) -> List[Subgraph]:
        return [Subgraph(v.id, v.degree, []) for v in root_vertices]
