from abc import abstractmethod, ABC
from typing import List

from Graph import Subgraph, Vertex

class Sampler(ABC):
    def __init__(self, layers_count: int):
        self._layers_count = layers_count

    @abstractmethod
    def sample(self, root_vertices: List[Vertex]) -> List[Subgraph]:
        raise NotImplementedError

