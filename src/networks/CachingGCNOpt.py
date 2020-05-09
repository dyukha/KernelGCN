from __future__ import annotations

import math
from typing import List, Dict

from mxnet.ndarray import NDArray

from Graph import Subgraph, Graph
from common import model_ctx
from networks.GCNBase import GCNBase, Mode

from mxnet import nd

from samplers import Sampler

class CachingGCNOpt(GCNBase):
    def __init__(self,
                 training_sampler: Sampler,
                 test_sampler: Sampler,
                 graph: Graph,
                 hidden_layer_sizes: List[int],
                 concatenate_features: bool):
        super().__init__(training_sampler, test_sampler, graph, hidden_layer_sizes, concatenate_features)
        self._sum_cache: List[List[NDArray]] = \
            [[]] + [[nd.zeros(x.shape) for x in v] for v in self._feature_layers[:-1]]
        """ Contains sum of features on the previous layer """
        # Initialize and fill the cache using all features with current weights
        for layer in range(1, self._num_layers):
            for vertex in graph.vertices:
                subgraph = Subgraph(vertex.id, len(vertex.neighbors),
                                    [Subgraph(v.id, len(v.neighbors), []) for v in vertex.neighbors])
                self._feature_layers[layer][vertex.id] = self.compute_vertex_layer(layer, vertex.id, subgraph)

    def compute(self, vertices_on_layer: List[Dict[int, Subgraph]]):
        """
        Compute features which must be computed, according to vertices_on_layer.

        :param vertices_on_layer: For each layer stores a map: vertex.id -> subgraph.
        :return: None
        """

        if self.mode == Mode.TRAINING:
            # Subtract old values from sum
            for layer in range(1, self._num_layers):
                for (vertex, subgraph) in vertices_on_layer[layer].items():
                    for n_subgraph in subgraph.neighbors:
                        prev = self._feature_layers[layer - 1][n_subgraph.vertex]
                        prev_act = prev if layer == 1 else self._act(prev)
                        self._sum_cache[layer][vertex] = self._sum_cache[layer][vertex] \
                                                         - prev_act / math.sqrt(subgraph.degree * n_subgraph.degree)

        for layer in range(1, self._num_layers):
            for (vertex, subgraph) in vertices_on_layer[layer].items():
                self._feature_layers[layer][vertex] = self.compute_vertex_layer(layer, vertex, subgraph)

    def compute_vertex_layer(self, layer: int, vertex: int, subgraph: Subgraph) -> NDArray:
        feature_sum = nd.zeros(shape=(self._feature_layers[layer - 1][vertex].size, 1), ctx=model_ctx)
        for n_subgraph in subgraph.neighbors:
            prev = self._feature_layers[layer - 1][n_subgraph.vertex]
            prev_act = prev if layer == 1 else self._act(prev)
            feature_sum = feature_sum + prev_act / math.sqrt(subgraph.degree * n_subgraph.degree)
        # Take sampling into account
        if self.mode == Mode.TRAINING:
            feature_sum = feature_sum + self._sum_cache[layer][vertex]
            self._sum_cache[layer][vertex] = feature_sum
        else:
            feature_sum = feature_sum * (subgraph.degree / len(subgraph.neighbors))

        return self._b[layer - 1].data() + nd.dot(self._W[layer - 1].data(), feature_sum)
