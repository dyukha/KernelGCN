from __future__ import annotations

import math
from typing import List, Dict

from mxnet.ndarray import NDArray

from Graph import Subgraph, Graph
from common import model_ctx, data_ctx
from networks.GCNBase import GCNBase

from mxnet import nd

from samplers import Sampler

class NaiveGCN(GCNBase):
    def __init__(self,
                 training_sampler: Sampler,
                 test_sampler: Sampler,
                 graph: Graph,
                 hidden_layer_sizes: List[int],
                 concatenate_features: bool):
        super().__init__(training_sampler, test_sampler, graph, hidden_layer_sizes, concatenate_features)

    def compute(self, vertices_on_layer: List[Dict[int, Subgraph]]):
        for layer in range(1, self._num_layers):
            for (vertex, subgraph) in vertices_on_layer[layer].items():
                self._feature_layers[layer][vertex] = self.compute_vertex_layer(layer, vertex, subgraph)

    def compute_vertex_layer(self, layer: int, vertex: int, subgraph: Subgraph) -> NDArray:
        feature_sum = nd.zeros(shape=(self._feature_layers[layer - 1][vertex].size, 1), ctx=self.features_ctx)
        for n_subgraph in subgraph.neighbors:
            prev = self._feature_layers[layer - 1][n_subgraph.vertex]
            prev_act = prev if layer == 1 else self._act(prev)
            feature_sum = feature_sum + prev_act / math.sqrt(subgraph.degree * n_subgraph.degree)
        # Take sampling into account
        feature_sum = feature_sum * (subgraph.degree / len(subgraph.neighbors))
        res = self._b[layer - 1].data() + nd.dot(self._W[layer - 1].data(), feature_sum.as_in_context(model_ctx))
        res = res.as_in_context(self.features_ctx)
        # print(self._feature_layers[layer - 1][vertex].shape, res.shape)
        return res \
            if not self._concatenate_features or layer == self._num_layers - 1 \
            else nd.concat(self._feature_layers[layer - 1][vertex], res, dim=0)
