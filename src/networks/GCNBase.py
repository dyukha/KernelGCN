from __future__ import annotations

from abc import abstractmethod

import mxnet as mx
from typing import List, Dict
from mxnet import nd
from mxnet.gluon import Parameter

from Graph import Subgraph, Graph
from mxnet.ndarray import NDArray, Activation

from common import model_ctx
from networks.GraphNetwork import GraphNetwork, Mode
from samplers import Sampler


class GCNBase(GraphNetwork):
    def __init__(self,
                 training_sampler: Sampler,
                 test_sampler: Sampler,
                 graph: Graph,
                 hidden_layer_sizes: List[int],
                 concatenate_features: bool):
        super().__init__()
        layer_sizes = [graph.num_features] + hidden_layer_sizes + [graph.num_classes]

        parameter_dict = self.parameter_dict
        sizes_sum = [0] + list(layer_sizes)
        for i in range(1, len(sizes_sum)):
            sizes_sum[i] += sizes_sum[i - 1]
        sizes_sum[len(layer_sizes) - 1] = 0
        additional_sizes: List[int] = sizes_sum if concatenate_features else [0] * len(layer_sizes)

        # noinspection PyTypeChecker
        self._W: List[Parameter] = [parameter_dict.get(f"W{i}",
                                                       shape=(layer_sizes[i + 1], layer_sizes[i] + additional_sizes[i]))
                                    for i in range(len(layer_sizes) - 1)]

        # noinspection PyTypeChecker
        self._b: List[Parameter] = [parameter_dict.get(f"b{i}", shape=(layer_sizes[i + 1], 1))
                                    for i in range(len(layer_sizes) - 1)]
        parameter_dict.initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)

        # for w in self._W:
        #     print(w.data().shape)
        self.features_ctx = model_ctx

        self._feature_layers: List[List[NDArray]] = []
        for layer in range(len(layer_sizes)):
            features = [v.features.as_in_context(self.features_ctx)
                        if layer == 0 else
                        nd.zeros(shape=(layer_sizes[layer] + additional_sizes[layer], 1), ctx=self.features_ctx)
                        for v in graph.vertices]
            self._feature_layers.append(features)

        self.__training_sampler: Sampler = training_sampler
        self.__test_sampler: Sampler = test_sampler
        self._graph = graph
        self._num_layers = len(self._feature_layers)
        self._concatenate_features = concatenate_features

    def bound(self):
        for p in self._b + self._W:
            p.data()[:] = nd.clip(p.data(), -1e10, 1e10)

    def bound_layers(self):
        for layer in self._feature_layers:
            for arr in layer:
                arr[:] = nd.clip(arr, -1e10, 1e10)

    def forward(self, root_vertices: NDArray) -> NDArray:
        sampler = self.__training_sampler if self.mode == Mode.TRAINING else self.__test_sampler
        subgraphs = sampler.sample([self._graph.vertices[int(i.asscalar())] for i in root_vertices])
        vertices_on_layer: List[Dict[int, Subgraph]] = [{} for _ in range(self._num_layers)]
        self.__collect_vertices(subgraphs, len(self._feature_layers) - 1, vertices_on_layer)
        self.compute(vertices_on_layer)
        last_layer = [self._feature_layers[self._num_layers - 1][v.vertex] for v in subgraphs]
        return nd.stack(*last_layer).reshape(len(last_layer), -1)

    def __collect_vertices(self, subgraphs: List[Subgraph], layer: int, vertices_on_layer: List[Dict[int, Subgraph]]):
        """
        Populates vertices_on_layer: for each layers stores which vertices must be computed.
        The main purpose is to avoid recomputation.

        :param subgraphs: subgraphs at the current layer
        :param layer: current layer
        :param vertices_on_layer: for each layers stores which vertices must be computed.
        :return: None
        """
        for subgraph in subgraphs:
            vertices_on_layer[layer][subgraph.vertex] = subgraph
        if layer > 1:
            for subgraph in vertices_on_layer[layer].values():
                self.__collect_vertices(subgraph.neighbors, layer - 1, vertices_on_layer)

    @abstractmethod
    def compute(self, vertices_on_layer: List[Dict[int, Subgraph]]):
        """
        Compute features which must be computed, according to vertices_on_layer.

        :param vertices_on_layer: For each layer stores a map: vertex.id -> subgraph.
        :return: None
        """
        pass

    @abstractmethod
    def compute_vertex_layer(self, layer: int, vertex: int, subgraph: Subgraph) -> NDArray:
        pass

    def regularization(self):
        return 0.0

    @staticmethod
    def _act(x):
        return Activation(x, 'softrelu')
