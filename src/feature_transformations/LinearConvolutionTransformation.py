import math

import Graph
from feature_transformations.FeatureTransformation import FeatureTransformation
import mxnet.ndarray as nd

class LinearConvolutionTransformation(FeatureTransformation):
    def __init__(self, graph: Graph, depth: int, neighbor_coef: float, concatenate: bool):
        self._graph = graph
        self._neighbor_coef = neighbor_coef
        self._depth = depth
        self._concatenate = concatenate

    def name(self) -> str:
        return "Linear convolution"

    def transform(self, features):
        graph = self._graph
        assert features.shape[0] == graph.n
        depth = self._depth

        if depth == 0:
            return features

        adj = graph.adj_matrix(features.context, lambda u, v: 1 / math.sqrt(u.degree * v.degree))
        # adj = graph.adj_matrix(features.context, lambda u, v: 1 / u.degree)

        feature_layers = [features]

        for i in range(depth):
            feature_layers.append(nd.dot(adj, feature_layers[-1]))
            if not self._concatenate:
                feature_layers = [feature_layers[0] + self._neighbor_coef * feature_layers[1]]

        return nd.concat(*feature_layers, dim=1) if self._concatenate else feature_layers[-1]

