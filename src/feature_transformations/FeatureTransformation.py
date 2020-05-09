from abc import ABC, abstractmethod
from typing import Callable

from mxnet.ndarray import NDArray
import mxnet.ndarray as nd

class FeatureTransformation(ABC):
    @abstractmethod
    def transform(self, features):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @staticmethod
    def to_matrix_and_back(ctx, graph, transformation: Callable[[NDArray], NDArray]):
        nd_features =\
            nd.stack(*[v.features for v in graph.vertices])\
            .reshape(graph.n, graph.num_features)\
            .as_in_context(ctx)

        transformed = transformation(nd_features)

        for v in graph.vertices:
            v.features = nd.array(transformed[v.id, :]).reshape(-1, 1)
        features_shape = graph.vertices[0].features.shape
        graph.num_features = features_shape[0]
        print(features_shape)

    @staticmethod
    def to_matrix(graph, ctx):
        return\
            nd.stack(*[v.features for v in graph.vertices])\
            .reshape(graph.n, graph.num_features)\
            .as_in_context(ctx)

    @staticmethod
    def update_features(features, graph):
        for v in graph.vertices:
            v.features = nd.array(features[v.id, :]).reshape(-1, 1)
        features_shape = graph.vertices[0].features.shape
        graph.num_features = features_shape[0]
        print(features_shape)
