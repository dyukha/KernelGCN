from __future__ import annotations

from enum import Enum
from timeit import default_timer as timer

import mxnet as mx
from mxnet import nd
from mxnet.gluon import Parameter, ParameterDict
from mxnet.ndarray import NDArray, Activation

from Graph import Graph
from common import model_ctx, data_ctx, sqr, feature_scaling
from networks.GraphNetwork import GraphNetwork

class Mode(Enum):
    TRAINING = 1
    TEST = 2

class Kernels(GraphNetwork):
    def rbf_kernels(self, x: NDArray, y: NDArray):
        """
        Computes exp(-c ||x - y||^2).
        ||x - y||^2 = x . x + y . y - 2 x . y
        Compute each term separately. x is are original features, y are features used for similarity
        """

        cross_products = nd.dot(x, y)

        x_products = nd.sum(sqr(x), axis=1, keepdims=True)
        x_products = nd.broadcast_axis(x_products, axis=1, size=y.shape[1])

        y_products = nd.sum(sqr(y), axis=0, keepdims=True)
        y_products = nd.broadcast_axis(y_products, axis=0, size=x.shape[0])

        sqr_difs = x_products + y_products - 2 * cross_products
        print(nd.mean(x_products), nd.mean(y_products), nd.mean(cross_products))
        print(nd.mean(sqr_difs))
        res = nd.exp(-0.5 * sqr_difs)
        print(res.shape)
        return res

    def linear_kernels(self, x: NDArray, y: NDArray):
        return nd.dot(x, y)

    def poly_kernels(self, x: NDArray, y: NDArray):
        prod = nd.dot(x, y)
        return nd.sign(prod) * nd.abs(prod) ** 2
        # return nd.abs(prod) ** 2
        # return (nd.abs(prod) + 3) ** 2

    # noinspection PyPep8Naming
    def __init__(self, graph: Graph, depth: int, train_size: int, concatenate_features: bool):
        super().__init__()
        parameter_dict = ParameterDict()

        # TODO
        # noinspection PyTypeChecker
        self._W: Parameter = parameter_dict.get(f"W", shape=(train_size * (1 + depth), graph.num_classes))

        # TODO
        # noinspection PyTypeChecker
        self._b: Parameter = parameter_dict.get(f"b", shape=(1, graph.num_classes))

        parameter_dict.initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)
        start_time = timer()
        kernel_ctx = model_ctx

        nd_features: NDArray =\
            nd.stack(*[v.features for v in graph.vertices])\
            .reshape(graph.n, graph.num_features)\
            .as_in_context(kernel_ctx)

        print(nd_features.shape)

        # features used for similarity measure
        sim_features = nd.random.normal(scale=0.1, shape=(graph.num_features, train_size), ctx=kernel_ctx)

        features = self.rbf_kernels(nd_features, sim_features)
        # features = self.linear_kernels(nd_features, sim_features)
        # features = self.poly_kernels(nd_features, sim_features)

        features = feature_scaling(features, 0, 0.1)
        features = graph.linear_convolution(features, depth, 1, concatenate_features)

        self._features: NDArray = feature_scaling(features, 0, 0.1).as_in_context(data_ctx)
        self.parameter_dict = parameter_dict
        self.mode = Mode.TRAINING
        self.reg_const = 0.03
        end_time = timer()
        print("Kernel computation time=", end_time - start_time)
        # self.train_indices = indices

    def bound(self):
        pass
        for p in [self._b, self._W]:
            p.data()[:] = nd.clip(p.data(), -1e5, 1e5)

    def forward(self, vertices: NDArray) -> NDArray:
        vertex_list = [int(v.asscalar()) for v in vertices]
        X = self._features[vertex_list].as_in_context(model_ctx)
        # return X
        return nd.dot(X, self._W.data()) + nd.broadcast_axis(self._b.data(), axis=0, size=len(vertices))

    def reg_fun(self, array):
        # return self.reg_const * nd.mean(nd.minimum(nd.abs(array), 1))
        # return self.reg_const * nd.sqrt(nd.mean(nd.abs(array) ** 2))
        return self.reg_const * nd.mean(nd.abs(array) ** 2)
        # return self.reg_const * nd.mean(nd.abs(array))
        # return self.reg_const * nd.mean(nd.sqrt(nd.abs(array)))
        # return 0

    def regularization(self):
        return self.reg_fun(self._W.data()) + self.reg_fun(self._b.data())

    @staticmethod
    def _act(x):
        return Activation(x, 'softrelu')
