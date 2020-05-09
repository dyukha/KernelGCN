from enum import Enum

import mxnet.ndarray as nd
from mxnet.ndarray import NDArray

from common import sqr
from feature_transformations.FeatureTransformation import FeatureTransformation

class KernelTransformation(FeatureTransformation):
    class Kernel(Enum):
        LINEAR = 1
        POLY = 2
        RBF = 3

    def __init__(self, num_kernels: int, kernel: Kernel):
        self._num_kernels = num_kernels
        self._kernel = kernel

    def name(self) -> str:
        return "Apply kernels"

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
        res = nd.exp(-0.05 * sqr_difs)
        print(res.shape)
        return res

    def linear_kernels(self, x: NDArray, y: NDArray):
        return nd.dot(x, y)

    def poly_kernels(self, x: NDArray, y: NDArray):
        prod = nd.dot(x, y)
        return nd.sign(prod) * nd.abs(prod) ** 2
        # return nd.abs(prod) ** 2
        # return (nd.abs(prod) + 3) ** 2

    def transform(self, features):
        # features used for similarity measure
        sim_features = nd.random.normal(scale=0.1, shape=(features.shape[1], self._num_kernels), ctx=features.context)

        if self._kernel == self.Kernel.LINEAR:
            return self.linear_kernels(features, sim_features)
        elif self._kernel == self.Kernel.POLY:
            return self.poly_kernels(features, sim_features)
        elif self._kernel == self.Kernel.RBF:
            return self.rbf_kernels(features, sim_features)
        else:
            assert False

