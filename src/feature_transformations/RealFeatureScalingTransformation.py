import mxnet.ndarray as nd

import Graph
from common import sqr, data_ctx
from feature_transformations.FeatureTransformation import FeatureTransformation

class RealFeatureScalingTransformation(FeatureTransformation):
    def __init__(self, std: float):
        self._std = std

    def name(self) -> str:
        return "Real feature scaling"

    def transform(self, features):
        shifted = features - nd.mean(features, axis=0)
        deviation = nd.sqrt(nd.mean(sqr(shifted), axis=0))
        return self._std * shifted / deviation

