import mxnet.ndarray as nd

import Graph
from common import sqr, data_ctx
from feature_transformations.FeatureTransformation import FeatureTransformation

class FeatureScalingTransformation(FeatureTransformation):
    def __init__(self, std: float):
        self._std = std

    def name(self) -> str:
        return "Feature scaling"

    def transform(self, features):
        shifted = features - nd.mean(features)
        deviation = nd.sqrt(nd.mean(sqr(shifted)))
        print(deviation)
        return self._std * shifted / deviation

