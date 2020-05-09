from feature_transformations.FeatureTransformation import FeatureTransformation
import mxnet.ndarray as nd

class PcaTransformation(FeatureTransformation):
    def __init__(self, num_components: int):
        self._num_components = num_components

    def name(self) -> str:
        return "PCA"

    def transform(self, features):
        from sklearn.decomposition import PCA
        np_features = features.asnumpy()
        pca = PCA(n_components=self._num_components)
        return nd.array(pca.fit_transform(np_features))
