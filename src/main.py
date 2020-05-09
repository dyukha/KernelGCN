import random
from typing import Optional, List

from mxnet.gluon.data import DataLoader

import mxnet as mx
from mxnet import nd, gluon

from mxnet import profiler

import Graph
from learners.IterativeLearner import learn_iterative
from common import data_ctx, measure_time
from feature_transformations import FeatureTransformation
from feature_transformations.FeatureScalingTransformation import FeatureScalingTransformation
from feature_transformations.KernelTransformation import KernelTransformation
from feature_transformations.LinearConvolutionTransformation import LinearConvolutionTransformation
from feature_transformations.PcaTransformation import PcaTransformation

############## PARAMETERS ###################
from feature_transformations.RealFeatureScalingTransformation import RealFeatureScalingTransformation
from learners.IterativeLogisticLearner import learn_iterative_logistic

profiler.set_config(aggregate_stats=True, filename='profile_output.json')

mx.random.seed(101)
random.seed(101)


def learn(graph, net_type, training_set, test_set, iterations_per_epoch, batch_size):
    def try_standard_approach(approach):
        def get_all_vertices(data_loader: DataLoader):
            res_X = []
            for X, y in data_loader:
                for x in X:
                    res_X.append(round(float(x.asscalar())))
            return res_X

        import sklearn.metrics
        train_vertices = get_all_vertices(training_set)
        test_vertices = get_all_vertices(test_set)
        X = nd.stack(*[v.features for v in graph.vertices]).reshape(graph.n, graph.num_features).asnumpy()
        y = nd.array([v.clazz for v in graph.vertices], ctx=data_ctx).asnumpy()
        X_train = X[train_vertices]
        y_train = y[train_vertices]
        X_test = X[test_vertices]
        y_test = y[test_vertices]
        approach.fit(X_train, y_train)
        train_accuracy = sklearn.metrics.accuracy_score(y_train, approach.predict(X_train))
        test_accuracy = sklearn.metrics.accuracy_score(y_test, approach.predict(X_test))
        print("  train_accuracy=", "%.4f" % train_accuracy, "test_accuracy=", "%.4f" % test_accuracy)
        # train_f1_score = sklearn.metrics.f1_score(y_train, approach.predict(X_train), average='micro')
        # test_f1_score = sklearn.metrics.f1_score(y_test, approach.predict(X_test), average='micro')
        # print("  train_f1_score=", "%.4f" % train_f1_score, "test_f1_score=", "%.4f" % test_f1_score)

    def naive_learn():
        from sklearn.naive_bayes import GaussianNB
        try_standard_approach(GaussianNB())
        # from sklearn.naive_bayes import ComplementNB
        # try_standard_approach(ComplementNB())

    def logistic_learn():
        from sklearn.linear_model import LogisticRegression
        try_standard_approach(LogisticRegression(solver='saga', max_iter=100, n_jobs=-1))

    def random_forest_learn():
        from sklearn.ensemble import RandomForestClassifier
        random_forest_parameters = {
            'n_estimators': 100,
            'max_depth': 10
        }
        print(random_forest_parameters)
        # noinspection PyTypeChecker
        try_standard_approach(RandomForestClassifier(n_jobs=-1, **random_forest_parameters))

    # print("learner:", learner)

    def run_learn():
        if net_type == 'simple_gcn':
            logistic_learn()
        elif net_type in ['naive_gcn', 'caching_gcn']:
            # regularization = 1
            hidden_layer_sizes = [50]
            # hidden_layer_sizes = [200, 50]
            # hidden_layer_sizes = [1] * 100
            sampling = 'random'
            learn_iterative(graph, hidden_layer_sizes, sampling, net_type, training_set, test_set, batch_size,
                            iterations_per_epoch)

            # net_type = 'naive_gcn'
            # net_type = 'logistic'
        #     learn_iterative_logistic(graph, training_set, test_set, batch_size, iterations_per_epoch)
        # elif learner == 'naive_bayes':
        #     naive_learn()
        # elif learner == 'logistic_regression':
        #     logistic_learn()
        # elif learner == 'random_forest':
        #     random_forest_learn()
        else:
            assert False

    measure_time("Learning", run_learn)


def run_transform(graph, kernel_count, kernel_kernel, pca_count, feature_scaling):
    def transform_data():
        transformations: List[FeatureTransformation] = []
        if feature_scaling:
            transformations.append(FeatureScalingTransformation(0.1))
        if pca_count is not None:
            transformations.append(PcaTransformation(pca_count))
        if feature_scaling:
            transformations.append(FeatureScalingTransformation(0.1))
        if kernel_count is not None:
            transformations.append(KernelTransformation(kernel_count, kernel_kernel))
        if feature_scaling:
            transformations.append(FeatureScalingTransformation(0.1))
        transformations.append(LinearConvolutionTransformation(graph, 2, 1, False))

        transformations.append(RealFeatureScalingTransformation(1))

        features = FeatureTransformation.FeatureTransformation.to_matrix(graph, data_ctx)

        def apply_transform(transform):
            nonlocal features
            features = transform.transform(features)

        for transformation in transformations:
            measure_time(transformation.name(), lambda: apply_transform(transformation))

        FeatureTransformation.FeatureTransformation.update_features(features, graph)

    measure_time("Data transform", transform_data)


def prepare_datasets(graph: Graph):
    train_end = int(graph.n * 0.8)
    test_end = graph.n

    shuffled_vertices = list(graph.vertices)
    random.shuffle(shuffled_vertices)
    X_all = nd.array([v.id for v in shuffled_vertices], ctx=data_ctx).reshape(-1, 1)
    Y_all = nd.array([v.clazz for v in shuffled_vertices], ctx=data_ctx).reshape(-1, 1)

    batch_size = 100

    # noinspection PyShadowingNames
    def create_set(start, end, batch_size, shuffle):
        return gluon.data.DataLoader(gluon.data.ArrayDataset(X_all[start:end], Y_all[start:end]),
                                     batch_size=batch_size, shuffle=shuffle, num_workers=10)

    training_set = create_set(0, train_end, batch_size, True)
    test_set = create_set(train_end, test_end, test_end - train_end, False)
    iterations_per_epoch = train_end // batch_size
    # all_set = create_set(0, X_all.size, X_all.size, False)
    return training_set, test_set, iterations_per_epoch, batch_size


def hpo():
    for graph_name in ['cora', 'pubmed', 'reddit']:
        print("graph:", graph_name)
        graph: Optional[Graph] = None

        def read_data():
            nonlocal graph
            graph = Graph.read_data(f"../data/{graph_name}.cites", f"../data/{graph_name}.content", apply_kernel=False)

        measure_time("Data read", read_data)

        training_set: Optional[DataLoader] = None
        test_set: Optional[DataLoader] = None
        iterations_per_epoch: Optional[int] = None
        batch_size: Optional[int] = None

        def prepare_data_inner():
            nonlocal training_set, test_set, iterations_per_epoch, batch_size
            training_set, test_set, iterations_per_epoch, batch_size = prepare_datasets(graph)

        measure_time("Prepare data", prepare_data_inner)

        # for kernel_count in [None, 100, 500]:
        for kernel_count in [None, 300]:
            kernels = [None] if kernel_count is None else [KernelTransformation.Kernel.POLY,
                                                           KernelTransformation.Kernel.RBF]
            for kernel in kernels:
                # for pca_count in [None, 100]:
                for pca_count in [None]:
                    # for feature_scaling in [False, True]:
                    for feature_scaling in [True]:
                        run_transform(graph.copy(), kernel_count, kernel, pca_count, feature_scaling)
                        for net_type in ['simple_gcn', 'naive_gcn', 'caching_gcn']:
                        # for net_type in ['simple_gcn']:
                            message = " graph: " + graph_name + \
                                      " net_type: " + net_type + \
                                      " kernel_count: " + str(kernel_count) + \
                                      " kernel " + str(kernel) + \
                                      " pca_count: " + str(pca_count) + \
                                      " feature_scaling: " + str(feature_scaling)
                            print("Processing:", message)
                            learn(graph, net_type, training_set, test_set, iterations_per_epoch, batch_size)
                            print("Finished:", message)


if __name__ == '__main__':
    measure_time("Total", hpo)
