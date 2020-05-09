from typing import Optional

from timeit import default_timer as timer

import mxnet as mx
from mxnet import gluon, ndarray as nd, autograd
from mxnet.gluon.data import DataLoader

from common import measure_time, model_ctx, data_ctx
from networks import GraphNetwork
from networks.GraphNetwork import Mode

from samplers.EmptySampler import EmptySampler
from samplers.RandomSampler import RandomSampler
from samplers.TrivialSampler import TrivialSampler

from networks.NaiveGCN import NaiveGCN
from networks.CachingGCN import CachingGCN

def learn_iterative_logistic(
        graph,
        training_set,
        test_set,
        batch_size,
        iterations_per_epoch
):
    def get_all_vertices(data_loader: DataLoader):
        res_X = []
        for X, y in data_loader:
            for x in X:
                res_X.append(round(float(x.asscalar())))
        return res_X

    X_all = nd.stack(*[v.features for v in graph.vertices]).reshape(graph.n, graph.num_features).asnumpy()
    y_all = nd.array([v.clazz for v in graph.vertices], ctx=data_ctx).asnumpy()

    train_vertices = get_all_vertices(training_set)
    test_vertices = get_all_vertices(test_set)
    X_train = X_all[train_vertices]
    y_train = y_all[train_vertices]
    X_test = X_all[test_vertices]
    y_test = y_all[test_vertices]
    training_set = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train),
                                         batch_size=batch_size, shuffle=False, num_workers=10)
    test_set = gluon.data.DataLoader(gluon.data.ArrayDataset(X_test, y_test),
                                     batch_size=batch_size, shuffle=False, num_workers=10)

    net = gluon.nn.Dense(graph.num_classes, in_units=graph.num_features)

    """
    Trainer
    """
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    schedule = mx.lr_scheduler.FactorScheduler(step=iterations_per_epoch, factor=0.95)

    net.initialize(mx.init.Xavier(), ctx=mx.cpu())
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01, 'lr_scheduler': schedule})

    """
    Learn
    """

    def learn():
        max_test_accuracy = 0.0
        max_train_accuracy = 0.0

        def accuracy(data):
            good = 0
            total = 0
            for (X, Y) in data:
                features = X.as_in_context(model_ctx)
                label = Y.as_in_context(model_ctx).reshape(Y.size, -1)
                prediction = nd.argmax(net(features), axis=1).reshape(Y.size, -1)
                good += nd.sum(prediction == label).asscalar()
                total += len(X)
            return good / total

        def recalculate_accuracy():
            # noinspection PyShadowingNames
            start_time = timer()
            train_accuracy = accuracy(training_set)
            test_accuracy = accuracy(test_set)
            # test_accuracy = accuracy(all_set)
            # noinspection PyShadowingNames
            end_time = timer()

            nonlocal max_train_accuracy, max_test_accuracy
            max_train_accuracy = max(max_train_accuracy, train_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
            print("  train_accuracy=", "%.4f" % train_accuracy,
                  "test_accuracy=", "%.4f" % test_accuracy,
                  "time=", end_time - start_time)
            return train_accuracy

        recalculate_accuracy()

        # noinspection PyShadowingNames
        def run(epochs, loss, training_set):
            for e in range(epochs):
                total_loss = 0
                cnt = 0
                start_time = timer()
                for (X, Y) in training_set:
                    vertices = X.as_in_context(model_ctx)
                    label = Y.as_in_context(model_ctx)
                    with autograd.record():
                        res = net(vertices)
                        cur_loss = nd.mean(loss(res, label))
                    cur_loss.backward()
                    trainer.step(batch_size)
                    total_loss += cur_loss.asscalar()
                    cnt += 1
                    # net.bound()
                end_time = timer()
                print("epoch=", e, "loss=", total_loss / cnt, "time=", end_time - start_time)
                recalculate_accuracy()

        run(epochs=101, loss=softmax_cross_entropy, training_set=training_set)
        print("Max train accuracy=", max_train_accuracy, "Max test accuracy=", max_test_accuracy)
        # print("Reg const:", net.reg_const)
        return max_train_accuracy

    learn()
