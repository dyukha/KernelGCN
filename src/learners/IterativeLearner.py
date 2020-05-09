from typing import Optional

from timeit import default_timer as timer

import mxnet as mx
from mxnet import gluon, ndarray as nd, autograd

from common import measure_time, model_ctx
from networks import GraphNetwork
from networks.GraphNetwork import Mode

from samplers.EmptySampler import EmptySampler
from samplers.RandomSampler import RandomSampler
from samplers.TrivialSampler import TrivialSampler

from networks.NaiveGCN import NaiveGCN
from networks.CachingGCN import CachingGCN

def learn_iterative(
        graph,
        hidden_layer_sizes,
        sampling,
        net_type,
        training_set,
        test_set,
        batch_size,
        iterations_per_epoch
):
    net: Optional[GraphNetwork] = None

    def create_network():
        nonlocal net
        num_layers = 2 + len(hidden_layer_sizes)

        if sampling == 'trivial':
            training_sampler = TrivialSampler(num_layers)
        elif sampling == 'random':
            training_sampler = RandomSampler(num_layers, 2)
        elif sampling == 'empty':
            training_sampler = EmptySampler(num_layers)
        else:
            assert False, 'unknown sampler'
        # training_sampler = OnlyThisVertexSampler(num_layers)

        test_sampler = TrivialSampler(num_layers)

        if net_type == 'naive_gcn':
            net = NaiveGCN(training_sampler, test_sampler, graph, hidden_layer_sizes, True)
        elif net_type == 'caching_gcn':
            net = CachingGCN(training_sampler, test_sampler, graph, hidden_layer_sizes, True)
        elif net_type == 'logistic':
            net = gluon.nn.Dense(graph.num_classes, in_units=graph.num_features)
        else:
            assert False, 'unknown net type'
        # net = CachingGCNOpt(training_sampler, test_sampler, graph, hidden_layer_sizes, True)
        # net = Kernels(graph, 2, num_kernels, True)

        # nd.save(f"../data/{graph_name}.{num_kernels}_kernels", [net._features, Y_all])
        # net._features, Y_all = nd.load(f"../data/{graph_name}.{num_kernels}_kernels")
        # with open(graph_name + "_preprocessed.txt", 'w') as fout:
        #     for features in net._features:

    measure_time("Create network", create_network)

    """
    Trainer
    """

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    schedule = mx.lr_scheduler.FactorScheduler(step=iterations_per_epoch, factor=0.95)
    # trainer = gluon.Trainer(net.parameter_dict, 'adam', {'learning_rate': 0.01, 'lr_scheduler': schedule})

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
                # profiler.set_state('run')
                vertices = X.as_in_context(model_ctx)
                label = Y.as_in_context(model_ctx)
                prediction = nd.argmax(net(vertices), axis=1).reshape(Y.size, -1)
                # print(label.shape)
                # print(prediction.shape)
                good += nd.sum(prediction == label).asscalar()
                total += len(X)
                # profiler.set_state('stop')
                # print(profiler.dumps())
                # profiler.dump()
                # exit(0)
            return good / total

        def recalculate_accuracy():
            # noinspection PyShadowingNames
            start_time = timer()
            net.mode = Mode.TEST
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
                net.mode = Mode.TRAINING
                total_loss = 0
                cnt = 0
                start_time = timer()
                for (X, Y) in training_set:
                    vertices = X.as_in_context(model_ctx)
                    label = Y.as_in_context(model_ctx)
                    with autograd.record():
                        res = net(vertices)
                        cur_loss = nd.mean(loss(res, label)) + net.regularization()
                    cur_loss.backward()
                    trainer.step(batch_size)
                    total_loss += cur_loss.asscalar()
                    cnt += 1
                    # net.bound()
                end_time = timer()
                print("epoch=", e, "loss=", total_loss / cnt, "time=", end_time - start_time)
                if e % 9 == 0:
                    train_accuracy = recalculate_accuracy()
                    if train_accuracy > 0.99:
                        net.reg_const *= 1.1

        run(epochs=101, loss=softmax_cross_entropy, training_set=training_set)
        print("Max train accuracy=", max_train_accuracy, "Max test accuracy=", max_test_accuracy)
        # print("Reg const:", net.reg_const)
        return max_train_accuracy

    learn()
