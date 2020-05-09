from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import profiler
from timeit import default_timer as timer

data_ctx = mx.cpu()
model_ctx = mx.gpu()

batch_size = 60
num_inputs = 500
num_outputs = 100
num_examples = 6000
num_test = 600
X_train = nd.random_normal(shape=(num_examples, num_inputs))
y_train = nd.random_randint(1, num_outputs + 1,  shape=num_examples)
X_test = nd.random_normal(shape=(num_test, num_inputs))
y_test = nd.random_randint(1, num_outputs + 1,  shape=num_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
train_data = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train), batch_size, shuffle=True, num_workers=20)
test_data = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(X_test, y_test), batch_size, shuffle=False, num_workers=20)

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(5000))
net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

epochs = 10
moving_loss = 0

profiler.set_config(profile_all=True, aggregate_stats=True)

for e in range(epochs):
    profiler.set_state('run')
    start_time = timer()
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s, Time %f"
          % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy, timer() - start_time))
    profiler.set_state('stop')
    print(profiler.dumps(reset=True))
