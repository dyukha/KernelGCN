import mxnet as mx
from timeit import default_timer as timer

from mxnet.ndarray import NDArray
import mxnet.ndarray as nd

data_ctx = mx.cpu()
model_ctx = mx.cpu()

def measure_time(msg, f):
    if msg is not None:
        print(msg + " started")
    start_time = timer()
    f()
    end_time = timer()
    time = end_time - start_time
    if msg is not None:
        print(msg + " finished; time = " + str(time))
    return time

def sqr(x):
    return x * x

def feature_scaling(x: NDArray, mean: float, std: float):
    shifted = x - nd.mean(x)
    deviation = nd.sqrt(nd.mean(sqr(shifted)))
    # deviation = nd.mean(self.sqr(shifted))
    print(deviation)
    return mean + std * shifted / deviation
