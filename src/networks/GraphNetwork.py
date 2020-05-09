from abc import ABC

from mxnet.gluon import Block, ParameterDict

from enum import Enum

class Mode(Enum):
    TRAINING = 1
    TEST = 2

class GraphNetwork(Block, ABC):
    def __init__(self):
        super().__init__()
        self.mode = Mode.TRAINING
        self.parameter_dict = ParameterDict()

    def regularization(self):
        return 0.0
