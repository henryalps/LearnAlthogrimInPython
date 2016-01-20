# -*- coding: utf-8 -*-
import MLModelBase
from sknn.mlp import Regressor, Layer


class BPModel(MLModelBase.MLModelBase):
    def __init__(self):
        MLModelBase.MLModelBase.__init__(self)
        self.neural_network = Regressor

    # 需要归一化的输入向量。
    def train(self):
        self.neural_network = Regressor(
            layers=[
                Layer("Rectifier", units=6),
                Layer("Linear")
            ],
            learning_rate=0.0001,
            n_iter=20)
        self.neural_network.fit(self.x_train, self.get_train_res())

    def test(self):
        self.testResults = self.neural_network.predict(self.x_test)
        return self.testResults
