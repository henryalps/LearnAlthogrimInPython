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
                Layer("Rectifier", units=2),
                Layer("Linear")
            ],
            learning_rate=0.00001,
            n_iter=20)
        self.neural_network.fit(self.x_train, self.get_train_res())

    def test(self):
        tmp_list = self.neural_network.predict(self.x_test)
        for array_i in tmp_list:
            self.testResults.append(array_i[0])
        return self.testResults
