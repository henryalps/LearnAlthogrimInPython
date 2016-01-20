# -*- coding: utf-8 -*-
import MLModelBase
from sklearn import linear_model


class LinearModel(MLModelBase.MLModelBase):
    def __init__(self):
        MLModelBase.MLModelBase.__init__(self)
        self.linear_model = linear_model.LinearRegression()

    def train(self):
        self.linear_model.fit(self.x_train, self.get_train_res())

    def test(self):
        self.testResults = self.linear_model.predict(self.x_test)
