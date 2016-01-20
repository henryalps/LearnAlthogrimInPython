# -*- coding: utf-8 -*-
import MLModelBase
from enums import BPTypes
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel(MLModelBase.MLModelBase):
    def __init__(self):
        MLModelBase.MLModelBase.__init__(self)
        self.rf = RandomForestRegressor

    def train(self):
        self.reset_model()
        train_arr = self.x_train
        train_res = self.y_train[:, self.colsResTypes.index(self.type)]
        self.rf.fit(train_arr, train_res)

    def test(self):
        test_arr = self.x_test
        self.testResults = self.rf.predict(test_arr)

    def show_predict_result(self):
        tmp = self.testSet
        tmp['prediction'] = self.testResults
        # print(tmp.head())
        self.dt.generate_scatter_plt(list(tmp[self.colsRes[0]]), list(self.testResults),
                             "Measured " + BPTypes.get_type_name(self.type) + "(mmHg)", "Estimated " +
                                     BPTypes.get_type_name(self.type) + "(mmHg)",
                                     BPTypes.get_type_name(self.type) + " Regression Result")

    def reset_model(self):
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=4, warm_start=True)
