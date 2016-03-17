# -*- coding: utf-8 -*-
import MLModelBase
from enums import BPTypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score, ShuffleSplit
import numpy as np
import FileHelper as fh
import Constants as Constant


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

    def sort_features(self):
        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        scores = []
        X = self.x_train  # np.concatenate((self.x_train, self.x_test), axis=0)
        Y = self.y_train  # np.concatenate((self.y_train, self.y_test), axis=0)
        Y = Y[:, self.colsResTypes.index(self.type)]
        for i in range(self.x_train.shape[1]):
            score = cross_val_score(rf, X[:, i:i+1],
                                    Y, scoring='r2', cv=ShuffleSplit(len(X), 3, .3))
            scores.append((round(np.mean(score), 3), fh.FileHelper.cols_updated[i]))
        scores = sorted(scores, reverse=True)
        X = []
        Y = []
        for s in scores:
            X.append(s[0])
            Y.append(s[1])
        fh.FileHelper().write_test_result_in_file('sbp.mat', X, Y)
        return X, Y
