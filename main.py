# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import draw_figure as df
from sklearn.ensemble import RandomForestRegressor
#  类名 与 属性名 要采用 驼峰表达式
#  方法名 与 局部变量名 要采用 下划线表达式
#  使用‘\’来换行 但在[]/()/{}中无需这样使用


class RandomForest:
    cols = ['hr_miu', 'hr_delta', 'hr_iqr', 'hr_skew',
            'pwtt_mean', 'PH', 'PRT', 'PWA', 'RBAr', 'SLP1', 'SLP2',
            'RBW10', 'RBW25', 'RBW33', 'RBW50', 'RBW66', 'RBW75',
            'DBW10', 'DBW25', 'DBW33', 'DBW50', 'DBW66', 'DBW75',
            'KVAL', 'AmBE', 'DfAmBE',
            'kte_miu', 'kte_delta', 'kte_iqr', 'kte_skew',
            'h_miu', 'h_delta', 'h_iqr', 'h_skew',
            'loge_ar_1', 'loge_ar_2', 'loge_ar_3', 'loge_ar_4', 'loge_ar_5',
            'loge_delta', 'loge_iqr', 'ppg_fed_ar_1', 'ppg_fed_ar_2',
            'ppg_fed_ar_3', 'ppg_fed_ar_4', 'ppg_fed_ar_5']
    colsRes = ['dbps']  # 'sbps' , 'dbps'

    def __init__(self, train_set_file_name, test_set_file_name):
        self.trainSet = pd.read_csv(train_set_file_name)  # 训练集
        self.testSet = pd.read_csv(test_set_file_name)  # 测试集
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=4, warm_start=True)
        self.testResults = []

        self.dt = df.DrawToolkit()  # plot assistant
        self.type = 'SBP' if self.colsRes[0].__eq__('sbps') else 'DBP'

    def train(self):
        train_arr = self.trainSet.as_matrix(self.cols)
        train_res = self.trainSet.as_matrix(self.colsRes)
        # train_res = np.char.mod('%f', train_res)  # 把浮点数组转为字符串数组
        train_res = np.ravel(train_res)  # 数据整形
        self.rf.fit(train_arr, train_res)

    def test(self):
        test_arr = self.testSet.as_matrix(self.cols)
        self.testResults = self.rf.predict(test_arr)

    def show_predict_result(self):
        tmp = self.testSet
        tmp['prediction'] = self.testResults
        # print(tmp.head())
        self.dt.plot_scatter(list(tmp[self.colsRes[0]]), list(self.testResults),
                             "Measured " + self.type + "(mmHg)", "Estimated " + self.type + "(mmHg)",
                             self.type + " Regression Result")


if __name__ == "__main__":
    train_set_name = '/mnt/code/matlab/data/csv/train.csv'
    test_set_name = '/mnt/code/matlab/data/csv/test.csv'
    rf_obj = RandomForest(train_set_file_name=train_set_name, test_set_file_name= test_set_name)
    rf_obj.train()
    rf_obj.test()
    rf_obj.show_predict_result()
