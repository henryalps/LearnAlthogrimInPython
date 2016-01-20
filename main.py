# -*- coding: utf-8 -*-
import BPModel
import LinearModel
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import draw_figure as df
from toolkits import ToolKits
from enums import BPTypes, BHSTypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
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
    colsRes = ['sbps', 'dbps']  # 'sbps' , 'dbps'
    colsResTypes = [BPTypes.SBP, BPTypes.DBP]
    minFullSetSize = 30  # 只有全集大于等于30时才进行训练

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.testResults = []
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=4, warm_start=True)
        self.dt = df.DrawToolkit()  # plot assistant
        self.type = BPTypes.SBP

    def train(self):
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

    def save_predict_result(self, sig_name, root_path):
        tmp_plt = self.dt.generate_scatter_plt(list(self.y_test[:, self.colsResTypes.index(self.type)]), list(self.testResults),
                             "Measured " + BPTypes.get_type_name(self.type) +
                             "(mmHg)", "Estimated " + BPTypes.get_type_name(self.type) + "(mmHg)",
                                     sig_name + ": " + BPTypes.get_type_name(self.type) + " Regression Result")
        tmp_plt.savefig(root_path + '_' + BPTypes.get_type_name(self.type) + '_' + sig_name + '.png')

    def get_result_bhs_type(self):
        small_than_5, small_than_10, small_than_15 = [0] * 3
        for val in list(map(lambda x: x[0]-x[1],
                                zip(self.testResults, self.y_test[:, self.colsResTypes.index(self.type)]))):
            val = abs(val)
            if val <= 15:
                small_than_15 += 1
                if val <= 10:
                    small_than_10 += 1
                    if val <= 5:
                        small_than_5 += 1

        return ToolKits.which_bhs_standard(100.0 * small_than_5/len(self.testResults),
                                           100.0 * small_than_10/len(self.testResults),
                                           100.0 * small_than_15/len(self.testResults))

    def read_file(self, data_file_name):
        full_set = pd.read_csv(data_file_name)
        return full_set.as_matrix(self.cols), full_set.as_matrix(self.colsRes)

    def split_sets(self, src_set, res_set):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(src_set, res_set)

    def set_data_type(self, bp_type):
        self.type = bp_type

    def reset_model(self):
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=4, warm_start=True)

    def alter_type(self):
        self.type = self.colsResTypes[0] if self.type == self.colsResTypes[1] else self.colsResTypes[1]


if __name__ == "__main__":
    root_path = '/mnt/code/matlab/data/csv/'
    pic_sub_path = 'LR/'
    # 1 获取文件列表
    only_csv_files = [f for f in listdir(root_path) if isfile(join(root_path, f)) &
                      f.startswith('a') & f.endswith('.csv')]
    # 定义各类预测结果的数量
    type_sbp_nums = list([0] * 4)
    type_dbp_nums = list([0] * 4)
    # 2 遍历文件列表
    for csv_file_name in only_csv_files:
        # model = BPModel.BPModel()
        model = LinearModel.LinearModel()
        arr, res = model.read_file(join(root_path, csv_file_name))
        if np.size(arr, 0) >= RandomForest.minFullSetSize:
            # 3 收缩压模型训练
            model.split_sets(arr, res)
            # model.x_train, model.x_test = model.scale_data(model.x_train, model.x_test)
            model.train()
            model.test()
            bhs_type = model.get_result_bhs_type()
            type_sbp_nums[bhs_type] += 1
            model.save_predict_result(csv_file_name, root_path + pic_sub_path + BHSTypes.get_type_name(bhs_type))
            # 4 舒张压模型训练
            model.alter_type()
            model.train()
            model.test()
            bhs_type = model.get_result_bhs_type()
            type_dbp_nums[bhs_type] += 1
            model.save_predict_result(csv_file_name, root_path + pic_sub_path + BHSTypes.get_type_name(bhs_type))

            # rf.display_and_save_predict_result(csv_file_name)
            # input()
            # rf.display_and_save_predict_result()

        # rf.train()
        # rf.test()
    # rf_obj.train()
    # rf_obj.test()
    # rf_obj.show_predict_result()

    # 5 打印模型输出的各类血压估计结果
    print(type_sbp_nums)
    print(type_dbp_nums)
