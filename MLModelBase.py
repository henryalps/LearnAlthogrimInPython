# -*- coding: utf-8 -*-
from toolkits import ToolKits
from enums import BPTypes
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import draw_figure as df
import pandas as pd


class MLModelBase:
    colsResTypes = [BPTypes.SBP, BPTypes.DBP]
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
    colsRes = ['sbps', 'dbps']

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.testResults = []
        self.dt = df.DrawToolkit()  # plot assistant
        self.type = BPTypes.SBP

    def train(self):
        return

    def test(self):
        return

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

    def save_predict_result(self, sig_name, root_path):
        tmp_plt = self.dt.generate_scatter_plt(list(self.y_test[:, self.colsResTypes.index(self.type)]), list(self.testResults),
                             "Measured " + BPTypes.get_type_name(self.type) +
                             "(mmHg)", "Estimated " + BPTypes.get_type_name(self.type) + "(mmHg)",
                                     sig_name + ": " + BPTypes.get_type_name(self.type) + " Regression Result")
        tmp_plt.savefig(root_path + '_' + BPTypes.get_type_name(self.type) + '_' + sig_name + '.png')

    def alter_type(self):
        self.type = self.colsResTypes[0] if self.type == self.colsResTypes[1] else self.colsResTypes[1]

    def read_file(self, data_file_name):
            full_set = pd.read_csv(data_file_name)
            return full_set.as_matrix(self.cols), full_set.as_matrix(self.colsRes)

    def split_sets(self, src_set, res_set):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(src_set, res_set)

    # according to current bp type, get model output
    def get_train_res(self):
        return self.y_train[:, self.colsResTypes.index(self.type)]

    @staticmethod
    def scale_data(train_set_data, test_set_data):
        standard_scaler = StandardScaler()
        standard_scaler.fit(train_set_data)
        return standard_scaler.transform(train_set_data), standard_scaler.transform(test_set_data)