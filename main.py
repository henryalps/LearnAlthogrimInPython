# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import draw_figure as df
from StatsToolKits import StatsToolKits
from toolkits import ToolKits
from FileHelper import FileHelper

from os import listdir
from os.path import isfile, join
from enums import BPTypes, BHSTypes

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit, KFold

#  类名 与 属性名 要采用 驼峰表达式
#  方法名 与 局部变量名 要采用 下划线表达式
#  使用‘\’来换行 但在[]/()/{}中无需这样使用


class RegressionAlgorithm:
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

    def train_mbp(self):
        self.rf.fit(self.x_train, self.y_train)

    def test(self):
        test_arr = self.x_test
        self.testResults = self.rf.predict(test_arr)
        return self.testResults

    def get_test_set_original_results(self):
        return list(self.y_test[:, self.colsResTypes.index(self.type)])

    def show_full_set_result(self, title):
        tmp_plt = self.dt.generate_scatter_plt(self.get_test_set_original_results(), list(self.testResults),
                             "Measured " + BPTypes.get_type_name(self.type) +
                             "(mmHg)", "Estimated " + BPTypes.get_type_name(self.type) + "(mmHg)",
                                               title)
        tmp_plt.show()

    def show_mbp_full_set_result(self, title, mbp_local):
        tmp_plt = self.dt.generate_scatter_plt(mbp_local, list(self.testResults),
                             "Measured " + BPTypes.get_type_name(self.type) +
                             "(mmHg)", "Estimated " + BPTypes.get_type_name(self.type) + "(mmHg)",
                                               title)
        tmp_plt.show()

    def save_predict_result(self, sig_name, title, plt_root_path):
        tmp_plt = self.dt.generate_scatter_plt(list(self.y_test[:, self.colsResTypes.index(self.type)]), list(self.testResults),
                             "Measured " + BPTypes.get_type_name(self.type) +
                             "(mmHg)", "Estimated " + BPTypes.get_type_name(self.type) + "(mmHg)",
                                     sig_name + ": " + title)
        tmp_plt.savefig(plt_root_path + '_' + BPTypes.get_type_name(self.type) + '_' + sig_name + '.png')

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
        valid, arrs, indexes = self.pick_data(full_set.as_matrix(self.cols))
        return valid, arrs, full_set.as_matrix(self.colsRes) if valid else []

    def split_sets(self, src_set, res_set):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(src_set, res_set)

    def set_data_type(self, bp_type):
        self.type = bp_type

    def reset_model(self):
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=4, warm_start=True)

    def alter_type(self):
        self.type = self.colsResTypes[0] if self.type == self.colsResTypes[1] else self.colsResTypes[1]

    def pick_data(self, data): # find column which contains 0 and eliminate them
        is_legal = False
        for_return = np.matrix([])
        indexes = []
        for index_local in range(0, data.shape[0]):
            indexes.append(index_local)
            if np.where(data[index_local] == 0).size == 0:
                if for_return.size == 0:
                    is_legal = True
                    for_return = data[index_local]
                else:
                    for_return = np.concatenate((for_return, data[index_local]), axis=0)
        return is_legal, for_return, indexes


def get_title(rf_local):
    # when you are asked to "shadow name", you should change the name.
    x = rf_local.get_test_set_original_results()
    y = rf_local.testResults
    type_str = BPTypes.get_type_name(rf_local.type)
    stats_toolkit_obj = StatsToolKits(x, y)
    pearson_r, p_val = stats_toolkit_obj.get_pearson_corr()
    mic = stats_toolkit_obj.get_mic()
    return type_str + ' ' + 'Regression Results\n' + \
           'pearson regression: ' + '%.2f' % pearson_r + ' p value: ' + '%.2e' % p_val + '\n'\
            'maximal information coefficient: ' + '%.2f' % mic


def disp_stats_paras(rf_local):
    stats_toolkit_obj = StatsToolKits([], [])
    print("testset size:" + "%d" % (len(rf_local.get_test_set_original_results()) / 0.2))
    print("testset range:" + "%d" % stats_toolkit_obj.get_range(rf_local.get_test_set_original_results()))
    print("standard derivation:" + "%.2f" % stats_toolkit_obj.get_std(rf_local.get_test_set_original_results()))


def rank_features(x, y, feature_names):
    rf_local = RandomForestRegressor(n_estimators=20, max_depth=4)
    scores = []
    for i in range(x.shape[1]):
        score = cross_val_score(rf_local, x[:, i:i+1], y, scoring="r2", cv=ShuffleSplit(len(x), 3, .3))
        scores.append((round(np.mean(score), 3), feature_names[i]))
    return sorted(scores, reverse=True)


def disp_map(map_val):
    for val in map_val:
        print(str(val[0]) + '\t' + str(val[1]))

def get_mean_bp(sbp_and_dbp_mat):  # 获取到mbp # 给两列分别为SBP和DBP的matrix添加第三列（2dbp+sbp）/3
    mbp = (2 * sbp_and_dbp_mat[:, 0] + sbp_and_dbp_mat[:, 1]) / 3
    return mbp
    # return np.concatenate((sbp_and_dbp_mat, mbp), axis=1)


if __name__ == "__main__":
    # a = [(0.098, 'RBW10'), (0.095, 'RBW25'), (0.089, 'RBW33'), (0.087, 'kte_delta'), (0.076, 'RBW50'), (0.069, 'pwtt_mean'), (0.067, 'RBW66'), (0.064, 'kte_skew'), (0.058, 'RBW75'), (0.058, 'DBW50'), (0.057, 'SLP2'), (0.057, 'SLP1'), (0.056, 'h_miu'), (0.055, 'kte_iqr'), (0.055, 'PRT'), (0.055, 'DBW33'), (0.053, 'DBW25'), (0.051, 'DBW10'), (0.049, 'DBW66'), (0.042, 'KVAL'), (0.042, 'DBW75'), (0.034, 'hr_delta'), (0.028, 'hr_miu'), (0.028, 'PH'), (0.027, 'ppg_fed_ar_5'), (0.027, 'RBAr'), (0.027, 'AmBE'), (0.026, 'kte_miu'), (0.026, 'PWA'), (0.021, 'DfAmBE'), (0.017, 'hr_skew'), (0.014, 'hr_iqr'), (0.013, 'ppg_fed_ar_4'), (0.009, 'ppg_fed_ar_1'), (0.009, 'h_delta'), (0.008, 'ppg_fed_ar_2'), (0.008, 'loge_delta'), (0.006, 'loge_iqr'), (0.005, 'h_iqr'), (0.002, 'ppg_fed_ar_3'), (0.001, 'h_skew'), (-0.0, 'loge_ar_5'), (-0.0, 'loge_ar_4'), (-0.0, 'loge_ar_3'), (-0.0, 'loge_ar_2'), (0.0, 'loge_ar_1')]
    # b = [(0.086, 'kte_delta'), (0.064, 'hr_miu'), (0.055, 'kte_iqr'), (0.054, 'RBW75'), (0.052, 'RBW66'), (0.049, 'RBW50'), (0.045, 'hr_delta'), (0.043, 'RBW33'), (0.037, 'RBW25'), (0.035, 'PWA'), (0.035, 'DBW66'), (0.034, 'kte_miu'), (0.034, 'RBW10'), (0.034, 'DBW50'), (0.033, 'AmBE'), (0.032, 'pwtt_mean'), (0.032, 'DfAmBE'), (0.028, 'RBAr'), (0.028, 'PH'), (0.028, 'DBW75'), (0.026, 'DBW33'), (0.026, 'DBW25'), (0.026, 'DBW10'), (0.024, 'ppg_fed_ar_2'), (0.023, 'ppg_fed_ar_1'), (0.022, 'kte_skew'), (0.022, 'KVAL'), (0.021, 'SLP1'), (0.017, 'hr_skew'), (0.017, 'PRT'), (0.015, 'SLP2'), (0.014, 'hr_iqr'), (0.013, 'ppg_fed_ar_5'), (0.012, 'h_miu'), (0.01, 'ppg_fed_ar_3'), (0.009, 'loge_delta'), (0.009, 'h_iqr'), (0.007, 'loge_iqr'), (0.007, 'h_delta'), (0.003, 'ppg_fed_ar_4'), (0.002, 'loge_ar_1'), (0.002, 'h_skew'), (0.001, 'loge_ar_2'), (0.0, 'loge_ar_5'), (0.0, 'loge_ar_4'), (-0.0, 'loge_ar_3')]

    # disp_map(a)
    # print('*****************')
    # disp_map(b)
    # exit()

    fh = FileHelper()
    all_csv_names = fh.get_all_csv_names()
    full_set_arr = []
    full_set_res = []
    for csv_file_name in all_csv_names:
        is_valid, arr, res = fh.read_file(csv_file_name)
        if not is_valid:
            continue
        full_set_arr = np.concatenate((full_set_arr, arr.tolist()), 0) if len(full_set_arr) > 0 else arr.tolist()
        res = get_mean_bp(res)
        full_set_res = np.concatenate((full_set_res, res.tolist()), 0) if len(full_set_res) > 0 else res.tolist()
    # print '************' + fh.colsRes[0] + '***************'
    # print rank_features(full_set_arr, full_set_res[:, 0], fh.cols)
    # print '************' + fh.colsRes[1] + '***************'
    # print rank_features(full_set_arr, full_set_res[:, 1], fh.cols)
    kf = KFold(full_set_res.shape[0], 10)
    max_sbp_corr = 0
    max_dbp_corr = 0

    sbp_corrs = []
    dbp_corrs = []

    max_sbp_train_index = []
    max_sbp_test_index = []

    max_dbp_train_index = []
    max_dbp_test_index = []
    index = 0
    max_sbp_index = 0
    max_dbp_index = 0

    max_sbp_index_num = 6
    max_dbp_index_num = 9

    # for train_index, test_index in kf:
    #     # if not (index == max_sbp_index_num or index == max_dbp_index_num):
    #     #     index += 1
    #     #     continue
    #
    #     if len(max_sbp_train_index) == 0:
    #         max_sbp_train_index = train_index
    #     if len(max_dbp_train_index) == 0:
    #         max_dbp_train_index = train_index
    #     if len(max_sbp_test_index) == 0:
    #         max_sbp_test_index = test_index
    #     if len(max_dbp_test_index) == 0:
    #         max_dbp_test_index = test_index
    #
    #     rf = RegressionAlgorithm()
    #     # if index == max_sbp_index_num:
    #     rf.x_train = full_set_arr[train_index, :]
    #     rf.y_train = full_set_res[train_index]
    #     rf.x_test = full_set_arr[test_index, :]
    #     rf.y_test = full_set_res[test_index]
    #     # rf.train()
    #     rf.train_mbp()
    #     stk = StatsToolKits(rf.test(), full_set_res[test_index])
    #     cor = stk.get_pearson_corr()
    #     print('***************SBP CORR')
    #     print cor
    #     cor = cor[0]
    #     # rf.show_full_set_result('Tuning Model SBP Regression Result')
    #     sbp_corrs.append(cor)
    #     if cor > max_sbp_corr:
    #         max_sbp_index = index
    #         max_sbp_corr = cor
    #         max_sbp_train_index = train_index
    #         max_sbp_test_index = test_index

        # rf.alter_type()
        # rf.reset_model()
        # rf.x_train = full_set_arr[train_index, :]
        # rf.y_train = full_set_res[train_index, :]
        # rf.x_test = full_set_arr[test_index, :]
        # rf.y_test = full_set_res[test_index, :]
        # rf.train()
        # stk = StatsToolKits(rf.test(), full_set_res[test_index, 1])
        # cor = stk.get_pearson_corr()
        # cor = cor[0]
        # # rf.show_full_set_result('Tuning Model DBP Regression Result')
        # dbp_corrs.append(cor)
        # print('***************DBP CORR')
        # print(cor)
        # if cor > max_dbp_corr:
        #     max_dbp_index = index
        #     max_dbp_corr = cor
        #     max_dbp_train_index = train_index
        #     max_dbp_test_index = test_index
        # index += 1
    index = 0
    for train_index, test_index in kf:
        if index == 1:
            max_sbp_train_index = train_index
            max_sbp_test_index = test_index
            break
        else:
            index += 1
            continue

    print("sbp max index:" + str(max_sbp_index) + " ")
    print(sbp_corrs)
    print("dbp max index:" + str(max_dbp_index) + " ")
    print(dbp_corrs)

    rf = RegressionAlgorithm()
    rf.x_train = full_set_arr[max_sbp_train_index, :]
    rf.y_train = full_set_res[max_sbp_train_index]
    rf.x_test = full_set_arr[max_sbp_test_index, :]
    rf.y_test = full_set_res[max_sbp_test_index]
    rf.train_mbp()
    cor = rf.test()
    # rf.show_full_set_result("Best SBP Regression")

    rf.show_mbp_full_set_result("Best SBP Regression", list(rf.y_test))
    stk = StatsToolKits(cor, full_set_res[max_sbp_test_index])
    cor = stk.get_pearson_corr()
    print('***************SBP CORR')
    print cor
    # rf.reset_model()
    # rf.alter_type()
    # rf.x_train = full_set_arr[max_dbp_train_index, :]
    # rf.y_train = full_set_res[max_dbp_train_index, :]
    # rf.x_test = full_set_arr[max_dbp_test_index, :]
    # rf.y_test = full_set_res[max_dbp_test_index, :]
    # rf.train()
    # rf.test()
    # rf.show_full_set_result("Best DBP Regression")

    exit()

    #
    root_path = '/mnt/code/matlab/data/csv-long/'
    only_csv_files = [f for f in listdir(root_path) if isfile(join(root_path, f)) &
                      f.startswith('a') & f.endswith('.csv')]
    type_sbp_nums = list([0] * 4)
    type_dbp_nums = list([0] * 4)
    full_set_arr = []
    full_set_res = []
    rf = RegressionAlgorithm()
    for csv_file_name in only_csv_files:
        arr, res = rf.read_file(root_path + csv_file_name)
        if np.size(arr, 0) >= RegressionAlgorithm.minFullSetSize:
            full_set_arr = np.concatenate((full_set_arr, arr.tolist()), 0) if len(full_set_arr) > 0 else arr.tolist()
            full_set_res = np.concatenate((full_set_res, res.tolist()), 0) if len(full_set_res) > 0 else res.tolist()
            rf.split_sets(arr, res)
            rf.train()
            rf.test()
            bhs_type = rf.get_result_bhs_type()
            type_sbp_nums[bhs_type] += 1
            rf.save_predict_result(csv_file_name, get_title(rf), '/mnt/code/matlab/data/csv/' + 'pic-long/' +
                                   BHSTypes.get_type_name(bhs_type))
            rf.alter_type()
            rf.reset_model()
            rf.train()
            rf.test()
            bhs_type = rf.get_result_bhs_type()
            type_dbp_nums[bhs_type] += 1
            rf.save_predict_result(csv_file_name, get_title(rf), '/mnt/code/matlab/data/csv/' + 'pic-long/' +
                                   BHSTypes.get_type_name(bhs_type))

            # rf.display_and_save_predict_result(csv_file_name)
            # input()
            # rf.display_and_save_predict_result()

        # rf.train()
        # rf.test()
    exit()
    rf.split_sets(full_set_arr, full_set_res)
    rf.train()
    rf.test()
    # rf.show_full_set_result(get_title(rf))
    disp_stats_paras(rf)
    rf.reset_model()
    rf.alter_type()
    rf.train()
    rf.test()
    disp_stats_paras(rf)
    # rf.show_full_set_result(get_title(rf))
    # rf_obj.train()
    # rf_obj.test()
    # rf_obj.show_predict_result()
    # print(type_sbp_nums)
    # print(type_dbp_nums)
