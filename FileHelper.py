# -*- coding: utf-8 -*-
from Constants import _Const as Constant
from os import listdir
from os.path import isfile, join
from enums import BPTypes
import pandas as pd
import numpy as np
import scipy.io as io


class FileHelper:
    pathRoot = Constant.ROOT_PATH
    pathDbp = Constant.DBP_FOLDER_NAME
    pathSbp = Constant.SBP_FOLDER_NAME
    pathTrain = '/train'
    pathTest = '/test'

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

    # cols_updated = ['PH', 'PRT', 'PWA', 'RBAr', 'SLP1', 'SLP2',
    #         'RBW10', 'RBW25', 'RBW33', 'RBW50', 'RBW66', 'RBW75',
    #         'DBW10', 'DBW25', 'DBW33', 'DBW50', 'DBW66', 'DBW75',
    #         'KVAL', 'AmBE', 'DfAmBE', 'pwtt', 'hr']

    cols_updated = ['PRT', 'RBAr',
            'RBW10', 'RBW25', 'RBW33', 'RBW50', 'RBW66', 'RBW75',
            'DBW10', 'DBW25', 'DBW33', 'DBW50', 'DBW66', 'DBW75',
            'KVAL', 'AmBE', 'pwtt', 'hr']

    cols_updated_sbp = ['RBW10', 'RBW25', 'PRT', 'RBW33', 'pwtt', 'DBW50', 'RBW50']

    cols_updated_dbp = ['RBW66', 'RBW75', 'RBW10', 'RBW33', 'RBW25', 'RBW50', 'PRT', 'pwtt', 'hr', 'DBW50']

    cols_updated_updated = ['pwtt', 'hr']  # 'SLP1', 'SLP2'

    cols_updated_updated_updated = ['pwtt']  # 'SLP1', 'SLP2'

    colsRes = ['sbps', 'dbps']  # 'sbps' , 'dbps'

    bpType = BPTypes.SBP

    def __init__(self):
        self.long_csv_root_path = '/mnt/code/matlab/data/csv-long/'
        self.short_csv_root_path = '/mnt/code/matlab/data/csv/'

    def read_file(self, data_file_name):
        full_set = pd.read_csv(data_file_name)
        # return True, full_set.as_matrix(self.cols), full_set.as_matrix(self.colsRes)
        valid, arrs, indexes = self.pick_data(full_set.as_matrix(self.cols))
        return valid, np.matrix(arrs), full_set.as_matrix(self.colsRes)[indexes] if valid else []

    def write_file(self, file_name, data_list):
        f = open(file_name, "w")
        for data in data_list:
            try:
                f.write(str(data[0]) + "\t" + str(data[1]) + "\n")
            except:
                continue
        f.close()

    def write_file_names(self, file_name, name_list):
        f = open(file_name, "w")
        for name_i in name_list:
            try:
                f.write(name_i + "\n")
            except:
                continue
        f.close()

    def pick_data(self, data):  # find column which contains 0 and eliminate them
        is_legal = False
        for_return = []
        indexes = []
        for index_local in range(0, data.shape[0]):
            if np.where(data[index_local] == 0)[0].size == 0:
                if len(for_return) == 0:
                    is_legal = True
                indexes.append(index_local)
                for_return.append(data[index_local])
        return is_legal, for_return, indexes

    @staticmethod
    def get_all_csv_file_in_path(path):
        return [path + f for f in listdir(path) if isfile(join(path, f)) &
                      f.startswith('a') & f.endswith('.csv')]

    def get_short_csv_names(self):
        return self.get_all_csv_file_in_path(self.short_csv_root_path)

    def get_long_csv_names(self):
        return self.get_all_csv_file_in_path(self.long_csv_root_path)

    def get_all_csv_names(self):
        for_return = self.get_short_csv_names() + self.get_long_csv_names()
        return for_return

    def get_trainset_and_testset_from_file_with_name(self, bp_type, filename):
        self.bpType = bp_type
        if bp_type == BPTypes.DBP:
            full_name = self.pathRoot + self.pathDbp
        else:
            full_name = self.pathRoot + self.pathSbp
        trainset = pd.read_csv(join(full_name + self.pathTrain, filename))
        testset = pd.read_csv(join(full_name + self.pathTest, filename))
        return [trainset, testset]

    def split_original_data_matrix(self, set_matrix):
        if self.bpType == BPTypes.DBP:
            arr = set_matrix.as_matrix(self.cols_updated_dbp)
        else:
            arr = set_matrix.as_matrix(self.cols_updated_sbp)
        arr = set_matrix.as_matrix(self.cols_updated)
        res = set_matrix.as_matrix(self.colsRes)
        return [arr, res]

    def write_test_result_in_file(self, file_full_path, orig, est):
        try:
            io.savemat(file_full_path, dict(orig=orig, est=est))
        except:
            return

    @staticmethod
    def read_mat(mat_name):
        return io.loadmat(mat_name)