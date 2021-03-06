# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


class FileHelper:
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

    def __init__(self):
        self.long_csv_root_path = '/mnt/code/matlab/data/csv-long/'
        self.short_csv_root_path = '/mnt/code/matlab/data/csv/'

    def read_file(self, data_file_name):
        full_set = pd.read_csv(data_file_name)
        # return True, full_set.as_matrix(self.cols), full_set.as_matrix(self.colsRes)
        valid, arrs, indexes = self.pick_data(full_set.as_matrix(self.cols))
        return valid, np.matrix(arrs), full_set.as_matrix(self.colsRes)[indexes] if valid else []

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
