# -*- coding: utf-8 -*-
from enums import BHSTypes
import numpy as np


class ToolKits:

    def __init__(self):
        return

    @staticmethod
    def which_bhs_standard(small_than_5, small_than_10, small_than_15):
        if (small_than_5 >= 60) & (small_than_10 >= 85) & (small_than_15 >= 95):
            return BHSTypes.TYPE_A
        elif (small_than_5 >= 50) & (small_than_10 >= 75) & (small_than_15 >= 90):
            return BHSTypes.TYPE_B
        elif (small_than_5 >= 40) & (small_than_10 >= 65) & (small_than_15 >= 85):
            return BHSTypes.TYPE_C
        else:
            return BHSTypes.TYPE_ERROR

    @staticmethod
    def is_train_set_legal(train_set):
        return train_set.shape[0] >= 10

    @staticmethod
    def get_legal_test_set(test_set):
        test_size_threshold = 300
        if test_set.__len__() <= test_size_threshold:
            return test_set
        else:  # resample the test_set to 300 points
            return_val = list([0] * test_size_threshold)
            pace_len = int(np.floor(test_set.__len__() / test_size_threshold))
            for index in range(0, test_size_threshold):
                return_val[index] = test_set[pace_len * index]
            return np.asarray(return_val)
