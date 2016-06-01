# -*- coding: utf-8 -*-
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import pearsonr
from minepy import MINE
import numpy as np


class StatsToolKits:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_mean_square_error(self):
        return sqrt(mean_squared_error(self.x, self.y))

    def get_pearson_corr(self):  # return 0: corr, return1: p-index
        try:
            tmp = pearsonr(self.x, self.y)
        except:
            tmp = (0, 0)
        return tmp[0], tmp[1]

    def get_mic(self):
        m = MINE()
        m.compute_score(self.x, self.y)
        return m.mic()

    def get_range(self, data):
        return max(data) - min(data)

    def get_std(self, data):
        return np.std(data)
