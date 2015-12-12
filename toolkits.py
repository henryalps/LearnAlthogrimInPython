# -*- coding: utf-8 -*-
from enums import BHSTypes


class ToolKits:

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
