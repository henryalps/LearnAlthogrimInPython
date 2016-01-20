# -*- coding: utf-8 -*-
import MLModelBase
import BPModel
import LinearModel
import RandomForestModel
import numpy as np
from os import listdir
from os.path import isfile, join
from enums import BHSTypes
#  类名 与 属性名 要采用 驼峰表达式
#  方法名 与 局部变量名 要采用 下划线表达式
#  使用‘\’来换行 但在[]/()/{}中无需这样使用

if __name__ == "__main__":
    root_path = '/mnt/code/matlab/data/csv-long/'
    # !!此处可能随模型不同而发生变化
    pic_sub_path = 'LR/'  # 'LR/' 'RF/' 'NN/'
    # 1 获取文件列表
    only_csv_files = [f for f in listdir(root_path) if isfile(join(root_path, f)) &
                      f.startswith('a') & f.endswith('.csv')]
    # 定义各类预测结果的数量
    type_sbp_nums = list([0] * 4)
    type_dbp_nums = list([0] * 4)
    # 2 遍历文件列表
    for csv_file_name in only_csv_files:
        # !!此处更改使用的模型
        # model = RandomForestModel.RandomForestModel()
        # model = BPModel.BPModel()
        model = LinearModel.LinearModel()
        arr, res = model.read_file(join(root_path, csv_file_name))
        if np.size(arr, 0) >= MLModelBase.MLModelBase.minFullSetSize:
            # 3 收缩压模型训练
            model.split_sets(arr, res)
            # !!此处可能随模型不同而发生变化
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
