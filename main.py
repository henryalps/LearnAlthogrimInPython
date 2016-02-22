# -*- coding: utf-8 -*-
import MLModelBase
import LinearModel
import RandomForestModel
import BPModel
import FileHelper
import numpy as np
import StatsToolKits as stk
from os import listdir
from os.path import isfile, join
from enums import BHSTypes, BPTypes
#  类名 与 属性名 要采用 驼峰表达式
#  方法名 与 局部变量名 要采用 下划线表达式
#  使用‘\’来换行 但在[]/()/{}中无需这样使用


def use_model_and_get_result(model, csv_file_list, bp_type, pic_path):
    file_helper = FileHelper.FileHelper()
    type_nums = list([0] * 4)
    corr_list = list([0] * csv_file_list.__len__())
    corr_index = -1
    for csv_file_name in csv_file_list:
        try:
            corr_index += 1
            if bp_type == BPTypes.DBP:
                model.alter_type()
            trainset, testset = file_helper.get_trainset_and_testset_from_file_with_name(bp_type, csv_file_name)
            model.x_test, model.y_test = file_helper.split_original_data_matrix(testset)
            if model.x_test.__len__() <= 3:  # use 3 as testset size THRESHOLD
                corr_list[corr_index] = (0, 0)
                continue
            model.x_train, model.y_train = file_helper.split_original_data_matrix(trainset)
            model.train()
            model.test()
            bhs_type = model.get_result_bhs_type()
            type_nums[bhs_type] += 1
            model.save_predict_result(csv_file_name, pic_path + BHSTypes.get_type_name(bhs_type))
            corr_list[corr_index] = list(stk.StatsToolKits(list(model.y_test[:, model.colsResTypes.index(model.type)]),
                                                      list(model.testResults)).get_pearson_corr())
        except:
            corr_list[corr_index] = (0, 0)
            continue
    file_helper.write_file(pic_path + 'corr.txt', corr_list)
    return type_nums


def intersect_func(list_a, list_b):
    tmp = set(list_b)
    return [val for val in list_a if val in tmp]


if __name__ == "__main__":
    root_path = '/mnt/code/matlab/data/csv-pace-2-pace/long/'
    pic_sub_path = ('RF/', 'LF/', 'NN/')
    models = (RandomForestModel.RandomForestModel(), LinearModel.LinearModel(),
              BPModel.BPModel())
    type = BPTypes.SBP
    type_train_sub_path = 'sbp/train/'
    type_test_sub_path = 'sbp/test/'
    for i in range(0, pic_sub_path.__len__()):
        # 1 获取sbp文件列表
        only_train_csv_files = [f for f in listdir(root_path + type_train_sub_path) if isfile(join(root_path +
                                type_train_sub_path, f)) & f.startswith('a') & f.endswith('.csv')]
        only_test_csv_files = [f for f in listdir(root_path + type_test_sub_path) if isfile(join(root_path +
                                type_test_sub_path, f)) & f.startswith('a') & f.endswith('.csv')]
        only_train_csv_files = intersect_func(only_train_csv_files, only_test_csv_files)
        # 2 遍历得到所有训练-测试集对，进行学习
        type_sbp_nums = use_model_and_get_result(models[i], only_train_csv_files, type, root_path + pic_sub_path[i])

        type = BPTypes.DBP
        type_train_sub_path = 'dbp/train/'
        type_test_sub_path = 'dbp/test/'
        # 3 获取dbp文件列表
        only_train_csv_files = [f for f in listdir(root_path + type_train_sub_path) if isfile(join(root_path +
                                type_test_sub_path, f)) & f.startswith('a') & f.endswith('.csv')]
        only_test_csv_files = [f for f in listdir(root_path + type_test_sub_path) if isfile(join(root_path +
                                type_test_sub_path, f)) & f.startswith('a') & f.endswith('.csv')]
        only_train_csv_files = intersect_func(only_train_csv_files, only_test_csv_files)
        # 4 对dbp重复2
        type_dbp_nums = use_model_and_get_result(models[i], only_train_csv_files, type, root_path + pic_sub_path[i])
        # 5 打印各种血压估计结果
        print(type_sbp_nums)
        print(type_dbp_nums)

def unused_func():
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
            # model.x_train, model.x_test = model.scale_data(model.x_train, model. x_test)
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
