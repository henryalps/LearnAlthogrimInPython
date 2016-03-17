# -*- coding: utf-8 -*-
import MLModelBase
import LinearModel
import RandomForestModel
import BPModel
import FileHelper
import numpy as np
import toolkits as tk
import StatsToolKits as stk
from os import listdir
from os.path import isfile, join
from enums import BHSTypes, BPTypes
from Constants import _Const as Constant
import time
#  类名 与 属性名 要采用 驼峰表达式
#  方法名 与 局部变量名 要采用 下划线表达式
#  使用‘\’来换行 但在[]/()/{}中无需这样使用


def use_model_and_get_result_feature_sort(model, csv_file_list, bp_type, pic_path):
    file_helper = FileHelper.FileHelper()
    type_nums = list([0] * 4)
    corr_list = list([0] * csv_file_list.__len__())
    file_name_list = list()
    corr_index = -1
    all_data_x = []
    all_data_y = []
    bp_type_name = 'sbp'
    if bp_type == BPTypes.DBP:
        model.alter_type()
        bp_type_name = 'dbp'
    for csv_file_name in csv_file_list:
        try:
            corr_index += 1
            trainset, testset = file_helper.get_trainset_and_testset_from_file_with_name(bp_type, csv_file_name)
            #  handle train and test set first
            if not tk.ToolKits.is_train_set_legal(trainset):
                continue
            model.x_test, model.y_test = file_helper.split_original_data_matrix(testset)
            # if all_data_x.__len__() == 0:
            #     all_data_x = model.x_test
            #     all_data_y = model.y_test
            # else:
            #     all_data_x = np.concatenate((all_data_x, model.x_test), axis=0)
            #     all_data_y = np.concatenate((all_data_y, model.y_test), axis=0)
            # model.x_test = tk.ToolKits.get_legal_test_set(model.x_test)
            # model.y_test = tk.ToolKits.get_legal_test_set(model.y_test)  # without asarray, error would occur in get_result_bhs_type
            # if model.x_test.__len__() < 100:  # use 400(shortest trainset length)/4 as testset size THRESHOLD
            #     corr_list[corr_index] = (0, 0)
            #     continue
            model.x_train, model.y_train = file_helper.split_original_data_matrix(trainset)
            if all_data_x.__len__() == 0:
                all_data_x = model.x_train
                all_data_y = model.y_train
            else:
                all_data_x = np.concatenate((all_data_x, model.x_train), axis=0)
                all_data_y = np.concatenate((all_data_y, model.y_train), axis=0)
            #  !!! Use testset as trainset; trainset as testset
            # model.x_train, model.y_train, model.x_test, model.y_test = model.x_test, model.y_test, \
            #     model.x_train, model.y_train
            #  !!!
            # file_helper.write_test_result_in_file(Constant.ROOT_PATH + 'metadata/' + bp_type_name + '.mat',
            #                                       x, y)
        except:
            corr_list[corr_index] = (0, 0)
            continue
    FileHelper.FileHelper().write_test_result_in_file(bp_type_name + 'all_data.mat', all_data_x, all_data_y)
    model.x_train = all_data_x
    model.y_train = all_data_y
    # x, y = model.sort_features()
    # file_helper.write_test_result_in_file(Constant.ROOT_PATH + 'metadata/' + bp_type_name + '.mat', x, y)

    return type_nums


def use_model_and_get_result(model, csv_file_list, bp_type, pic_path):
    file_helper = FileHelper.FileHelper()
    type_nums = list([0] * 4)
    corr_list = list([0] * csv_file_list.__len__())
    file_name_list = list()
    corr_index = -1
    if bp_type == BPTypes.DBP:
        model.alter_type()
    for csv_file_name in csv_file_list:
        try:
            corr_index += 1
            trainset, testset = file_helper.get_trainset_and_testset_from_file_with_name(bp_type, csv_file_name)
            #  handle train and test set first
            if not tk.ToolKits.is_train_set_legal(trainset):
                continue
            model.x_test, model.y_test = file_helper.split_original_data_matrix(testset)
            # model.x_test = tk.ToolKits.get_legal_test_set(model.x_test)
            # model.y_test = tk.ToolKits.get_legal_test_set(model.y_test)  # without asarray, error would occur in get_result_bhs_type
            # if model.x_test.__len__() < 100:  # use 400(shortest trainset length)/4 as testset size THRESHOLD
            #     corr_list[corr_index] = (0, 0)
            #     continue
            model.x_train, model.y_train = file_helper.split_original_data_matrix(trainset)
            #  !!! Use testset as trainset; trainset as testset
            # model.x_train, model.y_train, model.x_test, model.y_test = model.x_test, model.y_test, \
            #     model.x_train, model.y_train
            #  !!!
            model.train()
            model.test()
            bhs_type = model.get_result_bhs_type()
            type_nums[bhs_type] += 1
            # model.save_predict_result(csv_file_name, pic_path + BHSTypes.get_type_name(bhs_type))
            corr_list[corr_index] = list(stk.StatsToolKits(list(model.y_test[:, model.colsResTypes.index(model.type)]),
                                                      list(model.testResults)).get_pearson_corr())

            file_name = csv_file_name.split('.')
            file_name = file_name[0]
            file_helper.write_test_result_in_file(pic_path + file_name, list(model.y_test[:, model.colsResTypes.index(model.type)]),
                                                      list(model.testResults))

            file_name_list.append(csv_file_name)  # After all have done
        except:
            corr_list[corr_index] = (0, 0)
            continue
    if bp_type == BPTypes.DBP:
        file_helper.write_file_names(pic_path + 'dbp_file_names.txt', file_name_list)
        file_helper.write_file(pic_path + 'dbp_corr.txt', corr_list)
    else:
        file_helper.write_file_names(pic_path + 'sbp_file_names.txt', file_name_list)
        file_helper.write_file(pic_path + 'sbp_corr.txt', corr_list)
    return type_nums


def intersect_func(list_a, list_b):
    tmp = set(list_b)
    return [val for val in list_a if val in tmp]


if __name__ == "__main__":
    root_path = Constant.ROOT_PATH
    pic_sub_path = list()
    # pic_sub_path.append('TMP/')
    # pic_sub_path.append('PTT-LF/')
    # pic_sub_path.append('LF' + Constant.TIME_LEN + '/')
    # pic_sub_path.append('RF' + Constant.TIME_LEN + '/')
    pic_sub_path.append(Constant.RESULT_FOLDER_NAME + '/' + Constant.SBP_FOLDER_NAME + '/' + 'RF/')
    # pic_sub_path.append('NN' + Constant.TIME_LEN + '/')

    # the model list should be of size with pic path list
    models = list()
    # models.append(LinearModel.LinearModel())
    models.append(RandomForestModel.RandomForestModel())
    # models.append(BPModel.BPModel())

    for i in range(0, pic_sub_path.__len__()):
        start_time = time.time()
        type_train_sub_path = Constant.SBP_FOLDER_NAME + '/train/'
        type_test_sub_path = Constant.SBP_FOLDER_NAME + '/test/'
        bp_type = BPTypes.SBP
        # 1 获取sbp文件列表
        only_train_csv_files = [f for f in listdir(root_path + type_train_sub_path) if isfile(join(root_path +
                                type_train_sub_path, f)) & f.startswith('a') & f.endswith('.csv') & (Constant.FILE_NAME_FILTER in f)]
        only_test_csv_files = [f for f in listdir(root_path + type_test_sub_path) if isfile(join(root_path +
                                type_test_sub_path, f)) & f.startswith('a') & f.endswith('.csv') & (Constant.FILE_NAME_FILTER in f)]
        only_train_csv_files = intersect_func(only_train_csv_files, only_test_csv_files)
        # 2 遍历得到所有训练-测试集对，进行学习
        type_sbp_nums = use_model_and_get_result_feature_sort(models[i], only_train_csv_files, bp_type, root_path + pic_sub_path[i])

        data_num = only_train_csv_files.__len__()

        models[i].reset_model()
        bp_type = BPTypes.DBP
        type_train_sub_path = Constant.DBP_FOLDER_NAME + '/train/'
        type_test_sub_path = Constant.DBP_FOLDER_NAME + '/test/'
        # 3 获取dbp文件列表
        only_train_csv_files = [f for f in listdir(root_path + type_train_sub_path) if isfile(join(root_path +
                                type_test_sub_path, f)) & f.startswith('a') & f.endswith('.csv') & (Constant.FILE_NAME_FILTER in f)]
        only_test_csv_files = [f for f in listdir(root_path + type_test_sub_path) if isfile(join(root_path +
                                type_test_sub_path, f)) & f.startswith('a') & f.endswith('.csv') & (Constant.FILE_NAME_FILTER in f)]
        only_train_csv_files = intersect_func(only_train_csv_files, only_test_csv_files)
        # 4 对dbp重复2
        type_dbp_nums = use_model_and_get_result_feature_sort(models[i], only_train_csv_files, bp_type, root_path +
                                                 pic_sub_path[i].replace(Constant.SBP_FOLDER_NAME, Constant.DBP_FOLDER_NAME))

        data_num += only_train_csv_files.__len__()

        # 5 打印各种血压估计结果
        print('************')
        print(type_sbp_nums)
        print(type_dbp_nums)
        print (time.time() - start_time)/data_num
        print('************')

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
