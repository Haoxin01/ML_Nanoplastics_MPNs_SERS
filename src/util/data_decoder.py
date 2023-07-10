import os
import random

import numpy as np
import pandas as pd
from src.util.feature_engineering import norm, zscore_norm

def batch_data_decoder(data_addr):
    """
    This function is used to decode data_reference from csv file in batches.
    """
    # loop all files in the directory
    data = {}
    print('processing data_reference in: ', data_addr, '...', 'filter data_reference with 811.69, 869.87, 998.37, 1295.78 of range -+ 5.')
    for file in os.listdir(data_addr):
        # get file name
        file_name = os.path.splitext(file)[0]
        print('processing and filter file: ', file_name, '...')
        # get file address
        file_addr = data_addr + '/' + file
        # input data_reference from csv file
        data_mid = data_input(file_addr)
        data[file_name] = return_feature_dict(data_mid)
    return data


def data_concat(data, if_shuffle: bool, shuffle_seed):
    X = []
    y = []
    Xe = []
    ye = []
    concat_data = {}
    for item in data:
        print('processing: ', item, '...')
        for key in data[item]:
            concat_data[key] = data[item][key]

    # print('concat_data', concat_data)

    print('shuffling concat_data with seed: ', shuffle_seed, '...')
    if if_shuffle:
        concat_data = shuffle_with_seed(concat_data, random_state=shuffle_seed)

    print('divide concat_data into X and y...')
    for key in concat_data:
        X.append(concat_data[key])
        y.append(label_identifier(key))
        if label_identifier(key) != 4:
            Xe.append(concat_data[key])
            ye.append(label_identifier(key))

    print('normalizing X...')
    X = norm(X)
    Xe = norm(Xe)

    return X, y, Xe, ye


def data_concat_mixture(data, if_shuffle: bool, shuffle_seed: int, mixture_type: str):
    X = []
    y = []
    concat_data = {}
    for item in data:
        if mixture_type == 'PS_PMMA':
            if item in ['PS_PMMA_augmented', 'PS_augmented', 'PMMA_augmented']:
                print('processing PS_PMMA-related Type: ', item, '...')
                for key in data[item]:
                    concat_data[key] = data[item][key]
        elif mixture_type == 'PS_PLA':
            if item in ['PS_PLA_augmented', 'PS_augmented', 'PLA_augmented']:
                print('processing PS_PLA-related Type: ', item, '...')
                for key in data[item]:
                    concat_data[key] = data[item][key]
        elif mixture_type == 'PS_PE':
            if item in ['PS_PE_augmented', 'PS_augmented', 'PE_augmented']:
                print('processing PS_PE-related Type: ', item, '...')
                for key in data[item]:
                    concat_data[key] = data[item][key]
        else:
            print('Error: mixture_type is not in the mixture_type list, please check ')
            exit(-1)

    # print('concat_data', concat_data)

    print('shuffling concat_data with seed: ', shuffle_seed, '...')
    if if_shuffle:
        concat_data = shuffle_with_seed(concat_data, random_state=shuffle_seed)

    print('divide concat_data into X and y...')
    for key in concat_data:
        X.append(concat_data[key])
        if mixture_type == 'PS_PMMA':
            y.append(label_identifier_PS_PMMA(key))
        elif mixture_type == 'PS_PLA':
            y.append(label_identifier_PS_PLA(key))
        elif mixture_type == 'PS_PE':
            y.append(label_identifier_PS_PE(key))
        else:
            print('Error: mixture_type is not in the mixture_type list, please check ')
            exit(-1)

    print('normalizing X...')
    X = norm(X)

    return X, y


def label_identifier_PS_PE(label):
    if 'PS_PE' in label:
        return 0
    elif 'PE' in label:
        return 1
    elif 'PS' in label:
        return 2
    else:
        print('Error: label is not in the label list, please check '
              'if the name of the csv files includes the following labels: '
              'PE, PS, UD (undetected group).')
        exit(-1)


def label_identifier_PS_PLA(label):
    if 'PS_PLA' in label:
        return 0
    elif 'PLA' in label:
        return 1
    elif 'PS' in label:
        return 2
    else:
        print('Error: label is not in the label list, please check '
              'if the name of the csv files includes the following labels: '
              'PLA, PS, UD (undetected group).')
        exit(-1)


def label_identifier_PS_PMMA(label):
    if 'PS_PMMA' in label:
        return 0
    elif 'PMMA' in label:
        return 1
    elif 'PS' in label:
        return 2
    else:
        print('Error: label is not in the label list, please check '
              'if the name of the csv files includes the following labels: '
              'PMMA, PS, UD (undetected group).')
        exit(-1)


def label_identifier(label):
    if 'UD' in label:
        return 4
    elif 'PE' in label:
        return 0
    elif 'PLA' in label:
        return 1
    elif 'PMMA' in label:
        return 2
    elif 'PS' in label:
        return 3
    else:
        print('Error: label is not in the label list, please check '
              'if the name of the csv files includes the following labels: '
              'PE, PLA, PMMA, PS, UD (undetected group).')
        exit(-1)

def data_input(addr):
    """
    This function is used to input data_reference from csv file.
    """
    # TODO: need to be optimized
    data = pd.read_csv(addr)
    # delete first column and first row
    # data_reference = data_reference.iloc[1:, 1:]
    # rename first column
    # data_reference.rename(columns={data_reference.columns[0]: 'wavenumber'}, inplace=True)
    return data


def return_feature_dict(data):
    dict = {}
    # return the number of column in data_reference
    sample_num = data.shape[1] - 1
    feature_loc = [1000.22, 809.73, 871.79, 1297.47]
    # feature_loc = [551.15, 811.69, 869.87, 998.37, 1134.67, 1295.78, 1451.36, 1468.78, 1541.88, 1600.84]
    feature_range = []
    for item in feature_loc:
        feature_range.append([item - 6, item + 6])

    for i in range(sample_num):
        key = data.columns[i + 1]
        # get clomun index based on column name key
        key_index = data.columns.get_loc(key)
        dict[key] = []
        # TODO: could to be optimized
        for list_range in feature_range:
            start = list_range[0]
            end = list_range[1]
            max_peak = 0
            # get all values in the range as a list
            temp_list = data[(data['wavenumber'] >= start) & (data['wavenumber'] <= end)]['wavenumber'].values
            # get the row index from temp_list
            temp_list_index = data[(data['wavenumber'] >= start) & (data['wavenumber'] <= end)].index.tolist()
            #  get the max peak in the range
            for i in range(len(temp_list)-2):
                if data.iloc[temp_list_index[i+1], key_index] > data.iloc[temp_list_index[i], key_index] and \
                    data.iloc[temp_list_index[i+1], key_index] > data.iloc[temp_list_index[i+2], key_index]:
                    if max_peak < data.iloc[temp_list_index[i+1], key_index]:
                        max_peak = data.iloc[temp_list_index[i+1], key_index]

            dict[key].append(max_peak)

        # for item in feature_loc:
        #     # print('processing feature: ', item, '...')
        #     # if str(item) in data_reference['wavenumber'].values:
        #     if item in data['wavenumber'].values:
        #         for j in range(len(data['wavenumber'])):
        #             # if data_reference.iloc[j, 0] == str(item):
        #             if data.iloc[j, 0] == item:
        #                 dict[key].append(float(data.iloc[j, i + 1]))
        #     else:
        #         print('Error: feature is not in the wavenumber list.')
        #         exit(-1)
    return dict


def mixture_data_decoder(X, mixture_type):
    if mixture_type == 'PS_PMMA':
        # only need first two columns from a 2D list
        X = [[sub_list[0], sub_list[1]] for sub_list in X]
    elif mixture_type == 'PS_PLA':
        # need first and third columns from a 2D list, not a numpy array
        X = [[sub_list[0], sub_list[2]] for sub_list in X]
    elif mixture_type == 'PS_PE':
        # need first and fourth columns
        X = [[sub_list[0], sub_list[3]] for sub_list in X]
    else:
        print('Error: mixture type is not in the mixture type list, please check ')
        exit(-1)
    return X



def shuffle(dict_data, random_state):
    """
    This function is used to shuffle a dict.
    """
    # Convert dict items to a list
    items = list(dict_data.items())
    # Shuffle list with seed
    random.shuffle(items)
    # Create new dictionary from shuffled list
    shuffled_dict = dict(items)
    return shuffled_dict


def shuffle_with_seed(dict_data, random_state):
    """
    This function is used to shuffle a dict with seed.
    """
    # Convert dict items to a list
    items = list(dict_data.items())
    # Shuffle list with seed
    random.Random(random_state).shuffle(items)
    # Create new dictionary from shuffled list
    shuffled_dict = dict(items)
    return shuffled_dict
