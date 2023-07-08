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
    print('processing data_reference in: ', data_addr, '...', 'filter data_reference with 811.69, 869.87, 998.37, 1295.78.')
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


def label_identifier(label):
    if 'UD' in label:
        return 4
    # elif 'PS_PMMA' in label:
    #     return 6
    # elif 'PS_PLA' in label:
    #     return 5
    # elif 'PS_PE' in label:
    #     return 4
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
    feature_loc = [811.69, 869.87, 998.37, 1295.78]
    # feature_loc = [551.15, 811.69, 869.87, 998.37, 1134.67, 1295.78, 1451.36, 1468.78, 1541.88, 1600.84]
    for i in range(sample_num):
        key = data.columns[i + 1]
        dict[key] = []
        # TODO: need to be optimized
        for item in feature_loc:
            # print('processing feature: ', item, '...')
            # if str(item) in data_reference['wavenumber'].values:
            if item in data['wavenumber'].values:
                for j in range(len(data['wavenumber'])):
                    # if data_reference.iloc[j, 0] == str(item):
                    if data.iloc[j, 0] == item:
                        dict[key].append(float(data.iloc[j, i + 1]))
            else:
                print('Error: feature is not in the wavenumber list.')
                exit(-1)
    return dict

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


def loop_csv(addr):
    """
    This function is used to loop csv files in a directory.
    """

    pass
