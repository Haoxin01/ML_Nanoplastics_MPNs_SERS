import pandas as pd
import os
from feature_engineering import norm, zscore_norm

def batch_data_decoder(data_addr):
    """
    This function is used to decode data from csv file in batches.
    """
    # loop all files in the directory
    data = {}
    for file in os.listdir(data_addr):
        # get file name
        file_name = os.path.splitext(file)[0]
        # get file address
        file_addr = data_addr + '\\' + file
        # input data from csv file
        data_mid = data_input(file_addr)
        data[file_name] = return_feature_dict(data_mid)
    return data


def data_concat(data):
    X = []
    y = []
    for item in data:
        for key in data[item]:
            X.append(data[item][key])
            y.append(label_identifier(key))

    X = zscore_norm(X)

    return X, y


def label_identifier(label):
    if 'PE' in label:
        return 0
    elif 'PMMA' in label:
        return 1
    elif 'PS' in label:
        return 2


def data_input(addr):
    """
    This function is used to input data from csv file.
    """
    # TODO: need to be optimized
    data = pd.read_csv(addr)
    # delete first column and first row
    data = data.iloc[1:, 1:]
    # rename first column
    data.rename(columns={data.columns[0]: 'wavenumber'}, inplace=True)
    return data


def return_feature_dict(data):
    dict = {}
    # return the number of column in data
    sample_num = data.shape[1] - 1
    feature_loc = [551.15, 615.29, 998.37, 1134.67]
    for i in range(sample_num):
        key = data.columns[i + 1]
        dict[key] = []
        # TODO: need to be optimized
        for item in feature_loc:
            if str(item) in data['wavenumber'].values:
                for j in range(len(data['wavenumber'])):
                    if data.iloc[j, 0] == str(item):
                        dict[key].append(float(data.iloc[j, i + 1]))
            else:
                print('Error: feature is not in the wavenumber list.')
                exit(-1)
    return dict


def loop_csv(addr):
    """
    This function is used to loop csv files in a directory.
    """

    pass
