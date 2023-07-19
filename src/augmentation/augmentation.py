import os
import random

import numpy as np
import pandas as pd
from src.util.data_decoder import batch_data_decoder, data_input


def data_augmentation(data_path, augmentation_ratio, times):
    for file in os.listdir(data_path):
        # init target df
        target_df = pd.DataFrame()

        file_name = os.path.splitext(file)[0]
        file_addr = data_path + '/' + file
        data_mid = data_input(file_addr)

        # convert all data_reference to float
        data_mid = data_mid.astype(float)

        # get rows where wavenumber between 498.72 and 2008.41
        data_mid = data_mid[(data_mid['wavenumber'] >= 498.72) &
                            (data_mid['wavenumber'] <= 2008.41)]
        data_mid = data_mid.reset_index(drop=True)

        # add wavenumber to target df
        target_df['wavenumber'] = data_mid['wavenumber']
        target_df = target_df.reset_index(drop=True)

        # loop all columns except wavenumber
        for j in range(1, data_mid.shape[1]):
            column_name = data_mid.columns[j]
            column_data = data_mid[column_name]
            target_df[column_name] = data_mid[column_name]
            target_df = augmentation_scalar(data_mid, target_df, column_name, column_data, augmentation_ratio, times)

        # save target df
        target_df.to_csv('result/' + file_name + '_augmented.csv', index=False)
        print('file augmentation: ' + file_name + ' finished.')
        print('-----------------------------------')
        print('-----------------------------------')

    print()


def augmentation_scalar(data_mid, target_df, column_name, column_data, augmentation_ratio, times):
    max_value = column_data.max()
    for i in range(times):
        new_column_name = column_name + '_' + str(i + 1) + '_augmented'
        target_df[new_column_name] = column_data
        for j in range(column_data.shape[0]):
            if j == 0:
                random_scalar = random_noise_scalar(augmentation_ratio)
                target_df.loc[j, new_column_name] = target_df.loc[j, new_column_name] + random_scalar * max_value
            else:
                if target_df.loc[j, new_column_name] == 0:
                    random_scalar = random_noise_scalar(augmentation_ratio)
                    target_df.loc[j, new_column_name] = target_df.loc[j, new_column_name] + random_scalar * max_value
                else:
                    variable_0_current = target_df.loc[j - 1, new_column_name]
                    variable_1_min = data_mid.loc[j, column_name]
                    variable_1_max = data_mid.loc[j, column_name] + max_value * augmentation_ratio
                    if data_mid.loc[j, column_name] > data_mid.loc[j-1, column_name]:
                        if variable_0_current <= variable_1_min:
                            random_scalar = random_noise_scalar(augmentation_ratio)
                            target_df.loc[j, new_column_name] = target_df.loc[j, new_column_name] + \
                                                                random_scalar * max_value
                        else:
                            target_df.loc[j, new_column_name] = target_df.loc[j, new_column_name] + \
                                (variable_1_max - variable_0_current) * random.random()
                    elif data_mid.loc[j, column_name] < data_mid.loc[j-1, column_name]:
                        if variable_0_current >= variable_1_max:
                            random_scalar = random_noise_scalar(augmentation_ratio)
                            target_df.loc[j, new_column_name] = target_df.loc[j, new_column_name] + \
                                                                random_scalar * max_value
                        else:
                            target_df.loc[j, new_column_name] = target_df.loc[j, new_column_name] + \
                                (variable_0_current - variable_1_min) * random.random()
                    else:
                        target_df.loc[j, new_column_name] = target_df.loc[j-1, new_column_name]

        print('augmentation ' + str(i + 1) + ' finished, for column: ' + column_name + ' finished.')

    return target_df


def random_noise_scalar(augmentation_ratio):
    random_scalar = random.random() * augmentation_ratio
    return random_scalar


if __name__ == '__main__':
    abs_path = 'data/original_data'
    ratio = 1 / 15
    times = 10
    data_augmentation(abs_path+'/data', ratio, times)
