import pandas as pd

def data_decoder(addr):
    """
    This function is used to decode data from csv file.
    """


    return



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
    feature_loc = [551.15, 600.89, 998.37, 1141.78]
    for i in range(sample_num):
        key = data.columns[i+1]
        dict[key] = []
        # TODO: need to be optimized
        for item in feature_loc:
            if str(item) in data['wavenumber'].values:
                for j in range(len(data['wavenumber'])):
                    if data.iloc[j, 0] == str(item):
                        dict[key].append(float(data.iloc[j, i+1]))
            else:
                print('Error: feature is not in the wavenumber list.')
                exit(-1)
    return dict


def loop_csv(addr):
    """
    This function is used to loop csv files in a directory.
    """

    pass


