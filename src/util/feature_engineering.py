import numpy as np

def norm(X):
    """
    This function is used to normalize the data inside the samples
    :param dict:
    :return: dict
    """
    # normalization x
    for i in range(len(X)):
        max_value = max(X[i])
        min_value = min(X[i])
        for j in range(len(X[i])):
            X[i][j] = (X[i][j] - min_value) / (max_value - min_value)
    return dict

def zscore_norm(X):
    """
    This function is used to normalize the data inside the samples
    :param dict:
    :return: dict
    """
    # normalization x
    for i in range(len(X)):
        mean_value = np.mean(X[i])
        std_value = np.std(X[i])
        for j in range(len(X[i])):
            X[i][j] = (X[i][j] - mean_value) / std_value
    return X