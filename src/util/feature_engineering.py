
def norm(dict):
    """
    This function is used to normalize the data inside the samples
    :param dict:
    :return: dict
    """
    for key in dict.keys():
        max_value = max(dict[key])
        min_value = min(dict[key])
        for i in range(len(dict[key])):
            dict[key][i] = (dict[key][i] - min_value) / (max_value - min_value)

    return dict