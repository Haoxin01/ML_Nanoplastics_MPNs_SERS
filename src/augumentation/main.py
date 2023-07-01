from augmentation import data_augmentation

if __name__ == '__main__':
    ratio = 1/15
    times = 10
    data_augmentation('data', ratio, times)