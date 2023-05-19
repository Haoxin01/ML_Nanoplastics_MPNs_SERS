from src.util.data_decoder import data_input, return_feature_dict
from src.util.feature_engineering import norm
from src.util.data_decoder import batch_data_decoder, data_concat, shuffle
from src.util.result_saver import build_result_dir
import argparse
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from src.model.pca import incre_pca
from src.model.svm_model import svm_model
from src.model.outliner_detection_model import isolation_forest_model
from src.model.knn_model import knn_model
from src.model.rf_model import rf_model
# from src.model.one_class_model import one_class_model
import numpy as np


import numpy as np
import random

def data_augmentation(X, noise_factor, times, seed=None):
    # convert list to numpy array for convenience
    X = np.array(X)
    original_data = X.tolist()
    augmented_data = []
    rng = np.random.RandomState(seed)  # Create a RandomState
    for x in X:
        for _ in range(times-1):
            # Add noise
            noise = rng.normal(loc=0.0, scale=noise_factor, size=x.shape)
            x_noise = x + noise
            augmented_data.append(x_noise.tolist())

    # Concatenate the original data and the augmented data
    narray = np.concatenate((original_data, augmented_data), axis=0)
    return narray



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest='address', help="data address directory")
    args = parser.parse_args()
    address_dir = args.address
    # get current system path
    sys_path = sys.path[0]
    # get data address
    data_addr = sys_path + '\\' + address_dir

    # build result directory
    build_result_dir(sys_path)

    # batch process all files and return in forms of dict
    data = batch_data_decoder(data_addr)

    # shuffle and concat data
    X, y = data_concat(data)

    # principal component analysis & visualization
    X_ipca, X_pca = incre_pca(X, y, n_components=2)

    # print('X for incremental PCA', X_ipca)
    # print('X for PCA', X_pca)
    # print('Original X', X)
    # print('Original y', y)

    # # data augmentation for X_pca
    # Aug3_X_pac = data_augmentation(X_pca, 0.01, 3)
    # Aug5_X_pac = data_augmentation(X_pca, 0.01, 5)
    # Aug7_X_pac = data_augmentation(X_pca, 0.01, 7)
    # Aug9_X_pac = data_augmentation(X_pca, 0.02, 9)
    #
    # # train outliner detection model
    # if_model1 = isolation_forest_model(X_pca, y, 'original', rng=9)
    # if_model3 = isolation_forest_model(Aug3_X_pac, y+y+y, 'augmented_3', rng=9)
    # if_model5 = isolation_forest_model(Aug5_X_pac, y+y+y+y+y, 'augmented_5', rng=9)
    # if_model7 = isolation_forest_model(Aug7_X_pac, y+y+y+y+y+y+y, 'augmented_7', rng=9)
    # if_model9 = isolation_forest_model(Aug9_X_pac, y+y+y+y+y+y+y+y+y, 'augmented_9', rng=9)


    rng = np.random.RandomState(9)
    X_outliers = rng.uniform(low=-1, high=1, size=(50, 2))


    # train 7------------------------------------
    print('train 7------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.01, 7, seed=i)
        model = isolation_forest_model(aug_data, y+y+y+y+y+y+y, 'augmented_7', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1)/len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.01, 7, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y + y, 'augmented_7', rng=9)
    # --------------------------------------------

    # train 6------------------------------------
    print('train 6------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.01, 6, seed=i)
        model = isolation_forest_model(aug_data, y+y+y+y+y+y, 'augmented_6', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1)/len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.01, 6, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y+ y, 'augmented_6', rng=9)
    # --------------------------------------------


    # train 7------------------------------------
    print('train 7------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.01, 7, seed=i)
        model = isolation_forest_model(aug_data, y+y+y+y+y+y, 'augmented_7', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1)/len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.01, 7, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y, 'augmented_7', rng=9)
    # --------------------------------------------



    # train 8------------------------------------
    print('train 8------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.01, 8, seed=i)
        model = isolation_forest_model(aug_data, y+y+y+y+y+y+y+y, 'augmented_8', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1)/len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.01, 8, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y + y+ y, 'augmented_8', rng=9)
    # --------------------------------------------


    # train 9------------------------------------
    print('train 9------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.01, 9, seed=i)
        model = isolation_forest_model(aug_data, y+y+y+y+y+y+y+y+y, 'augmented_9', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1)/len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.01, 9, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y+ y + y + y + y + y + y+ y, 'augmented_9', rng=9)
    # --------------------------------------------


    # train 10------------------------------------
    print('train 10------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.01, 10, seed=i)
        model = isolation_forest_model(aug_data, y+y+y+y+y+y+y+y+y+y, 'augmented_10', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1)/len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.01, 10, seed=rng_num)
    model = isolation_forest_model(aug_data, y+y+y+y+y+y+y+y+y+y, 'augmented_10', rng=9)
    # --------------------------------------------



    # train 6------------------------------------
    print('train 6------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.02, 6, seed=i)
        model = isolation_forest_model(aug_data, y + y + y + y + y + y, 'augmented_6_', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1) / len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.02, 6, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y, 'augmented_6_', rng=9)
    # --------------------------------------------

    # train 7------------------------------------
    print('train 7------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.02, 7, seed=i)
        model = isolation_forest_model(aug_data, y + y + y + y + y + y, 'augmented_7_', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1) / len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.02, 7, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y, 'augmented_7_', rng=9)
    # --------------------------------------------

    # train 8------------------------------------
    print('train 8------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.02, 8, seed=i)
        model = isolation_forest_model(aug_data, y + y + y + y + y + y + y + y, 'augmented_8_', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1) / len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.02, 8, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y + y + y, 'augmented_8_', rng=9)
    # --------------------------------------------

    # train 9------------------------------------
    print('train 9------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.02, 9, seed=i)
        model = isolation_forest_model(aug_data, y + y + y + y + y + y + y + y + y, 'augmented_9_', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1) / len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.02, 9, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y + y + y + y, 'augmented_9_', rng=9)
    # --------------------------------------------

    # train 10------------------------------------
    print('train 10------------------------------------')
    model_list = []
    for i in range(500):
        aug_data = data_augmentation(X_pca, 0.02, 10, seed=i)
        model = isolation_forest_model(aug_data, y + y + y + y + y + y + y + y + y + y, 'augmented_10_', rng=9)
        model_list.append(model)

    min_score = 1
    best_model = None
    rng_num = -1
    for i in range(500):
        res_list = model_list[i].predict(X_outliers)
        # return percentage of 1
        score = res_list.tolist().count(1) / len(res_list)
        if score < min_score:
            min_score = score
            rng_num = i
            best_model = model_list[i]

    print(best_model.predict(X_outliers))
    print(min_score)
    print(rng_num)

    aug_data = data_augmentation(X_pca, 0.02, 10, seed=rng_num)
    model = isolation_forest_model(aug_data, y + y + y + y + y + y + y + y + y + y, 'augmented_10_', rng=9)
    # --------------------------------------------




    # # predict outliner detection model
    # print(if_model3.predict(X_pca))
    # print(if_model5.predict(X_pca))
    # print(if_model7.predict(X_pca))
    #
    #
    # print(if_model3.predict(X_outliers))
    # print(if_model5.predict(X_outliers))
    # print(if_model7.predict(X_outliers))
    # print(if_model9.predict(X_outliers))




    # SVM
    # svm_model(X, y)

    # KNN
    # knn_model(X, y)

    # random forest, this use the dimension reduced data
    # rf_model(X, y)

    # Add non-dimensional reduced data with random forest

    # TODO: add more models


if __name__ == '__main__':

    main()






