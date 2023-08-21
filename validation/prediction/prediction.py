import pickle
import numpy as np
from src.model.pca_model import pca
from src.model.tsne_model import tsne_implementation_udexcluded
from sklearn.neighbors import KNeighborsClassifier
from src.model.knn_model import knn_model_cross_validation
from src.util.train_strategy import *
import copy
from src.util.data_decoder import (
    batch_data_decoder,
    data_concat,
    data_concat_mixture,
    mixture_data_decoder,
    data_input,
    return_feature_dict,
    shuffle,
)
from validation.cache.reload_test import *

def knn_prediction():
    # reload knn model pkl file from cache
    with open('D:/Nanoplastics-ML/validation/cache/model/knn/knn.pkl', 'rb') as f:
        knn = pickle.load(f)

    # reload variables npy file from cache
    X_test = np.load('D:/Nanoplastics-ML/validation/cache/variable/non-mixture/Xe.npy')
    y_test = np.load('D:/Nanoplastics-ML/validation/cache/variable/non-mixture/Xe.npy')

    y_pred = knn.predict(X_test)

    # print accuracy
    print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))



def random_forest_prediction():
    pass

def isoforest_prediction():
    pass

def regression_prediction():
    pass

def tsne_reduction(X, y):
    Xe = np.load('D:/Nanoplastics-ML/validation/cache/variable/non-mixture/Xe.npy')
    ye = np.load('D:/Nanoplastics-ML/validation/cache/variable/non-mixture/Xe.npy')

    # every set in X and y, concat with Xe and ye and do tsne once
    res_list = []
    Xe = Xe.tolist()
    ye = ye.tolist()
    for i in range(len(X)):
        Xe_temp = copy.deepcopy(Xe)
        ye_temp = copy.deepcopy(ye)
        Xe_temp.extend([X[i]])
        ye_temp.extend([y[i]])
        Xe_tsne = tsne_implementation_udexcluded(Xe_temp, ye_temp, 2)
        # use knn to predict
        # reload knn model pkl file from cache
        with open('D:/Nanoplastics-ML/validation/cache/model/knn/knn.pkl', 'rb') as f:
            knn = pickle.load(f)
        y_pred = knn.predict(Xe_tsne)
        res_list.append(y_pred[-1])
        # concentration_list.append(concentration[i])

    cm = create_confusion_matrix(y, res_list)
    plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])

    print()

    return res_list

def pca_reduction():

    pass


def data_reader():
    pass

if __name__ == '__main__':
    # load data
    path_lake = 'D:/Nanoplastics-ML/validation/prediction/data/lake'
    path_tap = 'D:/Nanoplastics-ML/validation/prediction/data/tap'

    # for lake
    data = batch_data_decoder(path_lake)
    X, y, con_list, Xe, ye, cone_list = data_concat(data, if_shuffle=True, shuffle_seed=0)
    # # load pca pkl file
    # with open('D:/Nanoplastics-ML/validation/cache/model/dimension_reduction/pca_outlier.pkl', 'rb') as f:
    #     pca = pickle.load(f)
    # # map to pca space
    # X_pca = pca.transform(X)

    # iso_forest = reload_self()
    # # out = iso_forest.predict(X_pca)
    # out = iso_forest.clf.predict(X_pca)
    # # print percentage of 1 in out
    # print(np.sum(out == 1) / len(out))

    # tsne
    res_list = tsne_reduction(Xe, ye)
    print()


    # for tap
    # data = batch_data_decoder(path_tap)
    # X, y, Xe, ye = data_concat(data, if_shuffle=True, shuffle_seed=0)