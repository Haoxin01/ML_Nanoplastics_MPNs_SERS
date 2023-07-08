import argparse
import os
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from src.model.iforest_mdoel import isoForest
from src.model.knn_model import knn_model
from src.model.lda_model import lda_all, lda_udexcluded
from src.model.pca_model import pca
from src.model.rf_model import rf_model
from src.model.svm_model import svm_model
from src.model.tsne_model import tsne_implementation_all, tsne_implementation_udexcluded
from src.util.confusion_matrix import (
    compute_metrics,
    create_confusion_matrix,
    plot_confusion_matrix,
)
from src.util.data_decoder import (
    batch_data_decoder,
    data_concat,
    data_input,
    return_feature_dict,
    shuffle,
)
from src.util.feature_engineering import norm, select_best_num_features
from src.util.result_saver import build_result_dir


def prediction():
    # Env setting and data_reference loading START ----------------------------------------
    # set data_reference directory
    data_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/sample_data_augumented'
    result_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/result'

    # batch process all files and return X and y with shuffling
    # if cache exist, load from cache; else, process data_reference and store to cache
    cache_dir = 'result/cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        data = batch_data_decoder(data_addr)
        X, y, Xe, ye = data_concat(data, if_shuffle=True, shuffle_seed=0)
        # store X, y and Xe, ye to Cache
        np.save(cache_dir + '/X.npy', X)
        np.save(cache_dir + '/y.npy', y)
        np.save(cache_dir + '/Xe.npy', Xe)
        np.save(cache_dir + '/ye.npy', ye)
    else:
        print('loading data_reference from cache...')
        X = np.load(cache_dir + '/X.npy')
        y = np.load(cache_dir + '/y.npy')
        Xe = np.load(cache_dir + '/Xe.npy')
        ye = np.load(cache_dir + '/ye.npy')

    # Env setting and data_reference loading END ------------------------------------------

    # Dimension reduction START -----------------------------------------------------
    # PCA dimension reduction
    print('PCA dimension reduction for data_reference including undetected data_reference...')
    X_pca = pca(X, y, 2, 'all')
    print('PCA dimension reduction for data_reference excluding undetected data_reference...')
    Xe_pca = pca(Xe, ye, 2, 'UD_excluded')

    # t-SNE dimension reduction
    # print('t-SNE dimension reduction for data_reference including undetected data_reference...')
    # X_tsne = tsne_implementation_all(X, y, 2)
    # print('t-SNE dimension reduction for data_reference excluding undetected data_reference...')
    # Xe_tsne = tsne_implementation_udexcluded(Xe, ye, 2)

    # LDA dimension reduction
    # print('LDA dimension reduction for data_reference including undetected data_reference...')
    # Xe_lda = lda_all(X, y, 2)
    # print('LDA dimension reduction for data_reference excluding undetected data_reference...')
    # X_lda = lda_udexcluded(Xe, ye, 2)
    # Dimension reduction END -------------------------------------------------

    # Outliner detection START -------------------------------------------------
    # isoforest model
    isf_model = isoForest(X_pca, y, result_addr)
    isf_model.pre_visualization()
    isf_model.train()
    isf_model.plot_discrete()
    isf_model.plot_path_length_decision_boundary()
    isf_model.predict()
    isf_model.find_best_params()
    # Outliner detection END ---------------------------------------------------

    # Nonaplastics classification START ----------------------------------------
    # list of your labels
    labels = ['PE', 'PS', 'PS_PE']

    # Support Vector Machine
    for seed in range(1000):
        svm_clf, svm_test, svm_pred = svm_model(Xe_pca, ye, seed)
    # confusion matrix and metrics for SVM
    svm_cm = create_confusion_matrix(svm_test, svm_pred, seed)
    print("SVM Confusion Matrix: ")
    print(svm_cm)
    plot_confusion_matrix(svm_cm, labels)
    svm_acc, svm_rec, svm_prec, svm_f1 = compute_metrics(svm_test, svm_pred)
    print(f"SVM Accuracy: {svm_acc}, Recall: {svm_rec}, Precision: {svm_prec}, F1-score: {svm_f1}")

    # KNN
    knn_clf, knn_test, knn_pred = knn_model(Xe_pca, ye)
    # confusion matrix and metrics for KNN
    knn_cm = create_confusion_matrix(knn_test, knn_pred)
    print("KNN Confusion Matrix: ")
    print(knn_cm)
    plot_confusion_matrix(knn_cm, labels)
    knn_acc, knn_rec, knn_prec, knn_f1 = compute_metrics(knn_test, knn_pred)
    print(f"KNN Accuracy: {knn_acc}, Recall: {knn_rec}, Precision: {knn_prec}, F1-score: {knn_f1}")

    # random forest, this use the dimension reduced data_reference
    rf_clf, rf_test, rf_pred = rf_model(Xe_pca, ye)
    # confusion matrix and metrics for RF
    rf_cm = create_confusion_matrix(rf_test, rf_pred)
    print("RF Confusion Matrix: ")
    print(rf_cm)
    plot_confusion_matrix(rf_cm, labels)
    rf_acc, rf_rec, rf_prec, rf_f1 = compute_metrics(rf_test, rf_pred)
    print(f"RF Accuracy: {rf_acc}, Recall: {rf_rec}, Precision: {rf_prec}, F1-score: {rf_f1}")
    # Add non-dimensional reduced data_reference with random forest

    # HPA

    # emsemble model: voting classifier of SVM, KNN and RF

    # Nonaplastics classification END ------------------------------------------


def prediction_mixture():
    pass


if __name__ == '__main__':
    # outliner detection and classification for single kind of plastics
    prediction()

    # classification for mixed kinds of plastics
    # prediction_mixture()
