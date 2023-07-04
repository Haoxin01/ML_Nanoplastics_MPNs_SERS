from src.util.data_decoder import data_input, return_feature_dict
from src.util.feature_engineering import norm, select_best_num_features
from src.util.data_decoder import batch_data_decoder, data_concat, shuffle
from src.util.result_saver import build_result_dir
import argparse
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from src.model.pca import pca
from src.model.svm_model import svm_model
from src.model.knn_model import knn_model
from src.model.rf_model import rf_model
from src.model.lda import lda_all, lda_udexcluded
from src.model.iforest_mdoel import isoForest
from src.model.tsne import tsne_implementation_udexcluded, tsne_implementation_all
from src.util.confusion_matrix import create_confusion_matrix, plot_confusion_matrix, compute_metrics
import numpy as np
import os

def main():
    # Env setting and data loading START ----------------------------------------
    # set data directory
    data_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/sample_data_augumented'
    result_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/result'

    # batch process all files and return X and y with shuffling
    # if cache exist, load from cache; else, process data and store to cache
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
        print('loading data from cache...')
        X = np.load(cache_dir + '/X.npy')
        y = np.load(cache_dir + '/y.npy')
        Xe = np.load(cache_dir + '/Xe.npy')
        ye = np.load(cache_dir + '/ye.npy')

    # Env setting and data loading END ------------------------------------------

    # Dimension reduction START -----------------------------------------------------
    # PCA dimension reduction
    print('PCA dimension reduction for data including undetected data...')
    # X_pca = pca(X, y, 2, 'all')
    print('PCA dimension reduction for data excluding undetected data...')
    # Xe_pca = pca(Xe, ye, 2, 'UD_excluded')

    # t-SNE dimension reduction
    print('t-SNE dimension reduction for data including undetected data...')
    X_tsne = tsne_implementation_all(X, y, 2)
    print('t-SNE dimension reduction for data excluding undetected data...')
    Xe_tsne = tsne_implementation_udexcluded(Xe, ye, 2)

    # LDA dimension reduction
    print('LDA dimension reduction for data including undetected data...')
    Xe_lda = lda_all(X, y, 2)
    print('LDA dimension reduction for data excluding undetected data...')
    X_lda = lda_udexcluded(Xe, ye, 2)
    # Dimension reduction END -------------------------------------------------

    # Outliner detection START -------------------------------------------------
    # isoforest model
    isf_model = isoForest(X_pca, y, "")
    isf_model.pre_visualization()
    isf_model.train()
    isf_model.plot_discrete()
    isf_model.plot_path_length_decision_boundary()
    isf_model.predict()
    # Outliner detection END ---------------------------------------------------

    # Nonaplastics classification START ----------------------------------------
    # list of your labels
    labels = ['PE', 'PS', 'PS_PE']

    # SVM
    svm_clf, svm_test, svm_pred = svm_model(X_pca, y)
    # confusion matrix and metrics for SVM
    svm_cm = create_confusion_matrix(svm_test, svm_pred)
    print("SVM Confusion Matrix: ")
    print(svm_cm)
    plot_confusion_matrix(svm_cm, labels)
    svm_acc, svm_rec, svm_prec, svm_f1 = compute_metrics(svm_test, svm_pred)
    print(f"SVM Accuracy: {svm_acc}, Recall: {svm_rec}, Precision: {svm_prec}, F1-score: {svm_f1}")

    # KNN
    knn_clf, knn_test, knn_pred = knn_model(X_pca, y)
    # confusion matrix and metrics for KNN
    knn_cm = create_confusion_matrix(knn_test, knn_pred)
    print("KNN Confusion Matrix: ")
    print(knn_cm)
    plot_confusion_matrix(knn_cm, labels)
    knn_acc, knn_rec, knn_prec, knn_f1 = compute_metrics(knn_test, knn_pred)
    print(f"KNN Accuracy: {knn_acc}, Recall: {knn_rec}, Precision: {knn_prec}, F1-score: {knn_f1}")

    # random forest, this use the dimension reduced data
    rf_clf, rf_test, rf_pred = rf_model(X_pca, y)
    # confusion matrix and metrics for RF
    rf_cm = create_confusion_matrix(rf_test, rf_pred)
    print("RF Confusion Matrix: ")
    print(rf_cm)
    plot_confusion_matrix(rf_cm, labels)
    rf_acc, rf_rec, rf_prec, rf_f1 = compute_metrics(rf_test, rf_pred)
    print(f"RF Accuracy: {rf_acc}, Recall: {rf_rec}, Precision: {rf_prec}, F1-score: {rf_f1}")
    #Add non-dimensional reduced data with random forest

    # Nonaplastics classification END ------------------------------------------


if __name__ == '__main__':
    main()




