import argparse
import os
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from src.model.hca_model import hca_model
from src.model.iforest_mdoel import isoForest
from src.model.knn_model import knn_model_cross_validation
from src.model.lda_model import lda_all, lda_udexcluded
from src.model.pca_model import pca, pca_mixture
from src.model.rf_model import rf_model_cross_validation
from src.model.svm_model import svm_model_cross_validation
from src.model.kmeans_model import kmeans_model
from src.model.tsne_model import tsne_implementation_all, tsne_implementation_udexcluded, tsne_mixture
from src.util.data_decoder import (
    batch_data_decoder,
    data_concat,
    data_concat_mixture,
    mixture_data_decoder,
    data_input,
    return_feature_dict,
    shuffle,
)
from src.util.feature_engineering import norm, select_best_num_features
from src.util.plot_categorical_correlation import plot_categorical_correlation
from src.util.train_strategy import search_best_model
from src.util.train_strategy import create_confusion_matrix, plot_confusion_matrix, compute_metrics


def prediction():
    # Env setting and data_reference loading START ----------------------------------------
    # set data_reference directory
    data_addr = 'data/sample_data_augmented'
    result_addr = 'result'

    # batch process all files and return X and y with shuffling
    # if cache exist, load from cache; else, process data_reference and store to cache
    cache_dir = 'result/cache'
    if not os.path.exists(cache_dir+'/variable'):
        os.makedirs(cache_dir+'/variable')
        data = batch_data_decoder(data_addr)
        X, y, Xe, ye = data_concat(data, if_shuffle=True, shuffle_seed=0)
        # store X, y and Xe, ye to Cache
        np.save(cache_dir + '/variable/X.npy', X)
        np.save(cache_dir + '/variable/y.npy', y)
        np.save(cache_dir + '/variable/Xe.npy', Xe)
        np.save(cache_dir + '/variable/ye.npy', ye)
    else:
        print('loading data_reference from cache...')
        X = np.load(cache_dir + '/variable/X.npy')
        y = np.load(cache_dir + '/variable/y.npy')
        Xe = np.load(cache_dir + '/variable/Xe.npy')
        ye = np.load(cache_dir + '/variable/ye.npy')

    # Env setting and data_reference loading END ------------------------------------------
    model_cache_path = 'validation/cache/model'
    variable_cache_path = 'validation/cache/variable/non_mixture'

    # categorical_correlation
    # plot_categorical_correlation(X, y)

    # # Feature selection
    # X_best = select_best_num_features(Xe, ye, score_func='mutual_info')


    # Dimension reduction START -----------------------------------------------------
    # PCA dimension reduction
    # print('PCA dimension reduction for data_reference including undetected data_reference...')
    X_pca = pca(X, y, 2, 'all')
    # np.save(variable_cache_path + '/X_pca.npy', X_pca)
    # print('PCA dimension reduction for data_reference excluding undetected data_reference...')
    # Xe_pca = pca(Xe, ye, 2, 'UD_excluded')
    # np.save(variable_cache_path + '/Xe_pca.npy', Xe_pca)


    # t-SNE dimension reduction
    print('t-SNE dimension reduction for data_reference including undetected data_reference...')
    # X_tsne = tsne_implementation_all(X, y, 2)
    print('t-SNE dimension reduction for data_reference excluding undetected data_reference...')
    Xe_tsne = tsne_implementation_udexcluded(Xe, ye, 2)

    # # LDA dimension reduction
    # print('LDA dimension reduction for data_reference including undetected data_reference...')
    # Xe_lda = lda_all(X, y, 2)
    # print('LDA dimension reduction for data_reference excluding undetected data_reference...')
    # X_lda = lda_udexcluded(Xe, ye, 2)
    # # Dimension reduction END -------------------------------------------------


    # # Outliner detection START -------------------------------------------------
    # # isoforest model
    # isf_model = isoForest(X_pca, y, result_addr)
    # isf_model.find_best_params()
    # isf_model.predict()
    # isf_model.analyze()
    # isf_model.pre_visualization()
    # isf_model.plot_discrete()
    # isf_model.plot_path_length_decision_boundary()
    # # Outliner detection END ---------------------------------------------------

    # Nonaplastics classification START ----------------------------------------
    # # Support Vector Machine
    # clf = svm_model_cross_validation(Xe_tsne, ye, 100)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)

    # K nearest neighbors
    clf = knn_model_cross_validation(Xe_tsne, ye, seed=100)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)

    # random forest, this use the dimension reduced data_reference
    clf = rf_model_cross_validation(Xe, ye, seed=100)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)

    # non-dimensional reduced data_reference with random forest
    # search_best_model(Xe, ye, rf_model, labels, model_cache_path, variable_cache_path)

    #  K-means model
    # clf, all_y_test, all_y_pred = kmeans_model(Xe_tsne, ye, n_clusters=4, seed=100)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)

    #  hca model
    clf, all_y_test, all_y_pred = hca_model(Xe_tsne, ye, n_clusters=4)
    # confusion matrix
    cm = create_confusion_matrix(all_y_test, all_y_pred)
    plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])
    accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    print (accuracy, recall, precision, f1)
    print(cm)

    # ensemble model: voting classifier of SVM, KNN, RF and K-means
    # Ensemble Model
    # clf, all_y_test, all_y_pred = ensemble_model(Xe_tsne, ye)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PE', 'PLA', 'PMMA', "PS"])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)

    print('Finished!')
    # Nonaplastics classification END ------------------------------------------


def prediction_mixture():
    '''
    This function is used to predict the mixture of plastics
    '''
    ## Env setting and data_reference loading START ----------------------------------------
    data_addr = 'D:/Nanoplastics-ML/data/mixture_data_augmented'
    variable_cache_path = 'D:/Nanoplastics-ML/validation/cache/variable/mixture'
    model_cache_path = 'D:/Nanoplastics-ML/validation/cache/model'
    if not os.path.exists(variable_cache_path):
        os.makedirs(variable_cache_path)
        os.makedirs(variable_cache_path+'/PS_PMMA')
        os.makedirs(variable_cache_path+'/PS_PLA')
        os.makedirs(variable_cache_path+'/PS_PE')
        data = batch_data_decoder(data_addr)
        X_ps_pe, y_ps_pe = data_concat_mixture(data, if_shuffle=True, shuffle_seed=0, mixture_type='PS_PE')
        # X_ps_pe = mixture_data_decoder(X_ps_pe, 'PS_PE')
        X_ps_pla, y_ps_pla = data_concat_mixture(data, if_shuffle=True, shuffle_seed=0, mixture_type='PS_PLA')
        # X_ps_pla = mixture_data_decoder(X_ps_pla, 'PS_PLA')
        X_ps_pmma, y_ps_pmma = data_concat_mixture(data, if_shuffle=True, shuffle_seed=0, mixture_type='PS_PMMA')
        # X_ps_pmma = mixture_data_decoder(X_ps_pmma, 'PS_PMMA')
        np.save(variable_cache_path+'/PS_PMMA/X_ps_pmma.npy', X_ps_pmma)
        np.save(variable_cache_path+'/PS_PMMA/y_ps_pmma.npy', y_ps_pmma)
        np.save(variable_cache_path+'/PS_PLA/X_ps_pla.npy', X_ps_pla)
        np.save(variable_cache_path+'/PS_PLA/y_ps_pla.npy', y_ps_pla)
        np.save(variable_cache_path+'/PS_PE/X_ps_pe.npy', X_ps_pe)
        np.save(variable_cache_path+'/PS_PE/y_ps_pe.npy', y_ps_pe)
    else:
        # reload
        X_ps_pmma = np.load(variable_cache_path+'/PS_PMMA/X_ps_pmma.npy')
        y_ps_pmma = np.load(variable_cache_path+'/PS_PMMA/y_ps_pmma.npy')
        X_ps_pla = np.load(variable_cache_path+'/PS_PLA/X_ps_pla.npy')
        y_ps_pla = np.load(variable_cache_path+'/PS_PLA/y_ps_pla.npy')
        X_ps_pe = np.load(variable_cache_path+'/PS_PE/X_ps_pe.npy')
        y_ps_pe = np.load(variable_cache_path+'/PS_PE/y_ps_pe.npy')


    ## Env setting and data_reference loading END ------------------------------------------
    # [1000.22, 809.73, 871.79, 1297.47]
    # PS, PMMA, PLA, PE
    # T-SNE dimension reduction START -----------------------------------------------
    ####################
    X_tsne_ps_pmma = tsne_mixture(X_ps_pmma, y_ps_pmma, n_components=2, mixture_type='PS_PMMA')
    X_tsne_ps_pla = tsne_mixture(X_ps_pla, y_ps_pla, n_components=2, mixture_type='PS_PLA')
    X_tsne_ps_pe = tsne_mixture(X_ps_pe, y_ps_pe, n_components=2, mixture_type='PS_PE')
    print('Finished T-SNE dimension reduction!')
    # T-SNE dimension reduction END -------------------------------------------------

    ## PS+PMMA, PS, PMMA
    # clf, y_test, y_pred = svm_model(X_tsne_ps_pe, y_ps_pe, 100)
    # print('PS+PE, PS, PE')
    # # confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # # ## PS+PE, PS, PE
    # clf, all_y_test, all_y_pred = knn_model_cross_validation(X_tsne_ps_pe, y_ps_pe, 100)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PS+PE', 'PS', 'PE'])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)

    # ## PS+PLA, PS, PLA
    # clf, all_y_test, all_y_pred = knn_model_cross_validation(X_tsne_ps_pla, y_ps_pla, 100)
    # # confusion matrix
    # cm = create_confusion_matrix(all_y_test, all_y_pred)
    # plot_confusion_matrix(cm, ['PS+PLA', 'PS', 'PLA'])
    # accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    # print (accuracy, recall, precision, f1)
    # print(cm)
    #
    ## PS+PMMA, PS, PMMA
    clf, all_y_test, all_y_pred = knn_model_cross_validation(X_tsne_ps_pmma, y_ps_pmma, 100)
    # confusion matrix
    cm = create_confusion_matrix(all_y_test, all_y_pred)
    plot_confusion_matrix(cm, ['PS+PMMA', 'PS', 'PMMA'])
    accuracy, recall, precision, f1 = compute_metrics(all_y_test, all_y_pred)
    print (accuracy, recall, precision, f1)
    print(cm)




if __name__ == '__main__':
    # outliner detection and classification for single kind of plastics
     prediction()

    # classification for mixed kinds of plastics
    #  prediction_mixture()

