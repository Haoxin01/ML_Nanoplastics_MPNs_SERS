import argparse
import os
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from src.model.iforest_mdoel import isoForest
from src.model.knn_model import knn_model, knn_grid_search
from src.model.lda_model import lda_all, lda_udexcluded
from src.model.pca_model import pca, pca_mixture
from src.model.rf_model import rf_model
from src.model.svm_model import svm_model
from src.model.ensemble_model import ensemble_model
from src.model.hpa_model import hpa_model
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
from src.util.result_saver import build_result_dir
from src.util.train_strategy import search_best_model


def prediction():
    # Env setting and data_reference loading START ----------------------------------------
    # set data_reference directory
    data_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/data/sample_data_augmented'
    mixture_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/data/mixture_data_augmented'
    result_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/result'

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
    model_cache_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model'
    variable_cache_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/non_mixture'

    # Dimension reduction START -----------------------------------------------------
    # PCA dimension reduction
    # print('PCA dimension reduction for data_reference including undetected data_reference...')
    X_pca = pca(X, y, 2, 'all')
    # np.save(variable_cache_path + '/X_pca.npy', X_pca)
    # print('PCA dimension reduction for data_reference excluding undetected data_reference...')
    Xe_pca = pca(Xe, ye, 2, 'UD_excluded')
    # np.save(variable_cache_path + '/Xe_pca.npy', Xe_pca)

    # t-SNE dimension reduction
    print('t-SNE dimension reduction for data_reference including undetected data_reference...')
    X_tsne = tsne_implementation_all(X_pca, y, 2)
    print('t-SNE dimension reduction for data_reference excluding undetected data_reference...')
    Xe_tsne = tsne_implementation_udexcluded(Xe_pca, ye, 2)

    # LDA dimension reduction
    # print('LDA dimension reduction for data_reference including undetected data_reference...')
    # Xe_lda = lda_all(X, y, 2)
    # print('LDA dimension reduction for data_reference excluding undetected data_reference...')
    # X_lda = lda_udexcluded(Xe, ye, 2)
    # Dimension reduction END -------------------------------------------------

    # Outliner detection START -------------------------------------------------
    # isoforest model
    # isf_model = isoForest(X_pca, y, result_addr)
    # isf_model.pre_visualization()
    # isf_model.train()
    # isf_model.plot_discrete()
    # isf_model.plot_path_length_decision_boundary()
    # isf_model.predict()
    # isf_model.find_best_params()
    # Outliner detection END ---------------------------------------------------

    # Nonaplastics classification START ----------------------------------------
    # list of your labels
    labels = ['PE', 'PLA', 'PMMA', 'PS']

    # Support Vector Machine
    model_param = {'kernel': 'rbf', 'C': 1, 'gamma': 0.1}
    search_best_model(Xe_tsne, ye, svm_model, model_param, labels, model_cache_path, variable_cache_path)

    # K nearest neighbors
    knn_best_param = knn_grid_search(Xe_pca, ye)
    search_best_model(Xe_tsne, ye, knn_model, knn_best_param, labels, model_cache_path, variable_cache_path)

    # random forest, this use the dimension reduced data_reference
    search_best_model(Xe_pca, ye, rf_model, labels, model_cache_path, variable_cache_path)

    # non-dimensional reduced data_reference with random forest
    search_best_model(Xe, ye, rf_model, labels, model_cache_path, variable_cache_path)

    # hierarchical clustering
    search_best_model(Xe_pca, ye, hpa_model, labels, model_cache_path, variable_cache_path)

    # ensemble model: voting classifier of SVM, KNN and RF
    search_best_model(Xe_pca, ye, ensemble_model, labels, model_cache_path, variable_cache_path)

    print('Finished!')
    # Nonaplastics classification END ------------------------------------------


def prediction_mixture():
    '''
    This function is used to predict the mixture of plastics
    '''
    ## Env setting and data_reference loading START ----------------------------------------
    data_addr = '/Users/shiyujiang/Desktop/Nanoplastics-ML/data/mixture_data_augmented'
    variable_cache_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/mixture'
    model_cache_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/data/mixture'
    if not os.path.exists(variable_cache_path):
        os.makedirs(variable_cache_path)
        os.makedirs(variable_cache_path+'/PS_PMMA')
        os.makedirs(variable_cache_path+'/PS_PLA')
        os.makedirs(variable_cache_path+'/PS_PE')
        data = batch_data_decoder(data_addr)
        X_ps_pe, y_ps_pe = data_concat_mixture(data, if_shuffle=True, shuffle_seed=0, mixture_type='PS_PE')
        X_ps_pe = mixture_data_decoder(X_ps_pe, 'PS_PE')
        X_ps_pla, y_ps_pla = data_concat_mixture(data, if_shuffle=True, shuffle_seed=0, mixture_type='PS_PLA')
        X_ps_pla = mixture_data_decoder(X_ps_pla, 'PS_PLA')
        X_ps_pmma, y_ps_pmma = data_concat_mixture(data, if_shuffle=True, shuffle_seed=0, mixture_type='PS_PMMA')
        X_ps_pmma = mixture_data_decoder(X_ps_pmma, 'PS_PMMA')
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
    X_tsne_ps_pmma = tsne_mixture(X_ps_pmma, y_ps_pmma, n_components=2, mixture_type='PS_PMMA')
    X_tsne_ps_pla = tsne_mixture(X_ps_pla, y_ps_pla, n_components=2, mixture_type='PS_PLA')
    X_tsne_ps_pe = tsne_mixture(X_ps_pe, y_ps_pe, n_components=2, mixture_type='PS_PE')
    print('Finished T-SNE dimension reduction!')
    # T-SNE dimension reduction END -------------------------------------------------

    ## PS+PMMA, PS, PMMA

    ## PS+PLA, PS, PLA

    ## PS+PE, PS, PE



if __name__ == '__main__':
    # outliner detection and classification for single kind of plastics
    # prediction()

    # classification for mixed kinds of plastics
    prediction_mixture()
