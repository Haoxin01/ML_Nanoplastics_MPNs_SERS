from src.util.data_decoder import data_input, return_feature_dict
from src.util.feature_engineering import norm
from src.util.data_decoder import batch_data_decoder, data_concat, shuffle
from src.util.result_saver import build_result_dir
import argparse
import sys
from sklearn.decomposition import PCA, IncrementalPCA
from src.model.pca import incre_pca
from src.model.svm_model import svm_model
from src.model.knn_model import knn_model
from src.model.rf_model import rf_model
from src.model.iforest_mdoel import isoForest


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
    X, y = data_concat(data, if_shuffle=True, shuffle_seed=0)

    print(X)
    print(y)

    # principal component analysis
    # incre_pca(X, y, 2)
    pca, X_pca = incre_pca(X, y, 2)

    # var = pca.explained_variance_ratio_
    # print("\ndata_remaining: ")
    # print(var)

    print('\nX_pca: ', X_pca)
    print('\ny: ', y)

    # isoforest model
    isf_model = isoForest(X_pca, y, '')
    isf_model.pre_visualization()
    isf_model.train()
    isf_model.plot_discrete()
    isf_model.plot_path_length_decision_boundary()
    isf_model.predict()


    # # SVM
    # svm_model(X_pca, y)
    #
    # # KNN
    # knn_model(X_pca, y)
    #
    # # random forest, this use the dimension reduced data
    # rf_model(X_pca, y)

    # Add non-dimensional reduced data with random forest


    # TODO: add more models


if __name__ == '__main__':

    main()




