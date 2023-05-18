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

    print(X)
    print(y)

    # principal component analysis
    incre_pca(X, y, 2)

    # SVM
    svm_model(X, y)

    # KNN
    knn_model(X, y)

    # random forest, this use the dimension reduced data
    rf_model(X, y)

    # Add non-dimensional reduced data with random forest

    # TODO: add more models


if __name__ == '__main__':

    main()



