from src.util.data_decoder import data_input, return_feature_dict
from src.util.feature_engineering import norm
from src.util.data_decoder import batch_data_decoder, data_concat
from src.util.result_saver import build_result_dir
import argparse
import sys
import os
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA
from src.model.pca import incre_pca

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


    X, y = data_concat(data)
    print(X)

    incre_pca(X, y, 2)
    # print(X)
    # print(y)




if __name__ == '__main__':
    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # print(X)
    # print(y)
    # print(iris.target_names)

    main()


    # addr = 'sample_data/PE01.csv'
    # data = data_input(addr)
    # print(data)
    # dict_feature = return_feature_dict(data)
    # print(dict_feature)


