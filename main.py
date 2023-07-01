from src.util.data_decoder import data_input, return_feature_dict
from src.util.feature_engineering import norm, select_best_num_features
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
from src.model.tsne import tsne_dim_reduction, tsne_visualization
from src.util.confusion_matrix import create_confusion_matrix, plot_confusion_matrix, compute_metrics

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

    # Select best features based on model evaluation
    # X = select_best_num_features(X, y, 'f_value')

    # print(X)
    # print(y)

    # principal component analysis
    incre_pca(X, y, 2)
    pca, X_pca = incre_pca(X, y, 2)

    var = pca.explained_variance_ratio_
    print("\ndata_remaining: ")
    print(var)
    #
    # print('\nX_pca: ', X_pca)
    # print('\ny: ', y)

    # # t-SNE dimension reduction
    # tsneDf = tsne_dim_reduction(X, y, 2)
    #
    # # t-SNE visualization
    # tsne_visualization(tsneDf, "your_label")

    # isoforest model
    # isf_model = isoForest(X_pca, y, "")
    # isf_model.pre_visualization()
    # isf_model.train()
    # isf_model.plot_discrete()
    # isf_model.plot_path_length_decision_boundary()
    # isf_model.predict()

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


    # TODO: add more models


if __name__ == '__main__':

    main()




