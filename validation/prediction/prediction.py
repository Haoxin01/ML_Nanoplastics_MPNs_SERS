import pickle
import numpy as np

def knn_prediction():
    # reload knn model pkl file from cache
    with open('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model/knn/knn.pkl', 'rb') as f:
        knn = pickle.load(f)

    # reload variables npy file from cache
    X_test = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/non-mixture/Xe.npy')
    y_test = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/non-mixture/ye.npy')

    y_pred = knn.predict(X_test)

    # print accuracy
    print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))



def random_forest_prediction():
    pass

def isoforest_prediction():
    pass

def regression_prediction():
    pass

def tsne_fusion_prediction():
    pass


def data_reader():
    pass

if __name__ == '__main__':
    knn_prediction()