from joblib import dump, load
import numpy as np
from src.model.iforest_mdoel import isoForest

def reload():
    model_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model'
    model_name = 'isoForest'
    model_full_path = model_path + '/' + model_name + '.joblib'
    clf = load(model_full_path)
    return clf

def reload_self():
    model_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model'
    model_name = 'isoForest'
    model_full_path = model_path + '/' + model_name + '_self_model.joblib'
    clf = load(model_full_path)
    return clf

if __name__ == '__main__':
    clf_object = reload_self()
    print(clf_object)
    out = clf_object.clf.predict(clf_object.X_test)
    caccuracy = np.sum(out == clf_object.y_test) / len(clf_object.y_test)

    print(caccuracy)
    clf_object.plot_discrete()
    clf_object.plot_path_length_decision_boundary()

    # load npy
    X = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/X.npy')
    y = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/y.npy')
    X_pca = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/X_pca.npy')
    Xe_pca = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/Xe_pca.npy')
    Xe = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/Xe.npy')
    ye = np.load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/variable/ye.npy')

    # isf_model = isoForest()
    # isf_model = load('/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model/isoForest_self_model.joblib')
    # isf_model.plot_discrete()