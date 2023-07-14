from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import shap
# Set global matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 22


def svm_model_cross_validation(X, y, seed, cv=5):
    kf = KFold(n_splits=cv, random_state=seed, shuffle=True)

    params = {'C': [0.1, 1, 10],
              'kernel': ['linear']
              }
    grid_search = GridSearchCV(SVC(), params, cv=kf, verbose=0)

    best_score = 0
    best_params = None

    all_y_test = []
    all_y_pred = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid_search.fit(X_train, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_

        clf = grid_search.best_estimator_
        y_pred = clf.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    print("\nCross-validation SVM Accuracy: ")
    print(accuracy_score(all_y_test, all_y_pred))

    # Train final model on all data with best parameters
    clf = SVC(C=best_params['C'], kernel=best_params['kernel'])
    clf.fit(X, y)
    y_pred_all = clf.predict(X)

    print("\nFinal SVM Accuracy on all data: ")
    print(accuracy_score(y, y_pred_all))

    # Continue with the rest of your code



