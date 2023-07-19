from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
from sklearn.model_selection import KFold

# Set global matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 24

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap


def rf_model_cross_validation(X, y, seed, cv=5):
    kf = KFold(n_splits=cv, random_state=seed, shuffle=True)

    params = {'n_estimators': [100],
              'max_depth': [1, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
    grid_search = GridSearchCV(RandomForestClassifier(), params, cv=kf, verbose=0)

    grid_search.fit(X, y)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    clf = grid_search.best_estimator_
    print('Best_score of random forest:', best_score)

    clf_with_all = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                          max_depth=best_params['max_depth'],
                                          min_samples_split=best_params['min_samples_split'],
                                          min_samples_leaf=best_params['min_samples_leaf'])
    clf_with_all.fit(X, y)
    # save clf_with_all model with pickle
    import pickle
    save_path = 'validation/cache/model/random_forest'
    with open(save_path + '/random_forest.pkl', 'wb') as f:
        pickle.dump(clf_with_all, f)

    # save best_params and best_score
    with open(save_path + '/rf_best_info.pkl', 'w') as f:
        f.write('best_score: ' + str(best_score) + '\n')
        f.write('best_params: ' + str(best_params) + '\n')
        f.close()

    # # Add your decision boundary plot and other visualizations here
    #
    # # Create a list to store the errors for each tree in the forest
    # cumulative_errors = []
    #
    # # Fit the model and calculate the error for each tree
    # for n_trees in range(1, clf.n_estimators + 1):
    #     clf.set_params(n_estimators=n_trees)
    #     clf.fit(X, y)
    #     y_pred_all = clf.predict(X)
    #     error = mean_squared_error(y, y_pred_all)
    #     cumulative_errors.append(error)
    #
    # # Plot the cumulative errors over the number of trees
    # plt.figure(figsize=(10, 7))
    # plt.plot(range(1, clf.n_estimators + 1), cumulative_errors, marker='o')
    # plt.xlabel('Number of Trees', fontsize=24, weight='bold')
    # plt.ylabel('Cumulative Error', fontsize=24, weight='bold')
    # plt.title('Random Forest Cumulative Errors over Number of Trees', fontsize=24, weight='bold')
    # plt.show()
    #
    # # Initialize JS for SHAP plots
    # shap.initjs()
    #
    # # Create a Kernel SHAP explainer
    # explainer = shap.KernelExplainer(clf.predict, X_train)
    #
    # # Calculate shap_values for all of X
    # shap_values = explainer.shap_values(X)
    #
    # # Plot the SHAP values
    # shap.summary_plot(shap_values, X)

    return clf
