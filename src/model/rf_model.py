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

    all_y_test = []
    all_y_pred = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid_search.fit(X_train, y_train)
        clf = grid_search.best_estimator_
        y_pred = clf.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    print("\nCross-validation RF Accuracy: ")
    print(accuracy_score(all_y_test, all_y_pred))

    # Train final model on all data with best parameters
    clf.fit(X, y)
    y_pred_all = clf.predict(X)

    print("\nFinal RF Accuracy on all data: ")
    print(accuracy_score(y, y_pred_all))

    # Add your decision boundary plot and other visualizations here

    # Create a list to store the errors for each tree in the forest
    cumulative_errors = []

    # Fit the model and calculate the error for each tree
    for n_trees in range(1, clf.n_estimators + 1):
        clf.set_params(n_estimators=n_trees)
        clf.fit(X, y)
        y_pred_all = clf.predict(X)
        error = mean_squared_error(y, y_pred_all)
        cumulative_errors.append(error)

    # Plot the cumulative errors over the number of trees
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, clf.n_estimators + 1), cumulative_errors, marker='o')
    plt.xlabel('Number of Trees', fontsize=24, weight='bold')
    plt.ylabel('Cumulative Error', fontsize=24, weight='bold')
    plt.title('Random Forest Cumulative Errors over Number of Trees', fontsize=24, weight='bold')
    plt.show()

    # Initialize JS for SHAP plots
    shap.initjs()

    # Create a Kernel SHAP explainer
    explainer = shap.KernelExplainer(clf.predict, X_train)

    # Calculate shap_values for all of X
    shap_values = explainer.shap_values(X)

    # Plot the SHAP values
    shap.summary_plot(shap_values, X)

    return clf, all_y_test, all_y_pred