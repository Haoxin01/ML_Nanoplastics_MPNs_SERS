from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl

# Set global matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 22


def knn_model_cross_validation(X, y, seed, cv=5):
    kf = KFold(n_splits=cv, random_state=seed, shuffle=True)

    params = {'n_neighbors': range(10, 50), 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=kf, verbose=0)

    grid_search.fit(X, y)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    clf = grid_search.best_estimator_
    print('Best_score of k nearest neighborhood:', best_score)

    clf_with_all = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    clf_with_all.fit(X, y)
    # save clf_with_all model with pickle
    import pickle
    save_path = 'validation/cache/model/knn'
    with open(save_path + '/' + 'knn.pkl', 'wb') as f:
        pickle.dump(clf_with_all, f)

    # save best_params and best_score
    with open(save_path + '/' + 'knn_best_params.txt', 'w') as f:
        f.write('Best_score of k nearest neighborhood:' + str(best_score) + '\n')
        f.write('Best_params of k nearest neighborhood:' + str(best_params) + '\n')
        f.close()

    # # decision boundary plot
    # h = .02  # step size in the mesh
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    #
    # plt.figure(figsize=(10, 7))
    # sns.set(font_scale=1.6, style="whitegrid")
    # cmap = plt.cm.YlGnBu
    # plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    # scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=50, edgecolors='k')
    # plt.title('KNN Decision Boundary', fontsize=28)
    # plt.xlabel('First Principal Component', fontsize=24, weight='bold')
    # plt.ylabel('Second Principal Component', fontsize=24, weight='bold')
    #
    #
    # # Define class labels and assign them to the legend
    # class_labels = ['PE', 'PLA', 'PMMA', 'PS']
    # handles, _ = scatter.legend_elements()
    # legend1 = plt.legend(handles, class_labels, title="Classes", fontsize=20, loc='upper right')
    # plt.setp(legend1.get_title(), fontsize='xx-large')
    #
    # # Ensure the plot is displayed correctly with all labels visible
    # plt.tight_layout()
    # plt.show()

    return clf
