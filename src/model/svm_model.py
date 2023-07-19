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

    grid_search.fit(X, y)
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    clf = grid_search.best_estimator_
    print('Best_score of support vector machine:', best_score)

    clf_with_all = SVC(C=best_params['C'], kernel=best_params['kernel'])
    clf_with_all.fit(X, y)
    # save clf_with_all model with pickle
    import pickle
    save_path = 'validation/cache/model/svm'
    with open(save_path + '/' + 'svm.pkl', 'wb') as f:
        pickle.dump(clf_with_all, f)

    # save best_score and best_params
    with open(save_path + '/' + 'svm_best_info.txt', 'w') as f:
        f.write(str(best_score))
        f.write('\n')
        f.write(str(best_params))
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
    # plt.title('SVM Decision Boundary', fontsize=28)
    # plt.xlabel('First Principal Component', fontsize=24, weight='bold')
    # plt.ylabel('Second Principal Component', fontsize=24, weight='bold')
    #
    # # Set x and y ticks as integers
    # x_ticks = np.linspace(X[:, 0].min(), X[:, 0].max(), 5).astype(int)
    # y_ticks = np.linspace(X[:, 1].min(), X[:, 1].max(), 5).astype(int)
    # x_formatter = mpl.ticker.FormatStrFormatter('%d')
    # y_formatter = mpl.ticker.FormatStrFormatter('%d')
    # plt.xticks(x_ticks, fontsize=24)
    # plt.yticks(y_ticks, fontsize=24)
    # plt.gca().xaxis.set_major_formatter(x_formatter)
    # plt.gca().yaxis.set_major_formatter(y_formatter)
    #
    # # Set x and y axis limits
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
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
