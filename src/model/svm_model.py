from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import pandas as pd

def svm_model(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nSVM Accuracy: ")
    print(accuracy_score(y_test, y_pred))

    # # decision boundary plot
    # if isinstance(X, pd.DataFrame):
    #     X = X.to_numpy()  # convert DataFrame to NumPy array if X is a DataFrame
    # h = .02  # step size in the mesh
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # plt.figure(figsize=(10, 7))
    # sns.set(font_scale=1.6, style="whitegrid")
    # cmap = plt.cm.YlGnBu
    # plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    # scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=50, edgecolors='k')
    # plt.title('SVM Decision Boundary', fontsize=28)
    # plt.xlabel('First Principal Component', fontsize=24)
    # plt.ylabel('Second Principal Component', fontsize=24)
    #
    # # Define class labels and assign them to the legend
    # class_labels = ['PE', 'PS', 'PS_PE']
    # handles, _ = scatter.legend_elements()
    # legend1 = plt.legend(handles, class_labels, title="Classes", fontsize=22)
    # plt.setp(legend1.get_title(), fontsize='xx-large')
    #
    # # Ensure the plot is displayed correctly with all labels visible
    # plt.tight_layout()
    # plt.show()

    return clf, y_test, y_pred



