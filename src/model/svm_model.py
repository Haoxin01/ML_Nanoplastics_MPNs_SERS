import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


def svm_model(X, y):
    clf = svm.SVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf.fit(X_train, y_train)

    # visualization(clf, X, y)

    # get support vectors
    print(clf.support_vectors_)
    # get indices of support vectors
    print(clf.support_)
    # get number of support vectors for each class
    print(clf.n_support_)

    y_pred = clf.predict(X_test)

    print("\nSVM Accuracy: ")
    print(accuracy_score(y_test, y_pred))
    return clf


