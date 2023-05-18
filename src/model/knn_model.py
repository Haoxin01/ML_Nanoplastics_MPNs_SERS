from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# fit the data into the model
def knn_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nKNN Accuracy: ")
    print(accuracy_score(y_test, y_pred))
    return clf
