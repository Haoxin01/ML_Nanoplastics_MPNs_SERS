from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def ensemble_model(X, y):
    # Create the individual models
    svm_clf = SVC()
    knn_clf = KNeighborsClassifier()
    rf_clf = RandomForestClassifier()

    # Create the ensemble model
    voting_clf = VotingClassifier(
        estimators=[('svm', svm_clf), ('knn', knn_clf), ('rf', rf_clf)],
        voting='hard')

    # Define the grid of parameters to search
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'knn__n_neighbors': range(10, 50),
        'knn__weights': ['uniform', 'distance'],
        'rf__n_estimators': [100],
        'rf__max_depth': [1, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    # Create the grid search object
    grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search object to the data
    grid_search.fit(X, y)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict the labels of the test set: y_pred
    y_pred = best_model.predict(X)

    return best_model, y, y_pred