from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint



# use ensemble learning to predict the label of test data, with svm adn random forest
def ensemble_learning(data, label):
    X, y = data, label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # standardize the data, since SVM are not scale invariant
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # initialize the classifiers
    svm_clf = SVC(kernel='rbf', gamma='auto', probability=True, random_state=1)
    rf_clf = RandomForestClassifier(n_estimators=20, random_state=12)

    # create ensemble classifier
    eclf = VotingClassifier(estimators=[('svm', svm_clf), ('rf', rf_clf)], voting='soft', weights=[2, 1])

    # fit the ensemble classifier
    eclf.fit(X_train, y_train)

    # predict the label of test data
    y_pred = eclf.predict(X_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)


def ensemble_learning_rsCV(data, label):
    rf = RandomForestClassifier(n_estimators=50, random_state=1)
    svm = SVC(probability=True, random_state=1)
    X, y = data, label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # standardize the data, since SVM are not scale invariant
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # define the parameter distribution for Random Forest
    rf_param_dist = {
        'n_estimators': randint(50, 200),
        'max_features': ['auto', 'sqrt'],
        'max_depth': randint(10, 110),
        'min_samples_split': uniform(.01, .199),
        'min_samples_leaf': uniform(.01, .199),
        'bootstrap': [True, False]
    }

    # define the parameter distribution for SVM
    svm_param_dist = {
        'C': uniform(0.1, 10),
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    # create a random search of the pipeline, the fit the best model
    rf_random_search = RandomizedSearchCV(rf, param_distributions=rf_param_dist, n_iter=100, cv=5, verbose=2,
                                          random_state=42, n_jobs=-1)
    svm_random_search = RandomizedSearchCV(svm, param_distributions=svm_param_dist, n_iter=100, cv=5, verbose=2,
                                           random_state=42, n_jobs=-1)

    # fit the random search model (this will take some time)
    rf_random_search.fit(X_train, y_train)
    svm_random_search.fit(X_train, y_train)

    # you can inspect the best parameters found by RandomizedSearchCV
    print(rf_random_search.best_params_)
    print(svm_random_search.best_params_)

    # Replace the 'rf' and 'svm' in the VotingClassifier with the best estimators
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_random_search.best_estimator_), ('svm', svm_random_search.best_estimator_)],
        voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


# def