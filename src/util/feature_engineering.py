import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC  # Import the model you're using
import pandas as pd
import matplotlib.pyplot as plt

def norm(X):
    """
    This function is used to normalize the data inside the samples
    :param dict:
    :return: dict
    """
    # normalization x
    for i in range(len(X)):
        max_value = max(X[i])
        min_value = min(X[i])
        for j in range(len(X[i])):
            X[i][j] = (X[i][j] - min_value) / (max_value - min_value)
    return X

def zscore_norm(X):
    """
    This function is used to normalize the data inside the samples
    :param dict:
    :return: dict
    """
    # normalization x
    for i in range(len(X)):
        mean_value = np.mean(X[i])
        std_value = np.std(X[i])
        for j in range(len(X[i])):
            X[i][j] = (X[i][j] - mean_value) / std_value
    return X

def feature_selection(X, y, k, score_func='mutual_info'):
    """
    This function is used to select k best features using Mutual Information or ANOVA F-value
    :param X: Feature Matrix
    :param y: Target Vector
    :param k: Number of best features to select
    :param score_func: Scoring function to use ('mutual_info' or 'f_value')
    :return: DataFrame of k best features
    """
    # Converting list of lists to DataFrame
    X = pd.DataFrame(X)

    # Choose scoring function
    if score_func == 'mutual_info':
        score_func = mutual_info_classif
    elif score_func == 'f_value':
        score_func = f_classif
    else:
        raise ValueError("score_func should be either 'mutual_info' or 'f_value'")

    # Apply SelectKBest class to extract top k best features
    bestfeatures = SelectKBest(score_func=score_func, k=k)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # Concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  # Naming the dataframe columns

    print(featureScores.nlargest(k,'Score'))  # Print k best features

    # Creating DataFrame of k best features
    X_best = X[featureScores.nlargest(k,'Score')['Feature'].values]

    return X_best


def plot_fsp_set(num_features_range, scores, final_scores):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(num_features_range, scores, marker='o')
    plt.title('Feature Selection Profile (FSP)')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Cross-Validation Score')

    plt.subplot(1, 2, 2)
    plt.plot(num_features_range, final_scores, marker='o')
    plt.title('Subset Evaluation Curve (SET)')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Final Model Score')

    plt.tight_layout()
    plt.show()


def select_best_num_features(X, y, score_func=f_classif):
    model = SVC(kernel='linear')  # Creating an instance of SVC model for feature selection
    num_features_range = list(range(1, 7))  # Change accordingly if you have different number of features
    scores = []
    final_scores = []

    for num_features in num_features_range:
        # Select features
        X_selected = feature_selection(X, y, num_features, score_func=score_func)

        # Create and evaluate model
        score = cross_val_score(model, X_selected, y, cv=5).mean()  # use 5-fold cross validation
        scores.append(score)

        # Train model on the whole dataset and get the final score
        model.fit(X_selected, y)
        final_score = model.score(X_selected, y)
        final_scores.append(final_score)

    # Plot FSP and SET
    plot_fsp_set(num_features_range, scores, final_scores)

    # Find the number of features that leads to the best performance
    best_num_features = num_features_range[np.argmax(scores)]
    print("Best number of features:", best_num_features)

    # Select the best number of features
    X_best = feature_selection(X, y, best_num_features, score_func=score_func)

    return X_best