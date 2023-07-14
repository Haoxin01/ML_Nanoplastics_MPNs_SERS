import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def compute_cramers_v_matrix(df):
    cramers_v_matrix = np.zeros((df.shape[1], df.shape[1]))
    for col1, column1 in enumerate(df):
        for col2, column2 in enumerate(df):
            cramers_v_value = cramers_v(df[column1], df[column2])
            cramers_v_matrix[col1, col2] = cramers_v_value
    return cramers_v_matrix


def plot_categorical_correlation(X, y):
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'label']

    X_df = pd.DataFrame(X, columns=feature_names[:-1])
    y_df = pd.DataFrame(y, columns=[feature_names[-1]])

    df = pd.concat([X_df, y_df], axis=1)

    cramers_v_matrix = compute_cramers_v_matrix(df)

    # Create a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap
    sns.heatmap(cramers_v_matrix, cmap=cmap, annot=True, fmt=".2f", xticklabels=df.columns, yticklabels=df.columns)
    plt.show()