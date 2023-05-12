import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np

mpl.use('TkAgg')
warnings.filterwarnings('ignore')


def dim_reduction(data, n_components=2):
    """
    This function is used to reduce the dimension of data.
    """
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    return principalDf


def pca_visualization(principalDf, label):
    """
    This function is used to visualize the data after PCA.
    """
    plt.figure(figsize=(8, 8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('PC1', fontsize=15)
    plt.ylabel('PC2', fontsize=15)
    plt.title('PCA of ' + label, fontsize=20)
    targets = ['PE', 'PA', 'PP', 'PVC']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = principalDf['label'] == target
        plt.scatter(principalDf.loc[indicesToKeep, 'PC1'],
                    principalDf.loc[indicesToKeep, 'PC2'],
                    c=color,
                    s=50)
    plt.legend(targets, prop={'size': 15})
    plt.show()

def incre_pca(X, y):
    n_components = 2
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)
    X_ipca = ipca.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    print(X_pca)
    print(X_ipca)

    colors = ["navy", "turquoise", "darkorange"]

    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        plt.figure(figsize=(8, 8))
        for color, i, target_name in zip(colors, [0, 1, 2], ['PE', 'PMMA', 'PS']):
            plt.scatter(
                X_transformed[y == i, 0],
                X_transformed[y == i, 1],
                color=color,
                lw=2,
                label=target_name,
            )

        if "Incremental" in title:
            err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
        else:
            plt.title(title + " of iris dataset")
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.axis([-500, 2000, -500, 2000])

    plt.show()

