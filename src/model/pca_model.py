import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np

# mpl.use('TkAgg')
# warnings.filterwarnings('ignore')

# PCA and incremental PCA
def pca(X, y, n_components, ie):
    ipca = IncrementalPCA(n_components=n_components, batch_size=3)
    X_ipca = ipca.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # list to array
    y = np.array(y)
    if ie == 'all':
        colors = ["navy", "turquoise", "darkorange", "red", "green"]
    else:
        colors = ["navy", "turquoise", "darkorange", "red"]

    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        plt.figure(figsize=(8, 8))
        for color, i, target_name in zip(colors, [0, 1, 2, 3, 4],
                                         ['PE', 'PLA', 'PMMA', 'PS', 'UD']):
            plt.scatter(
                X_transformed[y == i, 0],
                X_transformed[y == i, 1],
                color=color,
                lw=2,
                label=target_name,
            )

        if "Incremental" in title:
            err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            plt.title(title + " of Nano-plastic dataset\nMean absolute unsigned error %.6f" % err)
        else:
            plt.title(title + " of Nano-plastic dataset")
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.axis([-1, 1, -1, 1])

        # save
        plt.savefig('result/pca/' + title + ie + " of Nano-plastic.png")
        plt.close()

    # stor information of pca and incre_pca to two txt files
    with open('result/pca/'+ie+'_pca.txt', 'w') as f:
        # wirte pca
        f.write('PCA-------------------------------------------\n')
        f.write('explained_variance_ratio_-------------------------------------------\n')
        f.write(str(pca.explained_variance_ratio_) + '\n')
        f.write('explained_variance_-------------------------------------------\n')
        f.write(str(pca.explained_variance_) + '\n')
        f.write('singular_values_-------------------------------------------\n')
        f.write(str(pca.singular_values_) + '\n')
        f.write('components_-------------------------------------------\n')
        f.write(str(pca.components_) + '\n')
        f.write('n_components_-------------------------------------------\n')
        f.write(str(pca.n_components_) + '\n')
        f.write('n_samples_-------------------------------------------\n')
        f.write(str(pca.n_samples_) + '\n')
        f.write('mean_-------------------------------------------\n')
        f.write(str(pca.mean_) + '\n')
        f.write('noise_variance_-------------------------------------------\n')
        f.write(str(pca.noise_variance_) + '\n')
        f.write('whiten-------------------------------------------\n')
        f.write(str(pca.whiten) + '\n')
        f.write('copy-------------------------------------------\n')
        f.write(str(pca.copy) + '\n')
        f.close()

    with open('result/pca/'+ie+'_incre_pca.txt', 'w') as f:
        # write incre_pca
        f.write('Incremental PCA-------------------------------------------\n')
        f.write('explained_variance_ratio_-------------------------------------------\n')
        f.write(str(ipca.explained_variance_ratio_) + '\n')
        f.write('explained_variance_-------------------------------------------\n')
        f.write(str(ipca.explained_variance_) + '\n')
        f.write('singular_values_-------------------------------------------\n')
        f.write(str(ipca.singular_values_) + '\n')
        f.write('components_-------------------------------------------\n')
        f.write(str(ipca.components_) + '\n')
        f.write('n_components_-------------------------------------------\n')
        f.write(str(ipca.n_components_) + '\n')
        f.write('mean_-------------------------------------------\n')
        f.write(str(ipca.mean_) + '\n')
        f.write('noise_variance_-------------------------------------------\n')
        f.write(str(ipca.noise_variance_) + '\n')
        f.write('whiten-------------------------------------------\n')
        f.write(str(ipca.whiten) + '\n')
        f.write('copy-------------------------------------------\n')
        f.write(str(ipca.copy) + '\n')
        f.close()

    return X_pca

