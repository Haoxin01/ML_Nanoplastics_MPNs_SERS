import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import seaborn as sns

# Set global matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 22

# PCA and incremental PCA
def pca(X, y, n_components, ie):
    ipca = IncrementalPCA(n_components=n_components, batch_size=3)
    X_ipca = ipca.fit_transform(X)

    pca = PCA(n_components=n_components)

    save_pca = pca.fit(X)
    # save_path = 'validation/cache/model/dimension_reduction'
    # save pca model with pickle
    # import pickle
    # with open(save_path + '/' + ie + '_pca.pkl', 'wb') as f:
    #     pickle.dump(save_pca, f)
    X_pca = pca.fit_transform(X)

    # list to array
    y = np.array(y)

    colors = sns.color_palette("Set2", 5)  # Generates a palette with 5 distinct colors
    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        plt.figure(figsize=(10, 10))
        for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], ['PE', 'PLA', 'PMMA', 'PS', 'UD']):
            plt.scatter(
                X_transformed[y == i, 0],
                X_transformed[y == i, 1],
                color=color,  # We use the color from the palette
                lw=2,
                label=target_name,
            )

            # Set x and y ticks and their font size
            plt.xticks(np.linspace(-1, 1, 5), fontsize=28)
            plt.yticks(np.linspace(-1, 1, 5), fontsize=28)

            # Set x and y labels and their font size
            plt.xlabel("First Principal Component", fontsize=28, weight='bold')
            plt.ylabel("Second Principal Component", fontsize=28, weight='bold')

            if "Incremental" in title:
                err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
                plt.title(title + " of Nano-plastic dataset\nMean absolute unsigned error %.6f" % err)
            else:
                plt.title(title + " of Nano-plastic dataset")

            plt.legend(loc="upper right", shadow=False, scatterpoints=1)
            plt.axis([-1.5, 1.5, -1.5, 1.5])

        # save
        # plt.savefig('result/pca/' + title + ie + " of Nano-plastic.png")
        plt.close()

    # # stor information of pca and incre_pca to two txt files
    # with open('result/pca/'+ie+'_pca.txt', 'w') as f:
    #     # wirte pca
    #     f.write('PCA-------------------------------------------\n')
    #     f.write('explained_variance_ratio_-------------------------------------------\n')
    #     f.write(str(pca.explained_variance_ratio_) + '\n')
    #     f.write('explained_variance_-------------------------------------------\n')
    #     f.write(str(pca.explained_variance_) + '\n')
    #     f.write('singular_values_-------------------------------------------\n')
    #     f.write(str(pca.singular_values_) + '\n')
    #     f.write('components_-------------------------------------------\n')
    #     f.write(str(pca.components_) + '\n')
    #     f.write('n_components_-------------------------------------------\n')
    #     f.write(str(pca.n_components_) + '\n')
    #     f.write('n_samples_-------------------------------------------\n')
    #     f.write(str(pca.n_samples_) + '\n')
    #     f.write('mean_-------------------------------------------\n')
    #     f.write(str(pca.mean_) + '\n')
    #     f.write('noise_variance_-------------------------------------------\n')
    #     f.write(str(pca.noise_variance_) + '\n')
    #     f.write('whiten-------------------------------------------\n')
    #     f.write(str(pca.whiten) + '\n')
    #     f.write('copy-------------------------------------------\n')
    #     f.write(str(pca.copy) + '\n')
    #     f.close()
    #
    # with open('result/pca/'+ie+'_incre_pca.txt', 'w') as f:
    #     # write incre_pca
    #     f.write('Incremental PCA-------------------------------------------\n')
    #     f.write('explained_variance_ratio_-------------------------------------------\n')
    #     f.write(str(ipca.explained_variance_ratio_) + '\n')
    #     f.write('explained_variance_-------------------------------------------\n')
    #     f.write(str(ipca.explained_variance_) + '\n')
    #     f.write('singular_values_-------------------------------------------\n')
    #     f.write(str(ipca.singular_values_) + '\n')
    #     f.write('components_-------------------------------------------\n')
    #     f.write(str(ipca.components_) + '\n')
    #     f.write('n_components_-------------------------------------------\n')
    #     f.write(str(ipca.n_components_) + '\n')
    #     f.write('mean_-------------------------------------------\n')
    #     f.write(str(ipca.mean_) + '\n')
    #     f.write('noise_variance_-------------------------------------------\n')
    #     f.write(str(ipca.noise_variance_) + '\n')
    #     f.write('whiten-------------------------------------------\n')
    #     f.write(str(ipca.whiten) + '\n')
    #     f.write('copy-------------------------------------------\n')
    #     f.write(str(ipca.copy) + '\n')
    #     f.close()

    return X_pca


def pca_mixture():
    pass

