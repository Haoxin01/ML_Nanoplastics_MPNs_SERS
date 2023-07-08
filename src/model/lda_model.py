import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def lda_all(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data_reference class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''
    lda_model = LinearDiscriminantAnalysis(n_components=n_dim)
    lda_model.fit(data, target)
    data_ndim = lda_model.transform(data)

    # draw using data_ndim and target
    plt.figure(figsize=(9, 9))
    colors = ["navy", "turquoise", "darkorange", "red", "green", "blue", "yellow", "black"]
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7],
                                        ['PE', 'PLA', 'PMMA', 'PS', 'PS_PE', 'PS_PLA', 'PA_PMMA', 'UD']):
            plt.scatter(
                data_ndim[target == i, 0],
                data_ndim[target == i, 1],
                color=color,
                lw=2,
                label=target_name,
            )
    plt.title('LDA of Nano-plastic dataset')
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.savefig('result/lda/LDA all of Nano-plastic.png')
    plt.close()

    # store information to txt
    with open('result/lda/lda_all_info.txt', 'w') as f:
        # write
        f.write('LDA of Nano-plastic dataset\n')
        f.write('n_dim: ' + str(n_dim) + '\n')
        f.write('explained_variance_ratio_: ' + str(lda_model.explained_variance_ratio_) + '\n')
        f.write('coef_: ' + str(lda_model.coef_) + '\n')
        f.write('intercept_: ' + str(lda_model.intercept_) + '\n')
        f.write('means_: ' + str(lda_model.means_) + '\n')
        f.write('priors_: ' + str(lda_model.priors_) + '\n')
        f.write('scalings_: ' + str(lda_model.scalings_) + '\n')
        f.write('xbar_: ' + str(lda_model.xbar_) + '\n')
        f.write('classes_: ' + str(lda_model.classes_) + '\n')
        f.close()

    return data_ndim

def lda_udexcluded(data, target, n_dim):
    lda_model = LinearDiscriminantAnalysis(n_components=n_dim)
    lda_model.fit(data, target)
    data_ndim = lda_model.transform(data)

    # draw using data_ndim and target
    plt.figure(figsize=(9, 9))
    colors = ["navy", "turquoise", "darkorange", "red", "green", "blue", "yellow"]
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6],
                                        ['PE', 'PLA', 'PMMA', 'PS', 'PS_PE', 'PS_PLA', 'PA_PMMA']):
            plt.scatter(
                data_ndim[target == i, 0],
                data_ndim[target == i, 1],
                color=color,
                lw=2,
                label=target_name,
            )
    plt.title('LDA of Nano-plastic dataset')
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.savefig('result/lda/LDA udexcluded of Nano-plastic.png')
    plt.close()

    # store information to txt
    with open('result/lda/lda_udexcluded_info.txt', 'w') as f:
        # write
        f.write('LDA of Nano-plastic dataset\n')
        f.write('n_dim: ' + str(n_dim) + '\n')
        f.write('explained_variance_ratio_: ' + str(lda_model.explained_variance_ratio_) + '\n')
        f.write('coef_: ' + str(lda_model.coef_) + '\n')
        f.write('intercept_: ' + str(lda_model.intercept_) + '\n')
        f.write('means_: ' + str(lda_model.means_) + '\n')
        f.write('priors_: ' + str(lda_model.priors_) + '\n')
        f.write('scalings_: ' + str(lda_model.scalings_) + '\n')
        f.write('xbar_: ' + str(lda_model.xbar_) + '\n')
        f.write('classes_: ' + str(lda_model.classes_) + '\n')
        f.close()


    return data_ndim