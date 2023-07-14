import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib as mpl
import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg')

# Set global matplotlib parameters
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 24

def tsne_implementation_all(data, labels, n_components=2):
    # Convert your data to a numpy array if it's not already
    data = np.array(data)

    result_dir = 'result/tsne'
    X_embedded = TSNE(n_components=n_components,
                      perplexity=70,
                      learning_rate=10,
                      ).fit_transform(data)
    tsneDf = pd.DataFrame(data=X_embedded, columns=['TSNE1', 'TSNE2'])
    # add label to tsneDf
    tsneDf['label'] = labels
    # save
    tsneDf.to_csv(result_dir + '/all_tsne_to2d.csv', index=False)
    # use X_embedded and labels to plot
    colors = sns.color_palette("Set2", 5)
    plt.figure(figsize=(10, 10))
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4],
                                        ['PE', 'PLA', 'PMMA', 'PS', 'UD']):
        plt.scatter(
            X_embedded[labels == i, 0],
            X_embedded[labels == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel("First Principal Component", fontsize=28, weight='bold')
    plt.ylabel("Second Principal Component", fontsize=28, weight='bold')
    plt.title("t-SNE of Nano-plastic dataset", fontsize=28, weight='bold')
    plt.legend(loc="upper right", shadow=False, scatterpoints=1, fontsize=24)
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])

    plt.savefig(result_dir + '/all_tsne_to2d.png')
    plt.close()

    with open(result_dir + '/all_tsne_to2d.txt', 'w') as f:
        # write
        f.write('t-SNE-------------------------------------------\n')
        f.write('n_components_-------------------------------------------\n')
        f.write(str(n_components) + '\n')
        f.write('perplexity_-------------------------------------------\n')
        f.write(str(30) + '\n')
        f.write('early_exaggeration_-------------------------------------------\n')
        f.write(str(12) + '\n')
        f.write('learning_rate_-------------------------------------------\n')
        f.write(str(200) + '\n')
        f.write('n_iter_-------------------------------------------\n')
        f.write(str(1000) + '\n')
        f.write('n_iter_without_progress_-------------------------------------------\n')
        f.write(str(300) + '\n')
        f.write('min_grad_norm_-------------------------------------------\n')
        f.write(str(1e-07) + '\n')
        f.write('metric_-------------------------------------------\n')
        f.write(str('euclidean') + '\n')
        f.write('init_-------------------------------------------\n')
        f.write(str('random') + '\n')
        f.write('verbose_-------------------------------------------\n')
        f.write(str(0) + '\n')
        f.close()
    return X_embedded

def tsne_implementation_udexcluded(data, labels, n_components=2):
    # Convert your data to a numpy array if it's not already
    data = np.array(data)

    result_dir = 'result/tsne'
    X_embedded = TSNE(n_components=n_components,
                      perplexity=70,
                      learning_rate=10,
                      ).fit_transform(data)
    tsneDf = pd.DataFrame(data=X_embedded, columns=['TSNE1', 'TSNE2'])
    # add label to tsneDf
    tsneDf['label'] = labels
    # save
    tsneDf.to_csv(result_dir + '/all_tsne_to2d.csv', index=False)
    # use X_embedded and labels to plot
    colors = sns.color_palette("Set2", 5)
    plt.figure(figsize=(10, 10))
    for color, i, target_name in zip(colors, [0, 1, 2, 3],
                                     ['PE', 'PLA', 'PMMA', 'PS']):
        plt.scatter(
            X_embedded[labels == i, 0],
            X_embedded[labels == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )

    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel("First Principal Component", fontsize=28, weight='bold')
    plt.ylabel("Second Principal Component", fontsize=28, weight='bold')
    plt.title("t-SNE of Nano-plastic dataset", fontsize=28, weight='bold')
    plt.legend(loc="upper right", shadow=False, scatterpoints=1, fontsize=24)
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.savefig(result_dir + '/udexcluded_tsne_to2d.png')
    plt.close()

    # record information
    with open(result_dir + '/udexcluded_tsne_to2d.txt', 'w') as f:
        # write
        f.write('t-SNE-------------------------------------------\n')
        f.write('n_components_-------------------------------------------\n')
        f.write(str(n_components) + '\n')
        f.write('perplexity_-------------------------------------------\n')
        f.write(str(30) + '\n')
        f.write('early_exaggeration_-------------------------------------------\n')
        f.write(str(12) + '\n')
        f.write('learning_rate_-------------------------------------------\n')
        f.write(str(200) + '\n')
        f.write('n_iter_-------------------------------------------\n')
        f.write(str(1000) + '\n')
        f.write('n_iter_without_progress_-------------------------------------------\n')
        f.write(str(300) + '\n')
        f.write('min_grad_norm_-------------------------------------------\n')
        f.write(str(1e-07) + '\n')
        f.write('metric_-------------------------------------------\n')
        f.write(str('euclidean') + '\n')
        f.write('init_-------------------------------------------\n')
        f.write(str('random') + '\n')
        f.write('verbose_-------------------------------------------\n')
        f.write(str(0) + '\n')
        f.close()

    # # grid search
    # # Set the possible values of perplexity and learning rate
    # perplexities = [10, 30, 50, 70]
    # learning_rates = [10, 100, 200, 500]
    # grid_search_dir = 'result/tsne/grid_search/udexcluded'
    #
    # # Apply grid search
    # for perplexity in perplexities:
    #     for learning_rate in learning_rates:
    #         # Apply t-SNE with current parameter values
    #         tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
    #         X_embedded = tsne.fit_transform(data)
    #
    #         # Plot the result
    #         plt.figure(figsize=(9, 9))
    #         colors = ["navy", "turquoise", "darkorange", "red"]
    #         for color, i, target_name in zip(colors, [0, 1, 2, 3],
    #                                          ['PE', 'PLA', 'PMMA', 'PS']):
    #             plt.scatter(
    #                 X_embedded[labels == i, 0],
    #                 X_embedded[labels == i, 1],
    #                 color=color,
    #                 lw=2,
    #                 label=target_name,
    #             )
    #         plt.title(f'Perplexity: {perplexity}, Learning Rate: {learning_rate}')
    #         plt.legend(loc="best", shadow=False, scatterpoints=1)
    #         plt.savefig(grid_search_dir + f'/perplexity_{perplexity}_learning_rate_{learning_rate}.png')
    #         plt.close()

    return X_embedded


def tsne_mixture(data, labels, n_components: int, mixture_type: str):
    # Convert your data to a numpy array if it's not already
    data = np.array(data)

    X_embedded = TSNE(n_components=n_components,
                      perplexity=10,
                      learning_rate=10,
                      ).fit_transform(data)
    tsneDf = pd.DataFrame(data=X_embedded, columns=['TSNE1', 'TSNE2'])
    # add label to tsneDf
    tsneDf['label'] = labels
    # use X_embedded and labels to plot
    colors = sns.color_palette("Set2", 5)  # Use the same color palette
    plt.figure(figsize=(10, 10))
    for color, i, target_name in zip(colors, [0, 1, 2],
                                     ['Mixture', 'Mix1', 'Mix2']):
        plt.scatter(
            X_embedded[labels == i, 0],
            X_embedded[labels == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel("First Principal Component", fontsize=24, weight='bold')
    plt.ylabel("Second Principal Component", fontsize=24, weight='bold')
    plt.title("t-SNE of Nano-plastic mixture dataset", fontsize=24, weight='bold')
    plt.legend(loc="upper right", shadow=False, scatterpoints=1, fontsize=22)
    plt.xlim([-70, 70])
    plt.ylim([-70, 70])
    plt.show()
    plt.close()



    return X_embedded