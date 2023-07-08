import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')

def tsne_implementation_all(data, labels, n_components=2):
    result_dir = 'result/tsne'
    X_embedded = TSNE(n_components=n_components).fit_transform(data)
    tsneDf = pd.DataFrame(data=X_embedded, columns=['TSNE1', 'TSNE2'])
    # add label to tsneDf
    tsneDf['label'] = labels
    # save
    tsneDf.to_csv(result_dir + '/all_tsne_to2d.csv', index=False)
    # use X_embedded and labels to plot
    colors = ["navy", "turquoise", "darkorange", "red", "green"]
    plt.figure(figsize=(9, 9))
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4],
                                        ['PE', 'PLA', 'PMMA', 'PS', 'UD']):
        plt.scatter(
            X_embedded[labels == i, 0],
            X_embedded[labels == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )
    plt.title("t-SNE of Nano-plastic dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-70, 70, -70, 70])
    # save
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

    # grid search
    # Set the possible values of perplexity and learning rate
    # perplexities = [10, 30, 50, 70]
    # learning_rates = [10, 100, 200, 500]
    # grid_search_dir = 'result/tsne/grid_search/all'
    #
    # # Apply grid search
    # for perplexity in perplexities:
    #     for learning_rate in learning_rates:
    #         # Apply t-SNE with current parameter values
    #         tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
    #         X_embedded = tsne.fit_transform(data_reference)
    #
    #         # Plot the result
    #         plt.figure(figsize=(9, 9))
    #         colors = ["navy", "turquoise", "darkorange", "red", "green"]
    #         for color, i, target_name in zip(colors, [0, 1, 2, 3, 4],
    #                                          ['PE', 'PLA', 'PMMA', 'PS', 'UD']):
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

def tsne_implementation_udexcluded(data, labels, n_components=2):
    result_dir = 'result/tsne'
    X_embedded = TSNE(n_components=n_components, perplexity=30).fit_transform(data)
    tsneDf = pd.DataFrame(data=X_embedded, columns=['TSNE1', 'TSNE2'])
    # add label to tsneDf
    tsneDf['label'] = labels
    # save
    tsneDf.to_csv(result_dir + '/udexcluded_tsne_to2d.csv', index=False)
    # use X_embedded and labels to plot
    colors = ["navy", "turquoise", "darkorange", "red"]
    plt.figure(figsize=(9, 9))
    for color, i, target_name in zip(colors, [0, 1, 2, 3],
                                        ['PE', 'PLA', 'PMMA', 'PS']):
        plt.scatter(
            X_embedded[labels == i, 0],
            X_embedded[labels == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )
    plt.title("t-SNE of Nano-plastic dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-70, 70, -70, 70])
    # save
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

    # grid search
    # Set the possible values of perplexity and learning rate
    perplexities = [10, 30, 50, 70]
    learning_rates = [10, 100, 200, 500]
    grid_search_dir = 'result/tsne/grid_search/udexcluded'

    # Apply grid search
    for perplexity in perplexities:
        for learning_rate in learning_rates:
            # Apply t-SNE with current parameter values
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
            X_embedded = tsne.fit_transform(data)

            # Plot the result
            plt.figure(figsize=(9, 9))
            colors = ["navy", "turquoise", "darkorange", "red"]
            for color, i, target_name in zip(colors, [0, 1, 2, 3],
                                             ['PE', 'PLA', 'PMMA', 'PS']):
                plt.scatter(
                    X_embedded[labels == i, 0],
                    X_embedded[labels == i, 1],
                    color=color,
                    lw=2,
                    label=target_name,
                )
            plt.title(f'Perplexity: {perplexity}, Learning Rate: {learning_rate}')
            plt.legend(loc="best", shadow=False, scatterpoints=1)
            plt.savefig(grid_search_dir + f'/perplexity_{perplexity}_learning_rate_{learning_rate}.png')
            plt.close()

    return X_embedded


# def tsne_dim_reduction(data_reference, labels, n_components=2):
#     """
#     This function is used to reduce the dimension of data_reference using t-SNE.
#     """
#     data_reference = StandardScaler().fit_transform(data_reference)
#     tsne = TSNE(n_components=n_components)
#     tsneComponents = tsne.fit_transform(data_reference)
#     tsneDf = pd.DataFrame(data_reference=tsneComponents, columns=['TSNE1', 'TSNE2'])
#
#     # Add the labels back into the DataFrame
#     tsneDf['label'] = labels
#
#     return tsneDf
#
# def tsne_visualization(tsneDf, label):
#     """
#     This function is used to visualize the data_reference after t-SNE.
#     """
#     plt.figure(figsize=(8, 8))
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=14)
#     plt.xlabel('TSNE1', fontsize=15)
#     plt.ylabel('TSNE2', fontsize=15)
#     plt.title('t-SNE of ' + label, fontsize=20)
#     targets = ['PE', 'PS', 'PS_PE']
#     colors = ["navy", "turquoise", "darkorange"]
#     for target, color in zip(targets, colors):
#         indicesToKeep = tsneDf['label'] == target
#         plt.scatter(tsneDf.loc[indicesToKeep, 'TSNE1'],
#                     tsneDf.loc[indicesToKeep, 'TSNE2'],
#                     c=color,
#                     s=50)
#     plt.xlim([tsneDf['TSNE1'].min(), tsneDf['TSNE1'].max()])
#     plt.ylim([tsneDf['TSNE2'].min(), tsneDf['TSNE2'].max()])
#     plt.legend(targets, prop={'size': 15})
#
#     # save
#     plt.savefig('t-SNE of ' + label + '.png')
#
#
# def tsne_visualization(tsneDf, label):
#     """
#     This function is used to visualize the data_reference after t-SNE.
#     """
#     plt.figure(figsize=(8, 8))
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=14)
#     plt.xlabel('TSNE1', fontsize=15)
#     plt.ylabel('TSNE2', fontsize=15)
#     plt.title('t-SNE of ' + label, fontsize=20)
#     targets = ['PE', 'PS', 'PS_PE']
#     colors = ["navy", "turquoise", "darkorange"]
#     for target, color in zip(targets, colors):
#         indicesToKeep = tsneDf['label'] == target
#         print(f"Plotting points for target {target}:")   # Debugging line
#         print(tsneDf.loc[indicesToKeep, ['TSNE1', 'TSNE2']])   # Debugging line
#         plt.scatter(tsneDf.loc[indicesToKeep, 'TSNE1'],
#                     tsneDf.loc[indicesToKeep, 'TSNE2'],
#                     c=color,
#                     s=50)
#     plt.xlim([tsneDf['TSNE1'].min(), tsneDf['TSNE1'].max()])
#     plt.ylim([tsneDf['TSNE2'].min(), tsneDf['TSNE2'].max()])
#     plt.legend(targets, prop={'size': 15})
#
#     # save
#     plt.savefig('t-SNE of ' + label + '.png')
