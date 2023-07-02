import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def tsne_dim_reduction(data, labels, n_components=2):
    """
    This function is used to reduce the dimension of data using t-SNE.
    """
    data = StandardScaler().fit_transform(data)
    tsne = TSNE(n_components=n_components)
    tsneComponents = tsne.fit_transform(data)
    tsneDf = pd.DataFrame(data=tsneComponents, columns=['TSNE1', 'TSNE2'])

    # Add the labels back into the DataFrame
    tsneDf['label'] = labels

    return tsneDf

def tsne_visualization(tsneDf, label):
    """
    This function is used to visualize the data after t-SNE.
    """
    plt.figure(figsize=(8, 8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('TSNE1', fontsize=15)
    plt.ylabel('TSNE2', fontsize=15)
    plt.title('t-SNE of ' + label, fontsize=20)
    targets = ['PE', 'PS', 'PS_PE']
    colors = ["navy", "turquoise", "darkorange"]
    for target, color in zip(targets, colors):
        indicesToKeep = tsneDf['label'] == target
        plt.scatter(tsneDf.loc[indicesToKeep, 'TSNE1'],
                    tsneDf.loc[indicesToKeep, 'TSNE2'],
                    c=color,
                    s=50)
    plt.xlim([tsneDf['TSNE1'].min(), tsneDf['TSNE1'].max()])
    plt.ylim([tsneDf['TSNE2'].min(), tsneDf['TSNE2'].max()])
    plt.legend(targets, prop={'size': 15})

    # save
    plt.savefig('t-SNE of ' + label + '.png')


def tsne_visualization(tsneDf, label):
    """
    This function is used to visualize the data after t-SNE.
    """
    plt.figure(figsize=(8, 8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('TSNE1', fontsize=15)
    plt.ylabel('TSNE2', fontsize=15)
    plt.title('t-SNE of ' + label, fontsize=20)
    targets = ['PE', 'PS', 'PS_PE']
    colors = ["navy", "turquoise", "darkorange"]
    for target, color in zip(targets, colors):
        indicesToKeep = tsneDf['label'] == target
        print(f"Plotting points for target {target}:")   # Debugging line
        print(tsneDf.loc[indicesToKeep, ['TSNE1', 'TSNE2']])   # Debugging line
        plt.scatter(tsneDf.loc[indicesToKeep, 'TSNE1'],
                    tsneDf.loc[indicesToKeep, 'TSNE2'],
                    c=color,
                    s=50)
    plt.xlim([tsneDf['TSNE1'].min(), tsneDf['TSNE1'].max()])
    plt.ylim([tsneDf['TSNE2'].min(), tsneDf['TSNE2'].max()])
    plt.legend(targets, prop={'size': 15})

    # save
    plt.savefig('t-SNE of ' + label + '.png')
