from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def hca_model(X, y, n_clusters):
    # Ensure data is float64
    X = X.astype(np.float64)

    # Perform AgglomerativeClustering
    hca = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    y_pred = hca.labels_

    # Print the accuracy
    print("\nHCA Accuracy: ")
    print(accuracy_score(y, y_pred))

    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.6, style="whitegrid")
    cmap = plt.cm.YlGnBu
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=50, edgecolors='k')
    plt.title('HCA Clusters', fontsize=28)
    plt.xlabel('First Principal Component', fontsize=24, weight='bold')
    plt.ylabel('Second Principal Component', fontsize=24, weight='bold')

    # Define class labels and assign them to the legend
    class_labels = ['PE', 'PLA', 'PMMA', 'PS']
    handles, _ = scatter.legend_elements()
    legend1 = plt.legend(handles, class_labels, title="Classes", fontsize=20, loc='upper right')
    plt.setp(legend1.get_title(), fontsize='xx-large')

    # Ensure the plot is displayed correctly with all labels visible
    plt.tight_layout()
    plt.show()

    return hca, y, y_pred
