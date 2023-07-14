from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def kmeans_model(X, y, n_clusters, seed, n_init=10):
    # Ensure data is float64
    X = X.astype(np.float64)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit(X)
    y_pred = kmeans.labels_

    # Print the accuracy
    print("\nKMeans Accuracy: ")
    print(accuracy_score(y, y_pred))

    # Decision boundary plot
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float64))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.6, style="whitegrid")
    cmap = plt.cm.YlGnBu
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=50, edgecolors='k')
    plt.title('KMeans Decision Boundary', fontsize=28)
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

    return kmeans, y, y_pred

