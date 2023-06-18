import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay

class isoForest():
    def __init__(self, X_pca, y, result_path):
        self.X = X_pca
        self.y = y
        self.path = result_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vis_X = None
        self.vis_y = None
        self.handles = None
        self.clf = None
        rng = np.random.RandomState(42)
        self.preprocess()

    def preprocess(self):
        self.data_split()
        self.preprocess_vis()

    def data_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.02)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def preprocess_vis(self):
        self.vis_X = self.X
        # replace 0, 1, 2, 3, with 1, and 4 with -1 use lambda
        self.vis_y = np.array(list(map(lambda x: 1 if x < 4 else -1, self.y)))


    def pre_visualization(self):
        X = self.vis_X
        y = self.vis_y
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        handles, labels = scatter.legend_elements()
        self.handles = handles
        plt.axis("square")
        plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
        plt.title("Gaussian inliers with \nuniformly distributed outliers")
        plt.show()
        # save
        # fig.savefig(self.path + 'isoforest_visualization.png')

    def train(self):
        self.clf = IsolationForest().fit(self.X_train)

    def plot_discrete(self):
        X = self.vis_X
        disp = DecisionBoundaryDisplay.from_estimator(
            self.clf,
            X,
            response_method="predict",
            alpha=0.5,
        )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=self.vis_y, s=20, edgecolor="k")
        disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
        plt.axis("square")
        plt.legend(handles=self.handles, labels=["outliers", "inliers"], title="true class")
        plt.show()

    def plot_path_length_decision_boundary(self):
        disp = DecisionBoundaryDisplay.from_estimator(
            self.clf,
            self.X,
            response_method="decision_function",
            alpha=0.5,
        )
        disp.ax_.scatter(self.X[:, 0], self.X[:, 1], c=self.vis_y, s=20, edgecolor="k")
        disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
        plt.axis("square")
        plt.legend(handles=self.handles, labels=["outliers", "inliers"], title="true class")
        plt.colorbar(disp.ax_.collections[1])
        plt.show()

    def predict(self):
        out = self.clf.predict(self.X_train)
        print('\nout', out)
        self.y_train = np.array(list(map(lambda x: 1 if x < 4 else -1, self.y_train)))
        print('\nself.y_train', self.y_train)
        print("Accuracy: ")
        print(np.sum(out == self.y_train) / len(self.y_train))
