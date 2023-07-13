import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from src.util.train_strategy import create_confusion_matrix, plot_confusion_matrix, compute_metrics

class isoForest():
    def __init__(self, X_pca, y, result_path):
        self.X = X_pca
        self.y = y
        self.path = result_path + '/' + 'outlier_detection' + '/'
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vis_X = None
        self.vis_y = None
        self.handles = None
        self.clf = None
        self.original_path = result_path
        self.preprocess()

    def preprocess(self):
        self.data_split()
        self.preprocess_vis()

    def data_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.05)
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
        fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size here.
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        handles, labels = scatter.legend_elements()
        self.handles = handles
        ax.axis("square")
        legend = ax.legend(handles=handles, labels=["outliers", "inliers"], title="true class", loc="upper right",
                           fontsize=22)
        plt.setp(legend.get_title(), fontsize=22)
        ax.set_title("Undetected as outliers", fontname='Times New Roman', fontsize=24)
        ax.set_xlabel("First Principal Component", fontname='Times New Roman', fontsize=24, weight='bold')
        ax.set_ylabel("Second Principal Component", fontname='Times New Roman', fontsize=24, weight='bold')

        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        ax.set_xlim([-2, 2])  # Set x-axis range here.
        ax.set_ylim([-2, 2])  # Set y-axis range here.

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontname('Times New Roman')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontname('Times New Roman')
        plt.savefig(self.path + 'pre-visualization.png')
        plt.show()


    def train(self):
        seed = 0
        self.clf = IsolationForest(
            n_estimators=40,
            contamination=0.3,
            random_state=seed,
            max_samples='auto',
            max_features=2,
            # bootstrap=True,
        ).fit(self.X_train)


    def find_best_params(self):
        # Define the parameter grid
        param_grid = {
            'n_estimators': [10, 50, 100, 200, 300],
            'contamination': [0.35],
        }

        # Create the GridSearchCV object
        grid_search = GridSearchCV(IsolationForest(max_samples='auto', max_features=2),
                                    param_grid,
                                    cv=5,
                                    scoring='accuracy')

        # Fit to the data and find the best parameters
        grid_search.fit(self.X_train, self.y_train)

        # Save the best model
        joblib.dump(grid_search.best_estimator_, 'isoForest_best_model.joblib')

        # Replace the existing model with the best model
        self.clf = grid_search.best_estimator_



    def plot_discrete(self, if_loop=False, loop_path=None):
        X = self.vis_X
        fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size here.
        disp = DecisionBoundaryDisplay.from_estimator(
            self.clf,
            X,
            ax=ax,
            response_method="predict",
            alpha=0.4,
        )
        ax.scatter(X[:, 0], X[:, 1], c=self.vis_y, s=20, edgecolor="k")
        disp.ax_.set_title("Binary decision boundary \nof IsolationForest", fontname='Times New Roman', fontsize=24)
        legend = disp.ax_.legend(handles=self.handles, labels=["outliers", "inliers"], title="true class",
                                 loc="upper right", fontsize=22)
        plt.setp(legend.get_title(), fontsize=22)

        disp.ax_.set_xlabel("First Principal Component", fontname='Times New Roman', fontsize=24, weight='bold')
        disp.ax_.set_ylabel("Second Principal Component", fontname='Times New Roman', fontsize=24, weight='bold')

        disp.ax_.xaxis.set_major_locator(ticker.MaxNLocator(5))
        disp.ax_.yaxis.set_major_locator(ticker.MaxNLocator(5))

        for tick in disp.ax_.xaxis.get_major_ticks():
            tick.label.set_fontname('Times New Roman')
        for tick in disp.ax_.yaxis.get_major_ticks():
            tick.label.set_fontname('Times New Roman')
        plt.savefig(self.path + 'plot_discrete.png')
        plt.show()

        if if_loop:
            plt.savefig(loop_path + '/isoForest_discrete.png')

    def plot_path_length_decision_boundary(self, if_loop=False, loop_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))  # Set figure size here.
        disp = DecisionBoundaryDisplay.from_estimator(
            self.clf,
            self.X,
            ax=ax,
            response_method="decision_function",
            alpha=0.5,
        )
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.vis_y, s=20, edgecolor="k")
        disp.ax_.set_title("Path length decision boundary \nof IsolationForest", fontname='Times New Roman',
                           fontsize=24)
        legend = disp.ax_.legend(handles=self.handles, labels=["outliers", "inliers"], title="true class",
                                 loc="upper right", fontsize=22)
        plt.setp(legend.get_title(), fontsize=22)

        disp.ax_.set_xlabel("First Principal Component", fontname='Times New Roman', fontsize=24, weight='bold')
        disp.ax_.set_ylabel("Second Principal Component", fontname='Times New Roman', fontsize=24, weight='bold')

        disp.ax_.xaxis.set_major_locator(ticker.MaxNLocator(5))
        disp.ax_.yaxis.set_major_locator(ticker.MaxNLocator(5))

        for tick in disp.ax_.xaxis.get_major_ticks():
            tick.label.set_fontname('Times New Roman')
        for tick in disp.ax_.yaxis.get_major_ticks():
            tick.label.set_fontname('Times New Roman')

        plt.colorbar(disp.ax_.collections[1])
        plt.savefig(self.path + 'plot_path_length_decision_boundary.png')
        plt.show()

        if if_loop:
            plt.savefig(loop_path + '/isoForest__decision_boundary.png')

    def predict(self):
        # Load the best model from disk
        self.clf = joblib.load('isoForest_best_model.joblib')
        out = self.clf.predict(self.X_train)
        print('\nout', out)
        self.y_train = np.array(list(map(lambda x: 1 if x < 4 else -1, self.y_train)))
        print('\nself.y_train', self.y_train)
        accuracy, recall, precision, f1 = compute_metrics(self.y_train, out)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1 Score:", f1)



    def analyze(self):
        # analyze prediction result and save to txt file
        y_pred = self.clf.predict(self.X_train)

        # Convert prediction labels to match the actual labels
        y_pred = np.array(list(map(lambda x: 1 if x == -1 else 0, y_pred)))
        y_true = np.array(list(map(lambda x: 1 if x == -1 else 0, self.y_train)))

        # create a new file
        f = open(self.path + 'analyze.txt', 'w')
        f.write('out: ' + str(y_pred) + '\n')
        f.write('self.y_train: ' + str(y_true) + '\n')

        # Compute metrics
        accuracy, recall, precision, f1 = compute_metrics(y_true, y_pred)
        f.write("Accuracy: " + str(accuracy) + '\n')
        f.write("Recall: " + str(recall) + '\n')
        f.write("Precision: " + str(precision) + '\n')
        f.write("F1 Score: " + str(f1) + '\n')

        # create confusion matrix
        cm = create_confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, labels=["inliers", "outliers"])

        f.close()



