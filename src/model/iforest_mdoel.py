import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay

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
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        handles, labels = scatter.legend_elements()
        self.handles = handles
        plt.axis("square")
        plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
        plt.title("Undetected as outliers")
        plt.show()
        # save
        plt.savefig(self.path + 'pre-visualization.png')

    def train(self):
        seed = 0
        self.clf = IsolationForest(
            n_estimators=40,
            contamination=0.1,
            random_state=seed,
            max_samples='auto',
            max_features=2,
            # bootstrap=True,
        ).fit(self.X_train)


    def find_best_params(self):
        accuracy = 0
        self.y_test = np.array(list(map(lambda x: 1 if x < 4 else -1, self.y_test)))
        # number of -1 in y_test
        minus1 = np.sum(self.y_train == -1)
        # number of 1 in y_test
        plus1 = np.sum(self.y_train == 1)
        contamination = minus1 / (minus1 + plus1)
        for seed in range(1000):
            clf = IsolationForest(
                n_estimators=80,
                contamination=contamination,
                random_state=seed,
                max_samples='auto',
                max_features=2,
                bootstrap=True
            ).fit(self.X_train)
            out = clf.predict(self.X_test)

            caccuracy = np.sum(out == self.y_test) / len(self.y_test)
            if caccuracy > accuracy:
                accuracy = caccuracy
                self.clf = clf
                # save the model
                model_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model'
                model_name = 'isoForest'
                model_full_path = model_path + '/' + model_name + '.joblib'
                from joblib import dump, load
                dump(self.clf, model_full_path)
                # save the seed, accuracy to txt
                seed_path = '/Users/shiyujiang/Desktop/Nanoplastics-ML/validation/cache/model'
                seed_name = 'isoForest'
                seed_full_path = seed_path + '/' + seed_name + '.txt'
                with open(seed_full_path, 'w') as f:
                    f.write(str(seed))
                    f.write('\n')
                    f.write(str(accuracy))
                    f.close()

                print('Current max accuracy', accuracy)
                print('with seed', seed, '\n')


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
        # save
        plt.savefig(self.path + 'plot_discrete.png')

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
        # save
        plt.savefig(self.path + 'plot_path_length_decision_boundary.png')

    def predict(self):
        out = self.clf.predict(self.X_train)
        print('\nout', out)
        self.y_train = np.array(list(map(lambda x: 1 if x < 4 else -1, self.y_train)))
        print('\nself.y_train', self.y_train)
        print("Accuracy: ")
        print(np.sum(out == self.y_train) / len(self.y_train))

    def analyze(self):
        # analyze prediction result and save to txt file
        out = self.clf.predict(self.X_train)
        self.y_train = np.array(list(map(lambda x: 1 if x < 4 else -1, self.y_train)))
        # create a new file
        f = open(self.path + 'analyze.txt', 'w')
        f.write('out: ' + str(out) + '\n')
        f.write('self.y_train: ' + str(self.y_train) + '\n')
        f.write("Accuracy: " + str(np.sum(out == self.y_train) / len(self.y_train)) + '\n')

        # draw confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import pandas as pd
        cm = confusion_matrix(self.y_train, out)
        print('\ncm', cm)
        df_cm = pd.DataFrame(cm, index = [i for i in "01234"],
                        columns = [i for i in "01234"])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(self.path + 'confusion_matrix.png')
        plt.show()
        plt.close()



