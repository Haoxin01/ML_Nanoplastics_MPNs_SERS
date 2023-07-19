import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def cross_validation(Xe_pca, ye, model):
    socres = cross_val_score(model, Xe_pca, ye, cv=8)
    print(f"Cross validation scores: {socres}")
    print(f"Average cross validation score: {np.mean(socres)}")
    print()


def search_best_model(Xe_pca, ye, model, model_param, labels, model_cache_path, variable_cache_path):
    max_accuracy = 0
    acc_list = []
    for seed in range(1000):
        svm_clf, svm_test, svm_pred = model(Xe_pca, ye, seed)
        svm_acc, svm_rec, svm_prec, svm_f1 = compute_metrics(svm_test, svm_pred)
        acc_list.append(svm_acc)
        if svm_acc > max_accuracy:
            max_accuracy = svm_acc
            best_seed = seed
            print(
                f"Best seed: {best_seed}, Best accuracy: {max_accuracy}, Best precision: {svm_prec}, Best recall: {svm_rec}, Best f1: {svm_f1}")
            cm = create_confusion_matrix(svm_test, svm_pred)
            plot_confusion_matrix(cm, labels)
            print(f"Confusion matrix: {cm}")
            print('\n\n')
            # model_save(svm_clf, f"svm_model_{best_seed}.pkl")
    avg_acc = np.mean(acc_list)
    print(f"Average accuracy: {avg_acc}")


def compute_metrics(true, pred):
    accuracy = accuracy_score(true, pred)
    recall = recall_score(true, pred, average='weighted')
    precision = precision_score(true, pred, average='weighted')
    f1 = f1_score(true, pred, average='weighted')
    return accuracy, recall, precision, f1


def create_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred, normalize='true')  # normalize on true labels
    return cm


def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2%', xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label", weight='bold')
    plt.ylabel("True Label", weight='bold')
    plt.show()


def model_save(model, filename):
    import pickle
    pickle.dump(model, open(filename, 'wb'))
    print(f"Model saved as {filename}")
