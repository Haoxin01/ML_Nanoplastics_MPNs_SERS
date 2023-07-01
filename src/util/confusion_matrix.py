from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def create_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred, normalize='true')  # normalize on true labels
    return cm

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2%', xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def compute_metrics(true, pred):
    accuracy = accuracy_score(true, pred)
    recall = recall_score(true, pred, average='weighted')
    precision = precision_score(true, pred, average='weighted')
    f1 = f1_score(true, pred, average='weighted')
    return accuracy, recall, precision, f1