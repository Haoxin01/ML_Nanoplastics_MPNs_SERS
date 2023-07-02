from model import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model import device
from src.util.data_decoder import batch_data_decoder, data_concat
import numpy as np


# load model
model = MLP().to(device)

# load data
# prepare data
source_data_dir = 'sample_data'
data = batch_data_decoder(source_data_dir)
X, y = data_concat(data, if_shuffle=True, shuffle_seed=0)
label_map = {0: 'PE', 1: 'PMMA', 2: 'PS', 3: 'PLA', 4: 'UD'}



# load model
model.load_state_dict(torch.load('model.pt'))

# predict
model.eval()
with torch.no_grad():
    pred = model(X_test)
    pred_probab = nn.Softmax(dim=1)(pred)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

# accuracy
correct = (y_pred == Y_test).sum().item()
accuracy = correct / len(Y_test)
print(f"Accuracy: {accuracy:.3f}")

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(Y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# precision, recall, f1-score
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

# ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# Binarize the output
y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3, 4])
n_classes = y_test.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
