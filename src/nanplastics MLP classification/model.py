import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, intake_dim, num_classes=5):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(intake_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        # Apply initialization
        nn.init.kaiming_uniform_(self.linear_relu_stack[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear_relu_stack[2].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear_relu_stack[5].weight, nonlinearity='relu')

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# model = MLP().to(device)
# print(model)

loss_fn = nn.CrossEntropyLoss()

epochs = 1000


# X = torch.rand(1, 10, device=device)
# logits = model(X)
# print('shape of logits: ', logits.shape)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")