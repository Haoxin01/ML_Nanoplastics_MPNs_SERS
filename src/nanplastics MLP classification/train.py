import pandas as pd
from src.util.data_decoder import batch_data_decoder, data_concat, shuffle
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import MLP, loss_fn, epochs
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def build_dataloader(X, y, batch_size, shuffle, if_drop_last):
    """
    This function is used to build dataloader.
    """
    # convert X and y to dataframe
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)
    # normalize X
    X_df = (X_df - X_df.mean()) / X_df.std()
    # concat X and y
    data_df = pd.concat([X_df, y_df], axis=1)
    # convert data_reference to numpy array
    data = data_df.to_numpy()
    # split data_reference into X and y
    X = data[:, :-1]
    y = data[:, -1]
    # convert X and y to tensor
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    # build dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=if_drop_last)
    return dataloader


# prepare data_reference
source_data_dir = 'sample_data'
data = batch_data_decoder(source_data_dir)
X, y = data_concat(data, if_shuffle=True, shuffle_seed=0)
label_map = {0: 'PE', 1: 'PMMA', 2: 'PS', 3: 'PLA', 4: 'UD'}

print('\n-----------Finished data_reference preparation-----------\n')

# build dataloader
train_loader = build_dataloader(X, y, batch_size=32, shuffle=True, if_drop_last=True)
print("train_loader:", train_loader)

print('\n-----------Finished dataloader preparation-----------\n')

model = MLP(intake_dim=4, num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(model)

print('\n-----------Finished model preparation-----------\n')

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        logits = model(X)
        probes = nn.Softmax(dim=1)(logits)
        # pred = torch.argmax(probes, dim=1)
        loss = loss_fn(probes, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item():.4f}")

print('\n-----------Finished training-----------\n')

# save model
torch.save(model.state_dict(), 'model.pth')
print("Saved PyTorch Model State to model.pth")