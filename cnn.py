import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from tqdm import tqdm
import os

from load_data import load_datasets

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN = False


def load_model():
    model = CNN().to(DEVICE)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pickle')
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 16, 3)
        self.conv1 = nn.Conv2d(16, 32, 3)
        # self.conv2 = nn.Conv2d(64, 128, 3)
        # self.conv3 = nn.Conv2d(128, 256, 3)
        # self.conv4 = nn.Conv2d(256, 512, 3)
        # self.conv5 = nn.Conv2d(512, 512, 3)

        self.fc1 = nn.Linear(32 * 5 * 5, 16)
        self.fc_language = nn.Linear(16, 3)
        self.fc_numeral = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        language = self.fc_language(x)
        numeral = self.fc_numeral(x)
        return language, numeral


class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(np.expand_dims(X, axis=1))
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(model, dataloader, num_epochs) -> CNN:

    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-3
    print(f'lr = {lr}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dataloader, total=len(dataloader)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()

            optimizer.zero_grad()

            language, numeral = model(image)
            loss = loss_fn(language, label[:, 0].long()) + loss_fn(numeral, label[:, 1].long())
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        print(np.mean(losses))
        torch.save(model.state_dict(), "model.pickle")

    return model


def main():
    # load the datasets
    datasets = load_datasets()

    # get the train and test data
    X_train = np.concatenate([dataset.get_X_train() for dataset in datasets.values()])
    y_train = np.concatenate([dataset.get_y_train() for dataset in datasets.values()])
    X_test = np.concatenate([dataset.get_X_test() for dataset in datasets.values()])
    y_test = np.concatenate([dataset.get_y_test() for dataset in datasets.values()])

    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(PyTorchDataset(X_train, y_train), batch_size=32, num_workers=8)
    test_loader = torch.utils.data.DataLoader(PyTorchDataset(X_test, y_test), batch_size=32, num_workers=8)

    if TRAIN:
        model = CNN().to(DEVICE)
        model = train(model, train_loader, num_epochs=30)
    else:
        model = load_model()
        y_pred = list()
        for image, label in tqdm(test_loader, total=len(test_loader)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()

            with torch.no_grad():
                language_pred, numeral_pred = model(image.to(DEVICE))

            language_pred = torch.softmax(language_pred, dim=1)
            numeral_pred = torch.softmax(numeral_pred, dim=1)

            language_pred = list(torch.argmax(language_pred, dim=1).numpy())
            numeral_pred = list(torch.argmax(numeral_pred, dim=1).numpy())

            y_pred.extend(list(zip(language_pred, numeral_pred)))


if __name__ == "__main__":
    main()
