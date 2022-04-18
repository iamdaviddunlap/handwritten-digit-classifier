import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import os

from evaluate import evaluate
from load_data import load_dataloaders
from enum import Enum

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN = True

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def load_model():
    model = MODEL.to(DEVICE)
    model_path = os.path.join(os.path.dirname(__file__), PICKLE_PATH)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        return None


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 16, 3)
        self.conv1 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 5 * 5, 16)
        self.fc_language = nn.Linear(16, 3)
        self.fc_numeral = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        language = self.fc_language(x)
        numeral = self.fc_numeral(x)
        return language, numeral


class TransferCNN(nn.Module):
    class PretrainedModel(Enum):
        RESNET = 1
        EFFICIENT_NET = 2

    def __init__(self, pretrained_model):
        super().__init__()
        self.model_type = pretrained_model
        if pretrained_model == self.PretrainedModel.RESNET:
            self.pretrained_model = models.resnet18(pretrained=True)
            self.pretrained_model.conv1 = nn.Conv2d(1, 64, 3)
            self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, 16)
        else:
            self.pretrained_model = models.efficientnet_b0(pretrained=True)
            self.pretrained_model.features[0][0] = nn.Conv2d(1, 32, 3)
            self.pretrained_model.classifier[-1] = nn.Linear(self.pretrained_model.classifier[-1].in_features, 16)
        self.fc_language = nn.Linear(16, 3)
        self.fc_numeral = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pretrained_model(x)
        language = self.fc_language(x)
        numeral = self.fc_numeral(x)
        return language, numeral


def train(model, dataloader, num_epochs):

    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-3
    print(f'lr = {lr}')
    model.train()
    if isinstance(MODEL, CNN):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.pretrained_model.parameters(), lr=lr)

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
        torch.save(model.state_dict(), PICKLE_PATH)


def predict(model, images):
    images = images.to(DEVICE).float()

    with torch.no_grad():
        language_pred, numeral_pred = model(images.to(DEVICE))

    language_pred = torch.softmax(language_pred, dim=1)
    numeral_pred = torch.softmax(numeral_pred, dim=1)

    language_pred = list(torch.argmax(language_pred, dim=1).cpu().detach().numpy())
    numeral_pred = list(torch.argmax(numeral_pred, dim=1).cpu().detach().numpy())

    return list(zip(language_pred, numeral_pred))


def main():
    train_loader, test_loader = load_dataloaders()

    LOAD_MODEL = False

    if TRAIN:
        if LOAD_MODEL:
            model = load_model()
            if not model:
                model = MODEL
        else:
            model = MODEL
        model.to(DEVICE)
        train(model, train_loader, num_epochs=30)
    else:
        model = load_model()
        if not model:
            raise FileNotFoundError('There is no model file to load')

        y_pred = list()
        for images, _ in tqdm(test_loader, total=len(test_loader)):
            y_pred.extend(predict(model, images))

        evaluate(y_true=test_loader.dataset.y.numpy(), y_pred=np.array(y_pred))


if __name__ == "__main__":
    # MODEL = TransferCNN(TransferCNN.PretrainedModel.RESNET)
    MODEL = CNN()

    if isinstance(MODEL, CNN):
        pickle_name = 'model.pickle'
    elif MODEL.model_type == TransferCNN.PretrainedModel.RESNET:
        pickle_name = 'model_resnet.pickle'
    elif MODEL.model_type == TransferCNN.PretrainedModel.EFFICIENT_NET:
        pickle_name = 'model_efficient_net.pickle'

    PICKLE_PATH = os.path.join(MODELS_DIR, pickle_name)

    main()
