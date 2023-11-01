import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from custom_utils import AlzheimerDataset
from torch.utils.data import DataLoader

import pandas as pd


# Pretrained Model 
model = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.require_grad = False

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1000, bias=True),
    nn.Linear(in_features=1000, out_features=4, bias=True)
)


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
BATCH_SIZE = 16
NUM_CLASSES = 4
lr = 1e-4

# loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(params=model.parameters(), lr=lr)

model.to(device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataframe = pd.read_csv("Dataset/train.csv")
dataset = AlzheimerDataset(dataframe, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


# Training loop
for epoch in range(EPOCHS):
    for img, label in dataloader:
        # print(img.shape)
        # break
        y_pred = model(img)
        loss = criterion(y_pred, label)

        print(f"EPOCHS={epoch}/{EPOCHS}\tLOSS={loss}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
