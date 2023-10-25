import torch
from torchvision.transforms import transforms
from torch.utils.Dataset import Dataset
from torchvision.model import resnet50
from torchvision.model import ResNet50_Weights


# Pretrained Model 
model = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.require_grad = False

model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1000, bias=True),
    nn.Linear(in_features=1000, out_features=4, bias=True)
)


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
NUM_CLASSES = 4


model.to(device)


# loss function
criterion = nn.CrossEntropyLoss()


# Training loop
for epoch in range(EPOCHS):
    pass
