import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from architecture import AlexNet

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the transform using transforms.Compose() and apply transforms.ToTensor() to convert images into tensors
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# Load the TinyImageNet dataset (training set) and make sure to apply the defined transform
trainset = datasets.ImageFolder(root='./tiny-imagenet-200/train',
                                transform=transform)


# Create DataLoader for the training dataset
trainloader = DataLoader(trainset,
                         batch_size=64,
                         shuffle=True)

# Create an instance of the neural network
model = AlexNet().to(device)


# Define the Loss function 
criterion = nn.CrossEntropyLoss()


# Define the optimizer using Adam optimizer with Learning rate = 0.0003
optimizer = optim.Adam(model.parameters(), lr=0.0003)


# Implement the training loop
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")
