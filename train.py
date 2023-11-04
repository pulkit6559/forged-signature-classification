import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from PIL import Image

from dataloader import SignatureContrastiveDataset  # Import your custom dataset
from models import SiameseNetworkWithContrastiveLoss  # Import your Siamese network model
from loss import ContrastiveLoss

# Define hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Initialize the dataset
data_path = 'dataset/'
signature_dataset = SignatureContrastiveDataset(data_path=data_path)

# Create data loader
data_loader = DataLoader(signature_dataset, batch_size=batch_size, shuffle=True)

# Initialize the Siamese network model
siamese_net = SiameseNetworkWithContrastiveLoss()

# Define loss function and optimizer
criterion = ContrastiveLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (img1, img2, labels) in enumerate(data_loader):
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output1, output2 = siamese_net(img1, img2)
        loss = criterion(output1, output2, labels.float())

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f'Batch Loss: {loss.item()}')

    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(data_loader)}')

print('Training completed!')