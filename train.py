import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from PIL import Image

from dataloader import SignatureContrastiveDataset, SignatureTripletDataset  # Import your custom dataset
from models import SiameseNetworkWithContrastiveLoss, SiameseTripletNet, EncoderNet  # Import your Siamese network model
from loss import ContrastiveLoss, TripletLoss


def calculate_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in data_loader:
            img1, img2, labels  = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1, img2)
            euc = F.pairwise_distance(output1, output2)
            predicted = (euc<2).float()
            # print("Out shape: ", output1.shape)
            # predicted = torch.sigmoid(euc)  # Convert to binary predictions (0 or 1)
            # predicted = (predicted>0.95).float()
            # print(euc)
            # print(predicted, labels)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def train_contrastive(model, train_data_loader, val_data_loader, num_epochs, device='cpu'):
    
    # Define loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (img1, img2, labels) in enumerate(train_data_loader):
            # Clear the gradients
            optimizer.zero_grad()
            img1, img2, labels  = img1.to(device), img2.to(device), labels.to(device)
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels.float())

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f'Batch Loss: {loss.item()}')
            val_accuracy = calculate_accuracy(siamese_net, val_data_loader, device)
            print(f'Validation Accuracy: {val_accuracy * 100}%')

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(train_data_loader)}')
    print('Training completed!')
    

def train_triplets(model_triplet, model_cont, train_data_loader, val_data_loader, num_epochs, device='cpu'):
    
    # Define loss function and optimizer
    criterion = TripletLoss(margin=2)
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (img1, img2, img3, labels) in enumerate(train_data_loader):
            # Clear the gradients
            
            optimizer.zero_grad()
            img1, img2, img3  = img1.to(device), img2.to(device), img3.to(device)
            # Forward pass
            emb1, emb2, emb3 = model_triplet(img1, img2, img3)
            
            loss = criterion(emb1, emb2, emb3)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f'Batch Loss: {loss.item()}')
            val_accuracy = calculate_accuracy(model_cont, val_data_loader, device)
            print(f'Validation Accuracy: {val_accuracy * 100}%')

        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(train_data_loader)}')
    print('Training completed!')
    
    
if __name__ == '__main__':
    
    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Initialize the dataset
    train_data_path = 'drive/MyDrive/sig_classifier/dataset/train_bin'
    val_data_path = 'drive/MyDrive/sig_classifier/dataset/val_bin'
    # dataset_train = SignatureContrastiveDataset(data_path=train_data_path)
    dataset_train = SignatureTripletDataset(data_path=train_data_path)
    dataset_val = SignatureContrastiveDataset(data_path=val_data_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Create data loader
    train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Initialize the Siamese network model
    cnn_encoder = EncoderNet()
    cnn_encoder.to(device)
    siamese_triplet_net = SiameseTripletNet(cnn_encoder)
    siamese_net = SiameseNetworkWithContrastiveLoss(cnn_encoder)

    siamese_triplet_net.to(device)
    siamese_net.to(device)
    # train_contrastive(siamese_net, train_data_loader, val_data_loader, num_epochs)
    train_triplets(siamese_triplet_net, siamese_net, train_data_loader, val_data_loader, num_epochs, device)
    