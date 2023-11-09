import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 45, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNetworkWithContrastiveLoss(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetworkWithContrastiveLoss, self).__init__()
        # Define the CNN layers
        self.encoder = encoder
        # # Output layer for contrastive loss
        # self.output_layer = nn.Linear(256, 2)  # 2 output units for binary classification (genuine or forged)

    # def forward_once(self, x):
    #     # Forward pass through the CNN
    #     x = self.cnn(x)
    #     # Flatten the tensor for the fully connected layers
    #     # print(x.shape)
    #     x = x.view(x.size()[0], -1)
    #     # Forward pass through the fully connected layers
    #     x = self.fc(x)
    #     return x

    def forward(self, input1, input2):
        # Forward pass of input 1
        output1 = self.encoder(input1)
        # Forward pass of input 2
        output2 = self.encoder(input2)
        return output1, output2
    
    def predict_similarity(output1, output2, threshold):
        # Calculate the Euclidean distance between the output vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Check if the distance is below the threshold
        similarity = euclidean_distance < threshold
        return similarity

class SiameseTripletNet(nn.Module):
    def __init__(self, encoder):
        super(SiameseTripletNet, self).__init__()

        self.encoder = encoder
    
    def forward(self, input1, input2, input3):
        output1 = self.encoder(input1)
        output2 = self.encoder(input2)
        output3 = self.encoder(input3)
        return output1, output2, output3
    
    
if __name__ == '__main__':
    input_size = (1, 1, 400, 200)  # Input image size
    dummy_image1 = np.random.rand(*input_size).astype(np.float32)
    dummy_image2 = np.random.rand(*input_size).astype(np.float32)
    
    siamese_net = SiameseNetworkWithContrastiveLoss()

    input1 = torch.from_numpy(dummy_image1)
    input2 = torch.from_numpy(dummy_image2)

    siamese_net.eval()

    output1, output2 = siamese_net(input1, input2)

    print("Output 1 size:", output1.size())
    print("Output 2 size:", output2.size())