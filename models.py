import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SiameseNetworkWithContrastiveLoss(nn.Module):
    def __init__(self):
        super(SiameseNetworkWithContrastiveLoss, self).__init__()
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
        # # Output layer for contrastive loss
        # self.output_layer = nn.Linear(256, 2)  # 2 output units for binary classification (genuine or forged)

    def forward_once(self, x):
        # Forward pass through the CNN
        x = self.cnn(x)
        # Flatten the tensor for the fully connected layers
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # Forward pass through the fully connected layers
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # Forward pass of input 1
        output1 = self.forward_once(input1)
        # Forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
    
    def predict_similarity(output1, output2, threshold):
        # Calculate the Euclidean distance between the output vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Check if the distance is below the threshold
        similarity = euclidean_distance < threshold
        return similarity


# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         # Setting up the Sequential of CNN Layers
#         self.cnn1 = nn.Sequential( 
#         nn.Conv2d(1, 96, kernel_size=11,stride=1),
#         nn.ReLU(inplace=True),
#         nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#         nn.MaxPool2d(3, stride=2),

#         nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
#         nn.ReLU(inplace=True),
#         nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
#         nn.MaxPool2d(3, stride=2),
#         nn.Dropout2d(p=0.3),

#         nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(3, stride=2),
#         nn.Dropout2d(p=0.3),
#         )

#         # Defining the fully connected layers
#         self.fc1 = nn.Sequential(
#         # First Dense Layer
#         nn.Linear(264704, 1024),
#         nn.ReLU(inplace=True),
#         nn.Dropout2d(p=0.5),
#         # Second Dense Layer
#         nn.Linear(1024, 128),
#         nn.ReLU(inplace=True),
#         # Final Dense Layer
#         nn.Linear(128,2))

#     def forward_once(self, x):
#         # Forward pass 
#         output = self.cnn1(x)
#         print(output.size())
#         output = output.view(output.size()[0], -1)
#         output = self.fc1(output)
#         return output

#     def forward(self, input1, input2):
#         # forward pass of input 1
#         output1 = self.forward_once(input1)
#         # forward pass of input 2
#         output2 = self.forward_once(input2)
#         # returning the feature vectors of two inputs
#         return output1, output2
    
    
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