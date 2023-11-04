
# Forged Signature Classification

This project aims to build a classifier that can distinguish between genuine and forged signatures using one-shot learning method. One-shot learning is a technique that 
allows a machine learning model to learn from a single example of each class.

## Dataset

The dataset consists of 300 images of signatures, 150 genuine and 150 forged. The images are in grayscale and have a resolution of 300 x 200 pixels. 
The images are named according to the following convention:

Folder structure:
1.genuines : genuine signatures of 30 people, 5 sample each.
2.forged : forged signature of the same 30 people as in genuine folder, 5 sample each.

Naming of files : NFI-XXXYYZZZ
explaination    XXX - ID number of a person who has done the signature. 
		YY - Image smaple number.
		ZZZ - ID number of person whose signature is in photo. 

for example: NFI-00602023 is an image of signature of person number 023 done by person 006. This is a forged signature.
	     NFI-02103021 is an image of signature of person number 021 done by person 021. This is a genuine signature.   

## Model

The model used in this project is a Siamese network with contrastive loss. A Siamese network is a type of neural network that consists of two identical subnetworks 
that share the same weights and parameters. The subnetworks take two input images and produce two feature vectors that represent the images. The feature vectors are then 
compared by a distance function to measure the similarity between the images. The contrastive loss is a type of loss function that encourages the network to learn a low distance 
for similar images and a high distance for dissimilar images.

## Files

The files in this project are:

- dataloader.py: This file contains the code for loading and preprocessing the images from the dataset. It also defines the custom dataset and dataloader classes for PyTorch.
- models.py: This file contains the code for defining the Siamese network architecture and the contrastive loss function.
- preprocessing.py: This file contains the code for applying some image processing techniques to the images, such as resizing, cropping, thresholding, and noise removal.
- train.py: This file contains the code for training and testing the model on the dataset. It also saves the model checkpoints and plots the loss curves and accuracy scores.